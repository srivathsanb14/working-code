"""Scoring a candidate: physics checks, nested fitting, held-out evaluation.

Three nested levels, mirroring how the model is used in a submission:
  outer   the LLM's structure (the candidate source)            - fixed here
  middle  PARAMS, global constants shared by all buildings      - fitted on TRAIN scenarios
  inner   THETA, per-scenario constants                         - fitted per scenario:
            on TRAIN jointly with PARAMS, using whole episodes;
            on VALIDATION / TEST from day 1 only, PARAMS frozen (exactly what a policy can do online)
The score is the open-loop error on days 2-3 of held-out scenarios, so a model
only wins if its structure transfers and its THETA is identifiable from one day.
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares

DAY = 288


# ----------------------------------------------------------------------------- parameters

class ParamMap:
    """Maps between named constants and an unbounded-ish optimizer vector.
    Positive constants spanning more than a factor 3 are fitted in log space."""

    def __init__(self, table):
        self.names = list(table)
        self.init = np.array([float(table[k][0]) for k in self.names])
        self.lo = np.array([float(table[k][1]) for k in self.names])
        self.hi = np.array([float(table[k][2]) for k in self.names])
        self.log = (self.lo > 0) & (self.hi / np.where(self.lo > 0, self.lo, 1) > 3)

    def to_z(self, v):
        return np.where(self.log, np.log(np.maximum(v, 1e-300)), v)

    def from_z(self, z):
        return np.where(self.log, np.exp(z), z)

    def bounds(self):
        return self.to_z(self.lo), self.to_z(self.hi)

    def as_dict(self, v):
        return {k: float(x) for k, x in zip(self.names, v)}


def _simulate(mod, ep, theta, p, steps=None):
    inp = ep.inputs if steps is None else {k: v[:steps] for k, v in ep.inputs.items()}
    T = mod.simulate(ep.geo, inp, theta, p, ep.T0)
    T = np.asarray(T, float)
    n = ep.n if steps is None else steps
    if T.shape != (n, ep.V):
        raise ValueError(f"simulate returned shape {T.shape}, expected {(n, ep.V)}")
    if not np.all(np.isfinite(T)):
        raise ValueError("simulate returned non-finite values")
    return T


# ----------------------------------------------------------------------------- physics checks

def physics_checks(mod, ep):
    """Cheap sanity tests on a real building's geometry, with every constant at its
    initial value, at its lower bound and at its upper bound (the optimizer may go
    anywhere inside the bounds, so the structure must be sane everywhere).
    Returns a list of failure strings (empty = pass)."""
    fails = []
    for label, idx in (("initial values", 0), ("lower bounds", 1), ("upper bounds", 2)):
        p = {k: v[idx] for k, v in mod.PARAMS.items()}
        th = {k: v[idx] for k, v in mod.THETA.items()}
        fails += [f"[{label}] {f}" for f in _checks_at(mod, ep, p, th)]
        if fails:
            break
    return fails


def _checks_at(mod, ep, p, th):
    fails = []
    n, V = 576, ep.V
    base = {k: np.zeros_like(v[:n]) for k, v in ep.inputs.items()}
    base["T_supply"] = np.full((n, V), 290.0)

    class E:  # minimal episode stand-in
        pass

    def run(T_out, Q=None, T0=294.0):
        e = E(); e.geo = ep.geo; e.T0 = T0; e.n = n; e.V = V
        e.inputs = dict(base); e.inputs["T_out"] = np.full(n, T_out)
        if Q is not None:
            e.inputs["Q_gain"] = Q
        return _simulate(mod, e, th, p)
    try:
        T = run(294.0)
        if np.abs(T - 294.0).max() > 0.05:
            fails.append(f"equilibrium: with T_out = T0 and no heat the rooms drift by {np.abs(T - 294).max():.3f} K")
        T = run(274.0)
        if T.max() > 294.05 or T.min() < 273.95:
            fails.append(f"bounds: cooling from 294 K toward 274 K left [274, 294] (range {T.min():.2f}-{T.max():.2f} K)")
        if not (T[-1] < T[0] + 1e-6).all():
            fails.append("relaxation: some rooms did not cool toward a colder outside")
        Q = np.zeros((n, V)); Q[:, 0] = 2000.0
        T2 = run(274.0, Q)
        if not (T2[-1, 0] > T[-1, 0] + 1e-3):
            fails.append("heating: 2 kW into room 0 did not raise its temperature")
        if (T2[-1] < T[-1] - 1e-6).any():
            fails.append("heating: 2 kW into room 0 cooled some other room")
        # stiff air stream: full flow of 290 K air into every room must pull it toward 290 K, never past it
        e = E(); e.geo = ep.geo; e.T0 = 294.0; e.n = n; e.V = V
        e.inputs = dict(base); e.inputs["T_out"] = np.full(n, 294.0)
        e.inputs["g"] = np.full((n, V), 2400.0); e.inputs["T_supply"] = np.full((n, V), 290.0)
        T3 = _simulate(mod, e, th, p)
        if T3.min() < 289.95 or T3.max() > 294.05 or not (T3[-1] < 293.0).all():
            fails.append(f"stiff air stream: 2400 W/K of 290 K air per room gave {T3.min():.2f}-{T3.max():.2f} K "
                         "(must stay within [290, 294] and approach 290 K); use implicit integration")
    except Exception as e:  # noqa: BLE001
        fails.append(f"exception during checks: {type(e).__name__}: {e}")
    return fails


# ----------------------------------------------------------------------------- fitting

def fit_train(mod, eps, max_nfev=40, stride=1):
    """Joint least squares over PARAMS and one THETA per training scenario."""
    P, T = ParamMap(mod.PARAMS), ParamMap(mod.THETA)
    scen = sorted({e.scenario for e in eps})
    si = {s: k for k, s in enumerate(scen)}
    nP, nT = len(P.names), len(T.names)

    def unpack(z):
        p = P.as_dict(P.from_z(z[:nP]))
        ths = [T.as_dict(T.from_z(z[nP + k * nT: nP + (k + 1) * nT])) for k in range(len(scen))]
        return p, ths

    sizes = [len(range(0, e.n, stride)) * e.V for e in eps]

    def resid(z):
        p, ths = unpack(z)
        r = []
        for e, m in zip(eps, sizes):
            Tp = _simulate(mod, e, ths[si[e.scenario]], p)
            r.append(((Tp - e.T_meas)[::stride]).ravel() / np.sqrt(m))
        return np.concatenate(r)

    rows = sum(sizes)
    S = sp.lil_matrix((rows, nP + nT * len(scen)), dtype=int)
    r0 = 0
    for e, m in zip(eps, sizes):
        S[r0:r0 + m, :nP] = 1
        k = si[e.scenario]
        S[r0:r0 + m, nP + k * nT: nP + (k + 1) * nT] = 1
        r0 += m
    z0 = np.concatenate([P.to_z(P.init)] + [T.to_z(T.init)] * len(scen))
    lo = np.concatenate([P.bounds()[0]] + [T.bounds()[0]] * len(scen))
    hi = np.concatenate([P.bounds()[1]] + [T.bounds()[1]] * len(scen))
    z0 = np.clip(z0, lo + 1e-9, hi - 1e-9)
    fit = least_squares(resid, z0, bounds=(lo, hi), jac_sparsity=S if S.shape[1] > 1 else None,
                        max_nfev=max_nfev, diff_step=1e-3, x_scale="jac")
    p, ths = unpack(fit.x)
    return p, dict(zip(scen, ths)), fit


def fit_theta_day1(mod, eps_of_scenario, p, max_nfev=40):
    """What a deployed policy does: THETA from day 1 only, PARAMS frozen."""
    T = ParamMap(mod.THETA)
    if not T.names:
        return {}

    def resid(z):
        th = T.as_dict(T.from_z(z))
        return np.concatenate([((_simulate(mod, e, th, p, steps=DAY) - e.T_meas[:DAY])).ravel() / np.sqrt(DAY * e.V)
                               for e in eps_of_scenario])
    lo, hi = T.bounds()
    z0 = np.clip(T.to_z(T.init), lo + 1e-9, hi - 1e-9)
    fit = least_squares(resid, z0, bounds=(lo, hi), max_nfev=max_nfev, diff_step=1e-3, x_scale="jac")
    return T.as_dict(T.from_z(fit.x))


def score_heldout(mod, eps, p, max_nfev=40):
    """Fit THETA on day 1 per held-out scenario, then open-loop error on days 2-3."""
    out = {"episodes": {}, "scenarios": {}}
    for s in sorted({e.scenario for e in eps}):
        es = [e for e in eps if e.scenario == s]
        th = fit_theta_day1(mod, es, p, max_nfev)
        out["scenarios"][s] = {"theta": th}
        for e in es:
            Tp = _simulate(mod, e, th, p)
            err = Tp - e.T_meas
            out["episodes"][e.name] = dict(
                rmse_day1=float(np.sqrt((err[:DAY] ** 2).mean())),
                rmse_days23=float(np.sqrt((err[DAY:] ** 2).mean())),
                bias_days23=float(err[DAY:].mean()),
                pred=Tp, err=err)
    vals = [v["rmse_days23"] for v in out["episodes"].values()]
    out["rmse_days23_mean"] = float(np.mean(vals)) if vals else float("nan")
    return out


# ----------------------------------------------------------------------------- one full evaluation

def evaluate(src, train_eps, val_eps, complexity_weight=0.0001, max_nfev=40, stride=1):
    """Everything about one candidate, as a JSON-able record (plus 'val_detail' with arrays)."""
    from . import contract
    t0 = time.time()
    rec = {"status": "ok"}
    try:
        mod, cx = contract.load(src)
        rec["complexity"] = cx
        rec["n_params"] = len(mod.PARAMS); rec["n_theta"] = len(mod.THETA)
        fails = physics_checks(mod, train_eps[0])
        if fails:
            return {**rec, "status": "rejected", "reason": "physics checks failed", "failures": fails, "sec": round(time.time() - t0, 1)}
        p, thetas, fit = fit_train(mod, train_eps, max_nfev, stride)
        rec["params"] = p
        rec["theta_train"] = thetas
        tr = {}
        for e in train_eps:
            err = _simulate(mod, e, thetas[e.scenario], p) - e.T_meas
            tr[e.name] = float(np.sqrt((err ** 2).mean()))
        rec["train_rmse"] = tr
        rec["train_rmse_mean"] = float(np.mean(list(tr.values())))
        val = score_heldout(mod, val_eps, p, max_nfev)
        rec["val_rmse_days23"] = {k: v["rmse_days23"] for k, v in val["episodes"].items()}
        rec["val_rmse_day1"] = {k: v["rmse_day1"] for k, v in val["episodes"].items()}
        rec["val_theta"] = {k: v["theta"] for k, v in val["scenarios"].items()}
        rec["val_rmse_days23_mean"] = val["rmse_days23_mean"]
        rec["score"] = val["rmse_days23_mean"] + complexity_weight * cx
        rec["val_detail"] = val
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()
        rec.update(status="error", reason=f"{type(e).__name__}: {e}", traceback="\n".join(tb[-6:]))
    rec["sec"] = round(time.time() - t0, 1)
    return rec
