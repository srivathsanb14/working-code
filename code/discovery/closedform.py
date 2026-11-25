"""Closed-form candidates: no number is fitted, anywhere (the decided approach; see docs/DISCOVERY_WALKTHROUGH.md).

A candidate is a discrete-time map written by the LLM:

    HISTORY = 36                                   # steps of measured history it reads (1..288)
    def prepare(geo, U):                           # optional: geometry-only templates, once per building
        return coef
    def step(coef, geo, hist, u, U):               # one 5-minute step, batched
        return T_next                              # (B, V) room air temperatures at the end of the step

  geo   per-room geometry from floor_plan + zone_map (episodes.geometry)
  U     universal constants (universal_constants.json), same for every building and weather
  hist  measured history, latest last:  hist["T"] (B, H, V) room temperatures x_{t-H+1..t};
        hist[k] for every input k = the H inputs BEFORE this step: per-room (B, H, V), building-wide (B, H)
  u     this step's inputs (the step x_t -> x_{t+1}): per-room (B, V), building-wide (B,)
Per-building values (facade/partition materials, slab density, film coefficient, occupants per room)
must be inferred from hist by closed-form expressions inside the formulas.

Scoring (no fitting):
  stage 0  physics checks on a real floor plan (equilibrium, bounded cooling, heat response, stiff air stream)
  stage 1  quick screen: one-step (5-minute) RMSE, all steps at once (simulator data are noiseless: no filter)
  stage 2  decisive: free-run rollouts from measured history at every `every`-th step, horizons 1/3/6 h,
           with a stability check; objectives (objectives()) and score = main-horizon RMSE + complexity, all on
           the VALIDATION episodes (there is no training split; test tiers are scored once at the end)
Baselines on the same windows: persistence (floor) and, as a diagnostic only, a per-building fitted
linear model (ceiling(): how well a model with free per-room numbers can do on this data).
"""
from __future__ import annotations

import json
import time
import traceback
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from . import coefform, contract

HERE = Path(__file__).resolve().parent
DAY = 288
ROOM_KEYS = ("g", "T_supply", "Q_reheat", "Q_gain")


def universal_constants(path=HERE / "universal_constants.json"):
    raw = json.loads(Path(path).read_text())
    return {k: v["value"] for k, v in raw.items() if not k.startswith("_")}, raw


# ----------------------------------------------------------------------------- batched windows

class _Hist(Mapping):
    """hist[...] built on access (a candidate usually reads only a few inputs)."""

    def __init__(self, ep, idx, T):
        self._ep, self._idx, self._T, self._c = ep, idx, T, {}

    def __getitem__(self, k):
        if k == "T":
            return self._T
        if k not in self._c:
            if k not in self._ep.Ip:
                raise KeyError(k)
            self._c[k] = self._ep.Ip[k][self._idx]
        return self._c[k]

    def __iter__(self):
        return iter(["T", *self._ep.Ip])

    def __len__(self):
        return 1 + len(self._ep.Ip)


class Prepared:
    """One episode laid out for batched stepping.  X[k] = room temperatures at the START of step k
    (X[0] = T0, X[k+1] = T_meas[k]); inputs[k] drive X[k] -> X[k+1].  Both are padded in front with
    H copies of the first value, so a window can start at step 0 (the early-episode buffer is incomplete)."""

    def __init__(self, ep, H):
        self.ep, self.H, self.n, self.V = ep, H, ep.n, ep.V
        self.S = None                                # filtered states of a stateful candidate (filter_states)
        self.CF = None                               # coefficient form: (states, networks, network index) (filter_coef)
        self.X = np.vstack([np.full((1, ep.V), ep.T0), ep.T_meas])
        self.Xp = np.vstack([np.repeat(self.X[:1], H, 0), self.X])                       # Xp[H + k] = X[k]
        self.Ip = {k: np.concatenate([np.repeat(v[:1], H, 0), v]) for k, v in ep.inputs.items()}   # Ip[H + k] = inputs[k]

    def u(self, ts):
        return {k: v[ts] for k, v in self.ep.inputs.items()}

    def hist(self, ts, T):
        return _Hist(self, ts[:, None] + np.arange(self.H)[None, :], T)     # inputs[t-H .. t-1]

    def measured_window(self, ts):
        return self.Xp[ts[:, None] + 1 + np.arange(self.H)[None, :]]       # X[t-H+1 .. t]


def is_stateful(mod):
    return all(hasattr(mod, f) for f in ("init_state", "predict", "correct"))


def prepare_model(mod, geo, U):
    """Per-building setup: a CoefModel for coefficient-form candidates, else the candidate's own prepare()."""
    if coefform.is_coefform(mod):
        return coefform.make_model(mod, geo, U)
    return mod.prepare(geo, U) if hasattr(mod, "prepare") else None


def _u_at(prep, ts):
    return {k: v[ts] for k, v in prep.ep.inputs.items()}


def filter_coef(mod, cm, prep, U):
    """Coefficient form: run the model over the measured history.  Every cm.refresh steps the coefficients are
    recomputed from the measured history up to that step (nothing later), the network is rebuilt, and each step the
    room-air nodes are set to the measurement.  Returns (S, nets, idx): S[k] the state after correcting with X[k],
    nets[idx[k]] the network that steps k -> k+1."""
    if getattr(prep, "CF", None) is not None:
        return prep.CF
    n, R = prep.n, cm.refresh
    S = np.empty((prep.n + 1, cm.N)); idx = np.empty(prep.n + 1, int); nets = []
    for k in range(n + 1):
        if k % R == 0:
            ts = np.array([k])
            nets.append(cm.build(prep.hist(ts, prep.measured_window(ts).astype(float))))
        idx[k] = len(nets) - 1
        if k == 0:
            s = cm.correct(cm.init(prep.X[0])[None], prep.X[0][None])
        else:
            s = cm.correct(cm.predict(nets[idx[k - 1]], S[k - 1][None], _u_at(prep, np.array([k - 1]))), prep.X[k][None])
        S[k] = s[0]
    if not np.all(np.isfinite(S)):
        raise ValueError("coefficient-form candidate produced non-finite states while filtering the measured history")
    prep.CF = (S, nets, idx)
    return prep.CF


def _rollout_coef(mod, cm, prep, U, starts, horizon, freeze=None):
    S, nets, idx = filter_coef(mod, cm, prep, U)
    out = np.empty((len(starts), horizon, prep.V))
    for g in np.unique(idx[starts]):                 # starts that share a network are stepped together
        sel = np.where(idx[starts] == g)[0]; st = starts[sel]
        x = S[st]
        for k in range(horizon):
            x = cm.predict(nets[g], x, _freeze(_u_at(prep, st + k), prep, st, freeze))
            out[sel, k] = x[:, :prep.V]
    return out


def filter_states(mod, coef, prep, U):
    """Stateful candidates: state after correcting with each measurement, S[k] <-> X[k].
    Computed once per episode by the candidate's own fixed predict/correct rules (nothing fitted)."""
    if getattr(prep, "S", None) is not None:
        return prep.S
    geo = prep.ep.geo
    s = np.atleast_2d(np.asarray(mod.init_state(coef, geo, U), float))
    s = mod.correct(coef, geo, s, prep.X[0][None], U)
    S = np.empty((prep.n + 1, s.shape[-1])); S[0] = s[0]
    for k in range(prep.n):
        s = mod.predict(coef, geo, s, _u_at(prep, np.array([k])), U)
        s = mod.correct(coef, geo, s, prep.X[k + 1][None], U)
        S[k + 1] = s[0]
    if not np.all(np.isfinite(S)):
        raise ValueError("stateful candidate produced non-finite states while filtering the measured history")
    prep.S = S
    return S


EXOGENOUS = ("T_out", "occupants")   # not known ahead at test time: frozen at the last observed value in deployment rollouts


def _freeze(u, prep, ts0, freeze):
    if freeze:
        u0 = _u_at(prep, ts0)
        for k in freeze:
            if k in u:
                u[k] = u0[k]
    return u


def _rollout_stateful(mod, coef, prep, U, starts, horizon, freeze=None):
    S = filter_states(mod, coef, prep, U)
    geo, V = prep.ep.geo, prep.V
    out = np.empty((len(starts), horizon, V))
    s = S[starts]
    for k in range(horizon):
        s = np.asarray(mod.predict(coef, geo, s, _freeze(_u_at(prep, starts + k), prep, starts, freeze), U), float)
        out[:, k] = s[:, :V]
    return out


def rollout(mod, coef, prep, U, starts, horizon, chunk=None, freeze=None):
    """Free-run every start in `starts` for `horizon` steps from measured history.
    Returns pred (S, horizon, V): pred[s, k] is the forecast of X[starts[s] + k + 1]."""
    starts = np.asarray(starts, int)
    if coefform.is_coefform(mod):
        return _rollout_coef(mod, coef, prep, U, starts, horizon, freeze)
    if is_stateful(mod):
        return _rollout_stateful(mod, coef, prep, U, starts, horizon, freeze)
    chunk = chunk or int(max(8, min(1024, 2e6 // (prep.H * prep.V))))       # bounds the (B, H, V) windows in memory
    out = np.empty((len(starts), horizon, prep.V))
    for a in range(0, len(starts), chunk):
        ts0 = starts[a:a + chunk]
        Tb = prep.measured_window(ts0).astype(float)
        for k in range(horizon):
            ts = ts0 + k
            y = np.asarray(mod.step(coef, prep.ep.geo, prep.hist(ts, Tb), _freeze(prep.u(ts), prep, ts0, freeze), U), float)
            if y.shape != (len(ts0), prep.V):
                raise ValueError(f"step returned shape {y.shape}, expected {(len(ts0), prep.V)} (batch, rooms)")
            out[a:a + chunk, k] = y
            Tb = np.concatenate([Tb[:, 1:], y[:, None, :]], 1)
    return out


def unstable(pred, lo=230.0, hi=350.0):
    return (not np.all(np.isfinite(pred))) or pred.min() < lo or pred.max() > hi


# ----------------------------------------------------------------------------- stage 0: physics checks

class _Synth:
    def __init__(self, geo, inputs, T0=294.0):
        self.geo, self.inputs, self.T0 = geo, inputs, T0
        self.n = len(inputs["T_out"]); self.V = int(geo["V"])
        self.T_meas = np.full((self.n, self.V), T0)


def shift_warning(mod, coef, ep, U, n=288, c=5.0):
    """Warning only: shifting every temperature (start, outside, supply, water) by c should shift the
    forecast by c when no heat is added (the heat flows depend on temperature differences)."""
    V = ep.V
    base = {k: np.zeros((n, V)) if v.ndim == 2 else np.zeros(n) for k, v in ep.inputs.items()}

    def run(dT):
        inp = dict(base); inp["T_out"] = np.full(n, 280.0 + dT); inp["T_supply"] = np.full((n, V), 290.0 + dT)
        if "T_water" in inp:
            inp["T_water"] = np.full(n, 340.0 + dT)
        p = Prepared(_Synth(ep.geo, inp, T0=294.0 + dT), mod.HISTORY)
        return rollout(mod, coef, p, U, [0], n)[0]
    try:
        d = np.abs(run(c) - run(0.0) - c).max()
        return [f"shift: moving every temperature by {c:g} K moved the forecast by up to {d:.3f} K more or less than {c:g} K"] if d > 0.05 else []
    except Exception as e:  # noqa: BLE001
        return [f"shift check failed: {type(e).__name__}: {e}"]


def long_run(mod, coef, ep, U, horizon=288, warmup=36):
    """Diagnostic: 24 h free runs from measured history, one per day; RMSE and mean drift at 24 h."""
    prep = Prepared(ep, mod.HISTORY)
    starts = np.arange(max(warmup, 288), ep.n - horizon, 288)
    if not len(starts):
        return None
    pr = rollout(mod, coef, prep, U, starts, horizon)
    e = pr[:, -1] - prep.X[starts + horizon]
    return float(np.sqrt(np.mean(e ** 2))), float(e.mean())


def physics_checks(mod, coef, ep, U, n=576):
    """Free-run 2 days on a real floor plan under simple conditions; the history starts at 294 K everywhere."""
    V = ep.V
    base = {k: np.zeros((n, V)) if v.ndim == 2 else np.zeros(n) for k, v in ep.inputs.items()}
    base["T_supply"] = np.full((n, V), 290.0)
    if "T_water" in base:
        base["T_water"] = np.full(n, 340.0)

    def run(T_out=294.0, **over):
        inp = dict(base); inp["T_out"] = np.full(n, T_out); inp.update(over)
        p = Prepared(_Synth(ep.geo, inp), mod.HISTORY)
        return rollout(mod, coef, p, U, [0], n)[0]
    fails = []
    try:
        T = run(294.0)
        if np.abs(T - 294.0).max() > 0.05:
            fails.append(f"equilibrium: with T_out = 294 K, no heat and no air the rooms drift by {np.abs(T - 294).max():.3f} K")
        T = run(274.0)
        if T.max() > 294.05 or T.min() < 273.95:
            fails.append(f"bounds: cooling from 294 K toward a 274 K outside left [274, 294] K (range {T.min():.2f}-{T.max():.2f})")
        elif not (T[-1] < T[0] + 1e-6).all():
            fails.append("relaxation: some rooms did not cool toward a colder outside")
        Tw = run(284.0)                                   # a warmer outside never makes a room colder
        if (Tw[-1] < T[-1] - 1e-6).any():
            fails.append("monotone: a 10 K warmer outside left some room colder")
        Q = np.zeros((n, V)); Q[:, 0] = 2000.0
        T2 = run(274.0, Q_reheat=Q)
        if not T2[-1, 0] > T[-1, 0] + 1e-3:
            fails.append("heating: 2 kW of reheat into room 0 did not raise its temperature")
        if (T2[-1] < T[-1] - 1e-6).any():
            fails.append("heating: 2 kW into room 0 cooled some other room")
        T3 = run(294.0, g=np.full((n, V), 2400.0), T_supply=np.full((n, V), 290.0))
        if T3.min() < 289.95 or T3.max() > 294.05 or not (T3[-1] < 293.0).all():
            fails.append(f"stiff air stream: 2400 W/K of 290 K air per room gave {T3.min():.2f}-{T3.max():.2f} K "
                         "(must stay in [290, 294] and approach 290 K); treat the supply air implicitly (dlib.end_of_step)")
    except Exception as e:  # noqa: BLE001
        fails.append(f"exception during checks: {type(e).__name__}: {e}")
    return fails


# ----------------------------------------------------------------------------- stage 1 + 2 on one episode

TERM_NAMES = ("air", "outside", "neighbours", "slab", "wall", "inner", "heat")


def term_contributions(cm, prep, ts, with_steps=False):
    """Term form only.  For steps ts, the change each term of the fixed function makes to every room's air in one step,
    from the filtered state (measured room air, hidden stores as the model carries them), and the one-step residual
    (measured - predicted).  T+ - T = sum of the contributions exactly, so a residual that follows one contribution
    says that term is too weak (positive slope) or too strong (negative).  Returns (contrib (m, 7), residual (m,),
    room (m,)) over rooms x steps."""
    S, nets, idx = prep.CF
    V = prep.V
    C, R, I, K = [], [], [], []
    for g in np.unique(idx[ts]):
        st = ts[idx[ts] == g]
        c, x = nets[g], S[st]
        T, Sl, W, P = x[:, :V], x[:, V:2 * V], x[:, 2 * V:3 * V], x[:, 3 * V:]
        u = _u_at(prep, st)
        gg = np.asarray(u["g"], float); Ts = np.asarray(u["T_supply"], float); To = np.asarray(u["T_out"], float).reshape(-1, 1)
        wa = c["air"] * gg
        den = 1 + wa + c["outside"] + c["nb_sum"] + c["slab"] + c["wall"] + c["inner"]
        k = np.stack([wa * (Ts - T), c["outside"] * (To - T), T @ c["neighbours"].T - c["nb_sum"] * T, c["slab"] * (Sl - T),
                      c["wall"] * (W - T), c["inner"] * (P - T), c["heat"] * cm.heat(u)], -1) / den[..., None]
        C.append(k.reshape(-1, len(TERM_NAMES)))
        R.append((prep.X[st + 1] - (T + k.sum(-1))).reshape(-1))
        I.append(np.tile(np.arange(V), len(st))); K.append(np.repeat(st, V))
    out = (np.concatenate(C), np.concatenate(R), np.concatenate(I))
    return out + (np.concatenate(K),) if with_steps else out


def episode_scores(mod, coef, ep, U, horizons=(12, 36, 72), every=6, warmup=36, keep=36, n_attrib=600):
    """One-step and rollout errors on one episode, with persistence on the same windows.  Kept for the analysis:
    the main-horizon errors per start and room, the error-vs-horizon curve, and (term form) the term contributions."""
    prep = Prepared(ep, mod.HISTORY)
    n, X = ep.n, prep.X
    t1 = np.arange(warmup, n)
    p1 = rollout(mod, coef, prep, U, t1, 1)[:, 0]
    e1 = p1 - X[t1 + 1]
    hmax = max(horizons)
    starts = np.arange(warmup, n - hmax, every)
    r = {"one_step": float(np.sqrt(np.mean(e1 ** 2))),
         "persistence_one_step": float(np.sqrt(np.mean((X[t1] - X[t1 + 1]) ** 2))),
         "one_step_err": e1, "one_step_t": t1}
    if len(starts) == 0:
        raise ValueError(f"episode {ep.name} too short for a {hmax}-step rollout after {warmup} warm-up steps")
    pr = rollout(mod, coef, prep, U, starts, hmax)
    r["unstable"] = bool(unstable(pr))
    for h in horizons:
        e = pr[:, h - 1] - X[starts + h]
        r[f"{h}"] = float(np.sqrt(np.mean(e ** 2))) if np.all(np.isfinite(e)) else float("inf")
        r[f"persistence_{h}"] = float(np.sqrt(np.mean((X[starts] - X[starts + h]) ** 2)))
    with np.errstate(all="ignore"):
        cur = pr - X[starts[:, None] + np.arange(1, hmax + 1)[None, :]]
        r["curve_rmse"] = np.sqrt(np.mean(cur ** 2, axis=(0, 2))); r["curve_bias"] = cur.mean(axis=(0, 2))
        r["curve_persistence"] = np.sqrt(np.mean((X[starts][:, None] - X[starts[:, None] + np.arange(1, hmax + 1)[None, :]]) ** 2, axis=(0, 2)))
    pd_ = rollout(mod, coef, prep, U, starts, keep, freeze=EXOGENOUS)         # what a deployed model can actually do
    r[f"deploy_{keep}"] = float(np.sqrt(np.mean((pd_[:, -1] - X[starts + keep]) ** 2)))
    r["keep_err"] = pr[:, keep - 1] - X[starts + keep]
    r["keep_t"] = starts + keep - 1                  # index of the input step that ends at the target
    r["keep_start"] = starts
    if isinstance(coef, coefform.TermModel) and prep.CF is not None:
        ts = t1[:-1] if len(t1) <= n_attrib else np.sort(np.random.default_rng(0).choice(t1[:-1], n_attrib, replace=False))
        r["attrib"] = term_contributions(coef, prep, ts, with_steps=True)
    return r


def _hname(h):
    return f"{h / 12:g}h" if h % 12 == 0 else f"{h}st"


# ----------------------------------------------------------------------------- objectives (multi-objective selection)

# name -> (what it measures, the coefficients it mostly depends on).  Each error objective isolates the rooms, times
# or horizons where one group of coefficients decides the prediction, so a candidate that gets one coefficient
# group right survives even if another group is still wrong.
OBJECTIVES = {
    "main": ("main-horizon rollout RMSE as at test time (future weather and occupancy unknown) + the size price", "all terms"),
    "fast": ("shortest-horizon rollout RMSE", "air, heat, neighbours"),
    "slow": ("longest-horizon rollout RMSE", "slab, wall, inner and their _back terms"),
    "exterior": ("main-horizon RMSE in rooms with an outside wall", "outside, wall, wall_out, wall_back"),
    "interior": ("main-horizon RMSE in rooms without an outside wall", "neighbours, inner, inner_back"),
    "passive": ("main-horizon RMSE while the HVAC air is off", "outside and the stores"),
    "driven": ("main-horizon RMSE while the HVAC air is on", "air, heat"),
    "complexity": ("AST nodes of the code", "-"),
}
# complexity is not a default objective: as its own axis the smallest (and worst) program is never dominated and stays
# a parent forever; it enters through the size price inside "main" instead.
DEFAULT_OBJECTIVES = ("main", "fast", "slow", "exterior", "interior")


def objectives(rec, horizons, main_h, deploy_score=True, complexity_weight=0.0001):
    """All objective values of an evaluated candidate (lower is better; inf when unstable)."""
    G = rec.get("groups", {})
    bad = bool(rec.get("unstable_episodes"))

    def v(x):
        return float("inf") if bad or x is None or not np.isfinite(x) else float(x)
    hs = sorted(horizons)
    main = rec.get(f"val_{_hname(main_h)}_deploy_mean") if deploy_score else rec.get(f"val_{_hname(main_h)}_mean")
    return {"main": v(main + complexity_weight * rec.get("complexity", 0) if main is not None else None), "fast": v(rec.get(f"val_{_hname(hs[0])}_mean")), "slow": v(rec.get(f"val_{_hname(hs[-1])}_mean")),
            "exterior": v(G.get("space: exterior rooms", {}).get("rmse")), "interior": v(G.get("space: interior rooms", {}).get("rmse")),
            "passive": v(G.get("time: HVAC air off at the target", {}).get("rmse")),
            "driven": v(G.get("time: HVAC air on at the target", {}).get("rmse")),
            "complexity": float(rec.get("complexity", 0))}


# ----------------------------------------------------------------------------- the full evaluation

def evaluate(src, val_eps, U, complexity_weight=0.0001, horizons=(12, 36, 72), main_h=36, every=6,
             warmup=36, screen_factor=5.0, deploy_score=True):
    """Everything about one closed-form candidate on the VALIDATION episodes as a JSON-able record (+ 'detail' with
    arrays for the analysis).  There is no training split: nothing is fitted, so every episode the search uses is
    a validation episode; the test tiers are scored separately, once, at the end of a run."""
    t0 = time.time()
    rec = {"status": "ok", "deployment": "closedform"}
    try:
        mod, cx = contract.load_closedform(src)
        rec.update(complexity=cx, history=mod.HISTORY)
        coefs = {}

        def coef_of(ep):
            if ep.building not in coefs:
                coefs[ep.building] = prepare_model(mod, ep.geo, U)
            return coefs[ep.building]
        fails = physics_checks(mod, coef_of(val_eps[0]), val_eps[0], U)
        if fails:
            return {**rec, "status": "rejected", "stage": 0, "reason": "physics checks failed", "failures": fails,
                    "sec": round(time.time() - t0, 1)}
        # stage 1: one-step screen
        s1 = {}
        for e in val_eps:
            prep = Prepared(e, mod.HISTORY)
            t1 = np.arange(warmup, e.n)
            p1 = rollout(mod, coef_of(e), prep, U, t1, 1)[:, 0]
            s1[e.name] = (float(np.sqrt(np.mean((p1 - prep.X[t1 + 1]) ** 2))),
                          float(np.sqrt(np.mean((prep.X[t1] - prep.X[t1 + 1]) ** 2))))
        rec["stage1_one_step_mean"] = float(np.mean([v[0] for v in s1.values()]))
        rec["stage1_persistence_mean"] = float(np.mean([v[1] for v in s1.values()]))
        if not np.isfinite(rec["stage1_one_step_mean"]) or rec["stage1_one_step_mean"] > screen_factor * rec["stage1_persistence_mean"]:
            return {**rec, "status": "screened", "stage": 1,
                    "reason": f"one-step RMSE {rec['stage1_one_step_mean']:.4f} K is more than {screen_factor}x persistence "
                              f"({rec['stage1_persistence_mean']:.4f} K) on the validation buildings", "sec": round(time.time() - t0, 1)}
        # stage 2: rollouts (decisive)
        val = {e.name: episode_scores(mod, coef_of(e), e, U, horizons, every, warmup, keep=main_h) for e in val_eps}
        rec["unstable_episodes"] = [k for k, v in val.items() if v["unstable"]]
        rec["val_one_step_mean"] = float(np.mean([v["one_step"] for v in val.values()]))
        for h in horizons:
            rec[f"val_{_hname(h)}_mean"] = float(np.mean([v[f"{h}"] for v in val.values()]))
            rec[f"val_persistence_{_hname(h)}_mean"] = float(np.mean([v[f"persistence_{h}"] for v in val.values()]))
        lr = [long_run(mod, coef_of(e), e, U) for e in val_eps]
        lr = [x for x in lr if x is not None and np.isfinite(x[0])]
        if lr:
            rec["val_24h_mean"] = float(np.mean([x[0] for x in lr])); rec["val_24h_drift_mean"] = float(np.mean([x[1] for x in lr]))
        w = shift_warning(mod, coef_of(val_eps[0]), val_eps[0], U)
        if w:
            rec["warnings"] = w
        rec["val_by_episode"] = {k: {kk: v[kk] for kk in v if kk.startswith(("one_step", "persistence", "deploy")) and not kk.endswith(("_err", "_t"))
                                     or kk.isdigit()} for k, v in val.items()}
        rec["horizons"] = list(horizons); rec["main_horizon"] = main_h
        rec[f"val_{_hname(main_h)}_deploy_mean"] = float(np.mean([v[f"deploy_{main_h}"] for v in val.values()]))
        m = rec[f"val_{_hname(main_h)}_deploy_mean"] if deploy_score else rec[f"val_{_hname(main_h)}_mean"]
        if rec["unstable_episodes"]:
            rec.update(status="unstable", stage=2, reason=f"free-run rollout left [230, 350] K or went non-finite on "
                       f"{len(rec['unstable_episodes'])} validation episode(s)")
            m = float("inf")
        rec["score_metric"] = (f"val {_hname(main_h)} rollout RMSE from measured history, future outside temperature and occupancy "
                               f"unknown (held at their last observed values), as at test time" if deploy_score else
                               f"val {_hname(main_h)} rollout RMSE from measured history with the true future inputs (diagnostic)")
        rec["score"] = m + complexity_weight * cx
        rec["detail"] = val
        if not rec["unstable_episodes"]:
            rec["groups"] = _group_stats(val, val_eps)
        rec["objectives"] = objectives(rec, horizons, main_h, deploy_score, complexity_weight)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()
        rec.update(status="error", reason=f"{type(e).__name__}: {e}", traceback="\n".join(tb[-6:]))
    rec["sec"] = round(time.time() - t0, 1)
    return rec


# ----------------------------------------------------------------------------- analysis (pre-mutation / pre-crossover)

HVAC_ON = 50.0          # W/K of supply air, room mean: above this the air handlers count as on


def _hour(inp, t):
    return (np.degrees(np.arctan2(inp["hour_sin"][t], inp["hour_cos"][t])) / 15.0) % 24 if "hour_sin" in inp else None


def _room_features(geo):
    a = np.maximum(geo["area"], 1e-9)
    return {"outside-wall length per floor area": geo["facade_faces"] * 0.2 / a, "floor area": geo["area"],
            "shared-wall length per floor area": geo["wall_cells"].sum(1) * 0.2 / a,
            "door width per floor area": geo["door_cells"].sum(1) * 0.2 / a, "number of neighbours": geo["n_neighbours"].astype(float)}


def _group_stats(det, val_eps):
    """RMSE and bias of the main-horizon rollout error by WHERE (room class, facade side, floor, air handler) and WHEN
    (hour of day, occupancy, HVAC state and switching during the rollout, outside temperature, episode phase)."""
    acc = {}

    def add(key, e):
        e = np.asarray(e)
        if e.size:
            s = acc.setdefault(key, [0.0, 0.0, 0])
            s[0] += float((e ** 2).sum()); s[1] += float(e.sum()); s[2] += e.size
    for ep in val_eps:
        d = det[ep.name]; E = d["keep_err"]; t = d["keep_t"]; s0 = d["keep_start"]
        geo, inp = ep.geo, ep.inputs
        ext = geo["facade_faces"] > 0
        corner = geo["facade_sides"] >= 2 if "facade_sides" in geo else np.zeros_like(ext)
        add("all rooms", E)
        add("space: exterior rooms", E[:, ext]); add("space: interior rooms", E[:, ~ext])
        add("space: corner rooms (outside wall on 2+ sides)", E[:, ext & corner])
        add("space: one-side exterior rooms", E[:, ext & ~corner])
        big = geo["area"] >= np.median(geo["area"])
        add("space: larger half of rooms (floor area)", E[:, big]); add("space: smaller half of rooms", E[:, ~big])
        for side in ("top", "bottom", "left", "right"):
            if f"facade_{side}" in geo:
                add(f"space: outside wall on the {side} side of the plan", E[:, geo[f"facade_{side}"] > 0])
        for f in np.unique(geo["floor"]):
            add(f"space: floor {int(f) + 1}", E[:, geo["floor"] == f])
        hr = _hour(inp, t)
        if hr is not None:
            for h0 in range(0, 24, 4):
                add(f"time: target hour {h0:02d}-{h0 + 4:02d}", E[(hr >= h0) & (hr < h0 + 4)])
        com = inp["comfort"][t] > 0.5
        add("time: occupied (comfort) hours", E[com]); add("time: nights and weekends", E[~com])
        on = inp["g"].mean(1) > HVAC_ON
        add("time: HVAC air on at the target", E[on[t]]); add("time: HVAC air off at the target", E[~on[t]])
        add("time: HVAC switched off during the rollout", E[on[s0] & ~on[t]])
        add("time: HVAC switched on during the rollout", E[~on[s0] & on[t]])
        cold = inp["T_out"][t] < np.median(inp["T_out"])
        add("time: colder half of outside temperatures", E[cold]); add("time: warmer half of outside temperatures", E[~cold])
        add("time: first 3 days of the episode", E[t < 3 * DAY]); add("time: after day 3", E[t >= 3 * DAY])
    return {k: {"rmse": float(np.sqrt(s[0] / s[2])), "bias": float(s[1] / s[2]), "n": s[2]} for k, s in acc.items()}


def term_regression(det, val_eps, rows=None):
    """Pooled least squares of the one-step residual on the term contributions (term form).  `rows(ep, room, step)`
    selects rows (a room type, a condition).  Returns {term: {"slope", "se", "size"}} plus "_r2", or {} when the
    candidate is not in the term form or too few rows are selected."""
    C, R = [], []
    for ep in val_eps:
        a = det[ep.name].get("attrib")
        if a is None:
            return {}
        c, r, i, t = a
        m = np.ones(len(r), bool) if rows is None else np.asarray(rows(ep, i, t), bool)
        C.append(c[m]); R.append(r[m])
    C, R = np.concatenate(C), np.concatenate(R)
    if len(R) < 50:
        return {}
    with np.errstate(all="ignore"):           # Accelerate's matmul raises spurious FP flags on macOS
        return _regress(C, R)


def _regress(C, R):
    keep = [k for k in range(C.shape[1]) if np.std(C[:, k]) > 1e-9]
    X = C[:, keep]
    sc = X.std(0)
    Z = X / sc
    A = Z.T @ Z + 1e-6 * len(R) * np.eye(len(keep))
    b = np.linalg.solve(A, Z.T @ R)
    res = R - Z @ b
    s2 = float(res @ res) / max(len(R) - len(keep), 1)
    se = np.sqrt(np.maximum(np.diag(np.linalg.inv(A)) * s2, 0))
    out = {TERM_NAMES[k]: {"slope": float(b[j] / sc[j]), "se": float(se[j] / sc[j]), "size": float(np.sqrt(np.mean(C[:, k] ** 2)))}
           for j, k in enumerate(keep)}
    out["_r2"] = float(1 - (res @ res) / max(float(R @ R), 1e-30))
    out["_residual_rms"] = float(np.sqrt(np.mean(R ** 2)))
    return out


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def spatial_stats(det, val_eps):
    """Per-room RMSE/bias per episode, the correlation of room bias with room features, and whether neighbouring
    rooms' errors move together (correlation of their error series, adjacent vs not adjacent)."""
    rooms, B, F = {}, [], {}
    adj_c, far_c = [], []
    for ep in val_eps:
        E = det[ep.name]["keep_err"]
        rooms[ep.name] = {"rmse": np.sqrt((E ** 2).mean(0)).round(4).tolist(), "bias": E.mean(0).round(4).tolist()}
        B.append(E.mean(0))
        for k, v in _room_features(ep.geo).items():
            F.setdefault(k, []).append(v)
        if E.shape[1] >= 3 and E.shape[0] >= 10:
            Cm = np.corrcoef(E.T)
            A = (ep.geo["wall_cells"] + ep.geo["door_cells"]) > 0
            iu = np.triu_indices(E.shape[1], 1)
            ok = np.isfinite(Cm[iu])
            adj_c += list(Cm[iu][ok & A[iu]]); far_c += list(Cm[iu][ok & ~A[iu]])
    B = np.concatenate(B)
    corr = {k: _corr(np.concatenate(v), B) for k, v in F.items()}
    return {"rooms": rooms, "bias_corr": corr,
            "err_corr_adjacent": float(np.mean(adj_c)) if adj_c else None, "err_corr_not_adjacent": float(np.mean(far_c)) if far_c else None}


def temporal_stats(det, val_eps):
    """Error-vs-horizon curve (pooled) and the persistence of the one-step error (lag-1 h autocorrelation)."""
    cr = np.sqrt(np.mean([det[e.name]["curve_rmse"] ** 2 for e in val_eps], 0))
    cb = np.mean([det[e.name]["curve_bias"] for e in val_eps], 0)
    cp = np.sqrt(np.mean([det[e.name]["curve_persistence"] ** 2 for e in val_eps], 0))
    ac = []
    for e in val_eps:
        x = det[e.name]["one_step_err"]
        if len(x) > 24:
            ac += [_corr(x[:-12, i], x[12:, i]) for i in range(x.shape[1])]
    return {"curve_rmse": cr.tolist(), "curve_bias": cb.tolist(), "curve_persistence": cp.tolist(),
            "one_step_autocorr_1h": float(np.mean(ac)) if ac else None}


def _room_type_masks():
    def corner(g):
        return (g["facade_faces"] > 0) & (g.get("facade_sides", np.zeros(g["V"])) >= 2)

    def side(g):
        return (g["facade_faces"] > 0) & ~corner(g)

    def interior(g):
        return g["facade_faces"] == 0

    def large(g):
        return g["area"] >= np.median(g["area"])
    return {"corner rooms (outside wall, 2+ sides)": corner, "one-side exterior rooms": side, "interior rooms": interior,
            "larger half of rooms (by floor area)": large, "smaller half of rooms": lambda g: ~large(g)}


ROOM_TYPES = _room_type_masks()
CONDITIONS = {"HVAC air on (room mean > 50 W/K)": lambda inp, t: inp["g"][t].mean(1) > HVAC_ON,
              "HVAC air off": lambda inp, t: inp["g"][t].mean(1) <= HVAC_ON,
              "occupied (comfort) hours": lambda inp, t: inp["comfort"][t] > 0.5,
              "colder half of outside temperatures": lambda inp, t: inp["T_out"][t] < np.median(inp["T_out"])}


def _verdict(t):
    s, se = t["slope"], t["se"]
    if abs(s) < 2 * se or abs(s) < 0.03:
        return "consistent with its current size"
    return f"{'too weak' if s > 0 else 'too strong'} by about {abs(s) * 100:.0f}%"


def analysis(rec, val_eps):
    """Pre-mutation analysis of one candidate on the validation episodes: where (spatial), when (temporal) and through
    which term (term form) its error arises.  Returns text for the prompt and JSON-able numbers for crossover."""
    if rec.get("status") not in ("ok", "unstable") or "detail" not in rec:
        lines = [f"status: {rec.get('status')} (stage {rec.get('stage', '?')}) - {rec.get('reason', '')}"]
        lines += [f"  - {f}" for f in rec.get("failures", [])]
        if rec.get("traceback"):
            lines.append("  traceback:\n    " + rec["traceback"].replace("\n", "\n    "))
        return {"text": "\n".join(lines), "groups": {}, "terms": {}, "rooms": {}}
    det = rec["detail"]; hs = rec["horizons"]; mh = _hname(rec["main_horizon"])
    G = rec.get("groups") or {}
    L = ["OBJECTIVES (validation, lower is better): " + ", ".join(
        f"{k} {v:.3f}" if k != "complexity" else f"{k} {v:.0f}" for k, v in rec.get("objectives", {}).items())]
    # ---- temporal
    T = temporal_stats(det, val_eps)
    L.append("\nTEMPORAL")
    L.append("  error vs horizon (rmse / bias / persistence rmse, K):")
    hmax = len(T["curve_rmse"])
    for k in [1, 6, 12, 24, 36, 48, 72, 144, 288]:
        if k <= hmax:
            L.append(f"    {k * 5 / 60:5.2f} h   {T['curve_rmse'][k - 1]:.3f} / {T['curve_bias'][k - 1]:+.3f} / {T['curve_persistence'][k - 1]:.3f}")
    r_end, b_end = T["curve_rmse"][-1], T["curve_bias"][-1]
    if abs(b_end) > 0.6 * r_end:
        L.append(f"    -> at {hmax * 5 / 60:g} h most of the error is a common offset ({b_end:+.3f} K): a steady-state balance is off "
                 "(the ratio of the outside or store coefficients to the others), not the timing")
    elif T["curve_rmse"][min(11, hmax - 1)] > 0.5 * r_end:
        L.append("    -> half of the long-horizon error is already there after 1 h: the fast terms (air, heat, neighbours) are off")
    else:
        L.append("    -> error builds up slowly with horizon: the slow stores (slab, wall, inner and their _back terms) carry it")
    if T["one_step_autocorr_1h"] is not None:
        L.append(f"  one-step error autocorrelation at 1 h lag: {T['one_step_autocorr_1h']:+.2f}"
                 + (" (persistent: a slow state or a slowly varying input is missing or mis-scaled)" if T["one_step_autocorr_1h"] > 0.5 else ""))
    L.append(f"  {mh} rollout error by time (rmse, bias, K):")
    for k, v in G.items():
        if k.startswith("time: "):
            L.append(f"    {k[6:]:<44} {v['rmse']:.3f}  {v['bias']:+.3f}")
    # ---- spatial
    Sp = spatial_stats(det, val_eps)
    L.append("\nSPATIAL")
    L.append(f"  {mh} rollout error by place (rmse, bias, K):")
    for k, v in G.items():
        if k.startswith("space: "):
            L.append(f"    {k[7:]:<44} {v['rmse']:.3f}  {v['bias']:+.3f}")
    L.append("  correlation of a room's bias with its geometry (over all validation rooms):")
    for k, c in sorted(Sp["bias_corr"].items(), key=lambda x: -abs(x[1])):
        L.append(f"    {k:<44} {c:+.2f}" + ("   <- the formula's dependence on this is wrong" if abs(c) > 0.4 else ""))
    if Sp["err_corr_adjacent"] is not None:
        L.append(f"  error series correlation, adjacent rooms {Sp['err_corr_adjacent']:+.2f} vs non-adjacent {Sp['err_corr_not_adjacent']:+.2f}"
                 + (" (similar: a building-wide cause, e.g. supply air or outside term)" if abs(Sp['err_corr_adjacent'] - Sp['err_corr_not_adjacent']) < 0.1
                    else " (adjacent rooms err together: the neighbour coupling or a shared store is off)"))
    worst = []
    for i_ep, ep in enumerate(val_eps):
        rm_, bs = np.array(Sp["rooms"][ep.name]["rmse"]), np.array(Sp["rooms"][ep.name]["bias"])
        for i in np.argsort(-rm_)[:2]:
            worst.append((rm_[i], bs[i], i_ep + 1, i, int(ep.geo["facade_faces"][i]), float(ep.geo["area"][i]), int(ep.geo["n_doors"][i].sum())))
    worst.sort(key=lambda w: -w[0])
    L.append("  worst rooms: " + "; ".join(f"episode {w[2]} room #{w[3]} (rmse {w[0]:.2f}, bias {w[1]:+.2f}, outside-wall faces {w[4]}, "
                                         f"area {w[5]:.0f} m2, doors {w[6]})" for w in worst[:4]))
    # ---- terms
    Tm = term_regression(det, val_eps)
    terms = {"all rooms": Tm} if Tm else {}
    if Tm:
        L.append("\nTERMS (one-step residual regressed on each term's contribution; slope s means the term should be about (1+s) times as large)")
        L.append(f"  residual rms {Tm['_residual_rms']:.4f} K per step; the terms explain {100 * max(Tm['_r2'], 0):.0f}% of it"
                 + (" (low: the error is not a rescaling of an existing term; the structure or a store's dynamics is missing)" if Tm["_r2"] < 0.2 else ""))
        for k in TERM_NAMES:
            if k in Tm:
                t = Tm[k]
                L.append(f"    {k:<11} size {t['size']:.4f} K/step   slope {t['slope']:+.3f} +- {2 * t['se']:.3f}   -> {_verdict(t)}")
        L.append("  BY ROOM TYPE (slope per term; blank = term not present or too few rows):")
        L.append("    " + f"{'room type':<38}" + "".join(f"{k:>11}" for k in TERM_NAMES) + "   explained")
        for name, sel in ROOM_TYPES.items():
            Tt = term_regression(det, val_eps, rows=lambda ep, i, t, sel=sel: sel(ep.geo)[i])
            if Tt:
                terms[name] = Tt
                L.append("    " + f"{name:<38}" + "".join(f"{Tt[k]['slope']:>+11.2f}" if k in Tt else f"{'':>11}" for k in TERM_NAMES)
                         + f"   {100 * max(Tt['_r2'], 0):5.0f}%")
        L.append("  BY CONDITION:")
        for name, sel in CONDITIONS.items():
            Tt = term_regression(det, val_eps, rows=lambda ep, i, t, sel=sel: sel(ep.inputs, t))
            if Tt:
                terms[name] = Tt
                L.append("    " + f"{name:<38}" + "".join(f"{Tt[k]['slope']:>+11.2f}" if k in Tt else f"{'':>11}" for k in TERM_NAMES)
                         + f"   {100 * max(Tt['_r2'], 0):5.0f}%")
        L.append("  -> a slope that differs by room type means the formula's dependence on geometry is wrong for that type;")
        L.append("     one that differs by condition means the term depends on something the formula ignores (airflow, time, weather)")
        L.append("  (terms that move together, e.g. outside and wall, share the residual: read their slopes together)")
    L.append(f"\ncomplexity {rec['complexity']} AST nodes, history {rec['history']} steps, horizons {', '.join(_hname(h) for h in hs)}")
    return {"text": "\n".join(L), "groups": G, "terms": terms, "rooms": Sp["rooms"]}


def coefficient_values(src, ep, U, t=None):
    """A term-form candidate's coefficients evaluated on one episode at step t (default: the middle), from the measured
    history up to t only.  Returns {term: mean over rooms} ('neighbours': mean row sum), or {} for other forms."""
    try:
        mod, _ = contract.load_closedform(src)
    except contract.ContractError:
        return {}
    if not coefform.is_coefform(mod) or hasattr(mod, "nodes"):
        return {}
    tm = coefform.make_model(mod, ep.geo, U); prep = Prepared(ep, mod.HISTORY)
    ts = np.array([ep.n // 2 if t is None else t])
    c = tm.build(prep.hist(ts, prep.measured_window(ts).astype(float)))
    out = {k: float(np.mean(c[k])) for k in coefform.TERMS_ROOM}
    out["neighbours"] = float(np.mean(c["nb_sum"]))
    return out


def crossover_analysis(a, b, names=("A", "B"), ep=None, U=None):
    """Pre-crossover analysis: where (rooms, room classes) and when (hours, HVAC state, horizons) each parent is better,
    how much a per-room best-of-both would gain, which term each parent gets right (term regression slopes), which
    functions differ, and how far apart the coefficient values are.  a, b: {"analysis": {...}, "code": src}."""
    import ast
    A, Bn = names
    ga, gb = a["analysis"].get("groups", {}), b["analysis"].get("groups", {})
    L = [f"WHERE AND WHEN EACH PARENT IS BETTER ({A} vs {Bn}, rollout RMSE, K)"]
    wins = {A: [], Bn: []}
    for part, label in (("space: ", "SPATIAL"), ("time: ", "TEMPORAL")):
        L.append(f"  {label}")
        for k in ga:
            if k.startswith(part) and k in gb:
                d = ga[k]["rmse"] - gb[k]["rmse"]
                who = Bn if d > 0 else A
                clear = abs(d) > 0.05 * max(ga[k]["rmse"], gb[k]["rmse"], 1e-9)
                L.append(f"    {k[len(part):]:<44} {ga[k]['rmse']:.3f} vs {gb[k]['rmse']:.3f}  {('-> ' + who) if clear else '(tie)'}")
                if clear:
                    wins[who].append(k[len(part):])
    n_groups = sum(1 for k in ga if k.startswith(("space: ", "time: ")) and k in gb)
    for w, ks in wins.items():
        if ks and len(ks) == n_groups:
            L.append(f"  {w} is clearly better everywhere: combine by taking {w}'s formulas and only what the TERMS lines favour from the other")
        elif ks:
            L.append(f"  {w} is clearly better on: {', '.join(ks)}")
    ra, rb = a["analysis"].get("rooms", {}), b["analysis"].get("rooms", {})
    common = [k for k in ra if k in rb and len(ra[k]["rmse"]) == len(rb[k]["rmse"])]
    if common:
        xa = np.concatenate([np.array(ra[k]["rmse"]) for k in common]); xb = np.concatenate([np.array(rb[k]["rmse"]) for k in common])
        L.append(f"  per room: {A} better in {int((xa < xb).sum())} of {len(xa)} validation rooms; rmse {np.sqrt(np.mean(xa ** 2)):.3f} vs "
                 f"{np.sqrt(np.mean(xb ** 2)):.3f}, best-of-both per room {np.sqrt(np.mean(np.minimum(xa, xb) ** 2)):.3f} "
                 "(the most a combination that keeps each parent's good rooms could reach)")
    ta, tb = a["analysis"].get("terms", {}).get("all rooms", {}), b["analysis"].get("terms", {}).get("all rooms", {})
    if ta and tb:
        L.append("TERMS (one-step residual slope per term; nearer 0 = that term's formula is closer to right)")
        for k in TERM_NAMES:
            if k in ta and k in tb:
                better = A if abs(ta[k]["slope"]) < abs(tb[k]["slope"]) else Bn
                L.append(f"    {k:<11} {ta[k]['slope']:+.3f} vs {tb[k]['slope']:+.3f}   -> take '{k}' from {better}")
        TA, TB = a["analysis"].get("terms", {}), b["analysis"].get("terms", {})
        sub = [g for g in TA if g != "all rooms" and g in TB]
        if sub:
            L.append("  by room type and condition: which parent's formula is closer to right (|slope| smaller) for each term")
            L.append("    " + f"{'':<38}" + "".join(f"{k:>11}" for k in TERM_NAMES))
            for g in sub:
                cells = []
                for k in TERM_NAMES:
                    if k in TA[g] and k in TB[g]:
                        sa, sb = TA[g][k]["slope"], TB[g][k]["slope"]
                        cells.append(f"{(A if abs(sa) < abs(sb) else Bn) + f' {min(abs(sa), abs(sb)):.2f}':>11}")
                    else:
                        cells.append(f"{'':>11}")
                L.append("    " + f"{g:<38}" + "".join(cells))

    def funcs(src):
        try:
            return {n.name: ast.unparse(n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
        except SyntaxError:
            return {}
    fa, fb = funcs(a["code"]), funcs(b["code"])
    L.append("FUNCTIONS: same in both: " + (", ".join(k for k in fa if k in fb and fa[k] == fb[k]) or "none")
             + "; different: " + (", ".join(k for k in fa if k in fb and fa[k] != fb[k]) or "none")
             + f"; only in {A}: " + (", ".join(k for k in fa if k not in fb) or "none")
             + f"; only in {Bn}: " + (", ".join(k for k in fb if k not in fa) or "none"))
    if ep is not None and U is not None:
        va, vb = coefficient_values(a["code"], ep, U), coefficient_values(b["code"], ep, U)
        if va and vb:
            L.append(f"COEFFICIENT VALUES on a validation building, mean over rooms ({A} vs {Bn}); most different first:")
            rel = {k: abs(va[k] - vb[k]) / max(abs(va[k]), abs(vb[k]), 1e-12) for k in va}
            for k in sorted(rel, key=lambda k: -rel[k]):
                if va[k] or vb[k]:
                    L.append(f"    {k:<12} {va[k]:.4g} vs {vb[k]:.4g}")
    return {"text": "\n".join(L), "wins": wins}


# ----------------------------------------------------------------------------- profiles for plots

def profile(src, ep, U, horizon=36, rooms=None):
    """Measured vs predicted series on one episode: one-step and `horizon`-ahead forecasts at every step."""
    mod, _ = contract.load_closedform(src)
    coef = prepare_model(mod, ep.geo, U)
    prep = Prepared(ep, mod.HISTORY)
    n = ep.n
    t1 = np.arange(0, n)
    one = rollout(mod, coef, prep, U, t1, 1)[:, 0]                       # forecast of X[t+1]
    starts = np.arange(0, n - horizon + 1)
    hz = rollout(mod, coef, prep, U, starts, horizon)[:, -1]             # forecast of X[s+horizon]
    if rooms is None:                     # the most exposed exterior room and the largest interior room
        expo = ep.geo["facade_faces"] * 0.2 / np.maximum(ep.geo["area"], 1e-9)
        inter = np.where(ep.geo["facade_faces"] == 0)[0]
        rooms = [int(np.argmax(expo)), int(inter[np.argmax(ep.geo["area"][inter])]) if len(inter) else int(np.argmin(expo))]
    return dict(X=prep.X, one=one, hz=hz, hz_t=starts + horizon, horizon=horizon, rooms=rooms,
                T_out=ep.inputs["T_out"], comfort=ep.inputs["comfort"], hvac=ep.inputs["g"].mean(1))


# ----------------------------------------------------------------------------- ceiling (diagnostic only)

CEILING_SRC = '''
"""Per-building, per-room linear model FITTED on the first half of the episode (diagnostic only: shows how
well a model with free per-room numbers can do on this data; never a deployable candidate).  Weights are
non-negative and the step uses the stable end-of-step form, so the ceiling itself cannot blow up in free run."""
import numpy as np
import dlib
HISTORY = 36
def sources(geo, hist, u, U):
    T = hist["T"][:, -1]
    W = geo["door_cells"] + 0.2 * geo["wall_cells"]
    nb = (T @ W.T) / np.maximum(W.sum(1), 1e-9)
    w = np.exp(-np.arange(HISTORY)[::-1] / 12.0)
    slow = np.tensordot(hist["T"], w / w.sum(), axes=([1], [0]))
    return T, [u["T_out"][:, None] + 0 * T, u["T_supply"], nb, slow], [np.ones_like(T), u["g"] / 1e3, np.ones_like(T), np.ones_like(T)], (u["Q_reheat"] + u["Q_gain"]) / 1e4
def step(coef, geo, hist, u, U):
    T, src, scale, q = sources(geo, hist, u, U)
    return dlib.assemble(T, [(coef[None, :, k] * scale[k], src[k]) for k in range(len(src))], coef[None, :, -1] * q)
'''


def ceiling(ep, U, horizons=(12, 36, 72), every=6, warmup=36):
    """Fit CEILING_SRC per room on the first half of `ep` (non-negative least squares on the one-step change);
    score free-run rollouts on the second half.  A diagnostic upper reference, never deployed."""
    from scipy.optimize import nnls
    mod = contract._exec(__import__("ast").parse(CEILING_SRC)); mod.HISTORY = 36
    prep = Prepared(ep, mod.HISTORY)
    half = ep.n // 2
    ts = np.arange(warmup, half)
    T, src, scale, q = mod.sources(ep.geo, prep.hist(ts, prep.measured_window(ts)), prep.u(ts), U)
    A = np.stack([(s_ - T) * sc for s_, sc in zip(src, scale)] + [q], -1)       # (n, V, 5): explicit one-step regressors
    y = prep.X[ts + 1] - prep.X[ts]
    coef = np.zeros((ep.V, A.shape[-1]))
    for i in range(ep.V):
        coef[i] = nnls(A[:, i], y[:, i])[0]
    hmax = max(horizons)
    starts = np.arange(max(half, warmup), ep.n - hmax, every)
    out = {}
    if len(starts):
        pr = rollout(mod, coef, prep, U, starts, hmax)
        for h in horizons:
            out[_hname(h)] = float(np.sqrt(np.mean((pr[:, h - 1] - prep.X[starts + h]) ** 2)))
            out[f"persistence_{_hname(h)}"] = float(np.sqrt(np.mean((prep.X[starts] - prep.X[starts + h]) ** 2)))
    return out
