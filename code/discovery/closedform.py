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
           with a stability check; score = mean RMSE at the main horizon on held-out buildings + complexity
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

from . import contract

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


def _u_at(prep, ts):
    return {k: v[ts] for k, v in prep.ep.inputs.items()}


def filter_states(mod, coef, prep, U):
    """Stateful candidates (backlog O3): state after correcting with each measurement, S[k] <-> X[k].
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


def _rollout_stateful(mod, coef, prep, U, starts, horizon):
    S = filter_states(mod, coef, prep, U)
    geo, V = prep.ep.geo, prep.V
    out = np.empty((len(starts), horizon, V))
    s = S[starts]
    for k in range(horizon):
        s = np.asarray(mod.predict(coef, geo, s, _u_at(prep, starts + k), U), float)
        out[:, k] = s[:, :V]
    return out


def rollout(mod, coef, prep, U, starts, horizon, chunk=None):
    """Free-run every start in `starts` for `horizon` steps from measured history.
    Returns pred (S, horizon, V): pred[s, k] is the forecast of X[starts[s] + k + 1]."""
    starts = np.asarray(starts, int)
    if is_stateful(mod):
        return _rollout_stateful(mod, coef, prep, U, starts, horizon)
    chunk = chunk or int(max(8, min(1024, 2e6 // (prep.H * prep.V))))       # bounds the (B, H, V) windows in memory
    out = np.empty((len(starts), horizon, prep.V))
    for a in range(0, len(starts), chunk):
        ts0 = starts[a:a + chunk]
        Tb = prep.measured_window(ts0).astype(float)
        for k in range(horizon):
            ts = ts0 + k
            y = np.asarray(mod.step(coef, prep.ep.geo, prep.hist(ts, Tb), prep.u(ts), U), float)
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
    """Warning only (backlog F4): shifting every temperature (start, outside, supply, water) by c should shift the
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
    """Diagnostic (backlog F5): 24 h free runs from measured history, one per day; RMSE and mean drift at 24 h."""
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
        Tw = run(284.0)                                   # backlog F4: a warmer outside never makes a room colder
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

def episode_scores(mod, coef, ep, U, horizons=(12, 36, 72), every=6, warmup=36, keep=36):
    """One-step and rollout errors on one episode, with persistence on the same windows.
    Errors at horizon `keep` are kept (per start and room) for the analysis."""
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
    r["keep_err"] = pr[:, keep - 1] - X[starts + keep]
    r["keep_t"] = starts + keep - 1                  # index of the input step that ends at the target
    return r


def _hname(h):
    return f"{h / 12:g}h" if h % 12 == 0 else f"{h}st"


# ----------------------------------------------------------------------------- the full evaluation

def evaluate(src, train_eps, val_eps, U, complexity_weight=0.0001, horizons=(12, 36, 72), main_h=36, every=6,
             warmup=36, screen_factor=5.0, max_train=8):
    """Everything about one closed-form candidate as a JSON-able record (+ 'detail' with arrays for the analysis)."""
    t0 = time.time()
    rec = {"status": "ok", "deployment": "closedform"}
    try:
        mod, cx = contract.load_closedform(src)
        rec.update(complexity=cx, history=mod.HISTORY)
        coefs = {}

        def coef_of(ep):
            if ep.building not in coefs:
                coefs[ep.building] = mod.prepare(ep.geo, U) if hasattr(mod, "prepare") else None
            return coefs[ep.building]
        fails = physics_checks(mod, coef_of(train_eps[0]), train_eps[0], U)
        if fails:
            return {**rec, "status": "rejected", "stage": 0, "reason": "physics checks failed", "failures": fails,
                    "sec": round(time.time() - t0, 1)}
        # stage 1: one-step screen on held-out buildings
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
                              f"({rec['stage1_persistence_mean']:.4f} K) on held-out buildings", "sec": round(time.time() - t0, 1)}
        # stage 2: rollouts (decisive) on held-out buildings; train buildings reported for the diagnostics
        val = {e.name: episode_scores(mod, coef_of(e), e, U, horizons, every, warmup, keep=main_h) for e in val_eps}
        tr = {e.name: episode_scores(mod, coef_of(e), e, U, horizons, every, warmup, keep=main_h) for e in train_eps[:max_train]}
        rec["unstable_episodes"] = [k for k, v in val.items() if v["unstable"]]
        for part, D in (("val", val), ("train", tr)):
            rec[f"{part}_one_step_mean"] = float(np.mean([v["one_step"] for v in D.values()]))
            for h in horizons:
                rec[f"{part}_{_hname(h)}_mean"] = float(np.mean([v[f"{h}"] for v in D.values()]))
                rec[f"{part}_persistence_{_hname(h)}_mean"] = float(np.mean([v[f"persistence_{h}"] for v in D.values()]))
        lr = [long_run(mod, coef_of(e), e, U) for e in val_eps]
        lr = [x for x in lr if x is not None and np.isfinite(x[0])]
        if lr:
            rec["val_24h_mean"] = float(np.mean([x[0] for x in lr])); rec["val_24h_drift_mean"] = float(np.mean([x[1] for x in lr]))
        w = shift_warning(mod, coef_of(train_eps[0]), train_eps[0], U)
        if w:
            rec["warnings"] = w
        rec["val_by_episode"] = {k: {kk: v[kk] for kk in v if not kk.endswith(("_err", "_t")) and kk != "unstable"} for k, v in val.items()}
        rec["horizons"] = list(horizons); rec["main_horizon"] = main_h
        m = rec[f"val_{_hname(main_h)}_mean"]
        if rec["unstable_episodes"]:
            rec.update(status="unstable", stage=2, reason=f"free-run rollout left [230, 350] K or went non-finite on "
                       f"{len(rec['unstable_episodes'])} held-out episode(s)")
            m = float("inf")
        rec["score_metric"] = f"val {_hname(main_h)} rollout RMSE from measured history (no fitting)"
        rec["score"] = m + complexity_weight * cx
        rec["detail"] = val
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()
        rec.update(status="error", reason=f"{type(e).__name__}: {e}", traceback="\n".join(tb[-6:]))
    rec["sec"] = round(time.time() - t0, 1)
    return rec


# ----------------------------------------------------------------------------- analysis (pre-mutation / pre-crossover)

def _group_stats(rec, val_eps):
    """RMSE and bias of the main-horizon rollout error, by room class and by condition."""
    det = rec["detail"]
    acc = {}

    def add(key, e):
        if e.size:
            s = acc.setdefault(key, [0.0, 0.0, 0])
            s[0] += float((e ** 2).sum()); s[1] += float(e.sum()); s[2] += e.size
    for ep in val_eps:
        d = det[ep.name]; E = d["keep_err"]; t = d["keep_t"]
        geo, inp = ep.geo, ep.inputs
        ext = geo["facade_faces"] > 0
        expo = geo["facade_faces"] * 0.2 / np.maximum(geo["area"], 1e-9)
        add("all rooms", E)
        add("exterior rooms", E[:, ext]); add("interior rooms", E[:, ~ext])
        if ext.sum() >= 3:
            hi = expo >= np.quantile(expo[ext], 2 / 3)
            add("most exposed third of exterior rooms", E[:, ext & hi])
        act = inp["g"][t].mean(1) > 50.0
        add("HVAC air on", E[act]); add("HVAC air off", E[~act])
        com = inp["comfort"][t] > 0.5
        add("comfort hours", E[com]); add("night/weekend", E[~com])
        rh = (inp["Q_reheat"][t] > 1.0)
        add("room being reheated", E[rh]); add("room not reheated", E[~rh])
        cold = inp["T_out"][t] < np.median(inp["T_out"])
        add("colder half of outside temps", E[cold]); add("warmer half", E[~cold])
    return {k: {"rmse": float(np.sqrt(s[0] / s[2])), "bias": float(s[1] / s[2]), "n": s[2]} for k, s in acc.items()}


def analysis(rec, val_eps):
    """Pre-mutation analysis of one parent: where its error is, as numbers and as text for the prompt."""
    if rec.get("status") not in ("ok", "unstable") or "detail" not in rec:
        lines = [f"status: {rec.get('status')} (stage {rec.get('stage', '?')}) - {rec.get('reason', '')}"]
        lines += [f"  - {f}" for f in rec.get("failures", [])]
        if rec.get("traceback"):
            lines.append("  traceback:\n    " + rec["traceback"].replace("\n", "\n    "))
        return {"text": "\n".join(lines), "groups": {}}
    hs = rec["horizons"]; mh = _hname(rec["main_horizon"])
    G = _group_stats(rec, val_eps)
    L = ["held-out rollout RMSE by horizon (model / persistence), K: " +
         ", ".join(f"{_hname(h)} {rec[f'val_{_hname(h)}_mean']:.3f}/{rec[f'val_persistence_{_hname(h)}_mean']:.3f}" for h in hs),
         f"one-step (5 min) RMSE {rec['val_one_step_mean']:.4f} K (persistence {rec['stage1_persistence_mean']:.4f}); "
         f"train buildings {mh} {rec[f'train_{mh}_mean']:.3f} K",
         f"{mh} rollout error by group (rmse, bias), K:"]
    for k, v in G.items():
        L.append(f"  {k:<38} {v['rmse']:.3f}  {v['bias']:+.3f}")
    g = [rec[f"val_{_hname(h)}_mean"] / max(rec[f"val_persistence_{_hname(h)}_mean"], 1e-9) for h in hs]
    L.append("skill vs persistence by horizon (ratio <1 is better): " + ", ".join(f"{_hname(h)} {x:.2f}" for h, x in zip(hs, g)))
    if g[-1] > g[0] * 1.3:
        L.append("  -> ratio grows with horizon: slow heat stores or a steady-state gain are off (history window too short?)")
    worst = []
    for ep in val_eps:
        E = rec["detail"][ep.name]["keep_err"]
        rms = np.sqrt((E ** 2).mean(0)); b = E.mean(0)
        for i in np.argsort(-rms)[:2]:
            worst.append((rms[i], b[i], ep.name, i, int(ep.geo["facade_faces"][i]), float(ep.geo["area"][i]), int(ep.geo["n_doors"][i].sum())))
    worst.sort(key=lambda w: -w[0])
    L.append("worst rooms: " + "; ".join(f"{w[2]} room #{w[3]} (rmse {w[0]:.2f}, bias {w[1]:+.2f}, facade faces {w[4]}, "
                                          f"area {w[5]:.0f} m2, doors {w[6]})" for w in worst[:4]))
    L.append(f"complexity {rec['complexity']} AST nodes, history {rec['history']} steps")
    return {"text": "\n".join(L), "groups": G}


def crossover_analysis(a, b, names=("A", "B")):
    """Pre-crossover analysis: which parent is better where, and which templates (top-level functions) differ."""
    import ast
    L = [f"Where each parent is better ({names[0]} vs {names[1]}, {''}main-horizon rollout RMSE, K):"]
    ga, gb = a["analysis"].get("groups", {}), b["analysis"].get("groups", {})
    wins = {names[0]: [], names[1]: []}
    for k in ga:
        if k in gb:
            d = ga[k]["rmse"] - gb[k]["rmse"]
            who = names[1] if d > 0 else names[0]
            L.append(f"  {k:<38} {ga[k]['rmse']:.3f} vs {gb[k]['rmse']:.3f}  -> {who}")
            if abs(d) > 0.02 * max(ga[k]["rmse"], 1e-9):
                wins[who].append(k)
    for w, ks in wins.items():
        if ks:
            L.append(f"{w} is clearly better on: {', '.join(ks)}")

    def funcs(src):
        try:
            return {n.name: ast.unparse(n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
        except SyntaxError:
            return {}
    fa, fb = funcs(a["code"]), funcs(b["code"])
    L.append("templates (top-level functions): same in both: " + (", ".join(k for k in fa if k in fb and fa[k] == fb[k]) or "none")
             + "; different: " + (", ".join(k for k in fa if k in fb and fa[k] != fb[k]) or "none")
             + f"; only in {names[0]}: " + (", ".join(k for k in fa if k not in fb) or "none")
             + f"; only in {names[1]}: " + (", ".join(k for k in fb if k not in fa) or "none"))
    return {"text": "\n".join(L), "wins": wins}


# ----------------------------------------------------------------------------- profiles for plots

def profile(src, ep, U, horizon=36, rooms=None):
    """Measured vs predicted series on one episode: one-step and `horizon`-ahead forecasts at every step."""
    mod, _ = contract.load_closedform(src)
    coef = mod.prepare(ep.geo, U) if hasattr(mod, "prepare") else None
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
