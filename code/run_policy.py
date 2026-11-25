"""Run a submission-style policy on held-out scenarios, exactly as the hidden evaluation does.

The hidden evaluation (Kaggle Overview and Data tabs): the submitted notebook's Policy class is
instantiated once per unseen (building, weather) scenario with that building's metadata, then
get_action(obs) is called every 5 minutes for 8,640 steps (30 days); every step's reward counts.
This script does the same locally with the starter kit's own harness
(smart_buildings.evaluate.evaluate_policy_for_config), on held-out scenarios (a test tier of
the split computed by make_splits.py: buildings and stations the discovery search never saw).

The policy is runtime/model_policy.py: control from a base policy (default pid_water), with a closed-form model
(a discovery run's best.py or a hand-written model) built from the metadata and tracking the building; nothing fitted.
Reported per scenario: the competition score (mean per-step reward), seconds per step, and the
model's 1/3/6 h forecast error from its filtered state, computed live inside the policy.

    python code/run_policy.py --model outputs/discovery/main/best.py --on test_medium --check-replay
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, KIT, WC, import_kit  # noqa: E402
from runtime.model_policy import make_closedform_policy  # noqa: E402


def model_tag(path):
    """Report name for a model file: a discovery run's best.py is named after its run."""
    path = Path(path)
    return f"{path.parent.name}_{path.stem}" if path.stem == "best" else path.stem


def scenarios_for(split, part, scen_dir):
    pats = split.get(part) or []
    allp = sorted(f"{b.name}/{w.name}" for b in Path(scen_dir).iterdir() if b.is_dir() for w in b.iterdir() if (w / "config.gin").exists())
    inc = [p for p in pats if not p.startswith("!")]; exc = [p[1:] for p in pats if p.startswith("!")]

    def m(name, p):
        return fnmatch.fnmatch(name, p) if "/" in p else fnmatch.fnmatch(name.split("/")[0], p)
    return [s for s in allp if any(m(s, p) for p in inc) and not any(m(s, p) for p in exc)]


def replay_check(tracker, run_dir, scenario_dir, base_key):
    """J3: the live tracker's inputs and scores must equal the offline replay of the same run."""
    from discovery import closedform as cf
    from discovery import starterkit_data as skd
    ctx = skd.ScenarioContext(scenario_dir)
    R = skd.read_results(run_dir)
    C, Dd = skd.replay(ctx, base_key, R)
    inputs, T_meas, chk, _ = skd.episode_inputs(ctx, R, C, Dd)
    live = tracker.to_episode()
    n = min(live.n, len(T_meas))
    perm = np.array([ctx.rooms.index(r) for r in tracker.io.rooms])        # offline (simulator VAV) order -> observation order
    off = {k: (np.asarray(v)[:n][:, perm] if np.ndim(v) == 2 else np.asarray(v)[:n]) for k, v in inputs.items()}
    diffs = {k: float(np.nanmax(np.abs(np.asarray(live.inputs[k][:n]) - off[k])))
             for k in ("T_out", "g", "T_supply", "Q_reheat", "Q_gain", "occupants", "comfort") if k in off}
    diffs["T_meas"] = float(np.max(np.abs(live.T_meas[:n] - np.asarray(T_meas)[:n][:, perm])))
    geo_diff = {}                                   # geometry from the Kaggle-style metadata (live) vs the offline episodes
    for k, v in tracker.geo.items():
        o = np.asarray(ctx.geo.get(k)) if k in ctx.geo else None
        if o is None or np.ndim(v) == 0:
            continue
        o = o[np.ix_(perm, perm)] if np.ndim(v) == 2 else o[perm] if np.ndim(v) == 1 else o
        if k == "ahu":                              # AHU numbering: compare the grouping, not the labels
            same = all(len(set(o[np.asarray(v) == a])) == 1 for a in set(np.asarray(v).tolist()))
            geo_diff[k] = 0.0 if same else 1.0
        else:
            geo_diff[k] = float(np.max(np.abs(np.asarray(v, float) - o.astype(float))))
    ep_off = live.__class__(**{**live.__dict__, "inputs": {**live.inputs, **{k: off[k] for k in off if k in live.inputs}},
                               "T_meas": np.asarray(T_meas)[:n][:, perm]})
    r = cf.episode_scores(tracker.mod, tracker.coef, ep_off, tracker.U, (12, 36, 72), 12, 36, keep=36)
    return {"replay_max_abs_diff": diffs, "geometry_max_abs_diff": geo_diff, "replay_flow_max_rel_err": chk["flow_max_rel_err"],
            "offline_one_step": r["one_step"], "offline_3h": r["36"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=HERE / "discovery/seeds_cf/seed_cf_network.py",
                    help="a closed-form model: a discovery run's best.py or a hand-written one in discovery/seeds_cf/")
    ap.add_argument("--scenarios-dir", type=Path, default=DATA / "scenarios")
    ap.add_argument("--split", type=Path, default=DATA / "split.json")
    ap.add_argument("--on", default="test_medium", help="split tier: val, test_easy, test_medium, test_hard, test_station")
    ap.add_argument("--steps", type=int, default=8640, help="8640 = 30 days, the hidden evaluation's length")
    ap.add_argument("--base", default="pid_water", help="starter-kit policy key that makes the control decisions")
    ap.add_argument("--out", type=Path, help="default outputs/policy/<model>_<base>_<on>")
    ap.add_argument("--check-replay", action="store_true",
                    help="replay the run offline (starterkit_data) and compare inputs and scores with the live tracker")
    a = ap.parse_args()

    src_text = a.model.read_text()
    src = "closed-form: nothing fitted"
    scen = scenarios_for(json.loads(a.split.read_text()), a.on, a.scenarios_dir)
    if not scen:
        raise SystemExit(f"no scenarios of split '{a.on}' under {a.scenarios_dir}")
    out = (a.out or WC / "outputs" / "policy" / f"{model_tag(a.model)}_{a.base}_{a.on}").resolve()   # before any chdir into the kit
    out.mkdir(parents=True, exist_ok=True)
    a.scenarios_dir = a.scenarios_dir.resolve()

    os.environ.setdefault("NUMBA_CACHE_DIR", str(WC / ".numba_cache"))
    ev = import_kit()
    cls = make_closedform_policy(ev.POLICY_MAP[a.base], src_text, name=f"closedform_tracking_{a.base}")
    rows = []
    for s in scen:
        b, w = s.split("/")
        sdir = (a.scenarios_dir / b / w).resolve()
        cwd = os.getcwd(); os.chdir(KIT)                       # gin configs use paths relative to the kit root
        t0 = time.time()
        try:
            summ = ev.evaluate_policy_for_config((out / "results" / b / w).resolve(), sdir, a.base, cls, a.steps)
        finally:
            os.chdir(cwd)
        pol = cls.instances[-1]
        fc = pol.model.forecast_errors()
        if a.check_replay:
            fc.update(replay_check(pol.model, out / "results" / b / w / pol.name(), sdir, a.base))
        row = dict(scenario=s, rooms=pol.model.V, ahus=pol.model.K, steps=summ["total_steps"],
                   score=-summ["avg_reward"], model_init_s=pol.init_seconds,
                   model_s_per_step=float(np.mean(pol.step_seconds)), wall_s=time.time() - t0, **fc)
        ref = a.scenarios_dir.parent / "results" / b / w / a.base / "summary.json"
        if ref.exists():
            r = json.loads(ref.read_text())
            if r.get("total_steps") == summ["total_steps"]:
                row["base_policy_score"] = -r["avg_reward"]
        rows.append(row)
        print(f"{s}: score {row['score']:.5f}, forecast 1/3/6 h {fc['1h']:.3f}/{fc['3h']:.3f}/{fc['6h']:.3f} K, "
              f"model {row['model_s_per_step'] * 1e3:.1f} ms/step", flush=True)

    keys = ["score", "1h", "3h", "6h", "persistence_3h", "model_s_per_step"]
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    rep = dict(model=str(a.model), constants=src, base=a.base, split=a.on, steps=a.steps,
               mean=mean, runs=rows)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    L = [f"# `{a.model.name}` inside a submission-style policy, on `{a.on}`\n",
         f"Constants: {src} (no fitting on the tested buildings). Control: `{a.base}`; the model tracks the building "
         f"from the metadata, the observations and the policy's own actions only. {a.steps} steps per scenario.\n",
         "| scenario | rooms | score | base policy score | forecast 1 h | 3 h | 6 h | persistence 3 h | model ms/step |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['scenario']} | {r['rooms']} | {r['score']:.5f} | {r.get('base_policy_score', float('nan')):.5f} | "
                 f"{r['1h']:.3f} | {r['3h']:.3f} | {r['6h']:.3f} | {r['persistence_3h']:.3f} | {r['model_s_per_step'] * 1e3:.1f} |")
    L.append(f"| **mean** | | {mean['score']:.5f} | | {mean['1h']:.3f} | {mean['3h']:.3f} | {mean['6h']:.3f} | "
             f"{mean['persistence_3h']:.3f} | {mean['model_s_per_step'] * 1e3:.1f} |")
    L.append("\nScore = mean per-step loss (the leaderboard metric for one scenario; lower is better). The score equals "
             "the base policy's because the model does not yet make decisions; the forecasts are what a model-based "
             "controller would plan with.")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nwrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
