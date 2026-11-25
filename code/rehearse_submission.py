"""Local rehearsal of the Kaggle submission: the notebook's policy core through the kit's own harness.

    python code/rehearse_submission.py                       # all 30 shipped scenarios, 8,640 steps (comparable with baselines)
    python code/rehearse_submission.py --scenarios building_train_8/atlanta_1 --steps 576     # quick check
    python code/rehearse_submission.py --model outputs/discovery/main/best.py               # also carry the tracker

Writes outputs/rehearsal/<tag>/report.md: per-scenario score (mean loss per step, lower is better), wall time,
seconds per step, safety repairs, and, at 8,640 steps, the kit baselines for the same scenarios.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import KIT, WC, import_kit  # noqa: E402
from submission.transfer_policy import TransferPolicyCore  # noqa: E402


def make_cls(model_src, controller):
    from submission.rule_policy import RulePolicy
    from submission.vendor.pid_plus import PIDWaterResetPolicy
    if controller in ("ff", "mpc"):
        from control.policies import FeedForwardPolicy, MPCPolicy
        base = {"ff": FeedForwardPolicy, "mpc": MPCPolicy}[controller]

        def ctrl(meta, spec):                         # the model-based controllers carry the model themselves
            return base(meta, spec, model_src=model_src)
        core_model = None
    else:
        ctrl = {"pid_water": PIDWaterResetPolicy, "rule": RulePolicy}[controller]
        core_model = model_src

    class Core(TransferPolicyCore):
        instances = []

        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec, model_src=core_model, controller_cls=ctrl)
            Core.instances.append(self)
    return Core


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", help="building/weather pairs (default: the 30 in training_set.yaml)")
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--model", type=Path, help="closed-form best.py to carry as a tracker")
    ap.add_argument("--controller", default="pid_water", choices=["pid_water", "rule", "ff", "mpc"],
                    help="ff / mpc need --model (a closed-form best.py, e.g. code/discovery/seeds_cf/seed_cf_network.py)")
    ap.add_argument("--tag", help="default: the controller name")
    a = ap.parse_args()
    import yaml
    sd = KIT / "smart_buildings" / "training_scenarios"
    scen = a.scenarios or [f"{s['building']}/{s['weather']}" for s in yaml.safe_load(open(KIT / "smart_buildings" / "training_set.yaml"))["scenarios"]]
    a.tag = a.tag or a.controller
    out = (WC / "outputs" / "rehearsal" / a.tag).resolve(); out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(WC / ".numba_cache"))
    ev = import_kit()
    cls = make_cls(a.model.read_text() if a.model else None, a.controller)
    base = json.loads((KIT / "baseline_returns.json").read_text())
    rows = []
    for s in scen:
        b, w = s.split("/")
        cwd = os.getcwd(); os.chdir(KIT); t0 = time.time()
        try:
            summ = ev.evaluate_policy_for_config(out / "results" / b / w, (sd / b / w).resolve(), "transfer", cls, a.steps)
        finally:
            os.chdir(cwd)
        core = cls.instances[-1]
        r = dict(scenario=s, score=-summ["avg_reward"], steps=summ["total_steps"], wall_s=round(time.time() - t0, 1),
                 sec_per_step=summ.get("seconds_per_step"), repairs=core.safe.repairs, tracker_errors=core.tracker_errors)
        if a.steps == 8640 and s in base:
            r.update({f"base_{k}": -v / 8640 for k, v in base[s].items()})
        if core.tracker is not None and len(core.tracker.X) > 100:
            r["tracker_one_step"] = core.tracker.live_one_step_rmse()
        ctl = core.controller
        if hasattr(ctl, "log"):
            from collections import Counter
            r["modes"] = dict(Counter(m for _, m, _ in ctl.log))
            r["plan_ms_mean"] = 1e3 * float(np.mean(ctl.plan_seconds)) if ctl.plan_seconds else 0.0
        rows.append(r)
        print(f"{s}: score {r['score']:.5f}" + (f" (pid_water {r['base_pid_water']:.5f})" if "base_pid_water" in r else "")
              + f", repairs {r['repairs']}" + (f", modes {r['modes']}, plan {r['plan_ms_mean']:.0f} ms" if "modes" in r else ""), flush=True)
    mean = float(np.mean([r["score"] for r in rows]))
    L = [f"# Rehearsal `{a.tag}` ({a.steps} steps per scenario)\n", f"Mean score over {len(rows)} scenarios: **{mean:.5f}** (lower is better).\n",
         "| scenario | score | pid_water | best baseline | safety repairs | s/step |", "|---|---|---|---|---|---|"]
    for r in rows:
        bb = min((v for k, v in r.items() if k.startswith("base_")), default=float("nan"))
        L.append(f"| {r['scenario']} | {r['score']:.5f} | {r.get('base_pid_water', float('nan')):.5f} | {bb:.5f} | {r['repairs']} | {r['sec_per_step'] or float('nan'):.3f} |")
    (out / "report.md").write_text("\n".join(L) + "\n"); (out / "report.json").write_text(json.dumps(dict(mean=mean, runs=rows), indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
