"""Step 5: test a state-space model on held-out buildings and stations.

    python code/test_model.py outputs/discovery/main/best.py --params outputs/discovery/main/summary.json
    python code/test_model.py code/discovery/seeds/seed_slab_wall_network.py --on val      # fits PARAMS on train first

A model is any module that obeys the candidate contract (code/discovery/contract.py): PARAMS,
THETA, simulate(geo, inputs, theta, p, T0), and optionally network(geo, inputs, theta, p).

For every scenario of the chosen split (default: test), exactly as a deployed policy would:
  1. global constants PARAMS: from --params (a summary.json of a discovery run, or a JSON dict),
     else fitted on the train split;
  2. per-building THETA: fitted on DAY 1 of that scenario only (the Sunday);
  3. open-loop RMSE of room temperature on day 1 and on every later day, from the reset state;
  4. if the model defines network(): a Kalman filter on the measured room temperatures, and the
     1-, 3- and 6-hour forecast RMSE from the filtered state (what MPC uses), against persistence.
Results by run, by station, by policy and by building go to outputs/test/<model>_<split>/report.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, WC  # noqa: E402
from discovery import contract, dlib, evaluate  # noqa: E402
from discovery.episodes import _load_cached  # noqa: E402
from discovery.loop import split_episodes  # noqa: E402

DAY = 288


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset")
    ap.add_argument("--split", type=Path, help="default: <dataset>/split.json")
    ap.add_argument("--on", default="test", help="split to test on (test or val)")
    ap.add_argument("--params", type=Path, help="summary.json of a discovery run, or a JSON dict of PARAMS")
    ap.add_argument("--max-nfev", type=int, default=40)
    ap.add_argument("--no-forecast", action="store_true")
    a = ap.parse_args()
    mod, cx = contract.load(a.model.read_text())
    idx = json.loads((a.dataset / "index.json").read_text())
    split = json.loads((a.split or a.dataset / "split.json").read_text())
    eps = [_load_cached(a.dataset / v["file"]) for v in idx.values() if v.get("status") == "ok"]
    S = split_episodes(eps, split)
    target = S.get(a.on) or []
    if not target:
        raise SystemExit(f"split '{a.on}' has no usable runs")
    fails = evaluate.physics_checks(mod, target[0])
    if fails:
        print("physics checks failed:\n  " + "\n  ".join(fails)); sys.exit(1)

    if a.params:
        P = json.loads(a.params.read_text())
        p = P["best"]["params"] if "best" in P else P
        src = str(a.params)
    else:
        print(f"fitting PARAMS on train ({len(S['train'])} runs) ...")
        p, _, _ = evaluate.fit_train(mod, S["train"], a.max_nfev, stride=2)
        src = "fitted on train"
    print(f"PARAMS ({src}): {p}")

    res = evaluate.score_heldout(mod, target, p, a.max_nfev)
    rows = []
    for e in target:
        d = res["episodes"][e.name]
        err = d["err"]
        days = [float(np.sqrt((err[i * DAY:(i + 1) * DAY] ** 2).mean())) for i in range(int(np.ceil(len(err) / DAY)))]
        row = dict(run=e.name, building=e.building, station=e.weather.rsplit("_", 1)[0], weather=e.weather, policy=e.policy,
                   rooms=e.V, open_loop_day1=d["rmse_day1"], open_loop_after_day1=float(np.sqrt((err[DAY:] ** 2).mean())),
                   open_loop_by_day=days, bias_after_day1=float(err[DAY:].mean()))
        if hasattr(mod, "network") and not a.no_forecast:
            th = res["scenarios"][e.scenario]["theta"]
            net, f = mod.network(e.geo, e.inputs, th, p)
            fc = dlib.filtered_forecast(net, f["T_out"], f["g"], f["T_supply"], f["Q"], e.T_meas, e.T0)
            row.update({f"forecast_{h // 12}h": v[0] for h, v in fc.items()})
            row.update({f"persistence_{h // 12}h": v[1] for h, v in fc.items()})
        rows.append(row)
        print(f"  {e.name}: day1 {row['open_loop_day1']:.3f} K, after {row['open_loop_after_day1']:.3f} K"
              + (f", forecast 1/3/6 h {row['forecast_1h']:.3f}/{row['forecast_3h']:.3f}/{row['forecast_6h']:.3f} K" if "forecast_1h" in row else ""))

    keys = [k for k in rows[0] if k.startswith(("open_loop_day1", "open_loop_after", "forecast_", "persistence_"))]

    def agg(by):
        g = defaultdict(list)
        for r in rows:
            g[r[by]].append(r)
        return {k: {m: float(np.mean([x[m] for x in v])) for m in keys} | {"runs": len(v)} for k, v in sorted(g.items())}
    report = dict(model=str(a.model), split=a.on, params=p, params_source=src, complexity=cx,
                  theta=res["scenarios"], overall={m: float(np.mean([r[m] for r in rows])) for m in keys},
                  by_station=agg("station"), by_policy=agg("policy"), by_building=agg("building"), runs=rows)
    out = WC / "outputs" / "test" / f"{a.model.stem}_{a.on}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    L = [f"# Test of `{a.model}` on `{a.on}`\n", f"PARAMS ({src}): `{json.dumps(p)}`  \ncomplexity {cx} AST nodes\n",
         "Open loop from the reset state with THETA fitted on day 1; forecasts from a Kalman-filtered state. RMSE in K.\n"]
    for title, tab in (("Overall", {"all": report["overall"] | {"runs": len(rows)}}), ("By station", report["by_station"]),
                       ("By policy", report["by_policy"]), ("By building", report["by_building"])):
        L.append(f"\n## {title}\n\n| | runs | " + " | ".join(keys) + " |\n|---|---|" + "---|" * len(keys))
        for k, v in tab.items():
            L.append(f"| {k} | {v['runs']} | " + " | ".join(f"{v[m]:.3f}" for m in keys) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
