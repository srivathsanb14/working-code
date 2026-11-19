"""Step 5: test a state-space model on held-out buildings and stations.

    python code/test_model.py outputs/discovery/main/best.py --params outputs/discovery/main/summary.json
    python code/test_model.py code/discovery/seeds/seed_slab_wall_network.py --on val      # fits PARAMS on train first

A model is any module that obeys the candidate contract (code/discovery/contract.py): PARAMS,
THETA, simulate(geo, inputs, theta, p, T0), and optionally network(geo, inputs, theta, p).

For every scenario of the chosen split (default: test), exactly as a deployed policy would:
  1. global constants PARAMS: from --params (a summary.json of a discovery run, or a JSON dict),
     else fitted on the train split;
  2. THETA, by --theta:
       shared (default, OFFLINE deployment: no fitting on the new building) - one universal THETA:
              summary.json's theta_shared if given, else fitted jointly with PARAMS on train;
       trainmean - geometric mean of per-train-building THETA (no fitting on the new building);
       prior     - THETA at its initial values (no fitting on the new building);
       day1      - ONLINE identification: THETA fitted on DAY 1 of that scenario only (the Sunday);
  3. open-loop RMSE of room temperature on day 1 and on every later day, from the reset state;
  4. if the model defines network(): a Kalman filter on the measured room temperatures (the state
     corrected from measured history), and the 1-, 3- and 6-hour forecast RMSE from the filtered
     state (what MPC uses), against persistence.  Forecasts use the TRUE future inputs and weather:
     this tests the model, not a closed-loop controller.
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


def model_tag(path):
    """Report name for a model file: a discovery run's best.py is named after its run (main_offline_best)."""
    path = Path(path)
    return f"{path.parent.name}_{path.stem}" if path.stem == "best" else path.stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset")
    ap.add_argument("--split", type=Path, help="default: <dataset>/split.json")
    ap.add_argument("--on", default="test", help="split to test on (test or val)")
    ap.add_argument("--params", type=Path, help="summary.json of a discovery run, or a JSON dict of PARAMS")
    ap.add_argument("--max-nfev", type=int, default=40)
    ap.add_argument("--no-forecast", action="store_true")
    ap.add_argument("--theta", choices=["shared", "trainmean", "prior", "day1", "all"], default="shared",
                    help="shared (default, offline): one universal THETA (summary's theta_shared, else fitted with PARAMS on "
                         "train); trainmean: geometric mean of per-train-building THETA; prior: THETA initial values; "
                         "day1: fit THETA per building on its day 1 (online identification). Except day1, only the Kalman "
                         "filter's state correction adapts to the building. all: run every mode and write a comparison "
                         "(outputs/test/<model>_<split>_compare.md)")
    a = ap.parse_args()
    if a.theta == "all":
        return compare_all(a)
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

    theta_mode = a.theta
    P = json.loads(a.params.read_text()) if a.params else None
    if a.theta == "shared" and P and P.get("best", {}).get("theta_shared"):
        p, theta_mode = P["best"]["params"], P["best"]["theta_shared"]
        src = f"{a.params} (PARAMS and shared THETA)"
    elif a.theta == "shared":
        print(f"fitting PARAMS and ONE shared THETA on train ({len(S['train'])} runs) ...")
        p, ths, _ = evaluate.fit_train(mod, S["train"], a.max_nfev, stride=2, shared=True)
        theta_mode = next(iter(ths.values()))
        src = "fitted on train (with one THETA shared by all train buildings)"
    elif a.theta == "trainmean":
        print(f"fitting PARAMS and per-building THETA on train ({len(S['train'])} runs) to average THETA ...")
        p, ths, _ = evaluate.fit_train(mod, S["train"], a.max_nfev, stride=2)
        theta_mode = {k: float(np.exp(np.mean([np.log(t[k]) for t in ths.values()]))) for k in mod.THETA}
        src = "fitted on train (with THETA averaged over train buildings)"
    elif P:
        p = P["best"]["params"] if "best" in P else P
        src = str(a.params)
    else:
        print(f"fitting PARAMS on train ({len(S['train'])} runs) ...")
        p, _, _ = evaluate.fit_train(mod, S["train"], a.max_nfev, stride=2)
        src = "fitted on train"
    print(f"PARAMS ({src}): {p}")
    if isinstance(theta_mode, dict):
        print(f"THETA used for every tested building (no fitting on them): {theta_mode}")

    res = evaluate.score_heldout(mod, target, p, a.max_nfev, theta_mode=theta_mode)
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
    report = dict(model=str(a.model), split=a.on, theta_mode=a.theta, params=p, params_source=src, complexity=cx,
                  theta=res["scenarios"], overall={m: float(np.mean([r[m] for r in rows])) for m in keys},
                  by_station=agg("station"), by_policy=agg("policy"), by_building=agg("building"), runs=rows)
    out = WC / "outputs" / "test" / f"{model_tag(a.model)}_{a.on}_theta-{a.theta}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    L = [f"# Test of `{a.model}` on `{a.on}`\n", f"PARAMS ({src}): `{json.dumps(p)}`  \ncomplexity {cx} AST nodes\n",
         f"THETA mode `{a.theta}` ({'fitted on day 1 of each building: online identification' if a.theta == 'day1' else 'no fitting on the tested buildings: offline deployment'}). "
         "Open loop from the reset state; forecasts from a Kalman-filtered state with the true future inputs "
         "(a model test, not a closed-loop controller result). RMSE in K.\n"]
    for title, tab in (("Overall", {"all": report["overall"] | {"runs": len(rows)}}), ("By station", report["by_station"]),
                       ("By policy", report["by_policy"]), ("By building", report["by_building"])):
        L.append(f"\n## {title}\n\n| | runs | " + " | ".join(keys) + " |\n|---|---|" + "---|" * len(keys))
        for k, v in tab.items():
            L.append(f"| {k} | {v['runs']} | " + " | ".join(f"{v[m]:.3f}" for m in keys) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out / 'report.md'}")


def compare_all(a):
    """Run every THETA mode in turn (offline: shared, trainmean, prior; online: day1) and tabulate them."""
    import subprocess
    modes = ["shared", "trainmean", "prior", "day1"]
    base = [sys.executable, str(Path(__file__).resolve()), str(a.model), "--dataset", str(a.dataset), "--on", a.on,
            "--max-nfev", str(a.max_nfev)] + (["--split", str(a.split)] if a.split else []) \
        + (["--params", str(a.params)] if a.params else []) + (["--no-forecast"] if a.no_forecast else [])
    rows = {}
    for m in modes:
        print(f"\n=== --theta {m}")
        subprocess.run(base + ["--theta", m], check=True)
        rows[m] = json.loads((WC / "outputs" / "test" / f"{model_tag(a.model)}_{a.on}_theta-{m}" / "report.json").read_text())["overall"]
    keys = [k for k in rows["day1"] if not k.startswith("persistence")] + ["persistence_3h"]
    label = {"shared": "shared θ (offline)", "trainmean": "train-mean θ (offline)", "prior": "prior θ (offline)",
             "day1": "θ fitted on day 1 (online)"}
    L = [f"# Deployment options for `{a.model}` on `{a.on}`\n",
         "Offline modes fit nothing on the tested buildings; online fits θ on each building's day 1. "
         "Forecasts are from a Kalman-filtered state with the true future inputs. RMSE in K.\n",
         "| θ | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for m in modes:
        L.append(f"| {label[m]} | " + " | ".join(f"{rows[m].get(k, float('nan')):.3f}" for k in keys) + " |")
    out = WC / "outputs" / "test" / f"{model_tag(a.model)}_{a.on}_compare.md"
    out.write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L) + f"\n\nwrote {out}")


if __name__ == "__main__":
    main()
