"""Turn the starter kit's finished runs into episodes.  Nothing is generated or simulated here.

The data is made beforehand with the starter kit's own tools (run_floor_generator, run_scenario_generator,
evaluate.py).  For every run in the given results folders this reads the logged CSVs, re-runs only the shipped
controller's decision function on the logged observations to recover the per-room actions, checks the replay against the
logged per-AHU airflow and run commands, computes each room's physical inputs, and writes
one file per run:

    data/dataset/<building>__<weather>__<policy>.npz      (schema: data/dataset/schema.json, GUIDE.md B.8)
    data/dataset/index.json                                 per run: split, replay check, sizes, file
    data/dataset/split.json                                 copy of the split used

    python code/build_dataset.py --scenarios S --results R --out D            # S, R: folders the kit wrote
    python code/build_dataset.py --scenarios S1 S2 --results R1 R2 --out D    # several pairs, matched in order

Runs whose replay does not reproduce the logged airflow are indexed with status
"replay mismatch" and are not used.  Existing up-to-date files are skipped.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402

SCHEMA = {
    "per step k = 0..n-1 (one 300 s step; inputs of step k take the state from x_k to x_{k+1})": {
        "T_meas (n,V)": "measured room air temperature x_{k+1} [K] (= timeseries row k+1; rooms in VAV order)",
        "y_sensor (n+1,V)": "all sensor rows; row k = x_k (row 0 = reset, 294 K)",
        "in__T_out (n)": "outside temperature used in step k [K]  (disturbance d)",
        "in__occupants (n)": "building occupants during step k  (disturbance d)",
        "in__comfort (n)": "1 in comfort mode at the start of step k  (disturbance d)",
        "in__hour_sin/hour_cos/dow_sin/dow_cos (n)": "clock  (disturbance d)",
        "in__g (n,V)": "supply-air capacity rate into each room [W/K]  (physical input from u)",
        "in__T_supply (n,V)": "supply-air temperature reaching each room [K]",
        "in__Q_reheat (n,V)": "reheat coil heat [W]",
        "in__Q_gain (n,V)": "lighting/equipment gain [W] (schedule)",
        "act__damper, act__reheat (n,V)": "realized damper (>= 0.1) and reheat valve, 0..1  (control u, per room)",
        "act__ahu_on, ahu_pressure [Pa], ahu_sat_setpoint [K], ahu_recirculation, ahu_flow [m3/s] (n,K)": "AHU controls and resulting flow (u)",
        "act__hws_on, hws_water_setpoint [K], hws_dp, T_water [K] (n)": "hot-water loop controls (u) and tank temperature used in step k",
        "u_cont_native (n, 3K+2+2V), u_disc (n, K+1)": "the full action vector the policy sent, native units; names in u_cont_names / u_disc_names",
        "obs_lean (n, V+8)": "lean observation before step k: room temps x_k (V), T_water, T_out, occupants, hour sin/cos, dow sin/cos, comfort; names in obs_lean_names",
    },
    "static": {
        "geo__area (V) [m2], geo__facade_faces (V), geo__wall_cells (V,V), geo__door_cells (V,V), geo__n_doors (V,V), geo__bbox_x/y (V) [m], geo__ahu (V), geo__floor (V), geo__V": "geometry from floor_plan / zone_map / ahu_zone_assignments",
        "room_ids (V), ahu_ids (K)": "VAV order of rooms, AHU order of the K axis",
        "meta": "JSON: name, building, weather, policy, scenario, T0 (294 K)",
        "check": "JSON: replay check (airflow error, run-command agreement)",
    },
}


def split_of(scenario, split):
    def hit(p):
        return fnmatch(scenario, p) if "/" in p else fnmatch(scenario.split("/")[0], p)
    for name, pats in split.items():
        if name.startswith("_"):
            continue
        if any(hit(p[1:]) for p in pats if p.startswith("!")):
            continue
        if any(hit(p) for p in pats if not p.startswith("!")):
            return name
    return None


def _do_scenario(job):
    """Worker: all policies of one scenario (the scenario context is loaded once)."""
    sdir, rdir, b, w, pols, out = job
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from discovery.episodes import _save_cached
    from discovery.starterkit_data import load_starterkit_episode
    res, ctxs = {}, {}
    for pol in pols:
        name = f"{b}__{w}__{pol}"
        f = Path(out) / f"{name}.npz"
        try:
            ep, chk, X = load_starterkit_episode(sdir, rdir, b, w, pol, ctxs, full=True)
            ok = chk["flow_max_rel_err"] <= 0.02 and chk["run_cmd_agreement"] >= 0.995
            chk["status"] = "ok" if ok else "replay mismatch"
            ctx, R, C, Dd, act = X["ctx"], X["R"], X["C"], X["Dd"], X["act"]
            V = ep.V; n = ep.n
            lean = np.column_stack([R["temps"][:n],              # x_k: CSV row k (row 0 = reset)
                                    act["T_water"], ep.inputs["T_out"], ep.inputs["occupants"], ep.inputs["hour_sin"], ep.inputs["hour_cos"],
                                    ep.inputs["dow_sin"], ep.inputs["dow_cos"], ep.inputs["comfort"]])
            tmp = Path(out) / f".{name}.tmp.npz"
            _save_cached(ep, tmp)
            Z = dict(np.load(tmp)); tmp.unlink()
            Z.update({f"act__{k}": v for k, v in act.items()})
            Z.update(y_sensor=R["temps"], u_cont_native=C[:n], u_disc=Dd[:n], u_cont_names=np.array(ctx.cont),
                     u_disc_names=np.array([f"{dv}:{nm}" for dv, nm in ctx.disc]), room_ids=np.array(ctx.vav_ids),
                     ahu_ids=np.array(ctx.ahu_names), obs_lean=lean,
                     obs_lean_names=np.array([f"T_{v}" for v in ctx.vav_ids] + ["T_water", "T_out", "occupants", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "comfort"]),
                     check=np.array(json.dumps(chk)))
            np.savez_compressed(f, **Z)
            res[name] = dict(chk, n=n, V=V, K=len(ctx.ahu_names), file=f.name)
        except Exception as e:  # noqa: BLE001
            res[name] = dict(status="error", reason=f"{type(e).__name__}: {e}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--scenarios", type=Path, nargs="+", help="scenario folders the kit wrote (default: <data>/scenarios)")
    ap.add_argument("--results", type=Path, nargs="+", help="the kit's evaluate.py result folders, same order (default: <data>/results)")
    ap.add_argument("--split", type=Path, help="default: <data>/split.json")
    ap.add_argument("--out", type=Path, help="default: <data>/dataset")
    ap.add_argument("--policies", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from discovery.starterkit_data import discover_runs
    data = a.data.resolve(); out = (a.out or data / "dataset").resolve()
    scen = a.scenarios or [data / "scenarios"]; resu = a.results or [data / "results"]
    if len(scen) != len(resu):
        raise SystemExit("give as many --scenarios folders as --results folders (matched in order)")
    sources = [(s.resolve(), r.resolve()) for s, r in zip(scen, resu)]
    for s, r in sources:
        if not r.is_dir():
            raise SystemExit(f"no results folder {r}: generate the runs with the starter kit's evaluate.py first")
    out.mkdir(parents=True, exist_ok=True)
    split_path = a.split or data / "split.json"
    split = json.loads(split_path.read_text()) if split_path.exists() else {}
    jobs, skipped = [], {}
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for sdir, rdir in sources:
        runs = discover_runs(rdir, a.policies)
        by_scen = {}
        for b, w, pol in runs:
            f = out / f"{b}__{w}__{pol}.npz"
            src = rdir / b / w / pol / "timeseries.csv"
            if f.exists() and f.stat().st_mtime > src.stat().st_mtime and f"{b}__{w}__{pol}" in index:
                skipped[f"{b}__{w}__{pol}"] = index[f"{b}__{w}__{pol}"]; continue
            by_scen.setdefault((b, w), []).append(pol)
        jobs += [(str(sdir), str(rdir), b, w, pols, str(out)) for (b, w), pols in by_scen.items()]
    print(f"{len(jobs)} scenarios to build ({sum(len(j[4]) for j in jobs)} runs), {len(skipped)} runs up to date")
    t0 = time.time()
    res = dict(skipped)
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_do_scenario, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result(); res.update(r)
            bad = [k for k, v in r.items() if v.get("status") != "ok"]
            if bad or i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} scenarios, {time.time() - t0:.0f} s" + (f"; not ok: {bad}" if bad else ""))
    for name, rec in res.items():
        b, w, pol = name.split("__")
        rec.update(building=b, weather=w, policy=pol, scenario=f"{b}/{w}", split=split_of(f"{b}/{w}", split) if split else None)
    index_path.write_text(json.dumps(res, indent=1))
    (out / "schema.json").write_text(json.dumps(SCHEMA, indent=1))
    if split:
        (out / "split.json").write_text(json.dumps(split, indent=1))
    st = {}
    for r in res.values():
        st.setdefault(r.get("status"), 0); st[r.get("status")] += 1
    per_split = {}
    for r in res.values():
        if r.get("status") == "ok":
            per_split[r["split"]] = per_split.get(r["split"], 0) + 1
    print(f"index: {index_path}  status counts {st}  usable runs per split {per_split}")


if __name__ == "__main__":
    main()
