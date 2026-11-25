"""Step 2 (and again after step 3): audit the starter-kit data before trusting it.

    python code/audit_data.py                 # after the kit's runs exist: layout, coverage, leakage
    python code/audit_data.py                 # after build_dataset.py: adds replay and dataset checks

Writes data/audit/audit.md (readable) and data/audit/audit.json, and exits with code 1 if
any HARD check fails.  Checks:

  kit        the starter kit is unmodified (git status)
  layout     every planned scenario has config.gin + floor_plan.npy + zone_map.npy; every run has
             timeseries.csv, ahu_log.csv, reward_components.csv, summary.json; row counts reach the
             requested steps; no NaN; CSV row 0 is the reset state (all rooms 294 K) and the episode
             starts on a Sunday at 06:00 (the evaluation's convention)
  coverage   per split: buildings, scenarios, runs; building parameters against the generator's own
             ranges (run_floor_generator.DEFAULT_RANGES) and materials against materials.PROPERTIES;
             realized rooms V and AHUs K; station x strip matrix; policies
  leakage    no building in two splits; the test station never in train/val; the val station never in train
  replay     (needs build_dataset.py) every run's replay reproduces the logged per-AHU airflow and run commands
  dataset    (needs build_dataset.py) temperature and input ranges, NaN, and how much each policy
             excites the building (AHU-on fraction, damper and reheat activity, temperature spread)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, KIT, kit_git_status  # noqa: E402


class Report:
    def __init__(self):
        self.lines, self.hard, self.warn, self.data = [], [], [], {}

    def h(self, t):
        self.lines.append(f"\n## {t}\n")

    def ok(self, t):
        self.lines.append(f"- [ok] {t}")

    def fail(self, t):
        self.lines.append(f"- **[FAIL]** {t}"); self.hard.append(t)

    def warning(self, t):
        self.lines.append(f"- [warn] {t}"); self.warn.append(t)

    def text(self, t):
        self.lines.append(t)


def split_of(scen, split):
    def hit(p):
        return fnmatch(scen, p) if "/" in p else fnmatch(scen.split("/")[0], p)
    for name, pats in split.items():
        if name.startswith("_"):
            continue
        if any(hit(p[1:]) for p in pats if p.startswith("!")):
            continue
        if any(hit(p) for p in pats if not p.startswith("!")):
            return name
    return None


def kit_ranges():
    """The generator's own sampling ranges, read from the kit (import only)."""
    sys.path.insert(0, str(KIT))
    from smart_buildings import run_floor_generator as rfg
    from smart_buildings.smart_control.utils import materials as mat
    return dict(rfg.DEFAULT_RANGES), {k: (lo, hi) for k, (lo, hi, _) in mat.PROPERTIES.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--max-csv-checks", type=int, default=100000, help="cap on runs whose CSVs are opened")
    a = ap.parse_args()
    data = a.data.resolve()
    R = Report()
    R.text(f"# Data audit\n\ndata root: `{data}`  \nstarter kit: `{KIT}`")
    from discovery.starterkit_data import resolve

    # ------------------------------------------------------------------ kit
    R.h("Starter kit untouched")
    st = kit_git_status()
    if st is None:
        R.warning("kit is not a git checkout; cannot verify it is unmodified")
    elif st:
        R.fail(f"kit has changes:\n```\n{st[:1500]}\n```")
    else:
        R.ok("git status clean")

    plan = json.loads((data / "plan.json").read_text()) if (data / "plan.json").exists() else None   # optional manifest
    split = json.loads((data / "split.json").read_text()) if (data / "split.json").exists() else {}
    if plan is None:
        R.ok("no plan.json manifest; auditing every results folder under data/results")
        sources = [(data / "scenarios", data / "results")]
        steps_req = None
    else:
        sources = [(resolve(s["scenarios"], data), resolve(s["results"], data)) for s in plan["sources"]]
        steps_req = plan["steps"]

    # ------------------------------------------------------------------ layout
    R.h("Layout and episode conventions")
    runs = []                                # (scenarios_dir, results_dir, b, w, pol)
    missing_cfg, missing_csv, short, nan_runs, bad_reset, bad_start = [], [], [], [], [], []
    for sdir, rdir in sources:
        if not rdir.exists():
            R.fail(f"results folder missing: {rdir}"); continue
        for bdir in sorted(p for p in rdir.iterdir() if p.is_dir()):
            for wdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
                b, w = bdir.name, wdir.name
                cfg = sdir / b / w
                if not all((cfg / f).exists() for f in ("config.gin", "floor_plan.npy", "zone_map.npy")):
                    missing_cfg.append(f"{b}/{w}")
                for pdir in sorted(p for p in wdir.iterdir() if p.is_dir()):
                    need = ("timeseries.csv", "ahu_log.csv", "reward_components.csv", "summary.json")
                    if not all((pdir / f).exists() for f in need):
                        missing_csv.append(f"{b}/{w}/{pdir.name}"); continue
                    runs.append((sdir, rdir, b, w, pdir.name))
    if plan:
        expected = {(b, w, pol) for b, ws in plan.get("generated_scenarios", {}).items() for w in ws for pol in plan["policies"]}
        have = {(b, w, pol) for _, _, b, w, pol in runs}
        miss = sorted(expected - have)
        (R.fail if miss else R.ok)(f"planned runs present: {len(expected & have)}/{len(expected)}" + (f"; missing e.g. {miss[:5]}" if miss else ""))
        if plan.get("failures"):
            R.warning(f"{len(plan['failures'])} generation failures recorded in the manifest (e.g. rooms under 20 m2): {list(plan['failures'].items())[:5]}")
    rejected = sorted(p.parent.name for sdir, _ in sources for p in Path(sdir).glob("*/.failed"))
    if rejected:
        used = sorted({b for _, _, b, _, _ in runs if b in rejected})
        R.warning(f"{len(rejected)} buildings rejected by the scenario generator (marker .failed): {rejected[:8]}"
                  + (f"; results exist for {used[:5]} but they are in no split and are never used" if used else ""))
        runs = [r for r in runs if r[2] not in rejected]
        missing_cfg = [m for m in missing_cfg if m.split("/")[0] not in rejected]
    (R.fail if missing_cfg else R.ok)("scenario folders complete" + (f": missing for {missing_cfg[:5]}" if missing_cfg else f" ({len({(r[2], r[3]) for r in runs})} scenarios)"))
    (R.fail if missing_csv else R.ok)("run folders complete" + (f": incomplete {missing_csv[:5]}" if missing_csv else f" ({len(runs)} runs)"))
    lengths = []
    for sdir, rdir, b, w, pol in runs[: a.max_csv_checks]:
        ts = pd.read_csv(rdir / b / w / pol / "timeseries.csv")
        zc = [c for c in ts.columns if c.startswith("zone_")]
        lengths.append(len(ts))
        if steps_req and len(ts) < steps_req:
            short.append(f"{b}/{w}/{pol} ({len(ts)} rows)")
        if ts[zc].isna().any().any():
            nan_runs.append(f"{b}/{w}/{pol}")
        if np.abs(ts[zc].iloc[0].to_numpy(float) - 294.0).max() > 1e-6:
            bad_reset.append(f"{b}/{w}/{pol}")
        t0 = pd.Timestamp(ts["timestamp"].iloc[0]) - pd.Timedelta(minutes=5)
        if not (t0.dayofweek == 6 and t0.hour == 6 and t0.minute == 0):
            bad_start.append(f"{b}/{w}/{pol} starts {t0}")
    if lengths:
        R.ok(f"rows per run: min {min(lengths)}, median {int(np.median(lengths))}, max {max(lengths)}" + (f" (requested {steps_req})" if steps_req else ""))
    (R.fail if short else R.ok)("all runs reach the requested steps" + (f": short {short[:5]}" if short else ""))
    (R.fail if nan_runs else R.ok)("no NaN temperatures" + (f": {nan_runs[:5]}" if nan_runs else ""))
    (R.fail if bad_reset else R.ok)("CSV row 0 is the reset state (all rooms 294 K)" + (f": not for {bad_reset[:5]}" if bad_reset else ""))
    (R.warning if bad_start else R.ok)("every episode starts on a Sunday at 06:00, as in the evaluation" + (f": {bad_start[:5]}" if bad_start else ""))
    if min(lengths or [10 ** 9]) < 864:
        R.warning("some runs are shorter than 864 steps: day-1 identification plus two scoring days needs 3 days")

    # ------------------------------------------------------------------ coverage
    R.h("Coverage against the evaluation's distribution")
    rng_b, rng_m = kit_ranges()
    scen_split = {}
    for _, _, b, w, _ in runs:
        scen_split[f"{b}/{w}"] = split_of(f"{b}/{w}", split) if split else "unsplit"
    by_split = defaultdict(lambda: dict(buildings=set(), scenarios=set(), runs=0, policies=Counter(), stations=Counter()))
    for _, _, b, w, pol in runs:
        s = scen_split[f"{b}/{w}"]
        d = by_split[s]; d["buildings"].add(b); d["scenarios"].add(f"{b}/{w}"); d["runs"] += 1; d["policies"][pol] += 1
        d["stations"][w.rsplit("_", 1)[0]] += 1
    R.text("\n| split | buildings | scenarios | runs | stations (runs) | policies (runs) |\n|---|---|---|---|---|---|")
    for s, d in sorted(by_split.items(), key=lambda kv: str(kv[0])):
        R.text(f"| {s} | {len(d['buildings'])} | {len(d['scenarios'])} | {d['runs']} | {dict(d['stations'])} | {dict(d['policies'])} |")
    if None in by_split:
        R.ok(f"{len(by_split[None]['scenarios'])} scenarios are in no split under the current val station (train buildings on the "
             "val station, val buildings on other stations): kept on purpose so the val station can be rotated; not used now")
    for need in ("train", "val", "test"):
        if split and need not in by_split:
            R.fail(f"split '{need}' has no runs")
    # building parameters from the YAMLs
    ydirs = list((data / "buildings").glob("*/*.yaml")) + list((data / "buildings").glob("*.yaml"))
    rows = []
    for y in ydirs:
        c = yaml.safe_load(open(y))
        fp, m = c.get("floor_plan", {}), c.get("materials", {})
        rows.append(dict(building=c.get("name", y.stem), split=y.parent.name, width=fp.get("width"), height=fp.get("height"),
                         num_floors=fp.get("num_floors"), min_room_size=fp.get("min_room_size"),
                         room_size_span=(fp.get("max_room_size", 0) - fp.get("min_room_size", 0)), shape_ratio=fp.get("shape_ratio"),
                         composite_coverage=fp.get("composite_coverage"), num_ahus=c.get("ahu", {}).get("num_ahus"),
                         occupants_per_zone=c.get("simulation", {}).get("occupants_per_zone"), shape=fp.get("building_shape"),
                         has_materials=bool(m), **{k: m.get(k) for k in rng_m}))
    if rows:
        B = pd.DataFrame(rows)
        R.text("\nBuilding parameters (generated YAMLs) against the generator's ranges:\n\n| parameter | range in kit | train | val | test |\n|---|---|---|---|---|")
        out_of_range = []
        for k, (lo, hi) in list({kk: tuple(v) for kk, v in rng_b.items() if isinstance(v, tuple)}.items()) + list(rng_m.items()):
            if k not in B:
                continue
            cells = []
            for s in ("train", "val", "test"):
                v = pd.to_numeric(B.loc[B.split == s, k], errors="coerce").dropna()
                cells.append(f"{v.min():.3g}-{v.max():.3g}" if len(v) else "-")
                if len(v) and ((v < lo - 1e-9).any() or (v > hi + 1e-9).any()):
                    out_of_range.append(f"{k} in {s}")
            R.text(f"| {k} | {lo}-{hi} | " + " | ".join(cells) + " |")
        (R.warning if out_of_range else R.ok)("all generated building parameters inside the generator's ranges" + (f": outside for {out_of_range}" if out_of_range else ""))
        (R.fail if (~B.has_materials).any() else R.ok)("every generated building has a materials block" + ("" if B.has_materials.all() else f": missing for {B.loc[~B.has_materials, 'building'].tolist()[:5]}"))
        (R.warning if (B["shape"] != "composite").any() else R.ok)("every generated building has the composite shape")
    else:
        R.warning("no generated building YAMLs under data/buildings (only shipped scenarios?)")
    # station x strip matrix
    R.text("\nStation x strip (scenarios) per split:\n")
    for s, d in sorted(by_split.items(), key=lambda kv: str(kv[0])):
        mat = Counter((sc.split("/")[1].rsplit("_", 1)[0], sc.split("/")[1].rsplit("_", 1)[1]) for sc in d["scenarios"])
        sts = sorted({k[0] for k in mat}); strips = sorted({k[1] for k in mat}, key=int)
        R.text(f"\n{s}:\n\n| station | " + " | ".join(strips) + " |\n|---|" + "---|" * len(strips))
        for stn in sts:
            R.text(f"| {stn} | " + " | ".join(str(mat.get((stn, x), "")) for x in strips) + " |")

    # ------------------------------------------------------------------ leakage
    R.h("Leakage between splits")
    if split:
        bsplits = defaultdict(set)
        for sc, s in scen_split.items():
            if s:
                bsplits[sc.split("/")[0]].add(s)
        multi = {b: sorted(s) for b, s in bsplits.items() if len(s) > 1}
        (R.fail if multi else R.ok)("no building in two splits" + (f": {list(multi.items())[:5]}" if multi else ""))
        if plan:
            ts_, vs_ = plan["test_station"], plan["val_station"]
            bad = [sc for sc, s in scen_split.items() if s in ("train", "val") and sc.split("/")[1].startswith(ts_)]
            (R.fail if bad else R.ok)(f"test station '{ts_}' absent from train and val" + (f": {bad[:5]}" if bad else ""))
            bad = [sc for sc, s in scen_split.items() if s == "train" and sc.split("/")[1].startswith(vs_)]
            (R.fail if bad else R.ok)(f"val station '{vs_}' absent from train" + (f": {bad[:5]}" if bad else ""))
    else:
        R.warning("no split.json: train/val/test separation not checked")

    # ------------------------------------------------------------------ replay + dataset
    R.h("Replay and dataset (after build_dataset.py)")
    idx_path = data / "dataset" / "index.json"
    if not idx_path.exists():
        R.warning("dataset not built yet: run code/build_dataset.py, then audit again")
    else:
        idx = json.loads(idx_path.read_text())
        stc = Counter(v.get("status") for v in idx.values())
        bad = [k for k, v in idx.items() if v.get("status") != "ok"]
        (R.warning if bad else R.ok)(f"replay status: {dict(stc)}" + (f"; not ok (excluded from fitting): {bad[:8]}" if bad else ""))
        notin = sorted({f'{r[2]}__{r[3]}__{r[4]}' for r in runs} - set(idx))
        (R.warning if notin else R.ok)("every run is in the dataset" + (f": missing {notin[:5]}" if notin else ""))
        errs = [v["flow_max_rel_err"] for v in idx.values() if "flow_max_rel_err" in v]
        if errs:
            R.ok(f"replay airflow error: median {np.median(errs):.1e}, max {max(errs):.1e} (CSV rounding is ~1e-4)")
        Vs = [v["V"] for v in idx.values() if "V" in v]; Ks = [v["K"] for v in idx.values() if "K" in v]
        if Vs:
            R.ok(f"rooms per building V: {min(Vs)}-{max(Vs)}; AHUs K: {min(Ks)}-{max(Ks)} (shipped: V 30-85, K 3-9)")
        # dataset content: ranges and excitation per policy
        exc = defaultdict(list); nan_files, odd = [], []
        for name, v in idx.items():
            if v.get("status") != "ok":
                continue
            Z = np.load(data / "dataset" / v["file"])
            T = Z["T_meas"]
            if not np.isfinite(T).all() or not all(np.isfinite(Z[k]).all() for k in Z.files if k.startswith("in__")):
                nan_files.append(name)
            if T.min() < 250 or T.max() > 330:
                odd.append(f"{name} ({T.min():.0f}-{T.max():.0f} K)")
            exc[v["policy"]].append(dict(ahu_on=float(Z["act__ahu_on"].mean()), damper=float(Z["act__damper"].mean()),
                                         reheat_active=float((Z["in__Q_reheat"] > 1).mean()), T_std=float(T.std()),
                                         g_mean=float(Z["in__g"].mean())))
        (R.fail if nan_files else R.ok)("dataset arrays finite" + (f": not for {nan_files[:5]}" if nan_files else ""))
        (R.warning if odd else R.ok)("room temperatures within 250-330 K" + (f": {odd[:5]}" if odd else ""))
        R.text("\nExcitation by policy (means over runs) - identification needs movement:\n\n| policy | runs | AHU on | mean damper | reheat active | mean g (W/K) | temperature spread (K) |\n|---|---|---|---|---|---|---|")
        for pol, L in sorted(exc.items()):
            m = {k: np.mean([x[k] for x in L]) for k in L[0]}
            R.text(f"| {pol} | {len(L)} | {m['ahu_on']:.2f} | {m['damper']:.2f} | {m['reheat_active']:.2f} | {m['g_mean']:.0f} | {m['T_std']:.2f} |")

    # ------------------------------------------------------------------ verdict
    R.h("Verdict")
    R.text(f"{len(R.hard)} hard failures, {len(R.warn)} warnings.")
    out = data / "audit"; out.mkdir(parents=True, exist_ok=True)
    (out / "audit.md").write_text("\n".join(R.lines) + "\n")
    (out / "audit.json").write_text(json.dumps(dict(hard=R.hard, warnings=R.warn), indent=1))
    print("\n".join(R.lines))
    print(f"\nwrote {out / 'audit.md'}")
    sys.exit(1 if R.hard else 0)


if __name__ == "__main__":
    main()
