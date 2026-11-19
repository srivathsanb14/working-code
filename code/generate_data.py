"""Step 1: generate training data with the starter kit's own command-line tools.

Every file is written by an unmodified starter-kit CLI, run from the kit root, with every
output pointed into the data root (working-code/data, or $SB_DATA).  Nothing is written
into the kit.  The plan follows GUIDE.md Part A:

  buildings   run_floor_generator, one batch per split (disjoint seed ranges, so each
              split covers the materials' Latin-hypercube ranges on its own)
  scenarios   run_scenario_generator, one call per building:
                train / val buildings: one strip from each train/val station, strip index
                rotated by building seed so seasons spread across the set
                test buildings: only the test station (default Minneapolis, the one shipped
                station that is in no shipped scenario: the stand-in for an unseen station)
  runs        evaluate.py with the shipped policies (replayable: see starterkit_data.py)
  shipped     optionally evaluate.py on the 30 shipped scenarios too

It also writes  plan.json  (everything above, plus failures) and  split.json  (exact
train/val/test scenario lists: train excludes the val and test stations, val = val
buildings on the val station, test = test buildings on the test station).  Re-running
resumes: finished pieces are skipped.

    python code/generate_data.py --train-seeds 500:600 --val-seeds 700:720 --test-seeds 800:820 \
        --steps 2016 --workers 8
    python code/generate_data.py ... --dry-run          # print the plan and commands only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, KIT, kit_git_status, run_kit  # noqa: E402

ALL_STATIONS = ["fairbanks", "denver", "atlanta", "phoenix", "singapore", "minneapolis"]
SHIPPED_POLICIES = ["all_off", "bang_bang", "fixed_setpoint", "pid", "pid_water"]


def seeds(spec):
    a, b = spec.split(":")
    return list(range(int(a), int(b)))


def strip_for(seed, j, k=0):
    """Strip index 1..8 for station number j of a building (rotated by seed; distinct per k)."""
    return 1 + (seed + 3 * j + 3 * k) % 8


def make_plan(a):
    stations = [s for s in a.stations if s != a.test_station]
    plan = dict(created=time.strftime("%Y-%m-%d %H:%M"), data_root=str(a.out), steps=a.steps, policies=a.policies,
                train_seeds=a.train_seeds, val_seeds=a.val_seeds, test_seeds=a.test_seeds, stations=stations,
                val_station=a.val_station, test_station=a.test_station, test_strips=a.test_strips,
                include_shipped=a.include_shipped, splits={}, scenarios={}, failures={})
    for split, spec in (("train", a.train_seeds), ("val", a.val_seeds), ("test", a.test_seeds)):
        plan["splits"][split] = [f"building_{s:04d}" for s in seeds(spec)] if spec else []
        for s in (seeds(spec) if spec else []):
            b = f"building_{s:04d}"
            if split == "test":
                plan["scenarios"][b] = [f"{a.test_station}_{strip_for(s, 0, k)}" for k in range(a.test_strips)]
            else:
                plan["scenarios"][b] = [f"{st}_{strip_for(s, j)}" for j, st in enumerate(stations)]
    return plan


def split_json(plan, done_scen, shipped=()):
    val_st, test_st = plan["val_station"], plan["test_station"]
    out = {"_comment": "written by generate_data.py; exact building/weather names (glob patterns are allowed too)",
           "train": [], "val": [], "test": []}
    for split, blds in plan["splits"].items():
        for b in blds:
            for w in done_scen.get(b, []):
                st = w.rsplit("_", 1)[0]
                if split == "train" and st not in (val_st, test_st):
                    out["train"].append(f"{b}/{w}")
                elif split == "val" and st == val_st:
                    out["val"].append(f"{b}/{w}")
                elif split == "test" and st == test_st:
                    out["test"].append(f"{b}/{w}")
    for sc in shipped:                               # shipped scenarios join train unless on a held-out station
        if sc.split("/")[1].rsplit("_", 1)[0] not in (val_st, test_st):
            out["train"].append(sc)
    return out


def run_done(results_dir, b, w, policies, steps):
    for pol in policies:
        f = results_dir / b / w / pol / "timeseries.csv"
        if not f.exists():
            return False
        try:
            if sum(1 for _ in open(f)) - 1 < steps:
                return False
        except OSError:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--train-seeds", default="500:600", help="start:stop (stop excluded)")
    ap.add_argument("--val-seeds", default="700:720")
    ap.add_argument("--test-seeds", default="800:820")
    ap.add_argument("--stations", nargs="+", default=ALL_STATIONS[:5], help="train/val stations")
    ap.add_argument("--val-station", default="phoenix")
    ap.add_argument("--test-station", default="minneapolis")
    ap.add_argument("--test-strips", type=int, default=3)
    ap.add_argument("--policies", nargs="+", default=SHIPPED_POLICIES)
    ap.add_argument("--steps", type=int, default=2016, help=">= 864 (3 days); 2016 = 7 days; 8640 = full episode")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--include-shipped", action="store_true", help="also run the 30 shipped scenarios")
    ap.add_argument("--stage", choices=["all", "buildings", "scenarios", "runs"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.steps < 864:
        print("warning: --steps < 864 leaves less than day 1 + 2 scoring days")
    out = a.out.resolve(); a.out = out
    logs = out / "logs"; bdir = out / "buildings"; sdir = out / "scenarios"; rdir = out / "results"
    plan = make_plan(a)
    if a.dry_run:
        print(json.dumps({k: v for k, v in plan.items() if k != "scenarios"}, indent=1))
        n_sc = sum(len(v) for v in plan["scenarios"].values())
        print(f"{len(plan['scenarios'])} buildings, {n_sc} scenarios, {n_sc * len(a.policies)} runs of {a.steps} steps "
              f"= {n_sc * len(a.policies) * a.steps / 1e6:.2f} M simulator steps")
        return
    for d in (logs, bdir, sdir, rdir):
        d.mkdir(parents=True, exist_ok=True)
    before = kit_git_status()

    # ---------------------------------------------------------------- buildings
    if a.stage in ("all", "buildings"):
        for split, spec in (("train", a.train_seeds), ("val", a.val_seeds), ("test", a.test_seeds)):
            if not spec:
                continue
            s = seeds(spec)
            if all((bdir / split / f"building_{x:04d}.yaml").exists() for x in s):
                print(f"buildings/{split}: already there ({len(s)})"); continue
            print(f"buildings/{split}: generating seeds {s[0]}..{s[-1]}")
            code, tail = run_kit(["smart_buildings.run_floor_generator", "--start-seed", str(s[0]), "--count", str(len(s)),
                                  "--output-dir", str(bdir / split), "--no-pdf"], logs / f"buildings_{split}.log")
            if code:
                raise SystemExit(f"run_floor_generator failed (see {logs / f'buildings_{split}.log'}):\n{tail}")

    # ---------------------------------------------------------------- scenarios
    done_scen = {}
    if a.stage in ("all", "scenarios", "runs"):
        jobs = []
        for split, blds in plan["splits"].items():
            for b in blds:
                ws = plan["scenarios"][b]
                if (sdir / b / ".failed").exists():          # rejected earlier (e.g. a room under 20 m2)
                    plan["failures"][b] = (sdir / b / ".failed").read_text().strip(); continue
                if all((sdir / b / w / "config.gin").exists() for w in ws):
                    done_scen[b] = ws; continue
                if a.stage == "runs":
                    continue
                jobs.append((split, b, ws))

        def make(job):
            split, b, ws = job
            y = bdir / split / f"{b}.yaml"
            if not y.exists():
                return b, [], "building YAML missing"
            code, tail = run_kit(["smart_buildings.run_scenario_generator", "--buildings", str(y), "--weathers", *ws,
                                  "--output-dir", str(sdir)], logs / f"scenarios_{b}.log")
            if code != 0:
                # the generator rejects some seeds (a room under 20 m2) and may leave a partial folder behind:
                # mark the whole building failed so nothing of it is evaluated or used
                lines = tail.strip().splitlines()
                msg = next((l for l in reversed(lines) if "Error" in l), lines[-1] if lines else "failed")[:300]
                (sdir / b).mkdir(parents=True, exist_ok=True)
                (sdir / b / ".failed").write_text(msg + "\n")
                return b, [], msg
            got = [w for w in ws if (sdir / b / w / "config.gin").exists()]
            return b, got, None
        if jobs:
            print(f"scenarios: generating for {len(jobs)} buildings")
        with ThreadPoolExecutor(a.workers) as ex:
            for f in as_completed([ex.submit(make, j) for j in jobs]):
                b, got, err = f.result()
                if got:
                    done_scen[b] = got
                if err:
                    plan["failures"][b] = err
                    print(f"  {b}: FAILED ({err})")

    # ---------------------------------------------------------------- runs
    shipped = []
    if a.stage in ("all", "runs"):
        todo = [(b, w) for b, ws in done_scen.items() for w in ws if not run_done(rdir, b, w, a.policies, a.steps)]
        print(f"runs: {len(todo)} scenarios to evaluate x {len(a.policies)} policies ({a.steps} steps)")
        t0 = time.time()

        def ev(job):
            b, w = job
            code, tail = run_kit(["smart_buildings.evaluate", "--scenarios-dir", str(sdir), "--buildings", b, "--weathers", w,
                                  "--policies", *a.policies, "--steps", str(a.steps), "--results-dir", str(rdir)],
                                 logs / f"runs_{b}_{w}.log")
            return b, w, code
        n = 0
        with ThreadPoolExecutor(a.workers) as ex:
            for f in as_completed([ex.submit(ev, j) for j in todo]):
                b, w, code = f.result(); n += 1
                if code:
                    plan["failures"][f"{b}/{w}"] = f"evaluate exited {code} (see logs/runs_{b}_{w}.log)"
                if n % 10 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} done, {time.time() - t0:.0f} s")
        if a.include_shipped:
            sh_dir = out / "results_shipped"
            ship = sorted((KIT / "smart_buildings" / "training_scenarios").glob("*/*/config.gin"))
            todo = [(p.parent.parent.name, p.parent.name) for p in ship if not run_done(sh_dir, p.parent.parent.name, p.parent.name, a.policies, a.steps)]
            print(f"shipped: {len(todo)} scenarios to evaluate")

            def ev2(job):
                b, w = job
                return b, w, run_kit(["smart_buildings.evaluate", "--buildings", b, "--weathers", w, "--policies", *a.policies,
                                      "--steps", str(a.steps), "--results-dir", str(sh_dir)], logs / f"shipped_{b}_{w}.log")[0]
            with ThreadPoolExecutor(a.workers) as ex:
                for f in as_completed([ex.submit(ev2, j) for j in todo]):
                    b, w, code = f.result()
                    if code:
                        plan["failures"][f"shipped:{b}/{w}"] = f"evaluate exited {code}"
            shipped = [f"{p.parent.parent.name}/{p.parent.name}" for p in ship]

    plan["generated_scenarios"] = done_scen
    plan["sources"] = [{"scenarios": "data:scenarios", "results": "data:results"}]
    if a.include_shipped:
        plan["sources"].append({"scenarios": "kit:smart_buildings/training_scenarios", "results": "data:results_shipped"})
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    sj = split_json(plan, done_scen, shipped)
    (out / "split.json").write_text(json.dumps(sj, indent=1))
    print(f"wrote {out / 'plan.json'} and {out / 'split.json'} "
          f"(train {len(sj['train'])}, val {len(sj['val'])}, test {len(sj['test'])} scenarios; {len(plan['failures'])} failures)")
    after = kit_git_status()
    if after not in ("", None) and after != before:
        print(f"WARNING: the starter kit changed during generation:\n{after}")


if __name__ == "__main__":
    main()
