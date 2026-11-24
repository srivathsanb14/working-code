"""Write the discovery split: validation + three held-out test tiers (easy / medium / hard), no training split.

    .venv/bin/python code/make_splits.py                       # from the episodes in data/dataset (what is generated and available)
    .venv/bin/python code/make_splits.py --from-scenarios      # from scenario folders, before anything is simulated
    .venv/bin/python code/make_splits.py --from-scenarios --only-shipped --out outputs/split_shipped.json

Nothing is hardcoded: the hard station is the one whose outside temperatures are farthest (Wasserstein-1) from the
rest, the held-out buildings are the most atypical floor plans, and easy windows are picked so their building and
station stay in validation.  run_discovery.py computes the same split by itself when split.json has no test tiers.
    .venv/bin/python code/make_splits.py --hard-station fairbanks --hard-buildings building_train_5 building_train_2

How each tier is chosen is in code/discovery/splits.py; the reasons (Wasserstein distances, atypicality) are written
into the JSON under "_why".
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, KIT  # noqa: E402
from discovery import splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "dataset" / "split.json")
    ap.add_argument("--from-scenarios", action="store_true", help="use scenario folders instead of the built episodes")
    ap.add_argument("--only-shipped", action="store_true", help="with --from-scenarios: only the kit's shipped scenarios")
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset")
    ap.add_argument("--hard-station", help="override the Wasserstein choice")
    ap.add_argument("--hard-buildings", nargs="*", help="override the atypicality choice")
    ap.add_argument("--hard-frac", type=float, default=0.25, help="fraction of buildings held out entirely (rounded up)")
    ap.add_argument("--easy-frac", type=float, default=0.25, help="fraction of seen-building windows held out as test_easy")
    ap.add_argument("--days", type=int, default=30, help="length of each scenario's weather window")
    a = ap.parse_args()
    if not a.from_scenarios and (a.dataset / "index.json").exists():
        from discovery.episodes import _load_cached
        idx = json.loads((a.dataset / "index.json").read_text())
        eps = [_load_cached(a.dataset / v["file"]) for v in idx.values() if v.get("status") == "ok"]
        sp = splits.split_from_episodes(eps, a.hard_station, a.hard_buildings, a.hard_frac, a.easy_frac)
    else:
        if not a.from_scenarios:
            print(f"no dataset at {a.dataset}: using scenario folders")
        roots = [KIT / "smart_buildings" / "training_scenarios"] + ([] if a.only_shipped else [DATA / "scenarios"])
        scen = splits.scan_scenarios(roots, KIT)
        if not scen:
            raise SystemExit(f"no scenario folders under {roots}")
        sp = splits.make_split(scen, a.hard_station, a.hard_buildings, a.hard_frac, a.easy_frac, DATA / "buildings", a.days)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        a.out.with_name(a.out.stem + "_previous.json").write_text(a.out.read_text())
    a.out.write_text(json.dumps(sp, indent=1))
    w = sp["_why"]
    print(f"hard station: {w['hard_station']}  (Wasserstein-1 to the other stations pooled, K: {w['station_wasserstein_to_rest_K']})")
    print(f"held-out buildings: {w['held_out_buildings']}  (atypicality z: {w['building_atypicality_z']})")
    for k in ("val", "test_easy", "test_medium", "test_hard", "test_station"):
        print(f"  {k:<13} {len(sp[k]):>3}  {', '.join(sp[k][:6])}{' ...' if len(sp[k]) > 6 else ''}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
