"""Dataset summary (backlog C3): what the built dataset contains, by split, station, season, policy and size.

    python code/dataset_report.py                       # data/dataset -> docs/DATASET_REPORT.md
    python code/dataset_report.py --dataset /big/disk/data/dataset --out docs/DATASET_REPORT.md
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, KIT, WC  # noqa: E402

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def window_start(w):
    f = KIT / "smart_buildings" / "weather" / f"{w}.yaml"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        if line.startswith("start_timestamp"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


def season(date):
    if not date:
        return "?"
    m = int(date[5:7]); return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                                 6: "summer", 7: "summer", 8: "summer"}.get(m, "autumn")


def split_of(b, w, split):
    for part, pats in split.items():
        if part.startswith("_"):
            continue
        for p in pats:
            if p.startswith("!"):
                continue
            if fnmatch.fnmatch(f"{b}/{w}" if "/" in p else b, p):
                return part
    return "unassigned"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset")
    ap.add_argument("--out", type=Path, default=WC / "docs" / "DATASET_REPORT.md")
    a = ap.parse_args()
    idx = json.loads((a.dataset / "index.json").read_text())
    split = json.loads((a.dataset / "split.json").read_text()) if (a.dataset / "split.json").exists() else {}
    rows = []
    for name, r in idx.items():
        if "__" in name:
            b, w, pol = name.split("__")
        else:
            b, w, pol = r.get("building", name), r.get("weather", "?"), r.get("policy", "?")
        st = w.rsplit("_", 1)[0]; d = window_start(w)
        rows.append(dict(name=name, b=b, w=w, station=st, policy=pol, start=d, season=season(d), status=r.get("status", "?"),
                         n=r.get("n", 0), V=r.get("V"), K=r.get("K"), flow=r.get("flow_max_rel_err"), split=split_of(b, w, split)))
    ok = [r for r in rows if r["status"] == "ok"]
    L = [f"# Dataset report: `{a.dataset}`", "", f"{len(rows)} runs indexed, **{len(ok)} usable** (replay matched), "
         f"{len(rows) - len(ok)} rejected. Usable steps: **{sum(r['n'] for r in ok):,}**.", ""]

    def table(title, key):
        c = defaultdict(lambda: [0, 0, set()])
        for r in ok:
            c[r[key]][0] += 1; c[r[key]][1] += r["n"]; c[r[key]][2].add(r["b"])
        L.extend([f"## By {title}", "", f"| {title} | runs | steps | buildings |", "|---|---|---|---|"])
        for k in sorted(c, key=str):
            L.append(f"| {k} | {c[k][0]} | {c[k][1]:,} | {len(c[k][2])} |")
        L.append("")
    for t, k in (("split", "split"), ("station", "station"), ("season", "season"), ("policy", "policy")):
        table(t, k)
    by_b = {}
    for r in ok:
        by_b.setdefault(r["b"], r)
    V = np.array([r["V"] for r in by_b.values() if r["V"]]); K = np.array([r["K"] for r in by_b.values() if r["K"]])
    if len(V):
        L += ["## Building sizes", "", f"{len(by_b)} buildings. Rooms: min {V.min()}, median {int(np.median(V))}, max {V.max()}. "
              f"AHUs: min {K.min()}, median {int(np.median(K))}, max {K.max()}.", "",
              "| rooms | buildings |", "|---|---|"]
        for lo in range(0, int(V.max()) + 25, 25):
            k = int(((V >= lo) & (V < lo + 25)).sum())
            if k:
                L.append(f"| {lo}–{lo + 24} | {k} |")
        L.append("")
    bad = Counter(r["status"] for r in rows if r["status"] != "ok")
    if bad:
        L += ["## Rejected runs", "", *[f"- {k}: {v}" for k, v in bad.items()], ""]
    flows = [r["flow"] for r in ok if r["flow"] is not None]
    if flows:
        L += [f"Replay check: largest AHU flow error among usable runs {max(flows):.4f} (relative).", ""]
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
