"""Which floor slab each room sits on (connected pieces of the footprint) and how AHUs map to floors."""
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import ndimage
KIT = Path(__import__("os").environ.get("STARTER_KIT", Path(__file__).resolve().parent.parent.parent / "smart-buildings-starter-kit")); OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(KIT)); from smart_buildings import evaluate as ev
res = {}
for b in ["building_train_%d" % i for i in (1, 2, 4, 5, 7, 8)]:
    d = sorted((KIT / "smart_buildings/training_scenarios" / b).iterdir())[0]
    env = ev.load_hybrid_environment(d / "config.gin"); bld = env.building.simulator.building
    lab, n = ndimage.label(~np.asarray(bld._exterior_space).astype(bool))
    rows = list(csv.DictReader(open(OUT / f"{b}_rooms.csv")))
    for r in rows:
        rr, cc = np.array(bld.room_dict[r["room"]]).T; r["floor_slab"] = int(Counter(lab[rr, cc]).most_common(1)[0][0])
    order = {s: i + 1 for i, s in enumerate(sorted({r["floor_slab"] for r in rows}, key=lambda s: np.nonzero(lab == s)[0].mean()))}
    for r in rows: r["floor"] = order[r["floor_slab"]]; del r["floor_slab"]
    with open(OUT / f"{b}_rooms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    res[b] = {"floors": n, "rooms_per_floor": dict(Counter(r["floor"] for r in rows)),
              "ahu_by_floor": {a: dict(Counter(r["floor"] for r in rows if r["ahu"] == a)) for a in sorted({r["ahu"] for r in rows})},
              "type_by_floor": {str(f): dict(Counter(r["type"] for r in rows if r["floor"] == f)) for f in order.values()}}
    print(b, json.dumps(res[b]))
(OUT / "floors_and_ahus.json").write_text(json.dumps(res, indent=1))
