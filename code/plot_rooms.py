"""Colour each room by corner/edge/interior and label it, one PNG per building (into ../outputs)."""
import csv, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
KIT = Path("/Users/srivathsan/Desktop/smart-buildings-starter-kit"); OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(KIT)); from smart_buildings import evaluate as ev
col = {"corner": "#d95f02", "edge": "#7570b3", "interior": "#1b9e77"}
for b in sys.argv[1:] or ["building_train_%d" % i for i in (1, 2, 4, 5, 7, 8)]:
    d = sorted((KIT / "smart_buildings/training_scenarios" / b).iterdir())[0]
    env = ev.load_hybrid_environment(d / "config.gin"); bld = env.building.simulator.building
    img = np.ones(bld.floor_plan.shape + (3,)); img[np.asarray(bld._exterior_space).astype(bool)] = (0.92, 0.92, 0.92)
    img[np.asarray(bld._exterior_walls).astype(bool)] = (0, 0, 0); img[np.asarray(bld._interior_walls).astype(bool)] = (0.35, 0.35, 0.35)
    rows = list(csv.DictReader(open(OUT / f"{b}_rooms.csv")))
    for r in rows:
        rr, cc = np.array(bld.room_dict[r["room"]]).T; img[rr, cc] = matplotlib.colors.to_rgb(col[r["type"]])
    fig, ax = plt.subplots(figsize=(8, 8 * img.shape[0] / img.shape[1])); ax.imshow(img)
    for r in rows: ax.text(float(r["cx"]), float(r["cy"]), r["room"].split("_")[1], ha="center", va="center", fontsize=7, color="white")
    ax.set_title(f"{b} ({d.name}): orange=corner purple=edge green=interior"); ax.axis("off")
    fig.savefig(OUT / f"{b}_rooms.png", dpi=110, bbox_inches="tight"); plt.close(fig)
