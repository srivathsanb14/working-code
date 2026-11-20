"""Per-room and per-pair geometry from floor_plan.npy and zone_map.npy only (backlog D1/D2).

    python code/geometry_cli.py <scenario_dir>                       # a folder holding floor_plan.npy, zone_map.npy (and config.gin)
    python code/geometry_cli.py --floor-plan fp.npy --zone-map zm.npy --out geo/
    python code/geometry_cli.py --all-shipped                        # the six shipped buildings into outputs/geometry/

Writes <out>/rooms.csv (one row per room: area, facade faces in total and by side of the drawing, corner flag,
bounding box, floor, AHU, neighbours, doors, boundary faces, aspect) and <out>/pairs.csv (shared wall cells, door
cells, doors for every pair of rooms that touch).  The numbers are exactly what discovery/episodes.geometry gives a
candidate, so this is also how a policy sees a new building.  AHU assignment is read from config.gin when present
(ahu_N_rooms), otherwise every room is put on AHU 0.  Units: m2, 0.2 m faces/cells, m.
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import room_model as rm  # noqa: E402
from common import KIT, WC  # noqa: E402
from discovery.episodes import geometry  # noqa: E402

PER_ROOM = ["area", "facade_faces", "facade_top", "facade_bottom", "facade_left", "facade_right", "facade_sides",
            "boundary_faces", "bbox_x", "bbox_y", "aspect", "floor", "ahu", "n_neighbours"]


def ahu_rooms_from_gin(path):
    if not Path(path).exists():
        return None
    out = {}
    for m in re.finditer(r"^(ahu_\d+)_rooms\s*=\s*(\[.*?\])", Path(path).read_text(), re.M):
        out[m.group(1)] = ast.literal_eval(m.group(2))
    return out or None


def build(fp, zm, ahu=None):
    V = int(rm.grid_geometry(fp, zm)["nroom"])
    rooms = [f"room_{i + 1}" for i in range(V)]
    return rooms, geometry(fp, zm, rooms, ahu or {"ahu_0": rooms})


def write(rooms, g, out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rooms.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["room", *PER_ROOM, "corner", "doors_total", "wall_cells_total"])
        for i, r in enumerate(rooms):
            w.writerow([r, *[round(float(g[k][i]), 4) for k in PER_ROOM], int(g["facade_sides"][i] >= 2),
                        int(g["n_doors"][i].sum()), int(g["wall_cells"][i].sum())])
    with (out / "pairs.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["room_i", "room_j", "wall_cells", "door_cells", "n_doors"])
        V = len(rooms)
        for i in range(V):
            for j in range(i + 1, V):
                if g["wall_cells"][i, j] or g["door_cells"][i, j]:
                    w.writerow([rooms[i], rooms[j], int(g["wall_cells"][i, j]), int(g["door_cells"][i, j]), int(g["n_doors"][i, j])])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", type=Path)
    ap.add_argument("--floor-plan", type=Path); ap.add_argument("--zone-map", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--all-shipped", action="store_true")
    a = ap.parse_args()
    jobs = []
    if a.all_shipped:
        root = KIT / "smart_buildings" / "training_scenarios"
        for b in sorted(p for p in root.iterdir() if p.is_dir()):
            d = sorted(p for p in b.iterdir() if (p / "floor_plan.npy").exists())[0]
            jobs.append((d / "floor_plan.npy", d / "zone_map.npy", d / "config.gin", WC / "outputs" / "geometry" / b.name))
    elif a.scenario:
        jobs.append((a.scenario / "floor_plan.npy", a.scenario / "zone_map.npy", a.scenario / "config.gin", a.out or WC / "outputs" / "geometry" / a.scenario.name))
    elif a.floor_plan and a.zone_map:
        jobs.append((a.floor_plan, a.zone_map, Path("/nonexistent"), a.out or WC / "outputs" / "geometry" / a.floor_plan.stem))
    else:
        ap.error("give a scenario folder, --floor-plan/--zone-map, or --all-shipped")
    for fp, zm, gin, out in jobs:
        rooms, g = build(np.load(fp), np.load(zm), ahu_rooms_from_gin(gin))
        write(rooms, g, out)
        print(f"{out}: {len(rooms)} rooms, {int((g['facade_sides'] >= 2).sum())} corner, {int((g['facade_faces'] == 0).sum())} interior")


if __name__ == "__main__":
    main()
