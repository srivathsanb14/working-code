"""Room geometry from floor_plan.npy / zone_map.npy via the simulator's own room dictionary.

zone_map uses the same 0/1/2 codes as floor_plan (0 space, 1 wall, 2 exterior) but
with doors filled in as walls, so rooms are the connected components of zone_map==0.
Per room we measure area, exterior-wall contact per side (N/S/E/W), neighbours
through walls and through doors, and classify corner / edge / interior.

Run from the starter-kit root (gin paths are relative to it); writes only to ../outputs:
  cd <kit> && PYTHONDONTWRITEBYTECODE=1 <venv>/python <this file> [building ...]
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

KIT = Path("/Users/srivathsan/Desktop/smart-buildings-starter-kit")
OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(KIT))
from smart_buildings import evaluate as ev  # noqa: E402

REACH = 6          # cells (1.2 m at 20 cm/cell) searched for an exterior wall past a room edge
FACE_MIN = 5       # min cells of room boundary facing the exterior (>=1 m) to count a side as a face
DIRS = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}

def shift(mask, dr, dc):
    out = np.zeros_like(mask)
    h, w = mask.shape
    rs, re = max(dr, 0), h + min(dr, 0); cs, ce = max(dc, 0), w + min(dc, 0)
    out[rs:re, cs:ce] = mask[rs - dr:re - dr, cs - dc:ce - dc]
    return out

bl = sys.argv[1:] or ["building_train_%d" % i for i in (1, 2, 4, 5, 7, 8)]
summary = {}
for b in bl:
    d = sorted((KIT / "smart_buildings/training_scenarios" / b).iterdir())[0]
    fp = np.load(d / "floor_plan.npy"); zm = np.load(d / "zone_map.npy")
    env = ev.load_hybrid_environment(d / "config.gin")
    bld = env.building.simulator.building
    cv_m = bld.cv_size_cm / 100.0
    assign = ev.get_ahu_zone_assignments(env)
    room_ahu = {r: a for a, rs in assign.items() for r in rs}
    H, W = bld.floor_plan.shape
    rooms = {k: v for k, v in bld.room_dict.items() if k.startswith("room_")}
    label = np.zeros((H, W), int); names = {}
    for i, (k, cells) in enumerate(sorted(rooms.items(), key=lambda kv: int(kv[0].split("_")[1])), 1):
        rr, cc = np.array(cells).T; label[rr, cc] = i; names[i] = k
    ext_wall = np.asarray(bld._exterior_walls).astype(bool) | np.asarray(bld._exterior_space).astype(bool)
    # doors: cells open in floor_plan but wall in zone_map
    door = (fp == 0) & (zm == 1)
    door_lab, nd = ndimage.label(door, structure=np.ones((3, 3)))
    rows = []
    adj_wall = {i: {} for i in names}
    for i, k in names.items():
        m = label == i
        n = int(m.sum()); rr, cc = np.nonzero(m)
        faces = {}
        for dn, (dr, dc) in DIRS.items():
            # cells whose neighbour in direction dn lies outside the room (shift(m,-d)[x] == m[x+d])
            edge = m & ~shift(m, -dr, -dc)
            hit = np.zeros_like(m)
            for s in range(1, REACH + 1):
                hit |= edge & shift(ext_wall, -dr * s, -dc * s)
            faces[dn] = int(hit.sum())
        nfaces = sum(v >= FACE_MIN for v in faces.values())
        # neighbouring rooms through walls
        ring = ndimage.binary_dilation(m, iterations=REACH) & ~m
        nbr = np.unique(label[ring]); nbr = [int(x) for x in nbr if x not in (0, i)]
        # doors touching this room
        dl = np.unique(door_lab[ndimage.binary_dilation(m, iterations=2) & door]); dl = dl[dl > 0]
        door_to = set()
        for l in dl:
            dm = ndimage.binary_dilation(door_lab == l, iterations=3)
            door_to |= {int(x) for x in np.unique(label[dm]) if x not in (0, i)}
        rows.append(dict(room=k, vav=k.replace("room_", "VAV_"), ahu=room_ahu.get(k), cells=n,
                         area_m2=round(n * cv_m * cv_m, 1), row_min=int(rr.min()), row_max=int(rr.max()),
                         col_min=int(cc.min()), col_max=int(cc.max()), cy=round(float(rr.mean()), 1),
                         cx=round(float(cc.mean()), 1), ext_N=faces["N"], ext_S=faces["S"], ext_W=faces["W"],
                         ext_E=faces["E"], n_ext_faces=nfaces,
                         type="interior" if nfaces == 0 else ("edge" if nfaces == 1 else "corner"),
                         n_wall_neighbours=len(nbr), n_door_neighbours=len(door_to),
                         wall_neighbours=" ".join(names[x] for x in nbr),
                         door_neighbours=" ".join(names[x] for x in sorted(door_to))))
    with open(OUT / f"{b}_rooms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    types = {t: sum(r["type"] == t for r in rows) for t in ("corner", "edge", "interior")}
    perim = sum(r["ext_N"] + r["ext_S"] + r["ext_W"] + r["ext_E"] for r in rows)
    summary[b] = dict(scenario=d.name, grid=[H, W], rooms=len(rows), cv_m=cv_m, types=types,
                      area_min=min(r["area_m2"] for r in rows), area_max=max(r["area_m2"] for r in rows),
                      doors=int(nd), zone_map_vs_floor_plan_differing_cells=int((fp != zm).sum()))
    print(b, summary[b])
(OUT / "zone_map_summary.json").write_text(json.dumps(summary, indent=1))
