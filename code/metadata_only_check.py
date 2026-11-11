"""Can room identity and room type be rebuilt from what a policy is actually given?

Uses only the dict returned by evaluate.create_building_meta_data (floor_plan, zone_map,
ahu_zone_assignments, observation_field_names, zone_action_indices ...).  Compares with the
simulator's own room dictionary.  Run from the starter-kit root; writes ../outputs.
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
KIT = Path(__import__("os").environ.get("STARTER_KIT", Path(__file__).resolve().parent.parent.parent / "smart-buildings-starter-kit")); OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(KIT)); from smart_buildings import evaluate as ev
DIRS = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}
def shift(m, dr, dc):
    o = np.zeros_like(m); h, w = m.shape
    o[max(dr,0):h+min(dr,0), max(dc,0):w+min(dc,0)] = m[max(-dr,0):h+min(-dr,0), max(-dc,0):w+min(-dc,0)]; return o
res = {}
for b in ["building_train_%d" % i for i in (1, 2, 4, 5, 7, 8)]:
    d = sorted((KIT / "smart_buildings/training_scenarios" / b).iterdir())[0]
    env = ev.load_hybrid_environment(d / "config.gin"); meta = ev.create_building_meta_data(env, d)
    fp, zm = np.asarray(meta["floor_plan"]), np.asarray(meta["zone_map"])
    # 1. room numbering from the metadata arrays alone
    lab, n = ndimage.label(zm == 0)                      # 4-connectivity, raster-scan numbering
    sim = {k: set(map(tuple, v)) for k, v in env.building.simulator.building.room_dict.items() if k.startswith("room_")}
    same = sum(set(zip(*np.nonzero(lab == int(k.split("_")[1])))) == v for k, v in sim.items())
    # 2. VAV ids named in the observation / action metadata
    vav_obs = sorted({int(m.group(1)) for f in meta["observation_field_names"] if (m := re.match(r"VAV_(\d+)_", f))})
    vav_act = sorted(int(k.split("_")[1]) for k in meta["zone_action_indices"])
    assign = {int(r.split("_")[1]) for rs in meta["ahu_zone_assignments"].values() for r in rs}
    # 3. types from the metadata arrays only (exterior = code 2 in floor_plan)
    ext = fp == 2; types = {}; feats = {}
    for N in range(1, n + 1):
        m = lab == N; face = {}
        for dn, (dr, dc) in DIRS.items():
            edge = m & ~shift(m, -dr, -dc); hit = np.zeros_like(m)
            for s in range(1, 9): hit |= edge & shift(ext, -dr * s, -dc * s)
            face[dn] = int(hit.sum())
        nf = sum(v >= 5 for v in face.values()); types[N] = "interior" if nf == 0 else "edge" if nf == 1 else "corner"
        per = int((m & ~shift(m, 1, 0) | m & ~shift(m, -1, 0) | m & ~shift(m, 0, 1) | m & ~shift(m, 0, -1)).sum())
        feats[N] = dict(cells=int(m.sum()), ext_cells=sum(face.values()), perimeter_cells=per)
    ref = {int(r["room"].split("_")[1]): r["type"] for r in csv.DictReader(open(OUT / f"{b}_rooms.csv"))}
    agree = sum(types[k] == ref[k] for k in ref)
    # 4. floors from floor_plan alone
    fl, nfl = ndimage.label(~ext)
    res[b] = dict(rooms_in_zone_map=int(n), rooms_in_simulator=len(sim), identical_cell_sets=int(same),
                  vav_in_obs=len(vav_obs), vav_in_actions=len(vav_act), vav_ids_equal_room_ids=(vav_obs == vav_act == sorted(assign) == list(range(1, n + 1))),
                  type_agreement_with_simulator_based=f"{agree}/{len(ref)}", floors_from_floor_plan=int(nfl),
                  metadata_keys=sorted(meta))
    print(b, {k: v for k, v in res[b].items() if k != "metadata_keys"})
(OUT / "metadata_only_check.json").write_text(json.dumps(res, indent=1))
