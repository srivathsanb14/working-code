"""Check that room_model.grid_geometry/cell_properties, built from the metadata
arrays only, reproduce the simulator's own thermal grid cell for cell.
Run from the starter-kit root:  <venv>/python <this file> [building ...]
"""
import json
import sys
import os
os.environ.setdefault("NUMBA_CACHE_DIR", str(__import__('pathlib').Path(__file__).resolve().parent.parent / ".numba_cache"))  # keep the starter kit untouched
from pathlib import Path

import numpy as np

KIT = Path(__import__("os").environ.get("STARTER_KIT", Path(__file__).resolve().parent.parent.parent / "smart-buildings-starter-kit"))
OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from smart_buildings import evaluate as ev  # noqa: E402
import room_model as rm  # noqa: E402

res = {}
for b in sys.argv[1:] or ["building_train_8", "building_train_1", "building_train_2", "building_train_4", "building_train_5", "building_train_7"]:
    d = sorted((KIT / "smart_buildings/training_scenarios" / b).iterdir())[0]
    env = ev.load_hybrid_environment(d / "config.gin")
    meta = ev.create_building_meta_data(env, d)
    sim = env.building.simulator; bld = sim.building; S = sim._solver
    h = sim._weather_controller.get_air_convection_coefficient(sim._current_timestamp)
    theta = dict(k_iw=bld.inside_wall_properties.conductivity,
                 rc_iw=bld.inside_wall_properties.density * bld.inside_wall_properties.heat_capacity,
                 k_ext=bld.building_exterior_properties.conductivity,
                 rc_ext=bld.building_exterior_properties.density * bld.building_exterior_properties.heat_capacity,
                 rc_mass=bld.interior_mass.material.density * bld.interior_mass.material.heat_capacity, h=h)
    G = rm.grid_geometry(meta["floor_plan"], meta["zone_map"])
    assert G["shape"] == bld.temp.shape, (G["shape"], bld.temp.shape)
    k, C, Cm, Gm, amb = rm.cell_properties(G, theta)
    gc, gr = rm.face_conductances(G, k)
    solved_sim = np.zeros(bld.temp.size, bool); solved_sim[S._solved] = True
    solved_sim = solved_sim.reshape(bld.temp.shape)
    r = {}
    r["solved_mismatch_cells"] = int((solved_sim != G["solved"]).sum())
    r["material_k_mismatch_cells"] = int((np.abs(np.asarray(bld.conductivity) - k) > 1e-9).sum())
    sl = G["solved"].ravel()
    cap_sim = S._capacitance_rate * rm.DT
    r["capacitance_max_rel_err"] = float(np.max(np.abs(C.ravel()[sl] - cap_sim) / cap_sim))
    tot = np.zeros(bld.temp.shape)
    tot[:, :-1] += gc; tot[:, 1:] += gc; tot[:-1, :] += gr; tot[1:, :] += gr
    tot_sim = np.asarray(S._conduction.diagonal()) - S._capacitance_rate - S._mass_coupling
    r["face_conductance_sum_max_abs_err_WK"] = float(np.max(np.abs(tot.ravel()[sl] - tot_sim)))
    r["ambient_conductance_max_abs_err_WK"] = float(np.max(np.abs(amb.ravel()[sl] - S.ambient_conductance(h))))
    mc = (Cm / rm.DT) * Gm / np.maximum(Cm / rm.DT + Gm, 1e-30)
    r["mass_coupling_max_abs_err"] = float(np.max(np.abs(mc.ravel()[sl] - S._mass_coupling)))
    rooms = sorted((z for z in bld.room_dict if z.startswith("room_")), key=lambda s: int(s.split("_")[1]))
    bad = 0
    for z in rooms:
        n = int(z.split("_")[1]); mine = set(map(tuple, np.argwhere(G["lab"] == n))); sim_c = set(map(tuple, bld.room_dict[z]))
        bad += mine != sim_c
    r["rooms_mismatch"] = bad; r["rooms"] = len(rooms)
    r["theta"] = theta
    res[b] = r
    print(b, json.dumps({k2: v for k2, v in r.items() if k2 != "theta"}))
(OUT / "room_model_validate.json").write_text(json.dumps(res, indent=1))
