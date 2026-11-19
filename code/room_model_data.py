"""Roll out one scenario under a behaviour policy and log, per step, both what the
policy observes and what the thermal solver actually received.

The solver call is intercepted, so the log holds the exact per-room heat inputs
(air-side conductance g and supply temperature per VAV, coil + internal-gain +
occupant watts per room), the ambient temperature and film coefficient, and the
true zone-average temperature after the step.  These are the oracle inputs used
to test a room-level model against the grid; the observation is logged beside
them so the same test can be repeated with inference-time inputs only.

Run from the starter-kit root:
    <venv>/python <this file> <building> <weather> <policy> <steps> [seed]
policy: alloff | random | pidwater
Writes ../outputs/roomdata_<building>_<weather>_<policy>.npz
"""
import sys
from pathlib import Path

import numpy as np

KIT = Path(__import__("os").environ.get("STARTER_KIT", Path(__file__).resolve().parent.parent.parent / "smart-buildings-starter-kit"))
OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(KIT))
from smart_buildings import evaluate as ev  # noqa: E402

b, w, pol, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
d = KIT / "smart_buildings/training_scenarios" / b / w
env = ev.load_hybrid_environment(d / "config.gin")
meta = ev.create_building_meta_data(env, d)
sim = env.building.simulator
bld = sim.building
solver = sim._solver
hvac = sim._hvac

rooms = sorted((z for z in bld.room_dict if z.startswith("room_")), key=lambda s: int(s.split("_")[1]))
V = len(rooms)
rpos = {z: k for k, z in enumerate(rooms)}
cells = {z: tuple(np.transpose(bld.room_dict[z])) for z in rooms}
ncols = bld.temp.shape[1]
col_room = []                                   # air-side column -> room index (one zone per VAV here)
for v_id in sim._coupled_vavs:
    zs = [z for z in hvac.get_zones_for_vav(v_id) if z in rpos]
    assert len(zs) == 1, (v_id, zs)
    col_room.append(rpos[zs[0]])
col_room = np.array(col_room)

LOG = {k: [] for k in ("Tamb", "h", "g", "Tsup", "Qroom", "Ttrue_after", "Tsensor", "occ_true")}
_orig = solver.solve


def _logged_solve(temp, input_q, ambient_temperature, convection_coefficient, air_side_conductance=None, supply_temperature=None):
    LOG["Tamb"].append(float(ambient_temperature))
    LOG["h"].append(float(convection_coefficient))
    g = np.zeros(V); ts = np.zeros(V)
    if air_side_conductance is not None:
        g[col_room] = air_side_conductance; ts[col_room] = supply_temperature
    LOG["g"].append(g); LOG["Tsup"].append(ts)
    LOG["Qroom"].append(np.array([input_q[cells[z]].sum() for z in rooms]))
    return _orig(temp, input_q, ambient_temperature, convection_coefficient, air_side_conductance, supply_temperature)


solver.solve = _logged_solve

names = list(env.field_names)
tuples = list(env._device_action_tuples)
cont = [(dv, nm) for dv, nm in tuples if "run_command" not in nm]
disc = [(dv, nm) for dv, nm in tuples if "run_command" in nm]
nc, nd = len(cont), len(disc)
ts0 = env.reset()
obs = np.asarray(ts0[0] if isinstance(ts0, tuple) else ts0.observation)
rng = np.random.default_rng(seed)
policy = None
if pol == "pidwater":
    policy = ev.initialize_policy(ev.POLICY_MAP["pid_water"], meta, env.action_spec())

O, AC, AD = [obs.copy()], [], []
a_c = -np.ones(nc, np.float32); a_d = np.zeros(nd, np.int32)
for t in range(steps):
    if pol == "random" and t % 12 == 0:
        a_c = rng.uniform(-1, 1, nc).astype(np.float32)
        a_d = (rng.uniform(size=nd) > 0.15).astype(np.int32)
    elif pol == "pidwater":
        act = ev.get_policy_action(policy, obs)
        a_c = np.asarray(act["continuous_action"], np.float32); a_d = np.asarray(act["discrete_action"], np.int32)
    out = env.step({"continuous_action": a_c, "discrete_action": a_d})
    obs = np.asarray(out[0])
    O.append(obs.copy()); AC.append(a_c.copy()); AD.append(a_d.copy())
    avg = bld.get_zone_average_temps()
    LOG["Ttrue_after"].append(np.array([avg[z] for z in rooms]))
    LOG["Tsensor"].append(np.array([hvac.vavs[f"VAV_{z.split('_')[1]}"].zone_air_temperature if f"VAV_{z.split('_')[1]}" in hvac.vavs else np.nan for z in rooms]))
    start = sim._current_timestamp - __import__("pandas").Timedelta(sim._time_step_sec, unit="s")
    end = sim._current_timestamp
    LOG["occ_true"].append(np.array([sim._occupancy.average_zone_occupancy(z, start, end) for z in rooms]))

OUT.mkdir(exist_ok=True)
np.savez_compressed(
    OUT / f"roomdata_{b}_{w}_{pol}.npz",
    obs=np.array(O), cont=np.array(AC), disc=np.array(AD), names=np.array(names),
    rooms=np.array(rooms), ahu_json=np.array(__import__("json").dumps({k: list(v) for k, v in meta["ahu_zone_assignments"].items()})), floor_plan=np.asarray(meta["floor_plan"]), zone_map=np.asarray(meta["zone_map"]),
    T0=np.array([bld.get_zone_average_temps()[z] for z in rooms]) * 0 + bld.initial_temp,
    mat=np.array([bld.inside_wall_properties.conductivity, bld.inside_wall_properties.density * bld.inside_wall_properties.heat_capacity,
                  bld.building_exterior_properties.conductivity, bld.building_exterior_properties.density * bld.building_exterior_properties.heat_capacity,
                  bld.interior_mass.material.conductivity, bld.interior_mass.material.density * bld.interior_mass.material.heat_capacity,
                  bld.inside_air_properties.conductivity, bld.inside_air_properties.density * bld.inside_air_properties.heat_capacity]),
    **{k: np.array(v) for k, v in LOG.items()},
)
print("saved", b, w, pol, steps, "V", V)
