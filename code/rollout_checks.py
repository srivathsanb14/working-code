"""Short rollout on train_8 with piecewise-random actions to test which observation
fields are duplicates / echoes / constants / normalized.  Writes only to ../outputs.
Run from the starter-kit root:  <venv>/python <this file> [n_steps]
"""
import json, sys
from pathlib import Path
import numpy as np
KIT = Path(__import__("os").environ.get("STARTER_KIT", Path(__file__).resolve().parent.parent.parent / "smart-buildings-starter-kit")); OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(KIT)); from smart_buildings import evaluate as ev
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
d = KIT / "smart_buildings/training_scenarios/building_train_8/atlanta_1"
env = ev.load_hybrid_environment(d / "config.gin")
names = list(env.field_names); ix = {n: i for i, n in enumerate(names)}
tuples = list(env._device_action_tuples)
cont = [(dv, nm) for dv, nm in tuples if "run_command" not in nm]
disc = [(dv, nm) for dv, nm in tuples if "run_command" in nm]
cidx = {k: i for i, k in enumerate(cont)}; didx = {k: i for i, k in enumerate(disc)}
ts = env.reset(); obs = np.asarray(ts[0] if isinstance(ts, tuple) else ts.observation)
rng = np.random.default_rng(0)
nc, nd = len(cont), len(disc)
O, A, D = [obs.copy()], [], []
a_c = rng.uniform(-1, 1, nc).astype(np.float32); a_d = np.ones(nd, np.int32)
for t in range(N):
    if t % 12 == 0:
        a_c = rng.uniform(-1, 1, nc).astype(np.float32)
        a_d = (rng.uniform(size=nd) > 0.15).astype(np.int32)   # occasionally switch units off
    out = env.step({"continuous_action": a_c, "discrete_action": a_d})
    obs = np.asarray(out[0]); O.append(obs.copy()); A.append(a_c.copy()); D.append(a_d.copy())
O = np.array(O); A = np.array(A); D = np.array(D)
np.savez_compressed(OUT / "rollout_train_8.npz", obs=O, cont=A, disc=D, names=np.array(names))
Ox = O[1:]                       # observation after each step, aligned with the action just applied
col = lambda n: Ox[:, ix[n]]
R = {"n_steps": N}
def maxdiff(a, b): return float(np.max(np.abs(col(a) - col(b))))
for j in (1, 2, 3):
    p = f"ahs_ahu_{j}_"
    R[f"ahu{j}_exhaust_vs_supervisor_run"] = maxdiff(p + "exhaust_fan_run_command", p + "supervisor_run_command")
    R[f"ahu{j}_supplyfan_vs_supervisor_run"] = maxdiff(p + "supply_fan_run_command", p + "supervisor_run_command")
    R[f"ahu{j}_discharge_vs_supply_fan_speed"] = maxdiff(p + "discharge_fan_speed_percentage_command", p + "supply_fan_speed_percentage_command")
    R[f"ahu{j}_flow_sensor_vs_setpoint"] = maxdiff(p + "supply_air_flowrate_sensor", p + "supply_air_flowrate_setpoint")
    R[f"ahu{j}_supervisor_run_vs_discrete_action"] = float(np.max(np.abs(col(p + "supervisor_run_command") - D[:, didx[("ahs", f"ahu_{j}_supervisor_run_command")]])))
    sa = A[:, cidx[("ahs", f"ahu_{j}_supply_air_temperature_setpoint")]]; R[f"ahu{j}_satemp_echo_vs_action_native"] = float(np.max(np.abs(col(p + "supply_air_temperature_setpoint") - (285 + (sa + 1) / 2 * 20))))
    pa = A[:, cidx[("ahs", f"ahu_{j}_supply_air_static_pressure_setpoint")]]; nat = (pa + 1) / 2 * 2000
    R[f"ahu{j}_pressure_echo_vs_action_native"] = float(np.max(np.abs(col(p + "supply_air_static_pressure_setpoint") - nat)))
    R[f"ahu{j}_pressure_echo_vs_floor200"] = float(np.max(np.abs(col(p + "supply_air_static_pressure_setpoint") - np.maximum(nat, 200))))
    ra = A[:, cidx[("ahs", f"ahu_{j}_recirculation_percentage_command")]]; nat = (ra + 1) / 2
    R[f"ahu{j}_recirc_echo_vs_action"] = float(np.max(np.abs(col(p + "recirculation_percentage_command") - nat)))
    R[f"ahu{j}_recirc_echo_vs_cap0.7"] = float(np.max(np.abs(col(p + "recirculation_percentage_command") - np.minimum(nat, 0.7))))
    R[f"ahu{j}_oat_vs_ahu1_oat"] = maxdiff(p + "outside_air_temperature_sensor", "ahs_ahu_1_outside_air_temperature_sensor")
# hot water
w = A[:, cidx[("hws", "supply_water_temperature_setpoint")]]; R["hws_water_setpoint_echo_vs_action_native"] = float(np.max(np.abs(col("hws_supply_water_temperature_setpoint") - (310 + (w + 1) / 2 * 40))))
dp = A[:, cidx[("hws", "differential_pressure_setpoint")]]; nat = (dp + 1) / 2 * 20
echo = col("hws_differential_pressure_setpoint")
R["hws_dp_echo_range"] = [float(echo.min()), float(echo.max())]; R["hws_dp_native_action_range"] = [float(nat.min()), float(nat.max())]
c = np.polyfit(nat, echo, 1); R["hws_dp_echo_linear_fit_slope_intercept"] = [float(c[0]), float(c[1])]
R["hws_dp_echo_resid_after_fit"] = float(np.max(np.abs(np.polyval(c, nat) - echo)))
R["hws_dp_echo_vs_(native-10)/6_resid"] = float(np.max(np.abs((nat - 10) / 6 - echo)))
for f in ("run_status", "supervisor_run_command"): R[f"hws_{f}_unique"] = sorted({round(float(v), 3) for v in col("hws_" + f)})
R["hws_supervisor_vs_discrete_action"] = float(np.max(np.abs(col("hws_supervisor_run_command") - D[:, didx[("hws", "supervisor_run_command")]])))
R["hws_runstatus_vs_supervisor"] = maxdiff("hws_run_status", "hws_supervisor_run_command")
# VAVs
dmax_floor = dmax_raw = rmax = 0.0; fl = set(); hs = set(); cs = set()
for v in range(1, 31):
    p = f"VAV_{v}_"; da = A[:, cidx[(f"VAV_{v}", "supply_air_damper_percentage_command")]]; ra = A[:, cidx[(f"VAV_{v}", "reheat_valve_percentage_command")]]
    dr = col(p + "supply_air_damper_percentage_command")      # normalized (x-.5)/.5 == 2x-1
    dmax_raw = max(dmax_raw, float(np.max(np.abs(dr - da)))); dmax_floor = max(dmax_floor, float(np.max(np.abs(dr - np.maximum(da, -0.8)))))
    rmax = max(rmax, float(np.max(np.abs(col(p + "reheat_valve_percentage_command") - ra))))
    fl |= {round(float(x), 3) for x in col(p + "supply_air_flowrate_setpoint")}
    hs |= {round(float(x), 2) for x in col(p + "zone_air_heating_temperature_setpoint_sensor")}; cs |= {round(float(x), 2) for x in col(p + "zone_air_cooling_temperature_setpoint_sensor")}
R["vav_damper_realized_vs_action_raw"] = dmax_raw; R["vav_damper_realized_vs_action_with_floor0.1"] = dmax_floor
R["vav_reheat_realized_vs_action"] = rmax; R["vav_box_max_airflow_unique"] = sorted(fl)
R["vav_heating_setpoint_unique"] = sorted(hs); R["vav_cooling_setpoint_unique"] = sorted(cs)
# request counts and general ranges
for p in ("ahs_ahu_1_", "hws_"):
    for f in ("cooling_request_count", "heating_request_count"):
        if p + f in ix: R[p + f + "_range"] = [float(col(p + f).min()), float(col(p + f).max())]
zt = np.array([col(f"VAV_{v}_zone_air_temperature_sensor") for v in range(1, 31)]) * 5 + 295
R["zone_temp_K_range_after_denorm"] = [float(zt.min()), float(zt.max())]
for f in ("supply_air_temperature_sensor", "supply_air_static_pressure_sensor", "supply_air_flowrate_sensor", "outside_air_flowrate_sensor", "supply_fan_speed_percentage_command"):
    R[f"ahu1_{f}_range"] = [float(col('ahs_ahu_1_' + f).min()), float(col('ahs_ahu_1_' + f).max())]
R["hws_supply_water_temperature_sensor_range"] = [float(col("hws_supply_water_temperature_sensor").min()), float(col("hws_supply_water_temperature_sensor").max())]
R["num_occupants_range"] = [float(col("num_occupants").min()), float(col("num_occupants").max())]
(OUT / "rollout_train_8_checks.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))
