"""Annotate every entry of the observation and action vectors with a modelling role.

Reads the CSVs written by dump_vectors.py and writes *_annotated.csv plus a full
per-entry markdown listing for the smallest building. Role rules come from reading
vav.py, air_handler.py, hot_water_system.py and environment.py (see STATE_SPACE.md).
"""
import csv, re
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "outputs"

# (role, in_dynamics_model, units/normalization, what it is)
def obs_role(name):
    m = re.match(r"VAV_(\d+)_(.*)", name)
    if m:
        f = m.group(2)
        return {
         "zone_air_temperature_sensor": ("OUTPUT / STATE", "YES - target", "(T-295)/5 -> K", "zone-average air temperature"),
         "supply_air_damper_percentage_command": ("INPUT (realized)", "YES - input", "(x-0.5)/0.5 -> [0,1]", "damper position the box actually took = max(action, 0.1) (rollout-confirmed)"),
         "reheat_valve_percentage_command": ("INPUT (realized)", "YES - input", "(x-0.5)/0.5 -> [0,1]", "reheat valve position = the action exactly (rollout-confirmed)"),
         "supply_air_flowrate_setpoint": ("CONSTANT (box parameter)", "parameter only", "raw m3/s", "box max airflow; 2.0 at every step (rollout-confirmed)"),
         "zone_air_heating_temperature_setpoint_sensor": ("CONTEXT (comfort band)", "NO - reward/policy only", "raw K", "lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes"),
         "zone_air_cooling_temperature_setpoint_sensor": ("CONTEXT (comfort band)", "NO - reward/policy only", "raw K", "upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes"),
        }[f]
    m = re.match(r"ahs_ahu_(\d+)_(.*)", name)
    if m:
        f = m.group(2)
        return {
         "cooling_request_count": ("CONSTANT (rollout-confirmed)", "NO", "raw count", "= number of zones the AHU serves (10 for train_8) at every step of a 600-step random-action rollout"),
         "discharge_fan_speed_percentage_command": ("DUPLICATE (rollout-confirmed)", "NO (redundant)", "raw [0,1]", "identical to supply_fan_speed_percentage_command at every step"),
         "supply_fan_speed_percentage_command": ("ALGEBRAIC (tracks flow)", "optional", "raw [0,1]", "fan speed fraction; correlation 1.000 with supply airflow"),
         "exhaust_fan_run_command": ("DUPLICATE (rollout-confirmed)", "NO (redundant)", "raw {0,1}", "identical to supervisor_run_command at every step"),
         "supply_fan_run_command": ("DUPLICATE (rollout-confirmed)", "NO (redundant)", "raw {0,1}", "identical to supervisor_run_command at every step"),
         "supervisor_run_command": ("INPUT (echo)", "NO - use the action", "raw {0,1}", "AHU on/off; equals the discrete action exactly"),
         "outside_air_flowrate_sensor": ("ALGEBRAIC (derived)", "optional", "raw m3/s", "outside-air (ventilation) flow, follows total flow and recirculation"),
         "outside_air_temperature_sensor": ("EXOGENOUS (disturbance)", "YES - disturbance", "raw K", "outside air temperature; same number in every AHU block"),
         "recirculation_percentage_command": ("INPUT (echo)", "NO - use the action", "raw [0,0.7]", "= min(native action, 0.7) (rollout-confirmed)"),
         "supply_air_flowrate_sensor": ("ALGEBRAIC (derived from dampers + pressure)", "YES - intermediate signal", "raw m3/s", "total supply airflow; 0 when the unit is off"),
         "supply_air_flowrate_setpoint": ("DUPLICATE (rollout-confirmed)", "NO (redundant)", "raw m3/s", "identical to supply_air_flowrate_sensor at every step"),
         "supply_air_static_pressure_sensor": ("ALGEBRAIC (= setpoint)", "NO - function of the action", "raw Pa", "equals max(action, 200 Pa) when running and 0 when off, with no lag"),
         "supply_air_static_pressure_setpoint": ("INPUT (echo)", "NO - use the action", "raw Pa", "= max(native action, 200 Pa) when running, 0 when off (rollout-confirmed)"),
         "supply_air_temperature_sensor": ("ALGEBRAIC (= setpoint)", "NO - function of the action", "raw K", "mirrors the setpoint exactly in every step; the physical min(mixed air, setpoint) is NOT what is reported"),
         "supply_air_temperature_setpoint": ("INPUT (echo)", "NO - use the action", "raw K", "equals the native action exactly (rollout-confirmed)"),
        }[f]
    if name.startswith("hws_"):
        f = name[4:]
        return {
         "differential_pressure_sensor": ("ALGEBRAIC (= setpoint)", "NO - function of the action", "raw", "= native pump setpoint when running, 0 when off, no lag (rollout-confirmed)"),
         "differential_pressure_setpoint": ("INPUT (echo)", "NO - use the action", "normalized (x-10)/6", "native pump setpoint when running; reads 0 (-1.667 normalized) when off (rollout-confirmed)"),
         "heating_request_count": ("CONSTANT (rollout-confirmed)", "NO", "raw count", "= total zone count (30 for train_8) at every step"),
         "run_status": ("INPUT (echo)", "NO (equals the hot-water run action)", "{-1,+1} normalized", "identical to hws_supervisor_run_command at every step"),
         "supervisor_run_command": ("INPUT (echo)", "NO - use the action", "{-1,+1} normalized", "hot-water on/off: -1 = off, +1 = on"),
         "supply_water_temperature_sensor": ("PLANT STATE (true memory)", "YES - the one observed plant state", "raw K", "ramp-limited: +2.5 K/step max up, -0.5 K/step max down; equals the setpoint only 41% of the time"),
         "supply_water_temperature_setpoint": ("INPUT (echo)", "NO - use the action", "raw K", "equals the native action exactly (rollout-confirmed)"),
        }[f]
    if name.startswith(("hod_", "dow_")):
        return ("EXOGENOUS (clock)", "YES - as 2 phases", "[-1,1]", "phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant)")
    return {"comfort_mode_now": ("CONTEXT (schedule)", "YES - cheap known disturbance proxy", "{0,1}", "building in comfort mode now"),
            "comfort_mode_soon": ("CONTEXT (schedule lookahead)", "NO - policy only", "{0,1}", "comfort mode 60 min ahead"),
            "num_occupants": ("EXOGENOUS (disturbance)", "YES - disturbance", "raw count (normalization constant is 0)", "building-total occupant count, not per zone (0-57 of 60 seen in the rollout)")}[name]

def act_role(dev, nm):
    if "run_command" in nm: return ("DISCRETE INPUT", "YES - input", "{0,1}", "on/off for AHU" if dev == "ahs" else "on/off for hot-water loop")
    nm2 = nm.split("_", 2)[-1] if nm.startswith("ahu_") else nm
    d = {"supply_air_temperature_setpoint": "supply-air temperature target 285-305 K",
         "supply_air_static_pressure_setpoint": "fan static-pressure target 0-2000 Pa (floor 200 Pa)",
         "recirculation_percentage_command": "return-air share 0-1 (cap 0.7)",
         "supply_water_temperature_setpoint": "loop water temperature target 310-350 K",
         "differential_pressure_setpoint": "pump head target 0-20",
         "supply_air_damper_percentage_command": "VAV damper 0-1 (floor 0.1)",
         "reheat_valve_percentage_command": "VAV reheat valve 0-1"}
    return ("CONTINUOUS INPUT", "YES - input", "[-1,1] normalized", d.get(nm2, d.get(nm, "")))

for b in ["building_train_%d" % i for i in (1, 2, 4, 5, 7, 8)]:
    rows = list(csv.DictReader(open(OUT / f"{b}_observation.csv")))
    with open(OUT / f"{b}_observation_annotated.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["index", "field_id", "role", "use_in_dynamics_model", "units_as_delivered", "meaning", "initial_value"])
        for r in rows: w.writerow([r["index"], r["field_id"], *obs_role(r["field_id"]), r["initial_value"]])
    arows = list(csv.DictReader(open(OUT / f"{b}_action.csv")))
    ci = di = 0
    with open(OUT / f"{b}_action_annotated.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tuple_index", "kind_index", "kind", "device_id", "setpoint_name", "role", "use_in_dynamics_model", "units", "meaning"])
        for r in arows:
            disc = r["kind"] == "discrete"; k = di if disc else ci
            if disc: di += 1
            else: ci += 1
            w.writerow([r["tuple_index"], k, r["kind"], r["device_id"], r["setpoint_name"], *act_role(r["device_id"], r["setpoint_name"])])

print("ok (the full listing is appended to STATE_SPACE.md by build_appendix.py)")
