"""Per-room heat inputs rebuilt from what a policy sees: the observation vector,
its own previous action and the metadata.  Mirrors simulator_flexible_floor_plan
.execute_step_sim, vav.py, air_handler.py and hot_water_system.py.

Timing (measured against the solver's own inputs): the observation returned
after step n carries the realized damper, valve, pressure, recirculation and
setpoints used DURING step n, and the zone temperatures from the START of step n
(one step stale).  The outside temperature, the supply-water temperature and the
comfort flag that step n uses are the ones in the observation returned BEFORE
it.  With the true per-zone occupancy the rebuilt heat matches the solver to
5 W rms (max 70 W, one occupant); splitting the building total equally over
zones adds about 16 W rms per room.
"""
import numpy as np

RHO_C_AIR = 1.2 * 1006.0        # constants.AIR_DENSITY * AIR_HEAT_CAPACITY (VAV side)
RHO_C_WATER = 1000.0 * 4180.0
VAV_MAX_FLOW = 2.0              # m3/s, hard-coded
P_DESIGN = 2000.0               # Pa, VAV max_air_flow_static_pressure
AHU_MAX_FLOW = 20.0             # m3/s, class-B config value
REHEAT_KV = 0.108               # vav_reheat_water_flowrate, hard-coded
T_DISCHARGE_MAX = 308.15
GAIN_W_M2, STANDBY = 10.0, 0.2
OCC_W = 70.0


class InputModel:
    def __init__(self, names, rooms, ahu_rooms, area):
        ix = {n: i for i, n in enumerate(names)}
        self.V = len(rooms)
        nums = [int(r.split("_")[1]) for r in rooms]
        self.damper = np.array([ix[f"VAV_{n}_supply_air_damper_percentage_command"] for n in nums])
        self.reheat = np.array([ix[f"VAV_{n}_reheat_valve_percentage_command"] for n in nums])
        self.ztemp = np.array([ix[f"VAV_{n}_zone_air_temperature_sensor"] for n in nums])
        self.ahus = sorted(ahu_rooms)
        self.room_ahu = np.zeros(self.V, int)
        pos = {r: k for k, r in enumerate(rooms)}
        for a, ah in enumerate(self.ahus):
            for r in ahu_rooms[ah]:
                self.room_ahu[pos[r]] = a
        p = lambda a, f: ix[f"ahs_{a}_{f}"]
        self.press = np.array([p(a, "supply_air_static_pressure_sensor") for a in self.ahus])
        self.sat_sp = np.array([p(a, "supply_air_temperature_setpoint") for a in self.ahus])
        self.recirc = np.array([p(a, "recirculation_percentage_command") for a in self.ahus])
        self.run = np.array([p(a, "supervisor_run_command") for a in self.ahus])
        self.oat = p(self.ahus[0], "outside_air_temperature_sensor")
        self.tw = ix["hws_supply_water_temperature_sensor"]
        self.dp = ix["hws_differential_pressure_sensor"]
        self.comfort = ix["comfort_mode_now"]
        self.occ = ix["num_occupants"]
        self.area = np.asarray(area, float)

    def zone_T(self, obs):
        return obs[self.ztemp] * 5.0 + 295.0

    def step_inputs(self, obs_prev, obs_after, T_rooms):
        """Inputs of the step between obs_prev and obs_after.

        T_rooms: room temperatures at the start of the step (the model's own
        estimate; the sensors only deliver them one step later).
        Returns T_out, g (W/K per room), T_supply (per room), Q (W per room).
        """
        Tout = obs_prev[self.oat]
        damper = obs_after[self.damper] * 0.5 + 0.5
        reheat = obs_after[self.reheat] * 0.5 + 0.5
        P = np.maximum(obs_after[self.press], 0.0)
        flow_u = np.maximum(damper * VAV_MAX_FLOW * np.sqrt(P[self.room_ahu] / P_DESIGN), 1e-5)
        dem = np.bincount(self.room_ahu, flow_u, len(self.ahus))
        ratio = np.minimum(1.0, AHU_MAX_FLOW / np.maximum(dem, 1e-12))
        flow = flow_u * ratio[self.room_ahu]
        g = flow * RHO_C_AIR
        # supply air: area-weighted return temperature, mixed with outside, capped by setpoint when running
        w = self.area
        Tret = np.bincount(self.room_ahu, w * T_rooms, len(self.ahus)) / np.bincount(self.room_ahu, w, len(self.ahus))
        r = obs_after[self.recirc]
        mixed = r * Tret + (1 - r) * Tout
        on = obs_after[self.run] > 0.5
        sp = obs_after[self.sat_sp]
        Tsa = np.where(on & (mixed > sp), sp, mixed)
        Tsup = Tsa[self.room_ahu]
        # reheat coil
        Tw = obs_prev[self.tw]
        dp = max(float(obs_after[self.dp]), 0.0)
        ff = reheat * REHEAT_KV
        qw = ff * np.sqrt(dp) / 3600.0
        eff = np.minimum(1.0, RHO_C_WATER * qw / np.maximum(g, 1e-12))
        gap = Tw - Tsup
        head = T_DISCHARGE_MAX - Tsup
        eff = np.where(gap > 0, np.where(head <= 0, 0.0, np.minimum(eff, head / np.where(gap > 0, gap, 1))), eff)
        Qre = eff * g * gap
        comfort = obs_prev[self.comfort] > 0.5
        Qgain = GAIN_W_M2 * self.area * (1.0 if comfort else STANDBY)
        Qocc = OCC_W * obs_after[self.occ] / self.V * np.ones(self.V)
        return Tout, g, Tsup, Qre + Qgain + Qocc, dict(Qre=Qre, Qgain=Qgain, Qocc=Qocc)
