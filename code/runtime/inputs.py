"""Room inputs exactly as a submitted policy can compute them on a NEW building.

Used by the runtime model (closedform_tracker.ClosedFormTracker).  It reads only:
  1. building_meta_data (once): floor plan, zone map, AHU assignment, field names, action indices,
     normalisation metadata;
  2. the observation of each call;
  3. the policy's own actions.
The command-to-heat conversion is the starter kit's equipment law, the same one
discovery/starterkit_data.episode_inputs uses offline (replay is checked against the logged AHU flow).

Timing (GUIDE.md B.2): obs_t shows room temperatures x_{t-1}; the inputs of step k need x_k (return
air) and the action taken at k, so they become computable one call later.
"""
from __future__ import annotations

import numpy as np

from discovery.episodes import geometry, natural_key

RHO_C_AIR = 1.2 * 1006.0
RHO_C_WATER = 1000.0 * 4180.0
VAV_MAX_FLOW, P_DESIGN, AHU_MAX_FLOW, REHEAT_KV, T_DISCHARGE_MAX = 2.0, 2000.0, 20.0, 0.108, 308.15
GAIN_W_M2, STANDBY = 10.0, 0.2
TW_RESET, TW_UP, TW_DOWN = 340.0, 2.5, 0.5
OCC_NORM = 1.0            # num_occupants arrives as a head count (checked against reward_components.total_occupancy)
INPUT_KEYS = ("T_out", "g", "T_supply", "Q_reheat", "Q_gain", "occupants", "comfort",
              "hour_sin", "hour_cos", "dow_sin", "dow_cos", "T_water")
PER_ROOM = ("g", "T_supply", "Q_reheat", "Q_gain")


class RoomInputs:
    """Metadata parsing + the per-step command-to-heat conversion (stateful only in the boiler water temperature)."""

    def __init__(self, meta):
        names = list(meta["observation_field_names"]); self.ix = {n: i for i, n in enumerate(names)}
        suf = "_zone_air_temperature_sensor"
        self.vavs = [n[: -len(suf)] for n in names if n.endswith(suf)]          # observation (VAV) order
        self.rooms = [f"room_{v.split('_')[1]}" for v in self.vavs]
        self.ahus = sorted(meta["ahu_zone_assignments"], key=natural_key)        # positional order of the action lists
        self.geo = geometry(meta["floor_plan"], meta["zone_map"], self.rooms, meta["ahu_zone_assignments"], ahu_order=self.ahus)
        self.V, self.K = self.geo["V"], len(self.ahus)
        self.zt = np.array([self.ix[v + suf] for v in self.vavs])
        zn = meta["observation_normalization_metadata"]["zone_air_temperature_sensor"]
        self.zmu, self.zsd = zn["sample_mean"], zn["sample_std"]
        self.oat_ix = self.ix[f"ahs_{self.ahus[0]}_outside_air_temperature_sensor"]
        zai = meta["zone_action_indices"]
        self.damper_ix = np.array([zai[v]["supply_air_damper_percentage_command"] for v in self.vavs])
        self.reheat_ix = np.array([zai[v]["reheat_valve_percentage_command"] for v in self.vavs])
        self.temp_ix = np.array(meta["temp_indices"][: self.K]); self.press_ix = np.array(meta["pressure_indices"][: self.K])
        self.rec_ix = np.array(meta["recirculation_indices"][: self.K])
        self.water_ix, self.dp_ix = meta["water_indices"][0], meta["pressure_indices"][-1]
        self.n_cont, self.n_disc = meta["n_continuous_action"], meta["n_discrete_action"]
        anm = meta["action_normalization_metadata"]
        names_c = [None] * self.n_cont
        for j, a in enumerate(self.ahus):
            names_c[self.temp_ix[j]] = f"{a}_supply_air_temperature_setpoint"
            names_c[self.press_ix[j]] = f"{a}_supply_air_static_pressure_setpoint"
            names_c[self.rec_ix[j]] = f"{a}_recirculation_percentage_command"
        names_c[self.water_ix] = "supply_water_temperature_setpoint"; names_c[self.dp_ix] = "differential_pressure_setpoint"
        for i in range(self.V):
            names_c[self.damper_ix[i]] = "supply_air_damper_percentage_command"
            names_c[self.reheat_ix[i]] = "reheat_valve_percentage_command"
        self.lo = np.array([anm[n]["min_native_value"] for n in names_c]); self.hi = np.array([anm[n]["max_native_value"] for n in names_c])
        self.clock = {k: self.ix.get(f) for k, f in (("hour_sin", "hod_sin_000"), ("hour_cos", "hod_cos_000"),
                                                      ("dow_sin", "dow_sin_000"), ("dow_cos", "dow_cos_000"))}
        self.tw = TW_RESET

    def measured(self, obs):
        """Room air temperatures (K) shown by this observation (they are x_{t-1})."""
        return np.asarray(obs, float)[self.zt] * self.zsd + self.zmu

    def native(self, action):
        """The continuous action in native units (from [-1, 1])."""
        return self.lo + (np.clip(np.asarray(action["continuous_action"], float), -1, 1) + 1) / 2 * (self.hi - self.lo)

    def step_inputs(self, obs_k, c_native, d, x_k, run_now, aux=False, advance=True):
        """Inputs of step k (x_k -> x_{k+1}) as a dict of scalars / (V,) arrays; advances the boiler water state.
        aux=True also returns the equipment quantities the energy cost needs (control/costs.py)."""
        area, room_ahu = self.geo["area"], self.geo["ahu"]
        oat = float(obs_k[self.oat_ix]); oat = oat if oat > 150 else oat * 10.0 + 295.0     # the harness passes kelvin
        on = np.asarray(run_now, bool)
        P = np.where(on, np.maximum(c_native[self.press_ix], 200.0), 0.0)
        rec = np.minimum(c_native[self.rec_ix], 0.7); sp = c_native[self.temp_ix]
        damper = np.maximum(c_native[self.damper_ix], 0.1); reheat = c_native[self.reheat_ix]
        fu = np.maximum(damper * VAV_MAX_FLOW * np.sqrt(P[room_ahu] / P_DESIGN), 1e-5)
        dem = np.bincount(room_ahu, fu, self.K)
        flow = fu * np.minimum(1.0, AHU_MAX_FLOW / np.maximum(dem, 1e-12))[room_ahu]
        g = flow * RHO_C_AIR
        Tret = np.bincount(room_ahu, area * x_k, self.K) / np.bincount(room_ahu, area, self.K)
        mixed = rec * Tret + (1 - rec) * oat
        Tsa = np.where(on & (mixed > sp), sp, mixed)[room_ahu]
        dp = c_native[self.dp_ix] if d[-1] > 0.5 else 0.0
        qw = reheat * REHEAT_KV * np.sqrt(max(dp, 0.0)) / 3600.0
        eff = np.minimum(1.0, RHO_C_WATER * qw / np.maximum(g, 1e-12))
        gap = self.tw - Tsa; head = T_DISCHARGE_MAX - Tsa
        eff = np.where(gap > 0, np.where(head <= 0, 0.0, np.minimum(eff, head / np.where(gap > 0, gap, 1))), eff)
        comfort = float(obs_k[self.ix["comfort_mode_now"]]) > 0.5
        out = dict(T_out=oat, g=g, T_supply=Tsa, Q_reheat=eff * g * gap, Q_gain=GAIN_W_M2 * area * (1.0 if comfort else STANDBY),
                   occupants=float(obs_k[self.ix["num_occupants"]]) * OCC_NORM, comfort=float(comfort), T_water=self.tw)
        for k, i in self.clock.items():
            out[k] = float(obs_k[i]) if i is not None else 0.0
        tw_now = self.tw
        if advance:
            self.tw = self.tw + float(np.clip(c_native[self.water_ix] - self.tw, -TW_DOWN, TW_UP))
        if not aux:
            return out
        ahu_flow = np.bincount(room_ahu, flow, self.K) * on
        Tsa_ahu = np.where(on & (mixed > sp), sp, mixed)
        A = dict(ahu_flow=ahu_flow, ahu_pressure=P, recirculation=rec, mixed=mixed, T_supply_ahu=Tsa_ahu,
                 water_flow=float((reheat * REHEAT_KV * np.sqrt(max(dp, 0.0)) / 3600.0).sum()), pump_dp=dp,
                 Q_reheat_total=float(out["Q_reheat"].sum()), T_water=tw_now, T_water_setpoint=float(c_native[self.water_ix]),
                 hws_on=bool(d[-1] > 0.5))
        return out, A
