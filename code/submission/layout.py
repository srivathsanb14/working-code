"""Action and observation positions read from the building metadata only (never hard-coded)."""
from __future__ import annotations

import re

import numpy as np


def _nat(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


class Layout:
    def __init__(self, meta):
        names = list(meta["observation_field_names"]); self.ix = {n: i for i, n in enumerate(names)}
        suf = "_zone_air_temperature_sensor"
        self.vavs = [n[: -len(suf)] for n in names if n.endswith(suf)]
        self.rooms = [f"room_{v.split('_')[1]}" for v in self.vavs]
        self.ahus = sorted(meta["ahu_zone_assignments"], key=_nat)
        self.V, self.K = len(self.vavs), len(self.ahus)
        pos = {r: i for i, r in enumerate(self.rooms)}
        self.room_ahu = np.zeros(self.V, int)
        for a, ah in enumerate(self.ahus):
            for r in meta["ahu_zone_assignments"][ah]:
                if r in pos:
                    self.room_ahu[pos[r]] = a
        self.zt = np.array([self.ix[v + suf] for v in self.vavs])
        zn = meta["observation_normalization_metadata"]["zone_air_temperature_sensor"]
        self.zmu, self.zsd = zn["sample_mean"], zn["sample_std"]
        self.oat_ix = self.ix.get(f"ahs_{self.ahus[0]}_outside_air_temperature_sensor")
        zai = meta["zone_action_indices"]
        self.damper_ix = np.array([zai[v]["supply_air_damper_percentage_command"] for v in self.vavs])
        self.reheat_ix = np.array([zai[v]["reheat_valve_percentage_command"] for v in self.vavs])
        self.temp_ix = np.array(meta["temp_indices"][: self.K]); self.press_ix = np.array(meta["pressure_indices"][: self.K])
        self.rec_ix = np.array(meta["recirculation_indices"][: self.K])
        self.water_ix, self.dp_ix = meta["water_indices"][0], meta["pressure_indices"][-1]
        self.n_cont, self.n_disc = int(meta["n_continuous_action"]), int(meta["n_discrete_action"])
        anm = meta["action_normalization_metadata"]
        nm = [None] * self.n_cont
        for j, a in enumerate(self.ahus):
            nm[self.temp_ix[j]] = f"{a}_supply_air_temperature_setpoint"
            nm[self.press_ix[j]] = f"{a}_supply_air_static_pressure_setpoint"
            nm[self.rec_ix[j]] = f"{a}_recirculation_percentage_command"
        nm[self.water_ix] = "supply_water_temperature_setpoint"; nm[self.dp_ix] = "differential_pressure_setpoint"
        for i in range(self.V):
            nm[self.damper_ix[i]] = "supply_air_damper_percentage_command"; nm[self.reheat_ix[i]] = "reheat_valve_percentage_command"
        self.lo = np.array([anm[n]["min_native_value"] for n in nm], float)
        self.hi = np.array([anm[n]["max_native_value"] for n in nm], float)
        self.comfort = tuple(meta.get("comfort_temp_window", (294.0, 297.0)))
        self.eco = tuple(meta.get("eco_temp_window", (289.0, 298.0)))

    def rooms_K(self, obs):
        return np.asarray(obs, float)[self.zt] * self.zsd + self.zmu

    def outside_K(self, obs):
        if self.oat_ix is None:
            return 290.0
        v = float(obs[self.oat_ix]); return v if v > 150 else v * 10.0 + 295.0

    def flag(self, obs, name):
        i = self.ix.get(name); return bool(i is not None and float(obs[i]) > 0.5)

    def to_normalized(self, native):
        return np.clip((np.asarray(native, float) - self.lo) / np.maximum(self.hi - self.lo, 1e-12) * 2 - 1, -1, 1)
