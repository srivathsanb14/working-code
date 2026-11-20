"""Time, schedule and 'future observations' for planning (shared by the feed-forward controller and the MPC).

The policy only sees the clock as phase-shifted sin/cos copies, so time is decoded from hod_*_000 / dow_*_000, advanced
by whole steps, and re-encoded.  The comfort schedule comes from the metadata (morning/evening start hours, comfort and
eco windows); holidays are not decodable from the clock and are ignored.
FutureObs.make() writes an observation vector for a future step: predicted room temperatures, advanced clock, the band
(sensors and comfort flags) a plan wants the base controller to see, forecast occupants, and the outside temperature.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2 * np.pi


class Schedule:
    def __init__(self, meta, ix):
        self.ix = ix
        self.morning, self.evening = float(meta["morning_start_hour"]), float(meta["evening_start_hour"])
        self.comfort = tuple(map(float, meta["comfort_temp_window"])); self.eco = tuple(map(float, meta["eco_temp_window"]))

    def angles(self, obs):
        hs, hc = obs[self.ix["hod_sin_000"]], obs[self.ix["hod_cos_000"]]
        ds, dc = obs[self.ix["dow_sin_000"]], obs[self.ix["dow_cos_000"]]
        return np.arctan2(hs, hc) % TWO_PI, np.arctan2(ds, dc) % TWO_PI

    def time(self, obs, steps_ahead=0):
        """(hour as float, weekday 0=Monday) of the step that starts `steps_ahead` steps after this observation's step."""
        th, td = self.angles(obs)
        hour = th / TWO_PI * 24.0 + steps_ahead * 5.0 / 60.0
        day = int(round(td / (TWO_PI / 7))) % 7
        day = (day + int(hour // 24)) % 7
        return hour % 24.0, day

    def is_comfort(self, hour, day):
        return day < 5 and self.morning <= hour < self.evening

    def band(self, hour, day):
        return self.comfort if self.is_comfort(hour, day) else self.eco

    def steps_to_comfort(self, obs, max_steps=288 * 3):
        for k in range(max_steps):
            if self.is_comfort(*self.time(obs, k)):
                return k
        return None


class FutureObs:
    def __init__(self, meta, io):
        self.io, self.ix = io, io.ix
        self.sched = Schedule(meta, io.ix)
        sp = "_zone_air_heating_temperature_setpoint_sensor"; cp = "_zone_air_cooling_temperature_setpoint_sensor"
        self.hsp = np.array([self.ix[v + sp] for v in io.vavs if v + sp in self.ix])
        self.csp = np.array([self.ix[v + cp] for v in io.vavs if v + cp in self.ix])
        self.hod = [(self.ix[f"hod_sin_{i:03d}"], self.ix[f"hod_cos_{i:03d}"]) for i in range(12) if f"hod_sin_{i:03d}" in self.ix]
        self.dow = [(self.ix[f"dow_sin_{i:03d}"], self.ix[f"dow_cos_{i:03d}"]) for i in range(12) if f"dow_sin_{i:03d}" in self.ix]
        self.oat = [i for n, i in self.ix.items() if n.endswith("_outside_air_temperature_sensor")]

    def make(self, obs, T_rooms=None, steps_ahead=0, band=None, occupants=None, T_out=None):
        o = np.array(obs, float, copy=True)
        if T_rooms is not None:
            o[self.io.zt] = (np.asarray(T_rooms) - self.io.zmu) / self.io.zsd
        if steps_ahead:
            th, td = self.sched.angles(obs)
            th2 = (th + TWO_PI * steps_ahead * 5.0 / (24 * 60)) % TWO_PI
            td2 = (td + TWO_PI / 7 * ((th / TWO_PI * 24 + steps_ahead * 5 / 60) // 24)) % TWO_PI
            for i, (s, c) in enumerate(self.hod):
                o[s], o[c] = np.sin(th2 + TWO_PI * i / 12), np.cos(th2 + TWO_PI * i / 12)
            for i, (s, c) in enumerate(self.dow):
                o[s], o[c] = np.sin(td2 + TWO_PI * i / 12), np.cos(td2 + TWO_PI * i / 12)
        hour, day = self.sched.time(obs, steps_ahead)
        lo, hi = band if band is not None else self.sched.band(hour, day)
        o[self.hsp] = lo; o[self.csp] = hi
        comfort_like = (lo, hi) == self.sched.comfort
        o[self.ix["comfort_mode_now"]] = 1.0 if comfort_like else 0.0
        if "comfort_mode_soon" in self.ix:
            o[self.ix["comfort_mode_soon"]] = 1.0 if (comfort_like or self.sched.is_comfort(*self.sched.time(obs, steps_ahead + 12))) else 0.0
        if occupants is not None:
            o[self.ix["num_occupants"]] = occupants
        if T_out is not None:
            for i in self.oat:
                o[i] = T_out
        return o
