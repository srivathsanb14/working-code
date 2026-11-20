"""Simulate a closed-loop plan with the discovered model and price it with the reward (shared by K2 and K3).

A plan is a MODE for the base controller (the kit's pid_water, vendored) over the horizon:
  schedule  the band the schedule says (what pid_water does anyway)
  comfort   the comfort band now (pre-heat / pre-cool before the morning start)
  eco       the eco band now (set back / coast before the evening end)
  off       every AHU and the boiler off (ventilation floors still apply in the simulator)
For each mode a copy of the base controller is stepped on future observations built from the model's predicted
room temperatures (control/schedule.FutureObs); its commands become room inputs through the kit's equipment laws
(runtime/inputs.py); the model (runtime/closedform_tracker.py) predicts the next temperatures; control/costs.py prices
comfort (with the occupancy forecast), energy and carbon.  Nothing is fitted.
"""
from __future__ import annotations

import copy

import numpy as np

from control import costs

MODES = ("schedule", "comfort", "eco", "off")


class Planner:
    def __init__(self, meta, tracker, fobs, occ, horizon=24):
        self.tr, self.io, self.fobs, self.occ, self.H = tracker, tracker.io, fobs, occ, horizon
        self.sched = fobs.sched

    def band_for(self, mode, hour, day):
        return {"comfort": self.sched.comfort, "eco": self.sched.eco}.get(mode, self.sched.band(hour, day))

    def action_for(self, base, mode, obs, steps_ahead=0, T_rooms=None, occupants=None):
        hour, day = self.sched.time(obs, steps_ahead)
        o = self.fobs.make(obs, T_rooms=T_rooms, steps_ahead=steps_ahead, band=self.band_for(mode, hour, day), occupants=occupants)
        a = base.get_action(o)
        if mode == "off":
            a = {"continuous_action": np.asarray(a["continuous_action"], np.float32), "discrete_action": np.zeros_like(np.asarray(a["discrete_action"]))}
        return a

    def ready(self):
        return len(self.tr.inputs) >= 2 and len(self.tr.X) == len(self.tr.inputs)

    def cost(self, base, obs, mode, horizon=None):
        """Predicted -reward summed over the horizon for one mode (lower is better), and the predicted temperatures."""
        H = horizon or self.H
        b = copy.deepcopy(base)
        tw0 = self.io.tw
        x = self.tr.predict_next()                       # x_t: the state the current step starts from
        future, temps, total = [], [], 0.0
        try:
            for k in range(H):
                hour, day = self.sched.time(obs, k)
                slot = int(hour * 12) % 288
                occ_tot = float(self.occ.total(slot, day < 5, 1)[0])
                a = self.action_for(b, mode, obs, k, T_rooms=x, occupants=occ_tot)
                c = self.io.native(a); d = np.asarray(a["discrete_action"], float)
                o = self.fobs.make(obs, T_rooms=x, steps_ahead=k, band=self.band_for(mode, hour, day), occupants=occ_tot)
                inp, aux = self.io.step_inputs(o, c, d, x, d[: self.io.K] > 0.5, aux=True, advance=True)
                future.append(inp)
                x = self.tr.rollout(future)[-1]
                h1, d1 = self.sched.time(obs, k + 1)
                lo, hi = self.sched.band(h1, d1)
                reg = float(costs.comfort_regret(x, lo, hi, self.occ.rooms(int(h1 * 12) % 288, d1 < 5, 1)[0]))
                p = costs.power(aux)
                E, C = costs.normalized_energy_carbon(p["electric"], p["gas"], self.io.K, hour, day < 5, None)
                total -= costs.reward(reg, E, C)
                temps.append(x)
        finally:
            self.io.tw = tw0
        return total, np.asarray(temps)
