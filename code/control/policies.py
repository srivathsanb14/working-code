"""Model-based controllers with the submission interface (backlog K2, K3, K6).

    FeedForwardPolicy(meta, action_spec, model_src)   pid_water, plus model-chosen pre-heat before the morning start
                                                       and early set-back before the evening end
    MPCPolicy(meta, action_spec, model_src)            every `replan` steps, picks the mode (schedule / comfort / eco / off)
                                                       with the best predicted reward over the horizon
Both carry a ClosedFormTracker of the discovered model, an occupancy forecast and the cost model; both fall back to
plain pid_water while the model has no history yet or if planning fails.  On/off (K6) is part of the mode choice:
'off' switches every AHU and the boiler off when the model predicts that is cheapest and still comfortable enough.
"""
from __future__ import annotations

import time

import numpy as np

from control.occupancy import OccupancyForecaster
from control.planner import MODES, Planner
from control.schedule import FutureObs
from runtime.closedform_tracker import ClosedFormTracker
from submission.vendor.pid_plus import PIDWaterResetPolicy


class _ModelBased:
    modes_default = MODES

    def __init__(self, building_meta_data, action_spec=None, model_src=None, horizon=24, replan=3):
        assert model_src, "a model-based controller needs the discovered model (best.py)"
        self.base = PIDWaterResetPolicy(building_meta_data, action_spec)
        self.tr = ClosedFormTracker(building_meta_data, model_src)
        self.fobs = FutureObs(building_meta_data, self.tr.io)
        self.occ = OccupancyForecaster(self.tr.V, self.tr.geo["area"])
        self.plan = Planner(building_meta_data, self.tr, self.fobs, self.occ, horizon)
        self.replan, self.mode, self.t = replan, "schedule", 0
        self.slots = []                                   # (slot, workday) of each call, to learn occupancy 2 steps late
        self.log = []; self.plan_seconds = []

    def name(self):
        return self.__class__.__name__

    def _learn_occupancy(self, obs):
        hour, day = self.fobs.sched.time(obs)
        self.slots.append((int(hour * 12) % 288, day < 5))
        if len(self.slots) > 2:
            s, w = self.slots[-3]
            self.occ.observe(float(obs[self.tr.io.ix["num_occupants"]]), s, w)

    def candidate_modes(self, obs):
        return self.modes_default

    def get_action(self, obs):
        obs = np.asarray(obs, float)
        self._learn_occupancy(obs)
        if self.plan.ready() and self.t % self.replan == 0:
            t0 = time.time()
            try:
                cands = self.candidate_modes(obs)
                if len(cands) > 1:
                    scores = {m: self.plan.cost(self.base, obs, m, self.horizon_for(obs))[0] for m in cands}
                    self.mode = min(scores, key=scores.get)
                    self.log.append((self.t, self.mode, scores))
                else:
                    self.mode = cands[0]
            except Exception:  # noqa: BLE001
                self.mode = "schedule"
            self.plan_seconds.append(time.time() - t0)
        a = self.plan.action_for(self.base, self.mode, obs)
        self.tr.observe(obs, a)
        self.t += 1
        return a

    def horizon_for(self, obs):
        return self.plan.H


class FeedForwardPolicy(_ModelBased):
    """pid_water everywhere, except two model-decided transitions (backlog K2)."""

    def __init__(self, building_meta_data, action_spec=None, model_src=None, lead_steps=36, replan=1):
        super().__init__(building_meta_data, action_spec, model_src, horizon=lead_steps + 12, replan=replan)
        self.lead = lead_steps

    def candidate_modes(self, obs):
        sch = self.fobs.sched
        k = sch.steps_to_comfort(obs, self.lead + 1)
        if k is not None and 0 < k <= self.lead:
            return ("schedule", "comfort")               # start the comfort band early, or wait?
        hour, day = sch.time(obs)
        if sch.is_comfort(hour, day) and not sch.is_comfort(*sch.time(obs, self.lead // 2)):
            return ("schedule", "eco")                   # end of the day: coast on stored heat, or keep going?
        return ("schedule",)

    def horizon_for(self, obs):
        k = self.fobs.sched.steps_to_comfort(obs, self.lead + 1)
        return (k + 12) if k else self.lead // 2 + 6


class MPCPolicy(_ModelBased):
    """Mode MPC over a reduced action (backlog K3): schedule / comfort / eco / off, replanned every few steps."""
