"""A safe, self-contained rule controller (numpy only), used as the fallback controller of the submission.

Not a model-based controller: it reacts to the measured temperatures like the kit's baselines, but reads every
position from the metadata and pre-conditions when comfort mode is about to start (comfort_mode_soon).
  AHU on        comfort hours, the hour before them, or any of its rooms outside the eco band
  supply air    colder when its rooms need cooling; 294 K otherwise (the AHU cannot heat above mixed air)
  pressure      grows with the largest cooling need among its rooms (floor 200 Pa is applied by the simulator)
  recirculation 0.7 (the cap): least outside air to condition
  dampers       open with each room's cooling need (floor 0.1)
  reheat        opens with each room's heating need; boiler on and hot when any room needs heat
"""
from __future__ import annotations

import numpy as np

from submission.layout import Layout


class RulePolicy:
    def __init__(self, building_meta_data, action_spec=None):
        self.L = Layout(building_meta_data)

    def name(self):
        return "rule_policy"

    def band(self, obs):
        L = self.L
        return L.comfort if (L.flag(obs, "comfort_mode_now") or L.flag(obs, "comfort_mode_soon")) else L.eco

    def native_action(self, obs):
        L = self.L
        T = L.rooms_K(obs); lo, hi = self.band(obs); To = L.outside_K(obs)
        cool = np.clip((T - (hi - 0.7)) / 1.5, 0, 1)            # 0..1 per room
        heat = np.clip(((lo + 0.7) - T) / 1.0, 0, 1)
        occupied = L.flag(obs, "comfort_mode_now") or L.flag(obs, "comfort_mode_soon")
        need = (cool > 0) | (heat > 0)
        on = np.array([occupied or bool(need[L.room_ahu == a].any()) for a in range(L.K)])
        c = (L.lo + L.hi) / 2.0
        for a in range(L.K):
            m = L.room_ahu == a
            cmax = float(cool[m].max()) if m.any() else 0.0
            c[L.temp_ix[a]] = 294.0 - 7.0 * cmax
            c[L.press_ix[a]] = 400.0 + 1200.0 * max(cmax, float(heat[m].max()) if m.any() else 0.0)
            c[L.rec_ix[a]] = 0.7
        c[L.damper_ix] = np.maximum(0.1, np.maximum(cool, 0.4 * heat))
        c[L.reheat_ix] = heat
        heating = bool((heat > 0).any()) or To < 283.0
        c[L.water_ix] = 345.0 if heating else 320.0
        c[L.dp_ix] = 15.0 if heating else 5.0
        d = np.zeros(L.n_disc, np.int32); d[: L.K] = on.astype(np.int32); d[-1] = int(heating)
        return c, d

    def get_action(self, obs):
        c, d = self.native_action(np.asarray(obs, float))
        return {"continuous_action": self.L.to_normalized(c).astype(np.float32), "discrete_action": d}
