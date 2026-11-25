"""Occupancy forecast for the comfort term.

Only the building's total head count is observed (two steps late), so the forecast is a per-building daily profile:
  prior   the kit's schedule: arrivals spread over 06-12, departures over 13-19, workdays only, about 1.5 people
          per room at the peak (the generator draws 1-2 per zone);
  online  an exponential average of the observed head count in each 5-minute slot of a workday, blended with the
          prior until a few days are seen.  Weekends: the observed weekend profile (the prior is zero).
Rooms: the total is split between rooms in proportion to floor area (or equally), the open question D4.
"""
from __future__ import annotations

import numpy as np

SLOTS = 288


def prior_profile(peak):
    h = np.arange(SLOTS) * 24.0 / SLOTS
    arrive = np.clip((h - 6.0) / 6.0, 0, 1); leave = np.clip((h - 13.0) / 6.0, 0, 1)
    return peak * (arrive - leave).clip(0, 1)


class OccupancyForecaster:
    def __init__(self, V, area=None, split="area", alpha=0.3):
        self.V = V; self.alpha = alpha
        self.share = (np.asarray(area, float) / np.sum(area)) if (split == "area" and area is not None) else np.full(V, 1.0 / V)
        self.prof = {True: prior_profile(1.5 * V), False: np.zeros(SLOTS)}
        self.days_seen = {True: 0, False: 0}
        self._last_slot = None

    @staticmethod
    def slot_and_workday(hour_sin, hour_cos, dow_sin, dow_cos):
        h = (np.degrees(np.arctan2(hour_sin, hour_cos)) % 360) / 15.0
        d = int(round((np.degrees(np.arctan2(dow_sin, dow_cos)) % 360) / (360 / 7))) % 7
        return int(h * SLOTS / 24.0) % SLOTS, d < 5

    def observe(self, occupants, slot, workday):
        """Learn from the head count observed for (slot, workday).  The observation is two steps late, so callers
        pass the slot of two steps ago."""
        p = self.prof[workday]
        w = self.alpha if self.days_seen[workday] >= 1 else 0.6
        p[slot] = (1 - w) * p[slot] + w * occupants
        if self._last_slot is not None and slot < self._last_slot:
            self.days_seen[workday] += 1
        self._last_slot = slot

    def total(self, slot, workday, steps=1):
        return np.array([self.prof[workday][(slot + k) % SLOTS] for k in range(steps)])

    def rooms(self, slot, workday, steps=1):
        """Forecast occupants per room, (steps, V)."""
        return self.total(slot, workday, steps)[:, None] * self.share[None, :]
