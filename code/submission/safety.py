"""Action safety layer: whatever the controller returns, the gateway gets a valid action.

- exact shapes (n_continuous_action, n_discrete_action); float32 / int32
- every continuous value finite and inside [-1, 1]; NaN or inf replaced by the fallback's value
- discrete values in {0, 1}
- any exception in the controller -> the fallback action (the last valid action, else a safe default)
"""
from __future__ import annotations

import numpy as np


class SafeAction:
    def __init__(self, n_cont, n_disc, default_cont=None, default_disc=None):
        self.n_cont, self.n_disc = int(n_cont), int(n_disc)
        self.default_c = np.zeros(self.n_cont, np.float32) if default_cont is None else np.asarray(default_cont, np.float32)
        self.default_d = np.ones(self.n_disc, np.int32) if default_disc is None else np.asarray(default_disc, np.int32)
        self.last = None
        self.repairs = 0

    def fallback(self):
        if self.last is not None:
            return {"continuous_action": self.last[0].copy(), "discrete_action": self.last[1].copy()}
        return {"continuous_action": self.default_c.copy(), "discrete_action": self.default_d.copy()}

    def sanitize(self, action):
        fb = self.fallback()
        try:
            c = np.asarray(action["continuous_action"], np.float64).reshape(-1)
            d = np.asarray(action["discrete_action"], np.float64).reshape(-1)
        except Exception:  # noqa: BLE001
            self.repairs += 1; return fb
        if c.shape != (self.n_cont,) or d.shape != (self.n_disc,):
            self.repairs += 1; return fb
        bad = ~np.isfinite(c)
        if bad.any():
            self.repairs += 1; c = np.where(bad, fb["continuous_action"], c)
        badd = ~np.isfinite(d)
        if badd.any():
            self.repairs += 1; d = np.where(badd, fb["discrete_action"], d)
        out = {"continuous_action": np.clip(c, -1.0, 1.0).astype(np.float32),
               "discrete_action": (d > 0.5).astype(np.int32)}
        self.last = (out["continuous_action"].copy(), out["discrete_action"].copy())
        return out

    def call(self, fn, *args):
        try:
            return self.sanitize(fn(*args))
        except Exception:  # noqa: BLE001
            self.repairs += 1
            return self.fallback()
