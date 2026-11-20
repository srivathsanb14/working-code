"""A closed-form candidate (discovery/closedform.py contract) running inside a submitted policy.

Uses only the metadata, the observations and the policy's own actions (runtime/inputs.py).  Nothing is
fitted: the candidate's prepare(geo, U) runs once in __init__, and every call appends the newly measured
room temperatures and the newly computable step inputs to a history buffer, the same layout the offline
scorer uses (closedform.Prepared: X[k] = temperatures at the start of step k, inputs[k] drive X[k] -> X[k+1]).

    tr = ClosedFormTracker(meta, open("best.py").read())
    tr.observe(obs, action)            # every get_action, after choosing the action
    tr.predict_next()                  # forecast of the state the next observation will show
    tr.rollout(future_inputs)          # free-run forecast for a plan (list of input dicts), for MPC
    tr.to_episode()                    # the logged episode, scored with closedform.episode_scores
"""
from __future__ import annotations

import numpy as np

from discovery import closedform as cf
from discovery import contract
from discovery.episodes import Episode
from runtime.inputs import INPUT_KEYS, RoomInputs


class ClosedFormTracker:
    def __init__(self, meta, src, U=None):
        self.io = RoomInputs(meta)
        self.mod, _ = contract.load_closedform(src)
        self.U = U or cf.universal_constants()[0]
        self.geo, self.V, self.K, self.H = self.io.geo, self.io.V, self.io.K, self.mod.HISTORY
        self.coef = self.mod.prepare(self.geo, self.U) if hasattr(self.mod, "prepare") else None
        self.stateful = cf.is_stateful(self.mod)
        self.filt = []             # stateful candidates: state after correcting with X[k]
        self.X = []                # measured room temperatures, X[k] = start of step k
        self.inputs = []           # inputs[k] drive X[k] -> X[k+1]
        self.prev = None
        self.one_step = []         # (k, forecast of X[k]) made before X[k] was observed
        self._free_state = None

    # ------------------------------------------------------------------ bookkeeping
    def observe(self, obs, action):
        """Call once per get_action with the observation received and the action returned.
        Call t >= 1 shows x_{t-1}; the action of call t-1 drives x_{t-1} -> x_t, so its inputs (which need x_{t-1}
        for the return air) become computable now.  Call 0 shows the reset state, which call 1 shows again."""
        obs = np.asarray(obs, float)
        c_native, d = self.io.native(action), np.asarray(action["discrete_action"], float)
        if self.prev is not None:
            y = self.io.measured(obs)                   # x_{t-1}
            if self.inputs and len(self.X) == len(self.inputs):
                self.one_step.append((len(self.X), self._step(self.X, self.inputs), y))
            self.X.append(y)
            if self.stateful:
                self._filter_update()
            po, pc, pd, prun = self.prev
            self.inputs.append(self.io.step_inputs(po, pc, pd, y, prun))   # inputs of step t-1
        # the first run command takes effect one step late (GUIDE.md B.2)
        run_now = (d[: self.K] > 0.5) if self.prev is not None else np.zeros(self.K, bool)
        self.prev = (obs, c_native, d, run_now)

    # ------------------------------------------------------------------ forecasting
    def _window(self, X, U_list):
        """hist/u for a step from X[-1], inputs U_list[-1] (the step's own inputs), padded like closedform.Prepared."""
        H = self.H
        Xp = [X[0]] * max(0, H - len(X)) + list(X[-H:])
        past = U_list[:-1]
        Up = [U_list[0]] * max(0, H - len(past)) + list(past[-H:])
        hist = {"T": np.asarray(Xp)[None]}
        for k in INPUT_KEYS:
            if k in Up[0]:
                hist[k] = np.asarray([u[k] for u in Up])[None]
        u = {k: np.asarray(U_list[-1][k])[None] for k in U_list[-1]}
        return hist, u

    def _filter_update(self):
        k = len(self.X) - 1
        if k == 0:
            s = np.atleast_2d(np.asarray(self.mod.init_state(self.coef, self.geo, self.U), float))
        else:
            s = self.mod.predict(self.coef, self.geo, self.filt[-1][None], self._u1(self.inputs[k - 1]), self.U)
        self.filt.append(np.asarray(self.mod.correct(self.coef, self.geo, s, self.X[k][None], self.U), float)[0])

    @staticmethod
    def _u1(inp):
        return {k: np.asarray(v)[None] for k, v in inp.items()}

    def _step(self, X, U_list):
        if self.stateful:                                # state at X[-1] predicted through U_list[-1]
            s = self.filt[len(X) - 1] if len(X) - 1 < len(self.filt) else self._free_state
            s = np.asarray(self.mod.predict(self.coef, self.geo, s[None], self._u1(U_list[-1]), self.U), float)[0]
            self._free_state = s
            return s[: self.V]
        hist, u = self._window(X, U_list)
        y = np.asarray(self.mod.step(self.coef, self.geo, hist, u, self.U), float)[0]
        return y

    def rollout(self, future_inputs, start=None):
        """Free-run from the latest measured state through future_inputs[0], [1], ... (each a step-input dict).
        The first entry is the step from X[-1]; returns (len(future_inputs), V) forecasts."""
        X = list(self.X if start is None else self.X[:start + 1])
        U = list(self.inputs[: len(X) - 1])
        out = []
        if self.stateful:
            st = self.filt[len(X) - 1][None]
            for f in future_inputs:
                st = np.asarray(self.mod.predict(self.coef, self.geo, st, self._u1(f), self.U), float)
                out.append(st[0, : self.V])
            return np.asarray(out)
        for f in future_inputs:
            U.append(f)
            y = self._step(X, U)
            out.append(y); X.append(y)
        return np.asarray(out)

    def predict_next(self):
        """Forecast of x_t, the state the next observation will show (after X ends at x_{t-1}, inputs at step t-1)."""
        if not self.inputs or len(self.inputs) != len(self.X):
            return None
        return self._step(self.X, self.inputs)

    def live_one_step_rmse(self):
        """RMSE of the forecasts predict_next would have made, against what was then observed."""
        if not self.one_step:
            return float("nan")
        return float(np.sqrt(np.mean([(f - y) ** 2 for _, f, y in self.one_step])))

    # ------------------------------------------------------------------ evaluation of a logged episode
    def to_episode(self, name="tracked"):
        """The logged run as an Episode (T_meas[k] = X[k+1]), so the offline scorer can score it unchanged."""
        n = min(len(self.inputs), len(self.X) - 1)
        inp = {k: np.asarray([self.inputs[i][k] for i in range(n)]) for k in self.inputs[0]}
        T = np.asarray(self.X[1:n + 1])
        return Episode(name=name, building="b", weather="w", policy="p", scenario="b/w", geo=self.geo, inputs=inp,
                       T_meas=T, T0=float(np.mean(self.X[0])))

    def forecast_errors(self, horizons=(12, 36, 72), every=12, warmup=36):
        ep = self.to_episode()
        r = cf.episode_scores(self.mod, self.coef, ep, self.U, horizons, every, warmup, keep=horizons[min(1, len(horizons) - 1)])
        return {"one_step": r["one_step"], **{f"{h / 12:g}h": r[f"{h}"] for h in horizons},
                **{f"persistence_{h / 12:g}h": r[f"persistence_{h}"] for h in horizons}}
