"""The state-space model exactly as a submitted policy runs it on a NEW building.

At evaluation a policy gets only three things (Kaggle Data tab; starter kit evaluate.py):
  1. building_meta_data, once, in __init__: floor_plan, zone_map, ahu_zone_assignments, observation
     field names, action indices and normalization constants, schedule;
  2. obs, every 5 minutes, in get_action: a flat vector of sensors (stale by one step);
  3. its own previous actions, which it remembers.
OnlineBuildingModel uses nothing else.  It builds the room network from the metadata with constants
learned offline (PARAMS, THETA from a discovery run: no fitting on the new building), computes each
step's room inputs from its own action and the sensors, and corrects the hidden state from measured
history with a Kalman filter.  `state` is the filtered estimate a controller would plan from.

Timing (GUIDE.md B.2): obs_t shows room temperatures x_{t-1}.  So at call t the inputs of step t-1
become computable (they need x_{t-1} for the return air), and the filter can predict x_{t-1} from
x_{t-2} and correct it with the measurement: the estimate runs one step behind, as the sensors do.
"""
from __future__ import annotations

import numpy as np

from discovery import dlib  # noqa: F401
from runtime.inputs import RoomInputs, TW_RESET

W_PERSON = 70.0


class OnlineBuildingModel:
    def __init__(self, meta, model, params, theta, q_air=0.05, q_hidden=0.01, r=0.01):
        self.model, self.p, self.th = model, dict(params), dict(theta)
        self.io = RoomInputs(meta)
        for k in ("ix", "vavs", "ahus", "geo", "V", "K", "zt", "zmu", "zsd", "oat_ix", "damper_ix", "reheat_ix", "temp_ix",
                  "press_ix", "rec_ix", "water_ix", "dp_ix", "n_cont", "n_disc", "lo", "hi"):
            setattr(self, k, getattr(self.io, k))
        # the network, built once from the floor plan (no fitting)
        one = self._blank_inputs()
        self.net, _ = model.network(self.geo, one, self.th, self.p)
        V, N = self.V, self.net.N
        self.Qn = np.diag(np.r_[np.full(V, q_air ** 2), np.full(N - V, q_hidden ** 2)]); self.r2 = r ** 2
        self.x = None; self.P = None
        self.prev = None            # (obs, native continuous action, discrete action) of the previous call
        self.pending = None         # forcing of the step whose result the next observation shows
        self.log = dict(x_meas=[], x_filt=[], forcing=[])

    def _blank_inputs(self, n=1):
        V = self.V
        return dict(T_out=np.zeros(n), g=np.zeros((n, V)), T_supply=np.zeros((n, V)), Q_reheat=np.zeros((n, V)),
                    Q_gain=np.zeros((n, V)), occupants=np.zeros(n), comfort=np.zeros(n))

    def measured(self, obs):
        return self.io.measured(obs)

    @property
    def tw(self):
        return self.io.tw

    @tw.setter
    def tw(self, v):
        self.io.tw = v

    def _inputs(self, obs_k, c_native, d, x_k, run_now):
        """Room inputs of step k from obs_k (before the step), the action taken at k and x_k (seen one step later)."""
        one = self.io.step_inputs(obs_k, c_native, d, x_k, run_now)
        inp = self._blank_inputs()
        for k in inp:
            inp[k][0] = one[k]
        if hasattr(self.model, "forcing"):
            f = self.model.forcing(self.geo, inp, self.th, self.p)
        else:                                   # slower: rebuilds the network to get its forcing
            _, f = self.model.network(self.geo, inp, self.th, self.p)
        return {k: np.asarray(v)[0] for k, v in f.items()}

    def _predict(self, f):
        net = self.net; V = self.V
        K = np.linalg.solve(np.eye(V) + f["g"][:, None] * net.S, f["g"][:, None] * net.W.T)
        Mi = net.A0inv - net.W @ K
        rhs = net.Cdt * self.x + net.G_amb * f["T_out"]; rhs[:V] += f["Q"] + f["g"] * f["T_supply"]
        self.x = Mi @ rhs
        A = Mi * net.Cdt[None, :]
        self.P = A @ self.P @ A.T + self.Qn

    def _correct(self, y):
        V = self.V
        Kg = np.linalg.solve(self.P[:V, :V] + self.r2 * np.eye(V), self.P[:V, :]).T
        self.x = self.x + Kg @ (y - self.x[:V]); self.P = self.P - Kg @ self.P[:V, :]

    def observe(self, obs, action):
        """Call once per get_action with the observation received and the action returned."""
        obs = np.asarray(obs, float)
        c_native = self.lo + (np.clip(np.asarray(action["continuous_action"], float), -1, 1) + 1) / 2 * (self.hi - self.lo)
        d = np.asarray(action["discrete_action"], float)
        y = self.measured(obs)                                   # x_{t-1} (obs_0 shows the reset state x_0)
        if self.x is None:
            self.x = np.full(self.net.N, float(np.mean(y))); self.P = np.diag(np.full(self.net.N, 1e-6))
            self.tw = TW_RESET
        else:
            if self.pending is not None:                         # predict x_{t-1} from x_{t-2}, then correct
                self._predict(self.pending)
                self._correct(y)
                self.log["x_meas"].append(y); self.log["x_filt"].append(self.x.copy())
            po, pc, pd, prun = self.prev
            self.pending = self._inputs(po, pc, pd, y, prun)        # step t-1 is now computable
            self.log["forcing"].append(self.pending)
        # the first run command takes effect one step late (GUIDE.md B.2)
        run_now = (d[: self.K] > 0.5) if self.prev is not None else np.zeros(self.K, bool)
        self.prev = (obs, c_native, d, run_now)

    @property
    def state(self):
        return None if self.x is None else self.x.copy()

    def forecast_errors(self, horizons=(12, 36, 72), every=12, start=288):
        """After an episode: forecast RMSE from the filtered states, rolled forward with the logged inputs."""
        meas, filt, fs = np.array(self.log["x_meas"]), np.array(self.log["x_filt"]), self.log["forcing"]
        # filt[j] estimates x_{j+1}; fs[j+1] drives x_{j+1} -> x_{j+2}; meas[j] = x_{j+1}
        n = len(meas); out = {}
        errs = {h: [] for h in horizons}; pers = {h: [] for h in horizons}
        saved = (self.x, self.P)
        for t0 in range(start, n - max(horizons) - 1, every):
            self.x = filt[t0].copy(); self.P = np.zeros((self.net.N, self.net.N))
            for k in range(1, max(horizons) + 1):
                self._predict(fs[t0 + k])
                if k in horizons:
                    errs[k].append(self.x[: self.V] - meas[t0 + k]); pers[k].append(meas[t0] - meas[t0 + k])
        self.x, self.P = saved
        for h in horizons:
            out[f"{h // 12}h"] = float(np.sqrt(np.mean(np.square(errs[h])))) if errs[h] else float("nan")
            out[f"persistence_{h // 12}h"] = float(np.sqrt(np.mean(np.square(pers[h])))) if pers[h] else float("nan")
        return out
