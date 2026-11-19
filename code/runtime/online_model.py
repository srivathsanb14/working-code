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

from discovery import dlib
from discovery.episodes import geometry, natural_key

RHO_C_AIR = 1.2 * 1006.0
RHO_C_WATER = 1000.0 * 4180.0
VAV_MAX_FLOW, P_DESIGN, AHU_MAX_FLOW, REHEAT_KV, T_DISCHARGE_MAX = 2.0, 2000.0, 20.0, 0.108, 308.15
GAIN_W_M2, STANDBY, W_PERSON = 10.0, 0.2, 70.0
TW_RESET, TW_UP, TW_DOWN = 340.0, 2.5, 0.5
OCC_NORM = 1.0            # num_occupants arrives as a head count (checked against reward_components.total_occupancy)


class OnlineBuildingModel:
    def __init__(self, meta, model, params, theta, q_air=0.05, q_hidden=0.01, r=0.01):
        self.model, self.p, self.th = model, dict(params), dict(theta)
        names = list(meta["observation_field_names"]); self.ix = {n: i for i, n in enumerate(names)}
        suf = "_zone_air_temperature_sensor"
        self.vavs = [n[: -len(suf)] for n in names if n.endswith(suf)]          # observation (VAV) order
        rooms = [f"room_{v.split('_')[1]}" for v in self.vavs]
        self.ahus = sorted(meta["ahu_zone_assignments"], key=natural_key)        # positional order of the action lists
        self.geo = geometry(meta["floor_plan"], meta["zone_map"], rooms, meta["ahu_zone_assignments"], ahu_order=self.ahus)
        self.V, self.K = self.geo["V"], len(self.ahus)
        self.zt = np.array([self.ix[v + suf] for v in self.vavs])
        zn = meta["observation_normalization_metadata"]["zone_air_temperature_sensor"]
        self.zmu, self.zsd = zn["sample_mean"], zn["sample_std"]
        self.oat_ix = self.ix[f"ahs_{self.ahus[0]}_outside_air_temperature_sensor"]
        # action layout, the same convention the starter kit's baselines use
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
        # the network, built once from the floor plan (no fitting)
        one = self._blank_inputs()
        self.net, _ = model.network(self.geo, one, self.th, self.p)
        net, V, N = self.net, self.V, self.net.N
        self.Qn = np.diag(np.r_[np.full(V, q_air ** 2), np.full(N - V, q_hidden ** 2)]); self.r2 = r ** 2
        self.x = None; self.P = None; self.tw = None
        self.prev = None            # (obs, native continuous action, discrete action) of the previous call
        self.pending = None         # forcing of the step whose result the next observation shows
        self.log = dict(x_meas=[], x_filt=[], forcing=[])

    def _blank_inputs(self, n=1):
        V = self.V
        return dict(T_out=np.zeros(n), g=np.zeros((n, V)), T_supply=np.zeros((n, V)), Q_reheat=np.zeros((n, V)),
                    Q_gain=np.zeros((n, V)), occupants=np.zeros(n), comfort=np.zeros(n))

    def measured(self, obs):
        return np.asarray(obs, float)[self.zt] * self.zsd + self.zmu

    def _inputs(self, obs_k, c_native, d, x_k, run_now):
        """Room inputs of step k from obs_k (before the step), the action taken at k and x_k (seen one step later)."""
        area, room_ahu = self.geo["area"], self.geo["ahu"]
        oat = float(obs_k[self.oat_ix]); oat = oat if oat > 150 else oat * 10.0 + 295.0     # the harness passes kelvin
        on = run_now
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
        inp = self._blank_inputs()
        inp["T_out"][0] = oat; inp["g"][0] = g; inp["T_supply"][0] = Tsa; inp["Q_reheat"][0] = eff * g * gap
        inp["Q_gain"][0] = GAIN_W_M2 * area * (1.0 if comfort else STANDBY)
        inp["occupants"][0] = float(obs_k[self.ix["num_occupants"]]) * OCC_NORM; inp["comfort"][0] = comfort
        self.tw = self.tw + float(np.clip(c_native[self.water_ix] - self.tw, -TW_DOWN, TW_UP))
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
