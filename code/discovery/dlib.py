"""Helper library that candidate programs may import (`import dlib`).

Deliberately generic: it solves whatever linear RC network a candidate builds and
offers a few numerical conveniences.  It encodes no knowledge of this simulator,
so the structure a candidate uses is the candidate's own.
"""
import numpy as np

DT = 300.0                 # s, one control step
CELL = 0.2                 # m, grid cell of floor_plan / zone_map
HEIGHT = 3.0               # m, floor-to-ceiling
RHO_C_AIR = 1.2 * 1005.0   # J/m3K, room air


class LinearNetwork:
    """A thermal RC network stepped by backward Euler, exactly as stiff building
    models need (the air stream can turn a room's air over faster than one step).

    Nodes 0..V-1 must be the room-air nodes (the outputs).  Any further nodes are
    hidden (slab, walls, ...).  Each step every air node i receives:
        heat Q[i]  (W)  and  an air stream g[i] * (T_supply[i] - T_i)  treated implicitly.

    C     (N,)    node heat capacities, J/K, > 0
    G     (N, N)  symmetric non-negative conductances between nodes, W/K (diagonal ignored)
    G_amb (N,)    conductance from each node to the outside temperature, W/K
    """

    def __init__(self, C, G, G_amb, V, dt=DT):
        C = np.asarray(C, float); G = np.asarray(G, float); G_amb = np.asarray(G_amb, float)
        N = len(C)
        if G.shape != (N, N) or G_amb.shape != (N,):
            raise ValueError(f"shapes: C {C.shape}, G {G.shape}, G_amb {G_amb.shape}")
        if np.any(C <= 0) or np.any(G < 0) or np.any(G_amb < 0) or not np.all(np.isfinite(G)):
            raise ValueError("need C > 0, G >= 0, G_amb >= 0, all finite")
        G = 0.5 * (G + G.T); np.fill_diagonal(G, 0.0)
        L = np.diag(G.sum(1) + G_amb) - G
        self.V, self.N, self.dt = V, N, dt
        self.Cdt = C / dt
        A0inv = np.linalg.inv(np.diag(self.Cdt) + L)
        self.A0inv, self.W, self.S, self.G_amb = A0inv, A0inv[:, :V], A0inv[:V, :V], G_amb

    def simulate(self, T_out, g, T_supply, Q, T0, x0=None):
        """Roll out from a uniform temperature T0 (or state x0).
        T_out (n), g (n,V), T_supply (n,V), Q (n,V)  ->  room-air temperatures (n,V) after each step."""
        n, V = len(T_out), self.V
        x = np.full(self.N, float(T0)) if x0 is None else np.array(x0, float)
        out = np.empty((n, V)); I = np.eye(V)
        for t in range(n):
            rhs = self.Cdt * x + self.G_amb * T_out[t]
            rhs[:V] += Q[t] + g[t] * T_supply[t]
            y = self.A0inv @ rhs
            q = np.linalg.solve(I + g[t][:, None] * self.S, g[t] * y[:V])
            x = y - self.W @ q
            out[t] = x[:V]
        self.last_state = x
        return out


def end_of_step(T, a, T_supply, b):
    """The assembled room equation for closed-form candidates (air-side heat at the end of the step):
        T_next = (T + a * T_supply + b) / (1 + a)
    a = g * DT / C_air (share of supply air over the step, >= 0);  b = DT / C_air * (all other heat, W), in K.
    Implicit in the supply air, so it stays stable for any airflow; works on any leading batch shape."""
    a = np.maximum(a, 0.0)
    return (T + a * T_supply + b) / (1.0 + a)


def assemble(T, terms, b=0.0):
    """General end-of-step room equation: every exchange term pulls the room toward a source temperature.
        T_next = (T + sum_k w_k * T_k + b) / (1 + sum_k w_k)
    terms: list of (w_k, T_k) with w_k = DT * conductance_k / C (>= 0), e.g. (a, T_supply), (d, T_neighbours),
    (e, T_out), (s, T_slow_store);  b = DT / C * (direct heat, W).  A convex mix plus b: stable for any weights."""
    num = T + b
    den = 1.0
    for w, src in terms:
        w = np.maximum(w, 0.0)
        num = num + w * src
        den = den + w
    return num / den


def network_predict(net, x, T_out, g, T_supply, Q):
    """One backward-Euler step of a LinearNetwork for a batch: x (B, N); T_out (B,); g, T_supply, Q (B, V).
    The air stream is implicit, exactly as LinearNetwork.simulate.  Returns the new state (B, N)."""
    x = np.atleast_2d(np.asarray(x, float)); V = net.V
    rhs = x * net.Cdt[None, :] + np.asarray(T_out, float).reshape(-1, 1) * net.G_amb[None, :]
    rhs[:, :V] += Q + g * T_supply
    y = rhs @ net.A0inv.T
    M = np.eye(V)[None] + g[:, :, None] * net.S[None]
    q = np.linalg.solve(M, (g * y[:, :V])[..., None])[..., 0]
    return y - q @ net.W.T


def correct_rooms(x, T_measured, V):
    """Fixed measurement correction for stateful candidates: room-air nodes take the measured temperatures,
    hidden nodes (slab, walls, ...) keep their predicted values and so follow the measured air over time."""
    x = np.array(x, float, copy=True)
    x[..., :V] = T_measured
    return x


def pairs(M):
    """Upper-triangle (i, j, value) of a symmetric pair matrix where value > 0."""
    i, j = np.nonzero(np.triu(M, 1) > 0)
    return i, j, M[i, j]


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def filtered_forecast(net, T_out, g, T_supply, Q, T_meas, T0, horizons=(12, 36, 72), every=12, start=288,
                      q_air=0.05, q_hidden=0.01, r=0.01, max_steps=None):
    """k-step forecast error of a LinearNetwork from a Kalman-filtered state.

    The filter uses only the measured room temperatures (T_meas[k] = x_{k+1}).  From every
    `every`-th step after `start` the filtered state is rolled forward open loop with the
    actual future inputs, and compared with the measurement `h` steps later.
    Returns {h: (model_rmse, persistence_rmse)} in K.
    """
    V, N = net.V, net.N
    n = len(T_meas) if max_steps is None else min(len(T_meas), max_steps)
    I = np.eye(V)

    def Minv(gt):
        K = np.linalg.solve(I + gt[:, None] * net.S, gt[:, None] * net.W.T)
        return net.A0inv - net.W @ K

    def step(x, t, Mi=None):
        Mi = Minv(g[t]) if Mi is None else Mi
        rhs = net.Cdt * x + net.G_amb * T_out[t]
        rhs[:V] += Q[t] + g[t] * T_supply[t]
        return Mi @ rhs, Mi

    Qn = np.r_[np.full(V, q_air ** 2), np.full(N - V, q_hidden ** 2)]
    x = np.full(N, float(T0)); P = np.diag(np.full(N, 1e-6)); filt = np.zeros((n, N))
    for t in range(n):
        x, Mi = step(x, t)
        A = Mi * net.Cdt[None, :]
        P = A @ P @ A.T + np.diag(Qn)
        Kg = np.linalg.solve(P[:V, :V] + r ** 2 * I, P[:V, :]).T
        x = x + Kg @ (T_meas[t] - x[:V]); P = P - Kg @ P[:V, :]
        filt[t] = x
    errs = {h: [] for h in horizons}; pers = {h: [] for h in horizons}
    for t0 in range(start, n - max(horizons), every):
        xx = filt[t0].copy()
        for k in range(1, max(horizons) + 1):
            xx, _ = step(xx, t0 + k)
            if k in horizons:
                errs[k].append(xx[:V] - T_meas[t0 + k]); pers[k].append(T_meas[t0] - T_meas[t0 + k])
    return {h: (float(np.sqrt(np.mean(np.square(errs[h])))) if errs[h] else float("nan"),
                float(np.sqrt(np.mean(np.square(pers[h])))) if pers[h] else float("nan")) for h in horizons}
