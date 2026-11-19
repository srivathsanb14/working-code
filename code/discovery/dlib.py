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


def pairs(M):
    """Upper-triangle (i, j, value) of a symmetric pair matrix where value > 0."""
    i, j = np.nonzero(np.triu(M, 1) > 0)
    return i, j, M[i, j]


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
