"""Room air, floor slab, facade wall and partition walls as separate heat stores.

Per room: an air node (air capacity only), a slab node coupled to the air in
proportion to floor area, and a facade node between the air and the outside
(inner and outer conductances from a wall conductivity and a surface film).  Each
shared wall is a node between its two rooms.  Doors couple rooms directly.  Heat
entering a room through a short boundary segment must spread through the room's
air, modelled as a series resistance growing with log(sqrt(area)/length).
This is the structure found by auditing the simulator code (STATE_SPACE.md §14),
written only in terms of the geometry a policy can compute.
"""
import numpy as np
import dlib

PARAMS = {
    "h_slab": (5.0, 0.5, 50.0),        # W/m2K, slab-to-air coupling per m2 of floor
    "a_spread": (0.6, 0.01, 5.0),      # strength of the spreading resistance
}
THETA = {
    "k_ext": (0.3, 0.1, 1.0),          # W/mK, facade wall conductivity
    "h_film": (15.0, 5.0, 40.0),       # W/m2K, outside surface film
    "rc_ext": (1.0e6, 2.0e5, 3.0e6),   # J/m3K, facade volumetric heat capacity
    "k_iw": (0.5, 0.2, 1.5),           # W/mK, partition conductivity
    "rc_iw": (3.0e5, 5.0e4, 8.0e5),    # J/m3K, partition volumetric heat capacity
    "rc_mass": (2.0e4, 5.0e3, 6.0e4),  # J/m3K, slab heat capacity per m3 of room
}
K_AIR = 50.0                           # W/mK, effective conductivity of mixed room air


def spread(area, length, a):
    return a * np.log1p(np.sqrt(area) / np.maximum(length, 1e-9)) / (np.pi * K_AIR * dlib.HEIGHT)


def network(geo, inputs, theta, p):
    """The RC network and its per-step forcing (lets test_model.py run a Kalman filter)."""
    V = geo["V"]; A = geo["area"]; H = dlib.HEIGHT; c = dlib.CELL
    face = H * c                                                     # m2 of one 0.2 m segment
    wi, wj, wn = dlib.pairs(geo["wall_cells"])
    di, dj, dn = dlib.pairs(geo["door_cells"])
    nW = len(wi)
    N = 3 * V + nW                                                   # air, slab, facade, partitions
    C = np.empty(N); G = np.zeros((N, N)); G_amb = np.zeros(N)
    air, slab, fac = np.arange(V), V + np.arange(V), 2 * V + np.arange(V)
    C[air] = dlib.RHO_C_AIR * H * A
    C[slab] = theta["rc_mass"] * H * A
    G[air, slab] = G[slab, air] = p["h_slab"] * A
    nf = geo["facade_faces"]
    has_f = nf > 0
    C[fac] = np.where(has_f, theta["rc_ext"] * nf * face * c / 2, 1.0)
    g_in = nf * face / (c / (2 * K_AIR) + c / (4 * theta["k_ext"]))
    g_in = np.where(has_f, 1 / (1 / np.maximum(g_in, 1e-9) + spread(A, nf * c, p["a_spread"])), 0.0)
    G[air, fac] = G[fac, air] = g_in
    G_amb[fac] = np.where(has_f, nf * face / (c / (4 * theta["k_ext"]) + 1 / theta["h_film"]), 0.0)
    for k in range(nW):
        i, j, n = wi[k], wj[k], wn[k]; w = 3 * V + k
        C[w] = theta["rc_iw"] * n * face * c
        g_half = n * face / (c / (2 * K_AIR) + c / (2 * theta["k_iw"]))
        G[i, w] = G[w, i] = 1 / (1 / g_half + spread(A[i], n * c, p["a_spread"]))
        G[j, w] = G[w, j] = 1 / (1 / g_half + spread(A[j], n * c, p["a_spread"]))
    for i, j, n in zip(di, dj, dn):
        g_door = n * face / (2 * c / K_AIR)
        G[i, j] = G[j, i] = G[i, j] + 1 / (1 / g_door + spread(A[i], n * c, p["a_spread"]) + spread(A[j], n * c, p["a_spread"]))
    net = dlib.LinearNetwork(C, G, G_amb, V)
    Q = inputs["Q_reheat"] + inputs["Q_gain"] + 70.0 * inputs["occupants"][:, None] / V
    return net, dict(T_out=inputs["T_out"], g=inputs["g"], T_supply=inputs["T_supply"], Q=Q)


def simulate(geo, inputs, theta, p, T0):
    net, f = network(geo, inputs, theta, p)
    return net.simulate(f["T_out"], f["g"], f["T_supply"], f["Q"], T0)
