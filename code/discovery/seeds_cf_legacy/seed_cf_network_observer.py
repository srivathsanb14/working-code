"""Stateful seed: the audited room network (air, slab, facade, partitions, doors) with a measurement-corrected state.

Nothing is fitted.  Every material number is a universal constant: the middle (geometric mean for log-drawn
properties) of the starter-kit generator's range (smart_control/utils/materials.py), and the outdoor film is the
middle of the stations' range.  The state is corrected every step by the measured room temperatures (room-air
nodes take the measurement; slab, facade and partition nodes follow), so the slow stores are estimated from
history instead of being fitted.  Stateful contract: init_state / predict / correct (see prompts.CF_CONTRACT).
"""
import numpy as np
import dlib

K_EXT = (0.15 * 0.55) ** 0.5               # W/mK   facade conductivity, log-middle of 0.15-0.55
RC_EXT = (350 * 2400) ** 0.5 * 920.0       # J/m3K  facade density (log-middle 350-2400) x heat capacity (middle 840-1000)
K_IW = 0.525                               # W/mK   partition conductivity, middle of 0.45-0.60
RC_IW = (110 * 700) ** 0.5 * 900.0         # J/m3K  partition density (log-middle) x fixed heat capacity 900
RC_MASS = (10 * 40) ** 0.5 * 1000.0        # J/m3K  slab/furniture density (log-middle 10-40) x 1000
H_FILM = 17.75                             # W/m2K  outdoor film, middle of the stations' 8.8-26.7
H_SLAB, A_SPREAD, K_AIR = 5.0, 0.6, 50.0   # slab-air coupling per m2, spreading-resistance strength, mixed-air conductivity


def _spread(area, length):
    return A_SPREAD * np.log1p(np.sqrt(area) / np.maximum(length, 1e-9)) / (np.pi * K_AIR * dlib.HEIGHT)


def prepare(geo, U):
    V = geo["V"]; A = geo["area"]; H = U["HEIGHT"]; c = U["CELL"]; face = H * c
    wi, wj, wn = dlib.pairs(geo["wall_cells"]); di, dj, dn = dlib.pairs(geo["door_cells"])
    N = 3 * V + len(wi)
    C = np.empty(N); G = np.zeros((N, N)); G_amb = np.zeros(N)
    air, slab, fac = np.arange(V), V + np.arange(V), 2 * V + np.arange(V)
    C[air] = U["RHO_C_AIR"] * H * A
    C[slab] = RC_MASS * H * A
    G[air, slab] = G[slab, air] = H_SLAB * A
    nf = geo["facade_faces"]; has_f = nf > 0
    C[fac] = np.where(has_f, RC_EXT * nf * face * c / 2, 1.0)
    g_in = nf * face / (c / (2 * K_AIR) + c / (4 * K_EXT))
    G[air, fac] = G[fac, air] = np.where(has_f, 1 / (1 / np.maximum(g_in, 1e-9) + _spread(A, nf * c)), 0.0)
    G_amb[fac] = np.where(has_f, nf * face / (c / (4 * K_EXT) + 1 / H_FILM), 0.0)
    for k in range(len(wi)):
        i, j, n = wi[k], wj[k], wn[k]; w = 3 * V + k
        C[w] = RC_IW * n * face * c
        g_half = n * face / (c / (2 * K_AIR) + c / (2 * K_IW))
        G[i, w] = G[w, i] = 1 / (1 / g_half + _spread(A[i], n * c))
        G[j, w] = G[w, j] = 1 / (1 / g_half + _spread(A[j], n * c))
    for i, j, n in zip(di, dj, dn):
        g_door = n * face / (2 * c / K_AIR)
        G[i, j] = G[j, i] = G[i, j] + 1 / (1 / g_door + _spread(A[i], n * c) + _spread(A[j], n * c))
    return dict(net=dlib.LinearNetwork(C, G, G_amb, V), V=V)


def init_state(coef, geo, U):
    return np.full(coef["net"].N, U["T0"])          # every cell starts at 294 K


def predict(coef, geo, state, u, U):
    Q = u["Q_reheat"] + u["Q_gain"] + U["OCC_W"] * np.asarray(u["occupants"])[:, None] / geo["V"]
    return dlib.network_predict(coef["net"], state, u["T_out"], u["g"], u["T_supply"], Q)


def correct(coef, geo, state, T_measured, U):
    return dlib.correct_rooms(state, T_measured, coef["V"])
