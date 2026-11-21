"""Stateful seed: the audited room network (air, slab, facade, partitions, doors) with a measurement-corrected state.

Nothing is fitted.  Every material number is a universal constant: the middle (geometric mean for log-drawn
properties) of the starter-kit generator's range (smart_control/utils/materials.py), and the outdoor film is the
middle of the stations' range.  The state is corrected every step by the measured room temperatures (room-air
nodes take the measurement; slab, facade and partition nodes follow), so the slow stores are estimated from
history instead of being fitted.  Stateful contract: init_state / predict / correct (see prompts.CF_CONTRACT).
"""
import numpy as np
import dlib
K_EXT = (0.15 * 0.55) ** 0.5
RC_EXT = (350 * 2400) ** 0.5 * 920.0
K_IW = 0.525
RC_IW = (110 * 700) ** 0.5 * 900.0
RC_MASS = (10 * 40) ** 0.5 * 1000.0
H_FILM = 17.75
H_SLAB, A_SPREAD, K_AIR = (5.0, 0.6, 50.0)

def _spread(area, length):
    return A_SPREAD * np.log1p(np.sqrt(area) / np.maximum(length, 1e-09)) / (np.pi * K_AIR * dlib.HEIGHT)

def prepare(geo, U):
    V = geo['V']
    A = geo['area']
    H = U['HEIGHT']
    c = U['CELL']
    face = H * c
    wi, wj, wn = dlib.pairs(geo['wall_cells'])
    di, dj, dn = dlib.pairs(geo['door_cells'])
    N = 3 * V + len(wi)
    C = np.empty(N)
    G = np.zeros((N, N))
    G_amb = np.zeros(N)
    air, slab, fac = (np.arange(V), V + np.arange(V), 2 * V + np.arange(V))
    C[air] = U['RHO_C_AIR'] * H * A
    C[slab] = RC_MASS * H * A
    G[air, slab] = G[slab, air] = H_SLAB * A
    nf = geo['facade_faces']
    has_f = nf > 0
    C[fac] = np.where(has_f, RC_EXT * nf * face * c / 2, 1.0)
    g_in = nf * face / (c / (2 * K_AIR) + c / (4 * K_EXT))
    G[air, fac] = G[fac, air] = np.where(has_f, 1 / (1 / np.maximum(g_in, 1e-09) + _spread(A, nf * c)), 0.0)
    G_amb[fac] = np.where(has_f, nf * face / (c / (4 * K_EXT) + 1 / H_FILM), 0.0)
    for k in range(len(wi)):
        i, j, n = (wi[k], wj[k], wn[k])
        w = 3 * V + k
        C[w] = RC_IW * n * face * c
        g_half = n * face / (c / (2 * K_AIR) + c / (2 * K_IW))
        G[i, w] = G[w, i] = 1 / (1 / g_half + _spread(A[i], n * c))
        G[j, w] = G[w, j] = 1 / (1 / g_half + _spread(A[j], n * c))
    for i, j, n in zip(di, dj, dn):
        g_door = n * face / (2 * c / K_AIR)
        G[i, j] = G[j, i] = G[i, j] + 1 / (1 / g_door + _spread(A[i], n * c) + _spread(A[j], n * c))
    return dict(net=dlib.LinearNetwork(C, G, G_amb, V), V=V)

def init_state(coef, geo, U):
    return np.full(coef['net'].N, U['T0'])

def predict(coef, geo, state, u, U):
    Q = u['Q_reheat'] + u['Q_gain'] + U['OCC_W'] * np.asarray(u['occupants'])[:, None] / geo['V']
    return dlib.network_predict(coef['net'], state, u['T_out'], u['g'], u['T_supply'], Q)

def correct(coef, geo, state, T_measured, U):
    return dlib.correct_rooms(state, T_measured, coef['V'])
