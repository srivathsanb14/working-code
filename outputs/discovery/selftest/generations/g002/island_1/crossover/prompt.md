# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 2, island 1.  Operator: CROSSOVER.

### Parent A: mock_mut_g1_0  [g001_i1_0010, island 1, score 2.4331]
Hypothesis: [mock mutation, no LLM] rescaled 1 -> 1.25 in seed_cf_network_observer
```python
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
    C[fac] = np.where(has_f, RC_EXT * nf * face * c / 2, 1.25)
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
```


### Parent B: mock_exp_g1_0  [g001_i1_0014, island 1, score 2.2714]
Hypothesis: [mock explore, no LLM] rescaled 900 -> 540, 0.6 -> 0.36, 50 -> 30, 17.75 -> 22.1875 in seed_cf_network_observer
```python
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
RC_IW = (110 * 700) ** 0.5 * 540.0
RC_MASS = (10 * 40) ** 0.5 * 1000.0
H_FILM = 22.1875
H_SLAB, A_SPREAD, K_AIR = (5.0, 0.36, 30.0)

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
```


Pre-crossover analysis:
Where each parent is better (A vs B, main-horizon rollout RMSE, K):
  all rooms                              2.284 vs 2.111  -> B
  exterior rooms                         2.480 vs 2.290  -> B
  interior rooms                         1.603 vs 1.488  -> B
  most exposed third of exterior rooms   2.773 vs 2.567  -> B
  HVAC air on                            0.810 vs 0.727  -> B
  HVAC air off                           3.361 vs 3.113  -> B
  comfort hours                          0.810 vs 0.727  -> B
  night/weekend                          3.361 vs 3.113  -> B
  room being reheated                    0.746 vs 0.673  -> B
  room not reheated                      2.520 vs 2.331  -> B
  colder half of outside temps           3.324 vs 3.077  -> B
  warmer half                            0.833 vs 0.757  -> B
B is clearly better on: all rooms, exterior rooms, interior rooms, most exposed third of exterior rooms, HVAC air on, HVAC air off, comfort hours, night/weekend, room being reheated, room not reheated, colder half of outside temps, warmer half
templates (top-level functions): same in both: _spread, init_state, predict, correct; different: prepare; only in A: none; only in B: none

Recent failures on this island (do not repeat):
- mock_mut_g1_1 [screened]: one-step RMSE 1.0871 K is more than 5.0x persistence (0.2169 K) on held-out buildings
- mock_cro_g1_0 [error]: NameError: name 'RC_MASS' is not defined

Write 2 children that COMBINE the parents: take each template from the parent that is better where that template matters (e.g. the exterior-wall template from the parent better on exterior rooms), make the pieces consistent, and state in the hypothesis which template came from which parent.
