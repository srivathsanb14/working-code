# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 1.  Operator: MUTATION.

### Parent: seed_cf_network_observer  [g000_i1_0003, island 1, score 2.4331]
Hypothesis: Stateful seed: the audited room network (air, slab, facade, partitions, doors) with a measurement-corrected state.
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
```


Pre-mutation analysis of the parent (held-out buildings):
held-out rollout RMSE by horizon (model / persistence), K: 1h 1.961/1.464, 3h 2.268/3.005, 6h 2.512/4.722
one-step (5 min) RMSE 0.8822 K (persistence 0.2169); train buildings 3h 2.016 K
3h rollout error by group (rmse, bias), K:
  all rooms                              2.284  +1.167
  exterior rooms                         2.480  +1.301
  interior rooms                         1.603  +0.788
  most exposed third of exterior rooms   2.773  +1.482
  HVAC air on                            0.810  -0.228
  HVAC air off                           3.361  +3.026
  comfort hours                          0.810  -0.228
  night/weekend                          3.361  +3.026
  room being reheated                    0.746  +0.117
  room not reheated                      2.520  +1.423
  colder half of outside temps           3.324  +2.927
  warmer half                            0.833  -0.197
skill vs persistence by horizon (ratio <1 is better): 1h 1.34, 3h 0.75, 6h 0.53
worst rooms: held-out episode 1 room #6 (rmse 3.15, bias +0.78, facade faces 74, area 35 m2, doors 0); held-out episode 1 room #13 (rmse 3.04, bias +1.73, facade faces 72, area 44 m2, doors 0); held-out episode 2 room #6 (rmse 3.00, bias +1.66, facade faces 74, area 35 m2, doors 0); held-out episode 4 room #17 (rmse 2.98, bias +1.66, facade faces 135, area 77 m2, doors 1)
complexity 990 AST nodes, history 1 steps

Measured vs predicted (the parent's 3 h-ahead forecast) for two rooms:
a held-out episode; room #57 = most exposed exterior room, room #10 = largest interior room
time  | room #57 measured / predicted | room #10 measured / predicted | T_out | HVAC air
   0h |   5.02 /  10.23 C |  14.00 /  15.64 C |  -2.0 C | off
   2h |   2.91 /   8.44 C |  12.74 /  14.96 C |  -3.9 C | off
   4h |   1.70 /   6.69 C |  11.39 /  14.02 C |  -4.0 C | off
   6h |   1.68 /   5.27 C |  10.12 /  12.90 C |  -2.2 C | off
   8h |   9.05 /   8.48 C |  13.73 /  14.26 C |   1.0 C | on
  10h |  12.64 /  11.55 C |  14.71 /  14.42 C |   4.6 C | on
  12h |  13.26 /  12.08 C |  15.85 /  15.05 C |   7.7 C | on
  14h |  13.96 /  13.06 C |  16.21 /  15.44 C |   9.5 C | on
  16h |  13.74 /  13.28 C |  16.70 /  16.00 C |   9.4 C | on
  18h |  13.52 /  13.76 C |  16.91 /  16.38 C |   7.7 C | on
  20h |  10.26 /  12.58 C |  16.11 /  16.01 C |   4.5 C | off
  22h |   7.57 /  11.56 C |  15.34 /  15.93 C |   0.8 C | off

Write 2 children of this parent.  Each must change ONE thing that the analysis points to (a template's formula, a new template, a different history summary, a history-inferred material value) and keep the rest. Say in the hypothesis which diagnostic line motivates the change.
