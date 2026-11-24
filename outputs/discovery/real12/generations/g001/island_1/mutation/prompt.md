# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 1.  Operator: MUTATION.

### Parent: seed_cf_network  [g000_i1_0003, island 1, score 0.8174; objectives main 0.817, fast 0.544, slow 0.889, exterior 0.732, interior 0.814, passive 0.773, driven 0.582, complexity 772]
Hypothesis: Starting model: the audited room network, written as coefficient formulas of the fixed state-space function.
```python
"""Starting model: the audited room network, written as coefficient formulas of the fixed state-space function.

Per room: its air, a slab-and-furniture store and (rooms with an outside wall) an outside-wall store.  Every coefficient
is a floor-plan count x a material number, divided by a heat capacity and multiplied by the 300 s step.  The material
numbers are universal constants: the middle (geometric mean for log-drawn properties) of the kit generator's ranges
(smart_control/utils/materials.py); the outside film is the middle of the stations' range.  Shared walls carry heat
between rooms as two conductances in series; their heat storage is an inner-wall store per room (half of each wall).  Nothing is read from history; nothing is fitted.
"""
import numpy as np
import dlib

HISTORY = 1
REFRESH = 288                              # the coefficients do not depend on history, so one evaluation a day is enough

K_EXT = (0.15 * 0.55) ** 0.5               # W/mK   outside-wall conductivity, log-middle of 0.15-0.55
RC_EXT = (350 * 2400) ** 0.5 * 920.0       # J/m3K  outside-wall density (log-middle 350-2400) x heat capacity (middle 840-1000)
K_IW = 0.525                               # W/mK   shared-wall conductivity, middle of 0.45-0.60
RC_IW = (110 * 700) ** 0.5 * 900.0         # J/m3K  shared-wall density (log-middle 110-700) x fixed heat capacity 900
RC_MASS = (10 * 40) ** 0.5 * 1000.0        # J/m3K  slab/furniture density (log-middle 10-40) x 1000
H_FILM = 17.75                             # W/m2K  outside film, middle of the stations' 8.8-26.7
H_SLAB, A_SPREAD, K_AIR = 5.0, 0.6, 50.0   # slab-air coupling per m2, spreading-resistance strength, mixed-air conductivity


def _spread(area, length):
    """Spreading resistance (K/W): heat entering through an opening of this length spreads through the room's air."""
    return A_SPREAD * np.log1p(np.sqrt(area) / np.maximum(length, 1e-9)) / (np.pi * K_AIR * dlib.HEIGHT)


def coefficients(geo, hist, U):
    V, A, H, c, dt = geo["V"], geo["area"], U["HEIGHT"], U["CELL"], U["DT"]
    face = H * c
    C_air = U["RHO_C_AIR"] * H * A                                      # J/K
    C_slab = RC_MASS * H * A
    G_slab = H_SLAB * A                                                  # W/K
    nf = geo["facade_faces"]; has = nf > 0
    g_fac = nf * face / (c / (2 * K_AIR) + c / (4 * K_EXT))
    G_wall = np.where(has, 1 / (1 / np.maximum(g_fac, 1e-9) + _spread(A, np.maximum(nf, 1) * c)), 0.0)
    C_wall = np.where(has, RC_EXT * nf * face * c / 2, 1.0)
    G_wall_out = nf * face / (c / (4 * K_EXT) + 1 / H_FILM)
    n = geo["wall_cells"]                                                # shared-wall faces between rooms i and j
    g_half = n * face / (c / (2 * K_AIR) + c / (2 * K_IW))
    gi = 1 / (1 / np.maximum(g_half, 1e-12) + _spread(A[:, None], np.maximum(n, 1) * c))
    gi = np.where(n > 0, gi, 0.0)
    G_wall_ij = np.where(n > 0, 1 / (1 / np.maximum(gi, 1e-12) + 1 / np.maximum(gi.T, 1e-12)), 0.0)   # air -> wall -> air
    G_inner = gi.sum(1)                                                  # room air -> its half of every shared wall
    C_inner = np.maximum(RC_IW * n.sum(1) * face * c / 2, 1.0)            # half of every shared wall's heat capacity
    d = geo["door_cells"]
    g_door = d * face / (2 * c / K_AIR)
    G_door = np.where(d > 0, 1 / (1 / np.maximum(g_door, 1e-12) + _spread(A[:, None], np.maximum(d, 1) * c)
                                  + _spread(A[None, :], np.maximum(d, 1) * c)), 0.0)
    return {"air": dt / C_air, "heat": dt / C_air,
            "neighbours": dt * (G_wall_ij + G_door) / C_air[:, None],
            "slab": dt * G_slab / C_air, "slab_back": dt * G_slab / C_slab,
            "wall": dt * G_wall / C_air, "wall_back": dt * G_wall / C_wall, "wall_out": np.where(has, dt * G_wall_out / C_wall, 0.0),
            "inner": dt * G_inner / C_air, "inner_back": dt * G_inner / C_inner}
```


Pre-mutation analysis of the parent (validation buildings): where, when and through which term it errs
OBJECTIVES (validation, lower is better): main 0.817, fast 0.544, slow 0.889, exterior 0.732, interior 0.814, passive 0.773, driven 0.582, complexity 772

TEMPORAL
  error vs horizon (rmse / bias / persistence rmse, K):
     0.08 h   0.247 / -0.097 / 0.527
     0.50 h   0.522 / -0.237 / 0.578
     1.00 h   0.582 / -0.271 / 0.764
     2.00 h   0.673 / -0.321 / 0.784
     3.00 h   0.757 / -0.368 / 0.931
     4.00 h   0.836 / -0.413 / 1.120
     6.00 h   0.980 / -0.498 / 1.358
    -> half of the long-horizon error is already there after 1 h: the fast terms (air, heat, neighbours) are off
  one-step error autocorrelation at 1 h lag: +0.82 (persistent: a slow state or a slowly varying input is missing or mis-scaled)
  3h rollout error by time (rmse, bias, K):
    target hour 00-04                            0.756  -0.427
    target hour 04-08                            0.870  -0.578
    target hour 08-12                            0.842  -0.556
    target hour 12-16                            0.715  -0.346
    target hour 16-20                            0.649  -0.136
    target hour 20-24                            0.655  -0.224
    occupied (comfort) hours                     0.727  -0.312
    nights and weekends                          0.767  -0.416
    HVAC air on at the target                    0.582  -0.235
    HVAC air off at the target                   0.773  -0.396
    HVAC switched off during the rollout         0.752  -0.340
    HVAC switched on during the rollout          0.651  -0.243
    colder half of outside temperatures          0.852  -0.514
    warmer half of outside temperatures          0.644  -0.247
    first 3 days of the episode                  0.624  -0.262
    after day 3                                  0.763  -0.388

SPATIAL
  3h rollout error by place (rmse, bias, K):
    exterior rooms                               0.732  -0.324
    interior rooms                               0.814  -0.555
    corner rooms (outside wall on 2+ sides)      0.758  -0.320
    outside wall on the top side of the plan     0.704  -0.263
    outside wall on the bottom side of the plan  0.732  -0.290
    outside wall on the left side of the plan    0.739  -0.327
    outside wall on the right side of the plan   0.728  -0.326
    floor 1                                      0.751  -0.348
    floor 2                                      0.765  -0.388
    floor 3                                      0.815  -0.495
    floor 4                                      0.550  -0.176
  correlation of a room's bias with its geometry (over all validation rooms):
    floor area                                   -0.47   <- the formula's dependence on this is wrong
    outside-wall length per floor area           +0.25
    number of neighbours                         -0.24
    shared-wall length per floor area            +0.16
    door width per floor area                    -0.11
  error series correlation, adjacent rooms +0.87 vs non-adjacent +0.84 (similar: a building-wide cause, e.g. supply air or outside term)
  worst rooms: episode 2 room #42 (rmse 1.72, bias -1.65, outside-wall faces 0, area 116 m2, doors 4); episode 1 room #36 (rmse 1.64, bias -1.56, outside-wall faces 308, area 195 m2, doors 1); episode 1 room #49 (rmse 1.62, bias -1.54, outside-wall faces 155, area 135 m2, doors 1); episode 2 room #53 (rmse 1.56, bias -1.48, outside-wall faces 0, area 174 m2, doors 4)

TERMS (one-step residual regressed on each term's contribution; slope s means the term should be about (1+s) times as large)
  residual rms 0.2411 K per step; the terms explain 58% of it
    air         size 0.3168 K/step   slope +0.015 +- 0.002   -> consistent with its current size
    neighbours  size 0.2230 K/step   slope -0.742 +- 0.004   -> too strong by about 74%
    slab        size 0.1361 K/step   slope +0.418 +- 0.011   -> too weak by about 42%
    wall        size 0.4035 K/step   slope -0.450 +- 0.002   -> too strong by about 45%
    inner       size 0.0826 K/step   slope -0.722 +- 0.019   -> too strong by about 72%
    heat        size 0.4642 K/step   slope -0.001 +- 0.002   -> consistent with its current size
  exterior rooms only: wall -0.447 (too strong by about 45%)
  (terms that move together, e.g. outside and wall, share the residual: read their slopes together)

complexity 772 AST nodes, history 1 steps, horizons 1h, 3h, 6h

Measured vs predicted (the parent's 3 h-ahead forecast) for two rooms:
a held-out episode; room #15 = most exposed exterior room, room #37 = largest interior room
time  | room #15 measured / predicted | room #37 measured / predicted | T_out | HVAC air
   0h |   9.45 /   8.48 C |  20.55 /  19.49 C |  -1.7 C | on
   2h |  21.96 /  20.64 C |  21.66 /  21.21 C |   1.1 C | on
   4h |  17.37 /  15.72 C |  22.35 /  21.86 C |  12.2 C | off
   6h |  21.90 /  19.75 C |  22.94 /  22.46 C |  15.6 C | off
   8h |  19.19 /  17.79 C |  23.45 /  23.07 C |  16.1 C | off
  10h |  19.62 /  18.47 C |  23.95 /  23.70 C |  16.1 C | off
  12h |  19.58 /  18.75 C |  24.45 /  24.28 C |   9.9 C | off
  14h |  17.50 /  17.12 C |  23.79 /  23.61 C |   4.5 C | off
  16h |  15.87 /  15.53 C |  23.72 /  23.31 C |   2.7 C | off
  18h |  14.42 /  13.94 C |  23.62 /  22.98 C |   2.2 C | off
  20h |  13.27 /  12.62 C |  23.51 /  22.71 C |   1.8 C | off
  22h |  12.34 /  11.56 C |  23.38 /  22.46 C |   1.8 C | off

Write 2 children of this parent.  Each must change ONE thing that the analysis points to (one coefficient formula, a different history summary, a history-inferred material value) and keep the rest. Use the TERMS slopes, the SPATIAL correlations and the TEMPORAL breakdown to decide which coefficient to change and in which direction, and say in the hypothesis which diagnostic line motivates the change and which objective it should improve.
