# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 1.  Operator: MUTATION.

### Parent: seed_cf_network  [g000_i1_0003, island 1, score 2.2602; objectives main 2.183, fast 1.807, slow 2.384, exterior 2.339, interior 1.253, passive 3.050, driven 0.814, complexity 772]
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
OBJECTIVES (validation, lower is better): main 2.183, fast 1.807, slow 2.384, exterior 2.339, interior 1.253, passive 3.050, driven 0.814, complexity 772

TEMPORAL
  error vs horizon (rmse / bias / persistence rmse, K):
     0.08 h   0.774 / +0.335 / 0.565
     0.50 h   1.651 / +0.777 / 1.116
     1.00 h   1.827 / +0.882 / 1.330
     2.00 h   2.013 / +0.973 / 2.088
     3.00 h   2.149 / +1.046 / 2.774
     4.00 h   2.242 / +1.105 / 3.394
     6.00 h   2.408 / +1.236 / 4.399
    -> half of the long-horizon error is already there after 1 h: the fast terms (air, heat, neighbours) are off
  one-step error autocorrelation at 1 h lag: +0.92 (persistent: a slow state or a slowly varying input is missing or mis-scaled)
  3h rollout error by time (rmse, bias, K):
    target hour 00-04                            3.544  +3.314
    target hour 04-08                            3.158  +2.707
    target hour 08-12                            0.817  -0.158
    target hour 12-16                            0.852  -0.576
    target hour 16-20                            0.600  +0.025
    target hour 20-24                            2.340  +2.012
    occupied (comfort) hours                     0.814  -0.214
    nights and weekends                          3.050  +2.651
    HVAC air on at the target                    0.814  -0.214
    HVAC air off at the target                   3.050  +2.651
    HVAC switched off during the rollout         1.153  +0.896
    HVAC switched on during the rollout          1.122  +0.694
    colder half of outside temperatures          3.012  +2.562
    warmer half of outside temperatures          0.814  -0.204
    first 3 days of the episode                  2.089  +1.014

SPATIAL
  3h rollout error by place (rmse, bias, K):
    exterior rooms                               2.339  +1.186
    interior rooms                               1.253  +0.580
    corner rooms (outside wall on 2+ sides)      2.620  +1.325
    outside wall on the top side of the plan     2.429  +1.229
    outside wall on the bottom side of the plan  2.423  +1.221
    outside wall on the left side of the plan    2.366  +1.205
    outside wall on the right side of the plan   2.345  +1.194
    floor 1                                      2.065  +1.006
    floor 2                                      1.995  +0.951
    floor 3                                      2.243  +1.111
    floor 4                                      2.336  +1.170
  correlation of a room's bias with its geometry (over all validation rooms):
    outside-wall length per floor area           +0.80   <- the formula's dependence on this is wrong
    number of neighbours                         -0.68   <- the formula's dependence on this is wrong
    shared-wall length per floor area            -0.49   <- the formula's dependence on this is wrong
    door width per floor area                    -0.34
    floor area                                   -0.16
  error series correlation, adjacent rooms +0.96 vs non-adjacent +0.93 (similar: a building-wide cause, e.g. supply air or outside term)
  worst rooms: episode 6 room #22 (rmse 3.25, bias +1.70, outside-wall faces 181, area 63 m2, doors 1); episode 5 room #22 (rmse 3.22, bias +0.86, outside-wall faces 181, area 63 m2, doors 1); episode 7 room #6 (rmse 3.22, bias +0.73, outside-wall faces 74, area 35 m2, doors 0); episode 6 room #4 (rmse 3.14, bias +1.64, outside-wall faces 204, area 95 m2, doors 1)

TERMS (one-step residual regressed on each term's contribution; slope s means the term should be about (1+s) times as large)
  residual rms 0.7439 K per step; the terms explain 97% of it
    air         size 0.2645 K/step   slope -0.535 +- 0.002   -> too strong by about 54%
    neighbours  size 0.1842 K/step   slope -0.838 +- 0.003   -> too strong by about 84%
    slab        size 0.4439 K/step   slope -1.089 +- 0.003   -> too strong by about 109%
    wall        size 0.2775 K/step   slope -0.545 +- 0.003   -> too strong by about 55%
    inner       size 0.2463 K/step   slope -0.943 +- 0.005   -> too strong by about 94%
    heat        size 0.3088 K/step   slope -0.707 +- 0.002   -> too strong by about 71%
  exterior rooms only: wall -0.557 (too strong by about 56%)
  (terms that move together, e.g. outside and wall, share the residual: read their slopes together)

complexity 772 AST nodes, history 1 steps, horizons 1h, 3h, 6h

Measured vs predicted (the parent's 3 h-ahead forecast) for two rooms:
a held-out episode; room #0 = most exposed exterior room, room #35 = largest interior room
time  | room #0 measured / predicted | room #35 measured / predicted | T_out | HVAC air
   0h |   5.57 /  10.43 C |  13.59 /  15.67 C |  -2.2 C | off
   2h |   3.30 /   8.63 C |  11.88 /  14.68 C |  -4.1 C | off
   4h |   1.87 /   6.76 C |  10.19 /  13.39 C |  -4.1 C | off
   6h |   1.56 /   5.18 C |   8.72 /  11.96 C |  -2.3 C | off
   8h |   9.23 /   9.34 C |  10.18 /  12.15 C |   0.8 C | on
  10h |  11.61 /  10.66 C |  11.51 /  12.14 C |   4.4 C | on
  12h |  13.08 /  11.79 C |  13.07 /  12.51 C |   7.6 C | on
  14h |  13.57 /  12.13 C |  14.44 /  13.21 C |   9.5 C | on
  16h |  13.87 /  12.89 C |  15.57 /  14.09 C |   9.4 C | on
  18h |  13.68 /  13.53 C |  16.39 /  15.02 C |   7.6 C | on
  20h |  10.86 /  12.43 C |  15.37 /  14.72 C |   4.5 C | off
  22h |   8.33 /  11.61 C |  14.40 /  14.73 C |   0.9 C | off

Write 2 children of this parent.  Each must change ONE thing that the analysis points to (one coefficient formula, a different history summary, a history-inferred material value) and keep the rest. Use the TERMS slopes, the SPATIAL correlations and the TEMPORAL breakdown to decide which coefficient to change and in which direction, and say in the hypothesis which diagnostic line motivates the change and which objective it should improve.
