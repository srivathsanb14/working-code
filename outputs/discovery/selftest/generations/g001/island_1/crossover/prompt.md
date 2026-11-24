# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 1.  Operator: CROSSOVER.

### Parent A: seed_cf_one_node  [g000_i1_0005, island 1, score 2.8151; objectives main 2.799, fast 2.230, slow 3.475, exterior 3.052, interior 3.277, passive 3.879, driven 2.392, complexity 156]
Hypothesis: Starting model: one temperature per room (its air), nominal materials, nothing read from history.
```python
"""Starting model: one temperature per room (its air), nominal materials, nothing read from history.

Coefficients: supply air and heat through the air capacity (floor area); outside wall = facade area x a nominal
2 W/m2K; neighbour exchange through shared walls and doors = contact area x a nominal conductance per m2.  No slab or
wall store.  This is the baseline the other starting models and every LLM candidate must beat.
"""
import numpy as np

HISTORY = 1
REFRESH = 288


def coefficients(geo, hist, U):
    face = U["CELL"] * U["HEIGHT"]; dt = U["DT"]
    C_air = U["RHO_C_AIR"] * U["HEIGHT"] * geo["area"]
    G_nb = (geo["wall_cells"] * face * 2.5 + geo["door_cells"] * face * 25.0) * (1 - np.eye(geo["V"]))
    return {"air": dt / C_air, "heat": dt / C_air, "neighbours": dt * G_nb / C_air[:, None],
            "outside": dt * geo["facade_faces"] * face * 2.0 / C_air}
```


### Parent B: seed_cf_network  [g000_i1_0003, island 1, score 2.2602; objectives main 2.183, fast 1.807, slow 2.384, exterior 2.339, interior 1.253, passive 3.050, driven 0.814, complexity 772]
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


Pre-crossover analysis:
WHERE AND WHEN EACH PARENT IS BETTER (A vs B, rollout RMSE, K)
  SPATIAL
    exterior rooms                               3.052 vs 2.339  -> B
    interior rooms                               3.277 vs 1.253  -> B
    corner rooms (outside wall on 2+ sides)      3.001 vs 2.620  -> B
    outside wall on the top side of the plan     2.957 vs 2.429  -> B
    outside wall on the bottom side of the plan  3.025 vs 2.423  -> B
    outside wall on the left side of the plan    2.987 vs 2.366  -> B
    outside wall on the right side of the plan   3.049 vs 2.345  -> B
    floor 1                                      3.016 vs 2.065  -> B
    floor 2                                      3.046 vs 1.995  -> B
    floor 3                                      3.422 vs 2.243  -> B
    floor 4                                      3.252 vs 2.336  -> B
  TEMPORAL
    target hour 00-04                            4.584 vs 3.544  -> B
    target hour 04-08                            4.109 vs 3.158  -> B
    target hour 08-12                            3.129 vs 0.817  -> B
    target hour 12-16                            1.714 vs 0.852  -> B
    target hour 16-20                            1.247 vs 0.600  -> B
    target hour 20-24                            3.297 vs 2.340  -> B
    occupied (comfort) hours                     2.392 vs 0.814  -> B
    nights and weekends                          3.879 vs 3.050  -> B
    HVAC air on at the target                    2.392 vs 0.814  -> B
    HVAC air off at the target                   3.879 vs 3.050  -> B
    HVAC switched off during the rollout         1.242 vs 1.153  -> B
    HVAC switched on during the rollout          3.589 vs 1.122  -> B
    colder half of outside temperatures          4.125 vs 3.012  -> B
    warmer half of outside temperatures          1.994 vs 0.814  -> B
    first 3 days of the episode                  3.117 vs 2.089  -> B
  B is clearly better on: exterior rooms, interior rooms, corner rooms (outside wall on 2+ sides), outside wall on the top side of the plan, outside wall on the bottom side of the plan, outside wall on the left side of the plan, outside wall on the right side of the plan, floor 1, floor 2, floor 3, floor 4, target hour 00-04, target hour 04-08, target hour 08-12, target hour 12-16, target hour 16-20, target hour 20-24, occupied (comfort) hours, nights and weekends, HVAC air on at the target, HVAC air off at the target, HVAC switched off during the rollout, HVAC switched on during the rollout, colder half of outside temperatures, warmer half of outside temperatures, first 3 days of the episode
  per room: A better in 131 of 586 validation rooms; rmse 3.117 vs 2.089, best-of-both per room 1.979 (the most a combination that keeps each parent's good rooms could reach)
TERMS (one-step residual slope per term; nearer 0 = that term's formula is closer to right)
    air         -0.751 vs -0.535   -> take 'air' from B
    neighbours  -0.784 vs -0.838   -> take 'neighbours' from A
    heat        -0.860 vs -0.707   -> take 'heat' from B
FUNCTIONS: same in both: none; different: coefficients; only in A: none; only in B: _spread
COEFFICIENT VALUES on a validation building, mean over rooms (A vs B); most different first:
    outside      0.07423 vs 0
    slab         0 vs 0.4146
    slab_back    0 vs 0.025
    wall         0 vs 0.1773
    wall_back    0 vs 0.01479
    wall_out     0 vs 0.01346
    inner        0 vs 0.2806
    inner_back   0 vs 0.05412
    neighbours   0.1875 vs 0.2107
    air          0.0006678 vs 0.0006678
    heat         0.0006678 vs 0.0006678

Write 2 children that COMBINE the parents: take each coefficient formula from the parent that is better where that coefficient matters (the TERMS lines say which parent's formula is closer to right for each term; the SPATIAL and TEMPORAL lines say where and when each parent wins), make the formulas consistent, and state in the hypothesis which formula came from which parent.
