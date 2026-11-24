# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 1.  Operator: CROSSOVER.

### Parent A: seed_cf_history  [g000_i1_0001, island 1, score 1.1724; objectives main 1.172, fast 1.036, slow 1.438, exterior 1.496, interior 0.926, passive 1.434, driven 1.018, complexity 731]
Hypothesis: Starting model: room air plus a slab store, with the outside-wall U-value computed from the measured history.
```python
"""Starting model: room air plus a slab store, with the outside-wall U-value computed from the measured history.

Coefficients of the fixed state-space function: neighbour exchange through shared walls and doors (contact area x a
nominal conductance per m2), a slab store coupled through floor area x 4 W/m2K with a 6 h time constant, and the
outside air through the wall area x U.  U is not fixed: every REFRESH steps it is computed from the last 24 h of
measurements as a shrunk least-squares ratio (how much the exterior rooms' measured changes follow T_out - T beyond
what the other terms explain), pulled toward a nominal value while the history is short.  Nothing is fitted.
"""
import numpy as np
import dlib

HISTORY = 288                                            # one day of 5-minute steps
REFRESH = 36                                             # U is recomputed every 3 hours
U_NOMINAL, U_LO, U_HI, SHRINK = 2.0, 0.3, 8.0, 5.0        # W/m2K; SHRINK in units of sum(x^2)
H_SLOW, TAU_SLOW = 4.0, 6 * 3600.0                       # W/m2K store coupling; store time constant, s


def _geo(geo, U):
    face = U["CELL"] * U["HEIGHT"]
    G_nb = (geo["wall_cells"] * face * 2.5 + geo["door_cells"] * face * 25.0) * (1 - np.eye(geo["V"]))
    return face, G_nb


def facade_u(geo, hist, U):
    """Closed-form estimate of the outside-wall U-value from the history buffer, pooled over exterior rooms."""
    face, G_nb = _geo(geo, U)
    dt = U["DT"]; Cair = U["RHO_C_AIR"] * U["HEIGHT"] * geo["area"]
    Th = hist["T"][0]                                                 # (H, V); input j drives T[j-1] -> T[j]
    prev, nxt = Th[:-1], Th[1:]
    inp = {k: hist[k][0, 1:] for k in ("g", "T_supply", "Q_reheat", "Q_gain", "occupants", "T_out")}
    w = np.exp(-np.arange(len(Th))[::-1] * dt / TAU_SLOW)
    T_slow = (w / w.sum()) @ Th                                       # the store, summarised by an average of history
    G_sum = G_nb.sum(1)
    T_nb = (prev @ G_nb.T) / np.maximum(G_sum, 1e-9)
    Q = inp["Q_reheat"] + inp["Q_gain"] + U["OCC_W"] * inp["occupants"][:, None] / geo["V"]
    terms = [(inp["g"] * dt / Cair, inp["T_supply"]), (G_sum * dt / Cair, T_nb), (H_SLOW * geo["area"] * dt / Cair, T_slow)]
    base = dlib.assemble(prev, terms, Q * dt / Cair)
    den = 1 + sum(np.maximum(wk, 0) for wk, _ in terms)
    A_fac = geo["facade_faces"] * face
    x = A_fac * dt / Cair * (inp["T_out"][:, None] - prev) / den      # effect of U = 1 W/m2K
    r = nxt - base
    ext = A_fac > 0
    sxy, sxx = float((x * r)[:, ext].sum()), float((x * x)[:, ext].sum())
    return float(np.clip((sxy + SHRINK * U_NOMINAL) / (sxx + SHRINK), U_LO, U_HI))


def coefficients(geo, hist, U):
    face, G_nb = _geo(geo, U)
    dt = U["DT"]; C_air = U["RHO_C_AIR"] * U["HEIGHT"] * geo["area"]
    return {"air": dt / C_air, "heat": dt / C_air,
            "neighbours": dt * G_nb / C_air[:, None],
            "slab": dt * H_SLOW * geo["area"] / C_air, "slab_back": dt / TAU_SLOW,
            "outside": dt * geo["facade_faces"] * face * facade_u(geo, hist, U) / C_air}
```


### Parent B: seed_cf_network  [g000_i1_0003, island 1, score 0.8174; objectives main 0.817, fast 0.544, slow 0.889, exterior 0.732, interior 0.814, passive 0.773, driven 0.582, complexity 772]
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
    exterior rooms                               1.496 vs 0.732  -> B
    interior rooms                               0.926 vs 0.814  -> B
    corner rooms (outside wall on 2+ sides)      1.668 vs 0.758  -> B
    outside wall on the top side of the plan     1.549 vs 0.704  -> B
    outside wall on the bottom side of the plan  1.564 vs 0.732  -> B
    outside wall on the left side of the plan    1.539 vs 0.739  -> B
    outside wall on the right side of the plan   1.515 vs 0.728  -> B
    floor 1                                      1.298 vs 0.751  -> B
    floor 2                                      1.346 vs 0.765  -> B
    floor 3                                      1.522 vs 0.815  -> B
    floor 4                                      1.473 vs 0.550  -> B
  TEMPORAL
    target hour 00-04                            1.392 vs 0.756  -> B
    target hour 04-08                            1.263 vs 0.870  -> B
    target hour 08-12                            1.624 vs 0.842  -> B
    target hour 12-16                            1.585 vs 0.715  -> B
    target hour 16-20                            1.102 vs 0.649  -> B
    target hour 20-24                            1.293 vs 0.655  -> B
    occupied (comfort) hours                     1.507 vs 0.727  -> B
    nights and weekends                          1.309 vs 0.767  -> B
    HVAC air on at the target                    1.018 vs 0.582  -> B
    HVAC air off at the target                   1.434 vs 0.773  -> B
    HVAC switched off during the rollout         1.263 vs 0.752  -> B
    HVAC switched on during the rollout          1.295 vs 0.651  -> B
    colder half of outside temperatures          1.417 vs 0.852  -> B
    warmer half of outside temperatures          1.361 vs 0.644  -> B
    first 3 days of the episode                  1.638 vs 0.624  -> B
    after day 3                                  1.360 vs 0.763  -> B
  B is clearly better everywhere: combine by taking B's formulas and only what the TERMS lines favour from the other
  per room: A better in 18 of 338 validation rooms; rmse 1.388 vs 0.751, best-of-both per room 0.738 (the most a combination that keeps each parent's good rooms could reach)
TERMS (one-step residual slope per term; nearer 0 = that term's formula is closer to right)
    air         -0.187 vs +0.015   -> take 'air' from B
    neighbours  -0.938 vs -0.742   -> take 'neighbours' from B
    slab        +0.173 vs +0.418   -> take 'slab' from A
    heat        -0.341 vs -0.001   -> take 'heat' from B
FUNCTIONS: same in both: none; different: coefficients; only in A: _geo, facade_u; only in B: _spread
COEFFICIENT VALUES on a validation building, mean over rooms (A vs B); most different first:
    outside      0.02866 vs 0
    wall         0 vs 0.1617
    wall_back    0 vs 0.01449
    wall_out     0 vs 0.01315
    inner        0 vs 0.286
    inner_back   0 vs 0.05415
    slab_back    0.01389 vs 0.025
    slab         0.3317 vs 0.4146
    neighbours   0.1932 vs 0.2195
    air          0.0006831 vs 0.0006831
    heat         0.0006831 vs 0.0006831

Write 2 children that COMBINE the parents: take each coefficient formula from the parent that is better where that coefficient matters (the TERMS lines say which parent's formula is closer to right for each term; the SPATIAL and TEMPORAL lines say where and when each parent wins), make the formulas consistent, and state in the hypothesis which formula came from which parent.
