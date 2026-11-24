# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 0.  Operator: CROSSOVER.

### Parent A: seed_cf_network  [g000_i0_0002, island 0, score 0.8174; objectives main 0.817, fast 0.544, slow 0.889, exterior 0.732, interior 0.814, passive 0.773, driven 0.582, complexity 772]
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


### Parent B: seed_cf_history  [g000_i0_0000, island 0, score 1.1724; objectives main 1.172, fast 1.036, slow 1.438, exterior 1.496, interior 0.926, passive 1.434, driven 1.018, complexity 731]
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


Pre-crossover analysis:
WHERE AND WHEN EACH PARENT IS BETTER (A vs B, rollout RMSE, K)
  SPATIAL
    exterior rooms                               0.732 vs 1.496  -> A
    interior rooms                               0.814 vs 0.926  -> A
    corner rooms (outside wall on 2+ sides)      0.758 vs 1.668  -> A
    outside wall on the top side of the plan     0.704 vs 1.549  -> A
    outside wall on the bottom side of the plan  0.732 vs 1.564  -> A
    outside wall on the left side of the plan    0.739 vs 1.539  -> A
    outside wall on the right side of the plan   0.728 vs 1.515  -> A
    floor 1                                      0.751 vs 1.298  -> A
    floor 2                                      0.765 vs 1.346  -> A
    floor 3                                      0.815 vs 1.522  -> A
    floor 4                                      0.550 vs 1.473  -> A
  TEMPORAL
    target hour 00-04                            0.756 vs 1.392  -> A
    target hour 04-08                            0.870 vs 1.263  -> A
    target hour 08-12                            0.842 vs 1.624  -> A
    target hour 12-16                            0.715 vs 1.585  -> A
    target hour 16-20                            0.649 vs 1.102  -> A
    target hour 20-24                            0.655 vs 1.293  -> A
    occupied (comfort) hours                     0.727 vs 1.507  -> A
    nights and weekends                          0.767 vs 1.309  -> A
    HVAC air on at the target                    0.582 vs 1.018  -> A
    HVAC air off at the target                   0.773 vs 1.434  -> A
    HVAC switched off during the rollout         0.752 vs 1.263  -> A
    HVAC switched on during the rollout          0.651 vs 1.295  -> A
    colder half of outside temperatures          0.852 vs 1.417  -> A
    warmer half of outside temperatures          0.644 vs 1.361  -> A
    first 3 days of the episode                  0.624 vs 1.638  -> A
    after day 3                                  0.763 vs 1.360  -> A
  A is clearly better everywhere: combine by taking A's formulas and only what the TERMS lines favour from the other
  per room: A better in 320 of 338 validation rooms; rmse 0.751 vs 1.388, best-of-both per room 0.738 (the most a combination that keeps each parent's good rooms could reach)
TERMS (one-step residual slope per term; nearer 0 = that term's formula is closer to right)
    air         +0.015 vs -0.187   -> take 'air' from A
    neighbours  -0.742 vs -0.938   -> take 'neighbours' from A
    slab        +0.418 vs +0.173   -> take 'slab' from B
    heat        -0.001 vs -0.341   -> take 'heat' from A
FUNCTIONS: same in both: none; different: coefficients; only in A: _spread; only in B: _geo, facade_u
COEFFICIENT VALUES on a validation building, mean over rooms (A vs B); most different first:
    outside      0 vs 0.02866
    wall         0.1617 vs 0
    wall_back    0.01449 vs 0
    wall_out     0.01315 vs 0
    inner        0.286 vs 0
    inner_back   0.05415 vs 0
    slab_back    0.025 vs 0.01389
    slab         0.4146 vs 0.3317
    neighbours   0.2195 vs 0.1932
    air          0.0006831 vs 0.0006831
    heat         0.0006831 vs 0.0006831

Write 2 children that COMBINE the parents: take each coefficient formula from the parent that is better where that coefficient matters (the TERMS lines say which parent's formula is closer to right for each term; the SPATIAL and TEMPORAL lines say where and when each parent wins), make the formulas consistent, and state in the hypothesis which formula came from which parent.
