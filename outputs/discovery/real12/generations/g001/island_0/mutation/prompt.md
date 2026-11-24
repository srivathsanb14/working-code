# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 0.  Operator: MUTATION.

### Parent: seed_cf_history  [g000_i0_0000, island 0, score 1.1724; objectives main 1.172, fast 1.036, slow 1.438, exterior 1.496, interior 0.926, passive 1.434, driven 1.018, complexity 731]
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


Pre-mutation analysis of the parent (validation buildings): where, when and through which term it errs
OBJECTIVES (validation, lower is better): main 1.172, fast 1.036, slow 1.438, exterior 1.496, interior 0.926, passive 1.434, driven 1.018, complexity 731

TEMPORAL
  error vs horizon (rmse / bias / persistence rmse, K):
     0.08 h   0.334 / +0.035 / 0.527
     0.50 h   0.937 / +0.103 / 0.578
     1.00 h   1.099 / +0.117 / 0.764
     2.00 h   1.221 / +0.122 / 0.784
     3.00 h   1.314 / +0.123 / 0.931
     4.00 h   1.399 / +0.127 / 1.120
     6.00 h   1.533 / +0.139 / 1.358
    -> half of the long-horizon error is already there after 1 h: the fast terms (air, heat, neighbours) are off
  one-step error autocorrelation at 1 h lag: +0.78 (persistent: a slow state or a slowly varying input is missing or mis-scaled)
  3h rollout error by time (rmse, bias, K):
    target hour 00-04                            1.392  -0.695
    target hour 04-08                            1.263  -0.122
    target hour 08-12                            1.624  +1.094
    target hour 12-16                            1.585  +1.068
    target hour 16-20                            1.102  +0.354
    target hour 20-24                            1.293  -0.566
    occupied (comfort) hours                     1.507  +0.943
    nights and weekends                          1.309  -0.281
    HVAC air on at the target                    1.018  +0.377
    HVAC air off at the target                   1.434  +0.162
    HVAC switched off during the rollout         1.263  +0.479
    HVAC switched on during the rollout          1.295  +0.703
    colder half of outside temperatures          1.417  -0.159
    warmer half of outside temperatures          1.361  +0.514
    first 3 days of the episode                  1.638  +0.387
    after day 3                                  1.360  +0.169

SPATIAL
  3h rollout error by place (rmse, bias, K):
    exterior rooms                               1.496  +0.349
    interior rooms                               0.926  -0.362
    corner rooms (outside wall on 2+ sides)      1.668  +0.436
    outside wall on the top side of the plan     1.549  +0.378
    outside wall on the bottom side of the plan  1.564  +0.404
    outside wall on the left side of the plan    1.539  +0.375
    outside wall on the right side of the plan   1.515  +0.371
    floor 1                                      1.298  +0.124
    floor 2                                      1.346  +0.143
    floor 3                                      1.522  +0.261
    floor 4                                      1.473  +0.367
  correlation of a room's bias with its geometry (over all validation rooms):
    number of neighbours                         -0.64   <- the formula's dependence on this is wrong
    outside-wall length per floor area           +0.61   <- the formula's dependence on this is wrong
    floor area                                   -0.36
    shared-wall length per floor area            -0.23
    door width per floor area                    -0.12
  error series correlation, adjacent rooms +0.90 vs non-adjacent +0.90 (similar: a building-wide cause, e.g. supply air or outside term)
  worst rooms: episode 1 room #15 (rmse 2.76, bias +0.67, outside-wall faces 181, area 63 m2, doors 1); episode 2 room #15 (rmse 2.68, bias +1.07, outside-wall faces 181, area 63 m2, doors 1); episode 1 room #44 (rmse 2.50, bias +0.40, outside-wall faces 204, area 95 m2, doors 1); episode 2 room #29 (rmse 2.43, bias +1.17, outside-wall faces 68, area 39 m2, doors 1)

TERMS (one-step residual regressed on each term's contribution; slope s means the term should be about (1+s) times as large)
  residual rms 0.3538 K per step; the terms explain 61% of it
    air         size 0.3807 K/step   slope -0.187 +- 0.003   -> too strong by about 19%
    outside     size 0.4036 K/step   slope -0.533 +- 0.003   -> too strong by about 53%
    neighbours  size 0.2722 K/step   slope -0.938 +- 0.004   -> too strong by about 94%
    slab        size 0.1708 K/step   slope +0.173 +- 0.006   -> too weak by about 17%
    heat        size 0.5845 K/step   slope -0.341 +- 0.002   -> too strong by about 34%
  exterior rooms only: outside -0.591 (too strong by about 59%)
  (terms that move together, e.g. outside and wall, share the residual: read their slopes together)

complexity 731 AST nodes, history 288 steps, horizons 1h, 3h, 6h

Measured vs predicted (the parent's 3 h-ahead forecast) for two rooms:
a held-out episode; room #15 = most exposed exterior room, room #37 = largest interior room
time  | room #15 measured / predicted | room #37 measured / predicted | T_out | HVAC air
   0h |   9.45 /  15.74 C |  20.55 /  20.47 C |  -1.7 C | on
   2h |  21.96 /  25.46 C |  21.66 /  22.89 C |   1.1 C | on
   4h |  17.37 /  23.54 C |  22.35 /  23.76 C |  12.2 C | off
   6h |  21.90 /  25.23 C |  22.94 /  23.81 C |  15.6 C | off
   8h |  19.19 /  20.56 C |  23.45 /  24.11 C |  16.1 C | off
  10h |  19.62 /  19.81 C |  23.95 /  24.33 C |  16.1 C | off
  12h |  19.58 /  17.18 C |  24.45 /  24.46 C |   9.9 C | off
  14h |  17.50 /  12.79 C |  23.79 /  22.38 C |   4.5 C | off
  16h |  15.87 /  10.70 C |  23.72 /  21.77 C |   2.7 C | off
  18h |  14.42 /  10.03 C |  23.62 /  21.67 C |   2.2 C | off
  20h |  13.27 /   9.39 C |  23.51 /  21.57 C |   1.8 C | off
  22h |  12.34 /   9.88 C |  23.38 /  21.73 C |   1.8 C | off

Write 2 children of this parent.  Each must change ONE thing that the analysis points to (one coefficient formula, a different history summary, a history-inferred material value) and keep the rest. Use the TERMS slopes, the SPATIAL correlations and the TEMPORAL breakdown to decide which coefficient to change and in which direction, and say in the hypothesis which diagnostic line motivates the change and which objective it should improve.
