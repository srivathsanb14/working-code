# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 0.  Operator: CROSSOVER.

### Parent A: seed_cf_one_node  [g000_i0_0004, island 0, score 2.8151; objectives main 2.799, fast 2.230, slow 3.475, exterior 3.052, interior 3.277, passive 3.879, driven 2.392, complexity 156]
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


### Parent B: seed_cf_history  [g000_i0_0000, island 0, score 2.8173; objectives main 2.744, fast 2.591, slow 2.926, exterior 3.098, interior 1.546, passive 3.720, driven 1.685, complexity 731]
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
    exterior rooms                               3.052 vs 3.098  (tie)
    interior rooms                               3.277 vs 1.546  -> B
    corner rooms (outside wall on 2+ sides)      3.001 vs 3.480  -> A
    outside wall on the top side of the plan     2.957 vs 3.202  -> A
    outside wall on the bottom side of the plan  3.025 vs 3.217  -> A
    outside wall on the left side of the plan    2.987 vs 3.135  (tie)
    outside wall on the right side of the plan   3.049 vs 3.111  (tie)
    floor 1                                      3.016 vs 2.730  -> B
    floor 2                                      3.046 vs 2.647  -> B
    floor 3                                      3.422 vs 2.936  -> B
    floor 4                                      3.252 vs 2.923  -> B
  TEMPORAL
    target hour 00-04                            4.584 vs 3.435  -> B
    target hour 04-08                            4.109 vs 4.518  -> A
    target hour 08-12                            3.129 vs 2.132  -> B
    target hour 12-16                            1.714 vs 0.851  -> B
    target hour 16-20                            1.247 vs 1.244  (tie)
    target hour 20-24                            3.297 vs 3.065  -> B
    occupied (comfort) hours                     2.392 vs 1.685  -> B
    nights and weekends                          3.879 vs 3.720  (tie)
    HVAC air on at the target                    2.392 vs 1.685  -> B
    HVAC air off at the target                   3.879 vs 3.720  (tie)
    HVAC switched off during the rollout         1.242 vs 2.154  -> A
    HVAC switched on during the rollout          3.589 vs 3.076  -> B
    colder half of outside temperatures          4.125 vs 3.812  -> B
    warmer half of outside temperatures          1.994 vs 1.437  -> B
    first 3 days of the episode                  3.117 vs 2.748  -> B
  A is clearly better on: corner rooms (outside wall on 2+ sides), outside wall on the top side of the plan, outside wall on the bottom side of the plan, target hour 04-08, HVAC switched off during the rollout
  B is clearly better on: interior rooms, floor 1, floor 2, floor 3, floor 4, target hour 00-04, target hour 08-12, target hour 12-16, target hour 20-24, occupied (comfort) hours, HVAC air on at the target, HVAC switched on during the rollout, colder half of outside temperatures, warmer half of outside temperatures, first 3 days of the episode
  per room: A better in 228 of 586 validation rooms; rmse 3.117 vs 2.748, best-of-both per room 2.381 (the most a combination that keeps each parent's good rooms could reach)
TERMS (one-step residual slope per term; nearer 0 = that term's formula is closer to right)
    air         -0.751 vs -0.764   -> take 'air' from A
    outside     -0.730 vs -0.890   -> take 'outside' from A
    neighbours  -0.784 vs -0.918   -> take 'neighbours' from A
    heat        -0.860 vs -0.923   -> take 'heat' from A
FUNCTIONS: same in both: none; different: coefficients; only in A: none; only in B: _geo, facade_u
COEFFICIENT VALUES on a validation building, mean over rooms (A vs B); most different first:
    slab         0 vs 0.3317
    slab_back    0 vs 0.01389
    outside      0.07423 vs 0.09998
    air          0.0006678 vs 0.0006678
    heat         0.0006678 vs 0.0006678
    neighbours   0.1875 vs 0.1875

Write 2 children that COMBINE the parents: take each coefficient formula from the parent that is better where that coefficient matters (the TERMS lines say which parent's formula is closer to right for each term; the SPATIAL and TEMPORAL lines say where and when each parent wins), make the formulas consistent, and state in the hypothesis which formula came from which parent.
