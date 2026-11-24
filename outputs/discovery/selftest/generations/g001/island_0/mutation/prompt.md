# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 0.  Operator: MUTATION.

### Parent: seed_cf_one_node  [g000_i0_0004, island 0, score 2.8151; objectives main 2.799, fast 2.230, slow 3.475, exterior 3.052, interior 3.277, passive 3.879, driven 2.392, complexity 156]
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


Pre-mutation analysis of the parent (validation buildings): where, when and through which term it errs
OBJECTIVES (validation, lower is better): main 2.799, fast 2.230, slow 3.475, exterior 3.052, interior 3.277, passive 3.879, driven 2.392, complexity 156

TEMPORAL
  error vs horizon (rmse / bias / persistence rmse, K):
     0.08 h   0.481 / +0.163 / 0.565
     0.50 h   1.646 / +0.316 / 1.116
     1.00 h   2.263 / +0.084 / 1.330
     2.00 h   2.941 / -0.241 / 2.088
     3.00 h   3.259 / -0.415 / 2.774
     4.00 h   3.389 / -0.487 / 3.394
     6.00 h   3.515 / -0.637 / 4.399
    -> half of the long-horizon error is already there after 1 h: the fast terms (air, heat, neighbours) are off
  one-step error autocorrelation at 1 h lag: +0.75 (persistent: a slow state or a slowly varying input is missing or mis-scaled)
  3h rollout error by time (rmse, bias, K):
    target hour 00-04                            4.584  -4.365
    target hour 04-08                            4.109  -1.962
    target hour 08-12                            3.129  +2.561
    target hour 12-16                            1.714  +1.073
    target hour 16-20                            1.247  +0.823
    target hour 20-24                            3.297  -2.922
    occupied (comfort) hours                     2.392  +1.737
    nights and weekends                          3.879  -3.294
    HVAC air on at the target                    2.392  +1.737
    HVAC air off at the target                   3.879  -3.294
    HVAC switched off during the rollout         1.242  -0.522
    HVAC switched on during the rollout          3.589  +3.162
    colder half of outside temperatures          4.125  -2.475
    warmer half of outside temperatures          1.994  +1.198
    first 3 days of the episode                  3.117  -0.419

SPATIAL
  3h rollout error by place (rmse, bias, K):
    exterior rooms                               3.052  -0.219
    interior rooms                               3.277  -0.926
    corner rooms (outside wall on 2+ sides)      3.001  -0.058
    outside wall on the top side of the plan     2.957  -0.190
    outside wall on the bottom side of the plan  3.025  -0.120
    outside wall on the left side of the plan    2.987  -0.212
    outside wall on the right side of the plan   3.049  -0.215
    floor 1                                      3.016  -0.401
    floor 2                                      3.046  -0.369
    floor 3                                      3.422  -0.453
    floor 4                                      3.252  -0.731
  correlation of a room's bias with its geometry (over all validation rooms):
    shared-wall length per floor area            -0.46   <- the formula's dependence on this is wrong
    outside-wall length per floor area           +0.43   <- the formula's dependence on this is wrong
    number of neighbours                         -0.33
    door width per floor area                    -0.25
    floor area                                   +0.18
  error series correlation, adjacent rooms +0.97 vs non-adjacent +0.94 (similar: a building-wide cause, e.g. supply air or outside term)
  worst rooms: episode 1 room #35 (rmse 6.29, bias +1.06, outside-wall faces 0, area 193 m2, doors 3); episode 1 room #38 (rmse 6.12, bias +2.09, outside-wall faces 189, area 346 m2, doors 1); episode 2 room #35 (rmse 5.99, bias +0.90, outside-wall faces 0, area 193 m2, doors 3); episode 2 room #38 (rmse 5.97, bias +2.04, outside-wall faces 189, area 346 m2, doors 1)

TERMS (one-step residual regressed on each term's contribution; slope s means the term should be about (1+s) times as large)
  residual rms 0.4662 K per step; the terms explain 93% of it
    air         size 0.3812 K/step   slope -0.751 +- 0.001   -> too strong by about 75%
    outside     size 0.4267 K/step   slope -0.730 +- 0.001   -> too strong by about 73%
    neighbours  size 0.2717 K/step   slope -0.784 +- 0.002   -> too strong by about 78%
    heat        size 0.4666 K/step   slope -0.860 +- 0.001   -> too strong by about 86%
  exterior rooms only: outside -0.725 (too strong by about 72%)
  (terms that move together, e.g. outside and wall, share the residual: read their slopes together)

complexity 156 AST nodes, history 1 steps, horizons 1h, 3h, 6h

Measured vs predicted (the parent's 3 h-ahead forecast) for two rooms:
a held-out episode; room #0 = most exposed exterior room, room #35 = largest interior room
time  | room #0 measured / predicted | room #35 measured / predicted | T_out | HVAC air
   0h |   5.57 /   2.31 C |  13.59 /   6.84 C |  -2.2 C | off
   2h |   3.30 /  -0.06 C |  11.88 /   4.22 C |  -4.1 C | off
   4h |   1.87 /  -0.94 C |  10.19 /   2.66 C |  -4.1 C | off
   6h |   1.56 /  -0.16 C |   8.72 /   2.49 C |  -2.3 C | off
   8h |   9.23 /  13.84 C |  10.18 /  14.21 C |   0.8 C | on
  10h |  11.61 /  14.92 C |  11.51 /  20.31 C |   4.4 C | on
  12h |  13.08 /  14.64 C |  13.07 /  21.34 C |   7.6 C | on
  14h |  13.57 /  15.04 C |  14.44 /  21.57 C |   9.5 C | on
  16h |  13.87 /  15.08 C |  15.57 /  22.11 C |   9.4 C | on
  18h |  13.68 /  14.68 C |  16.39 /  21.61 C |   7.6 C | on
  20h |  10.86 /  10.13 C |  15.37 /  16.60 C |   4.5 C | off
  22h |   8.33 /   5.53 C |  14.40 /   9.84 C |   0.9 C | off

Write 2 children of this parent.  Each must change ONE thing that the analysis points to (one coefficient formula, a different history summary, a history-inferred material value) and keep the rest. Use the TERMS slopes, the SPATIAL correlations and the TEMPORAL breakdown to decide which coefficient to change and in which direction, and say in the hypothesis which diagnostic line motivates the change and which objective it should improve.
