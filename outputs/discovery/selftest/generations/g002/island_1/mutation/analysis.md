# Pre-mutation analysis of g001_i1_0011 mock_mut_g1_1

```
OBJECTIVES (validation, lower is better): main 2.199, fast 1.831, slow 2.383, exterior 2.363, interior 1.253, passive 3.083, driven 0.811, complexity 772

TEMPORAL
  error vs horizon (rmse / bias / persistence rmse, K):
     0.08 h   0.752 / +0.348 / 0.565
     0.50 h   1.662 / +0.833 / 1.116
     1.00 h   1.851 / +0.947 / 1.330
     2.00 h   2.038 / +1.043 / 2.088
     3.00 h   2.168 / +1.116 / 2.774
     4.00 h   2.255 / +1.172 / 3.394
     6.00 h   2.406 / +1.293 / 4.399
    -> half of the long-horizon error is already there after 1 h: the fast terms (air, heat, neighbours) are off
  one-step error autocorrelation at 1 h lag: +0.93 (persistent: a slow state or a slowly varying input is missing or mis-scaled)
  3h rollout error by time (rmse, bias, K):
    target hour 00-04                            3.566  +3.329
    target hour 04-08                            3.253  +2.826
    target hour 08-12                            0.807  +0.021
    target hour 12-16                            0.792  -0.514
    target hour 16-20                            0.586  +0.038
    target hour 20-24                            2.309  +1.977
    occupied (comfort) hours                     0.811  -0.109
    nights and weekends                          3.083  +2.667
    HVAC air on at the target                    0.811  -0.109
    HVAC air off at the target                   3.083  +2.667
    HVAC switched off during the rollout         1.127  +0.858
    HVAC switched on during the rollout          1.260  +0.898
    colder half of outside temperatures          3.057  +2.625
    warmer half of outside temperatures          0.773  -0.134
    first 3 days of the episode                  2.109  +1.081

SPATIAL
  3h rollout error by place (rmse, bias, K):
    exterior rooms                               2.363  +1.265
    interior rooms                               1.253  +0.613
    corner rooms (outside wall on 2+ sides)      2.648  +1.414
    outside wall on the top side of the plan     2.454  +1.310
    outside wall on the bottom side of the plan  2.449  +1.304
    outside wall on the left side of the plan    2.392  +1.285
    outside wall on the right side of the plan   2.370  +1.275
    floor 1                                      2.086  +1.073
    floor 2                                      2.013  +1.016
    floor 3                                      2.264  +1.184
    floor 4                                      2.355  +1.235
  correlation of a room's bias with its geometry (over all validation rooms):
    outside-wall length per floor area           +0.81   <- the formula's dependence on this is wrong
    number of neighbours                         -0.69   <- the formula's dependence on this is wrong
    shared-wall length per floor area            -0.52   <- the formula's dependence on this is wrong
    door width per floor area                    -0.35
    floor area                                   -0.14
  error series correlation, adjacent rooms +0.96 vs non-adjacent +0.93 (similar: a building-wide cause, e.g. supply air or outside term)
  worst rooms: episode 6 room #22 (rmse 3.29, bias +1.80, outside-wall faces 181, area 63 m2, doors 1); episode 5 room #22 (rmse 3.24, bias +0.99, outside-wall faces 181, area 63 m2, doors 1); episode 7 room #6 (rmse 3.22, bias +0.86, outside-wall faces 74, area 35 m2, doors 0); episode 6 room #4 (rmse 3.18, bias +1.75, outside-wall faces 204, area 95 m2, doors 1)

TERMS (one-step residual regressed on each term's contribution; slope s means the term should be about (1+s) times as large)
  residual rms 0.7221 K per step; the terms explain 97% of it
    air         size 0.2720 K/step   slope -0.557 +- 0.002   -> too strong by about 56%
    neighbours  size 0.1911 K/step   slope -0.862 +- 0.002   -> too strong by about 86%
    slab        size 0.3988 K/step   slope -1.086 +- 0.003   -> too strong by about 109%
    wall        size 0.2793 K/step   slope -0.572 +- 0.003   -> too strong by about 57%
    inner       size 0.2502 K/step   slope -0.978 +- 0.004   -> too strong by about 98%
    heat        size 0.3190 K/step   slope -0.729 +- 0.002   -> too strong by about 73%
  exterior rooms only: wall -0.582 (too strong by about 58%)
  (terms that move together, e.g. outside and wall, share the residual: read their slopes together)

complexity 772 AST nodes, history 1 steps, horizons 1h, 3h, 6h
```

## Measured vs predicted

```
a held-out episode; room #0 = most exposed exterior room, room #35 = largest interior room
time  | room #0 measured / predicted | room #35 measured / predicted | T_out | HVAC air
   0h |   5.57 /  10.46 C |  13.59 /  15.62 C |  -2.2 C | off
   2h |   3.30 /   8.71 C |  11.88 /  14.65 C |  -4.1 C | off
   4h |   1.87 /   6.90 C |  10.19 /  13.40 C |  -4.1 C | off
   6h |   1.56 /   5.36 C |   8.72 /  12.02 C |  -2.3 C | off
   8h |   9.23 /   9.66 C |  10.18 /  12.44 C |   0.8 C | on
  10h |  11.61 /  10.91 C |  11.51 /  12.51 C |   4.4 C | on
  12h |  13.08 /  11.93 C |  13.07 /  12.84 C |   7.6 C | on
  14h |  13.57 /  12.19 C |  14.44 /  13.46 C |   9.5 C | on
  16h |  13.87 /  12.91 C |  15.57 /  14.26 C |   9.4 C | on
  18h |  13.68 /  13.52 C |  16.39 /  15.11 C |   7.6 C | on
  20h |  10.86 /  12.33 C |  15.37 /  14.63 C |   4.5 C | off
  22h |   8.33 /  11.50 C |  14.40 /  14.57 C |   0.9 C | off
```
