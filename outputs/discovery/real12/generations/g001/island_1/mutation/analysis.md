# Pre-mutation analysis of g000_i1_0003 seed_cf_network

```
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
```

## Measured vs predicted

```
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
```
