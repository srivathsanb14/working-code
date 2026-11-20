# Pre-mutation analysis of g001_i0_0004 mock_mut_g1_0

```
held-out rollout RMSE by horizon (model / persistence), K: 1h 2.136/1.464, 3h 2.346/3.005, 6h 2.126/4.722
one-step (5 min) RMSE 0.5547 K (persistence 0.2169); train buildings 3h 2.297 K
3h rollout error by group (rmse, bias), K:
  all rooms                              2.345  +1.102
  exterior rooms                         2.583  +1.245
  interior rooms                         1.473  +0.696
  most exposed third of exterior rooms   2.863  +1.388
  HVAC air on                            1.703  +0.855
  HVAC air off                           2.994  +1.431
  comfort hours                          1.703  +0.855
  night/weekend                          2.994  +1.431
  room being reheated                    2.082  +1.622
  room not reheated                      2.405  +0.975
  colder half of outside temps           3.177  +2.077
  warmer half                            1.392  +0.346
skill vs persistence by horizon (ratio <1 is better): 1h 1.46, 3h 0.78, 6h 0.45
worst rooms: building_train_5_synth0_mixed room #6 (rmse 3.99, bias +2.19, facade faces 74, area 35 m2, doors 0); building_train_5_synth1_mixed room #22 (rmse 3.71, bias +2.04, facade faces 79, area 55 m2, doors 0); building_train_5_synth1_mixed room #63 (rmse 3.64, bias +2.02, facade faces 111, area 74 m2, doors 0); building_train_5_synth0_mixed room #45 (rmse 3.62, bias +1.96, facade faces 93, area 78 m2, doors 0)
complexity 793 AST nodes, history 288 steps
```

## Measured vs predicted

```
episode building_train_5_synth0_mixed; room #57 = most exposed exterior room, room #10 = largest interior room
time  | room #57 measured / predicted | room #10 measured / predicted | T_out | HVAC air
   0h |   5.02 /   7.49 C |  14.00 /  15.00 C |  -2.0 C | off
   2h |   2.91 /   5.18 C |  12.74 /  13.80 C |  -3.9 C | off
   4h |   1.70 /   4.31 C |  11.39 /  12.87 C |  -4.0 C | off
   6h |   1.68 /   4.74 C |  10.12 /  12.16 C |  -2.2 C | off
   8h |   9.05 /  12.09 C |  13.73 /  16.18 C |   1.0 C | on
  10h |  12.64 /  13.71 C |  14.71 /  16.11 C |   4.6 C | on
  12h |  13.26 /  13.25 C |  15.85 /  16.01 C |   7.7 C | on
  14h |  13.96 /  13.72 C |  16.21 /  15.91 C |   9.5 C | on
  16h |  13.74 /  13.25 C |  16.70 /  16.44 C |   9.4 C | on
  18h |  13.52 /  12.99 C |  16.91 /  16.63 C |   7.7 C | on
  20h |  10.26 /   8.17 C |  16.11 /  14.79 C |   4.5 C | off
  22h |   7.57 /   5.48 C |  15.34 /  13.73 C |   0.8 C | off
```
