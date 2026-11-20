# Pre-mutation analysis of g000_i1_0001 seed_cf_history_slab

```
held-out rollout RMSE by horizon (model / persistence), K: 1h 2.339/1.464, 3h 2.498/3.005, 6h 2.326/4.722
one-step (5 min) RMSE 0.6604 K (persistence 0.2169); train buildings 3h 2.389 K
3h rollout error by group (rmse, bias), K:
  all rooms                              2.500  +1.129
  exterior rooms                         2.739  +1.250
  interior rooms                         1.645  +0.786
  most exposed third of exterior rooms   3.026  +1.402
  HVAC air on                            1.520  +0.434
  HVAC air off                           3.391  +2.057
  comfort hours                          1.520  +0.434
  night/weekend                          3.391  +2.057
  room being reheated                    1.814  +1.232
  room not reheated                      2.640  +1.104
  colder half of outside temps           3.489  +2.612
  warmer half                            1.288  -0.019
skill vs persistence by horizon (ratio <1 is better): 1h 1.60, 3h 0.83, 6h 0.49
worst rooms: building_train_5_synth0_mixed room #6 (rmse 3.76, bias +1.75, facade faces 74, area 35 m2, doors 0); building_train_5_synth1_mixed room #22 (rmse 3.56, bias +1.65, facade faces 79, area 55 m2, doors 0); building_train_5_synth0_mixed room #45 (rmse 3.54, bias +1.61, facade faces 93, area 78 m2, doors 0); building_train_5_synth1_mixed room #63 (rmse 3.51, bias +1.64, facade faces 111, area 74 m2, doors 0)
complexity 793 AST nodes, history 288 steps
```

## Measured vs predicted

```
episode building_train_5_synth0_mixed; room #57 = most exposed exterior room, room #10 = largest interior room
time  | room #57 measured / predicted | room #10 measured / predicted | T_out | HVAC air
   0h |   5.02 /   7.94 C |  14.00 /  15.41 C |  -2.0 C | off
   2h |   2.91 /   5.85 C |  12.74 /  14.52 C |  -3.9 C | off
   4h |   1.70 /   5.21 C |  11.39 /  13.75 C |  -4.0 C | off
   6h |   1.68 /   5.76 C |  10.12 /  13.04 C |  -2.2 C | off
   8h |   9.05 /  11.90 C |  13.73 /  15.70 C |   1.0 C | on
  10h |  12.64 /  13.39 C |  14.71 /  15.47 C |   4.6 C | on
  12h |  13.26 /  12.84 C |  15.85 /  15.41 C |   7.7 C | on
  14h |  13.96 /  13.36 C |  16.21 /  15.50 C |   9.5 C | on
  16h |  13.74 /  12.86 C |  16.70 /  15.96 C |   9.4 C | on
  18h |  13.52 /  12.57 C |  16.91 /  16.21 C |   7.7 C | on
  20h |  10.26 /   8.02 C |  16.11 /  14.82 C |   4.5 C | off
  22h |   7.57 /   5.55 C |  15.34 /  14.29 C |   0.8 C | off
```
