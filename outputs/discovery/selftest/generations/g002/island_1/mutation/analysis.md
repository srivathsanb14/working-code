# Pre-mutation analysis of g001_i1_0010 mock_mut_g1_0

```
held-out rollout RMSE by horizon (model / persistence), K: 1h 1.961/1.464, 3h 2.268/3.005, 6h 2.512/4.722
one-step (5 min) RMSE 0.8822 K (persistence 0.2169); train buildings 3h 2.016 K
3h rollout error by group (rmse, bias), K:
  all rooms                              2.284  +1.167
  exterior rooms                         2.480  +1.301
  interior rooms                         1.603  +0.788
  most exposed third of exterior rooms   2.773  +1.482
  HVAC air on                            0.810  -0.228
  HVAC air off                           3.361  +3.026
  comfort hours                          0.810  -0.228
  night/weekend                          3.361  +3.026
  room being reheated                    0.746  +0.117
  room not reheated                      2.520  +1.423
  colder half of outside temps           3.324  +2.927
  warmer half                            0.833  -0.197
skill vs persistence by horizon (ratio <1 is better): 1h 1.34, 3h 0.75, 6h 0.53
worst rooms: held-out episode 1 room #6 (rmse 3.15, bias +0.78, facade faces 74, area 35 m2, doors 0); held-out episode 1 room #13 (rmse 3.04, bias +1.73, facade faces 72, area 44 m2, doors 0); held-out episode 2 room #6 (rmse 3.00, bias +1.66, facade faces 74, area 35 m2, doors 0); held-out episode 4 room #17 (rmse 2.98, bias +1.66, facade faces 135, area 77 m2, doors 1)
complexity 990 AST nodes, history 1 steps
```

## Measured vs predicted

```
a held-out episode; room #57 = most exposed exterior room, room #10 = largest interior room
time  | room #57 measured / predicted | room #10 measured / predicted | T_out | HVAC air
   0h |   5.02 /  10.23 C |  14.00 /  15.64 C |  -2.0 C | off
   2h |   2.91 /   8.44 C |  12.74 /  14.96 C |  -3.9 C | off
   4h |   1.70 /   6.69 C |  11.39 /  14.02 C |  -4.0 C | off
   6h |   1.68 /   5.27 C |  10.12 /  12.90 C |  -2.2 C | off
   8h |   9.05 /   8.48 C |  13.73 /  14.26 C |   1.0 C | on
  10h |  12.64 /  11.55 C |  14.71 /  14.42 C |   4.6 C | on
  12h |  13.26 /  12.08 C |  15.85 /  15.05 C |   7.7 C | on
  14h |  13.96 /  13.06 C |  16.21 /  15.44 C |   9.5 C | on
  16h |  13.74 /  13.28 C |  16.70 /  16.00 C |   9.4 C | on
  18h |  13.52 /  13.76 C |  16.91 /  16.38 C |   7.7 C | on
  20h |  10.26 /  12.58 C |  16.11 /  16.01 C |   4.5 C | off
  22h |   7.57 /  11.56 C |  15.34 /  15.93 C |   0.8 C | off
```
