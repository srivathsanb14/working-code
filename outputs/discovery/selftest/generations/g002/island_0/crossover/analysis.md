# Pre-crossover analysis: A = g001_i0_0006 mock_mut_g1_0, B = g001_i0_0016 mock_exp_g1_0

```
Where each parent is better (A vs B, main-horizon rollout RMSE, K):
  all rooms                              2.345 vs 2.111  -> B
  exterior rooms                         2.583 vs 2.290  -> B
  interior rooms                         1.473 vs 1.488  -> A
  most exposed third of exterior rooms   2.863 vs 2.567  -> B
  HVAC air on                            1.703 vs 0.727  -> B
  HVAC air off                           2.994 vs 3.113  -> A
  comfort hours                          1.703 vs 0.727  -> B
  night/weekend                          2.994 vs 3.113  -> A
  room being reheated                    2.082 vs 0.673  -> B
  room not reheated                      2.405 vs 2.331  -> B
  colder half of outside temps           3.177 vs 3.077  -> B
  warmer half                            1.392 vs 0.757  -> B
A is clearly better on: HVAC air off, night/weekend
B is clearly better on: all rooms, exterior rooms, most exposed third of exterior rooms, HVAC air on, comfort hours, room being reheated, room not reheated, colder half of outside temps, warmer half
templates (top-level functions): same in both: none; different: prepare; only in A: _known, _slow, facade_u, step; only in B: _spread, init_state, predict, correct
```
