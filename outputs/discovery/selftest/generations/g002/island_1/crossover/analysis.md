# Pre-crossover analysis: A = g001_i1_0010 mock_mut_g1_0, B = g001_i1_0014 mock_exp_g1_0

```
Where each parent is better (A vs B, main-horizon rollout RMSE, K):
  all rooms                              2.284 vs 2.111  -> B
  exterior rooms                         2.480 vs 2.290  -> B
  interior rooms                         1.603 vs 1.488  -> B
  most exposed third of exterior rooms   2.773 vs 2.567  -> B
  HVAC air on                            0.810 vs 0.727  -> B
  HVAC air off                           3.361 vs 3.113  -> B
  comfort hours                          0.810 vs 0.727  -> B
  night/weekend                          3.361 vs 3.113  -> B
  room being reheated                    0.746 vs 0.673  -> B
  room not reheated                      2.520 vs 2.331  -> B
  colder half of outside temps           3.324 vs 3.077  -> B
  warmer half                            0.833 vs 0.757  -> B
B is clearly better on: all rooms, exterior rooms, interior rooms, most exposed third of exterior rooms, HVAC air on, HVAC air off, comfort hours, night/weekend, room being reheated, room not reheated, colder half of outside temps, warmer half
templates (top-level functions): same in both: _spread, init_state, predict, correct; different: prepare; only in A: none; only in B: none
```
