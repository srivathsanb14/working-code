# Pre-crossover analysis: A = g000_i1_0001 seed_cf_history_slab, B = g001_i1_0013 mock_mut_g1_0

```
Where each parent is better (A vs B, main-horizon rollout RMSE, K):
  all rooms                              2.500 vs 2.345  -> B
  exterior rooms                         2.739 vs 2.583  -> B
  interior rooms                         1.645 vs 1.473  -> B
  most exposed third of exterior rooms   3.026 vs 2.863  -> B
  HVAC air on                            1.520 vs 1.703  -> A
  HVAC air off                           3.391 vs 2.994  -> B
  comfort hours                          1.520 vs 1.703  -> A
  night/weekend                          3.391 vs 2.994  -> B
  room being reheated                    1.814 vs 2.082  -> A
  room not reheated                      2.640 vs 2.405  -> B
  colder half of outside temps           3.489 vs 3.177  -> B
  warmer half                            1.288 vs 1.392  -> A
A is clearly better on: HVAC air on, comfort hours, room being reheated, warmer half
B is clearly better on: all rooms, exterior rooms, interior rooms, most exposed third of exterior rooms, HVAC air off, night/weekend, room not reheated, colder half of outside temps
templates (top-level functions): same in both: _known, _slow, facade_u, step; different: prepare; only in A: none; only in B: none
```
