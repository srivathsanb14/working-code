# Pre-crossover analysis: A = g000_i1_0001 seed_cf_history_slab, B = g000_i1_0003 seed_cf_network_observer

```
Where each parent is better (A vs B, main-horizon rollout RMSE, K):
  all rooms                              2.500 vs 2.284  -> B
  exterior rooms                         2.739 vs 2.480  -> B
  interior rooms                         1.645 vs 1.603  -> B
  most exposed third of exterior rooms   3.026 vs 2.773  -> B
  HVAC air on                            1.520 vs 0.810  -> B
  HVAC air off                           3.391 vs 3.361  -> B
  comfort hours                          1.520 vs 0.810  -> B
  night/weekend                          3.391 vs 3.361  -> B
  room being reheated                    1.814 vs 0.746  -> B
  room not reheated                      2.640 vs 2.520  -> B
  colder half of outside temps           3.489 vs 3.324  -> B
  warmer half                            1.288 vs 0.833  -> B
B is clearly better on: all rooms, exterior rooms, interior rooms, most exposed third of exterior rooms, HVAC air on, comfort hours, room being reheated, room not reheated, colder half of outside temps, warmer half
templates (top-level functions): same in both: none; different: prepare; only in A: _known, _slow, facade_u, step; only in B: _spread, init_state, predict, correct
```
