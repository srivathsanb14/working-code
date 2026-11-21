# Pre-crossover analysis: A = g000_i0_0002 seed_cf_network_observer, B = g000_i0_0000 seed_cf_history_slab

```
Where each parent is better (A vs B, main-horizon rollout RMSE, K):
  all rooms                              2.284 vs 2.500  -> A
  exterior rooms                         2.480 vs 2.739  -> A
  interior rooms                         1.603 vs 1.645  -> A
  most exposed third of exterior rooms   2.773 vs 3.026  -> A
  HVAC air on                            0.810 vs 1.520  -> A
  HVAC air off                           3.361 vs 3.391  -> A
  comfort hours                          0.810 vs 1.520  -> A
  night/weekend                          3.361 vs 3.391  -> A
  room being reheated                    0.746 vs 1.814  -> A
  room not reheated                      2.520 vs 2.640  -> A
  colder half of outside temps           3.324 vs 3.489  -> A
  warmer half                            0.833 vs 1.288  -> A
A is clearly better on: all rooms, exterior rooms, interior rooms, most exposed third of exterior rooms, HVAC air on, comfort hours, room being reheated, room not reheated, colder half of outside temps, warmer half
templates (top-level functions): same in both: none; different: prepare; only in A: _spread, init_state, predict, correct; only in B: _known, _slow, facade_u, step
```
