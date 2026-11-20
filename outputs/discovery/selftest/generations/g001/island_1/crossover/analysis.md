# Pre-crossover analysis: A = g000_i1_0001 seed_cf_history_slab, B = g000_i1_0003 seed_cf_one_node

```
Where each parent is better (A vs B, main-horizon rollout RMSE, K):
  all rooms                              2.500 vs 3.289  -> A
  exterior rooms                         2.739 vs 3.022  -> A
  interior rooms                         1.645 vs 3.950  -> A
  most exposed third of exterior rooms   3.026 vs 2.445  -> B
  HVAC air on                            1.520 vs 2.300  -> A
  HVAC air off                           3.391 vs 4.265  -> A
  comfort hours                          1.520 vs 2.300  -> A
  night/weekend                          3.391 vs 4.265  -> A
  room being reheated                    1.814 vs 2.523  -> A
  room not reheated                      2.640 vs 3.451  -> A
  colder half of outside temps           3.489 vs 4.488  -> A
  warmer half                            1.288 vs 1.896  -> A
A is clearly better on: all rooms, exterior rooms, interior rooms, HVAC air on, HVAC air off, comfort hours, night/weekend, room being reheated, room not reheated, colder half of outside temps, warmer half
B is clearly better on: most exposed third of exterior rooms
templates (top-level functions): same in both: none; different: prepare, step; only in A: _known, _slow, facade_u; only in B: none
```
