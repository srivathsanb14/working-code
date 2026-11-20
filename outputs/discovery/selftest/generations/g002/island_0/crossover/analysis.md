# Pre-crossover analysis: A = g001_i0_0005 mock_mut_g1_1, B = g000_i0_0002 seed_cf_one_node

```
Where each parent is better (A vs B, main-horizon rollout RMSE, K):
  all rooms                              2.501 vs 3.289  -> A
  exterior rooms                         2.740 vs 3.022  -> A
  interior rooms                         1.646 vs 3.950  -> A
  most exposed third of exterior rooms   3.028 vs 2.445  -> B
  HVAC air on                            1.520 vs 2.300  -> A
  HVAC air off                           3.394 vs 4.265  -> A
  comfort hours                          1.520 vs 2.300  -> A
  night/weekend                          3.394 vs 4.265  -> A
  room being reheated                    1.815 vs 2.523  -> A
  room not reheated                      2.642 vs 3.451  -> A
  colder half of outside temps           3.492 vs 4.488  -> A
  warmer half                            1.287 vs 1.896  -> A
A is clearly better on: all rooms, exterior rooms, interior rooms, HVAC air on, HVAC air off, comfort hours, night/weekend, room being reheated, room not reheated, colder half of outside temps, warmer half
B is clearly better on: most exposed third of exterior rooms
templates (top-level functions): same in both: none; different: prepare, step; only in A: _known, _slow, facade_u; only in B: none
```
