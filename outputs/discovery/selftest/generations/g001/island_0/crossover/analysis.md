# Pre-crossover analysis: A = g000_i0_0004 seed_cf_one_node, B = g000_i0_0000 seed_cf_history

```
WHERE AND WHEN EACH PARENT IS BETTER (A vs B, rollout RMSE, K)
  SPATIAL
    exterior rooms                               3.052 vs 3.098  (tie)
    interior rooms                               3.277 vs 1.546  -> B
    corner rooms (outside wall on 2+ sides)      3.001 vs 3.480  -> A
    outside wall on the top side of the plan     2.957 vs 3.202  -> A
    outside wall on the bottom side of the plan  3.025 vs 3.217  -> A
    outside wall on the left side of the plan    2.987 vs 3.135  (tie)
    outside wall on the right side of the plan   3.049 vs 3.111  (tie)
    floor 1                                      3.016 vs 2.730  -> B
    floor 2                                      3.046 vs 2.647  -> B
    floor 3                                      3.422 vs 2.936  -> B
    floor 4                                      3.252 vs 2.923  -> B
  TEMPORAL
    target hour 00-04                            4.584 vs 3.435  -> B
    target hour 04-08                            4.109 vs 4.518  -> A
    target hour 08-12                            3.129 vs 2.132  -> B
    target hour 12-16                            1.714 vs 0.851  -> B
    target hour 16-20                            1.247 vs 1.244  (tie)
    target hour 20-24                            3.297 vs 3.065  -> B
    occupied (comfort) hours                     2.392 vs 1.685  -> B
    nights and weekends                          3.879 vs 3.720  (tie)
    HVAC air on at the target                    2.392 vs 1.685  -> B
    HVAC air off at the target                   3.879 vs 3.720  (tie)
    HVAC switched off during the rollout         1.242 vs 2.154  -> A
    HVAC switched on during the rollout          3.589 vs 3.076  -> B
    colder half of outside temperatures          4.125 vs 3.812  -> B
    warmer half of outside temperatures          1.994 vs 1.437  -> B
    first 3 days of the episode                  3.117 vs 2.748  -> B
  A is clearly better on: corner rooms (outside wall on 2+ sides), outside wall on the top side of the plan, outside wall on the bottom side of the plan, target hour 04-08, HVAC switched off during the rollout
  B is clearly better on: interior rooms, floor 1, floor 2, floor 3, floor 4, target hour 00-04, target hour 08-12, target hour 12-16, target hour 20-24, occupied (comfort) hours, HVAC air on at the target, HVAC switched on during the rollout, colder half of outside temperatures, warmer half of outside temperatures, first 3 days of the episode
  per room: A better in 228 of 586 validation rooms; rmse 3.117 vs 2.748, best-of-both per room 2.381 (the most a combination that keeps each parent's good rooms could reach)
TERMS (one-step residual slope per term; nearer 0 = that term's formula is closer to right)
    air         -0.751 vs -0.764   -> take 'air' from A
    outside     -0.730 vs -0.890   -> take 'outside' from A
    neighbours  -0.784 vs -0.918   -> take 'neighbours' from A
    heat        -0.860 vs -0.923   -> take 'heat' from A
FUNCTIONS: same in both: none; different: coefficients; only in A: none; only in B: _geo, facade_u
COEFFICIENT VALUES on a validation building, mean over rooms (A vs B); most different first:
    slab         0 vs 0.3317
    slab_back    0 vs 0.01389
    outside      0.07423 vs 0.09998
    air          0.0006678 vs 0.0006678
    heat         0.0006678 vs 0.0006678
    neighbours   0.1875 vs 0.1875
```
