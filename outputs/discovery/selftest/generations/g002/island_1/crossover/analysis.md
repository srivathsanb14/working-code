# Pre-crossover analysis: A = g000_i1_0005 seed_cf_one_node, B = g000_i1_0003 seed_cf_network

```
WHERE AND WHEN EACH PARENT IS BETTER (A vs B, rollout RMSE, K)
  SPATIAL
    exterior rooms                               3.052 vs 2.339  -> B
    interior rooms                               3.277 vs 1.253  -> B
    corner rooms (outside wall on 2+ sides)      3.001 vs 2.620  -> B
    outside wall on the top side of the plan     2.957 vs 2.429  -> B
    outside wall on the bottom side of the plan  3.025 vs 2.423  -> B
    outside wall on the left side of the plan    2.987 vs 2.366  -> B
    outside wall on the right side of the plan   3.049 vs 2.345  -> B
    floor 1                                      3.016 vs 2.065  -> B
    floor 2                                      3.046 vs 1.995  -> B
    floor 3                                      3.422 vs 2.243  -> B
    floor 4                                      3.252 vs 2.336  -> B
  TEMPORAL
    target hour 00-04                            4.584 vs 3.544  -> B
    target hour 04-08                            4.109 vs 3.158  -> B
    target hour 08-12                            3.129 vs 0.817  -> B
    target hour 12-16                            1.714 vs 0.852  -> B
    target hour 16-20                            1.247 vs 0.600  -> B
    target hour 20-24                            3.297 vs 2.340  -> B
    occupied (comfort) hours                     2.392 vs 0.814  -> B
    nights and weekends                          3.879 vs 3.050  -> B
    HVAC air on at the target                    2.392 vs 0.814  -> B
    HVAC air off at the target                   3.879 vs 3.050  -> B
    HVAC switched off during the rollout         1.242 vs 1.153  -> B
    HVAC switched on during the rollout          3.589 vs 1.122  -> B
    colder half of outside temperatures          4.125 vs 3.012  -> B
    warmer half of outside temperatures          1.994 vs 0.814  -> B
    first 3 days of the episode                  3.117 vs 2.089  -> B
  B is clearly better on: exterior rooms, interior rooms, corner rooms (outside wall on 2+ sides), outside wall on the top side of the plan, outside wall on the bottom side of the plan, outside wall on the left side of the plan, outside wall on the right side of the plan, floor 1, floor 2, floor 3, floor 4, target hour 00-04, target hour 04-08, target hour 08-12, target hour 12-16, target hour 16-20, target hour 20-24, occupied (comfort) hours, nights and weekends, HVAC air on at the target, HVAC air off at the target, HVAC switched off during the rollout, HVAC switched on during the rollout, colder half of outside temperatures, warmer half of outside temperatures, first 3 days of the episode
  per room: A better in 131 of 586 validation rooms; rmse 3.117 vs 2.089, best-of-both per room 1.979 (the most a combination that keeps each parent's good rooms could reach)
TERMS (one-step residual slope per term; nearer 0 = that term's formula is closer to right)
    air         -0.751 vs -0.535   -> take 'air' from B
    neighbours  -0.784 vs -0.838   -> take 'neighbours' from A
    heat        -0.860 vs -0.707   -> take 'heat' from B
FUNCTIONS: same in both: none; different: coefficients; only in A: none; only in B: _spread
COEFFICIENT VALUES on a validation building, mean over rooms (A vs B); most different first:
    outside      0.07423 vs 0
    slab         0 vs 0.4146
    slab_back    0 vs 0.025
    wall         0 vs 0.1773
    wall_back    0 vs 0.01479
    wall_out     0 vs 0.01346
    inner        0 vs 0.2806
    inner_back   0 vs 0.05412
    neighbours   0.1875 vs 0.2107
    air          0.0006678 vs 0.0006678
    heat         0.0006678 vs 0.0006678
```
