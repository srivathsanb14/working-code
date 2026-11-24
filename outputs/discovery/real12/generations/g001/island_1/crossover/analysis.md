# Pre-crossover analysis: A = g000_i1_0001 seed_cf_history, B = g000_i1_0003 seed_cf_network

```
WHERE AND WHEN EACH PARENT IS BETTER (A vs B, rollout RMSE, K)
  SPATIAL
    exterior rooms                               1.496 vs 0.732  -> B
    interior rooms                               0.926 vs 0.814  -> B
    corner rooms (outside wall on 2+ sides)      1.668 vs 0.758  -> B
    outside wall on the top side of the plan     1.549 vs 0.704  -> B
    outside wall on the bottom side of the plan  1.564 vs 0.732  -> B
    outside wall on the left side of the plan    1.539 vs 0.739  -> B
    outside wall on the right side of the plan   1.515 vs 0.728  -> B
    floor 1                                      1.298 vs 0.751  -> B
    floor 2                                      1.346 vs 0.765  -> B
    floor 3                                      1.522 vs 0.815  -> B
    floor 4                                      1.473 vs 0.550  -> B
  TEMPORAL
    target hour 00-04                            1.392 vs 0.756  -> B
    target hour 04-08                            1.263 vs 0.870  -> B
    target hour 08-12                            1.624 vs 0.842  -> B
    target hour 12-16                            1.585 vs 0.715  -> B
    target hour 16-20                            1.102 vs 0.649  -> B
    target hour 20-24                            1.293 vs 0.655  -> B
    occupied (comfort) hours                     1.507 vs 0.727  -> B
    nights and weekends                          1.309 vs 0.767  -> B
    HVAC air on at the target                    1.018 vs 0.582  -> B
    HVAC air off at the target                   1.434 vs 0.773  -> B
    HVAC switched off during the rollout         1.263 vs 0.752  -> B
    HVAC switched on during the rollout          1.295 vs 0.651  -> B
    colder half of outside temperatures          1.417 vs 0.852  -> B
    warmer half of outside temperatures          1.361 vs 0.644  -> B
    first 3 days of the episode                  1.638 vs 0.624  -> B
    after day 3                                  1.360 vs 0.763  -> B
  B is clearly better everywhere: combine by taking B's formulas and only what the TERMS lines favour from the other
  per room: A better in 18 of 338 validation rooms; rmse 1.388 vs 0.751, best-of-both per room 0.738 (the most a combination that keeps each parent's good rooms could reach)
TERMS (one-step residual slope per term; nearer 0 = that term's formula is closer to right)
    air         -0.187 vs +0.015   -> take 'air' from B
    neighbours  -0.938 vs -0.742   -> take 'neighbours' from B
    slab        +0.173 vs +0.418   -> take 'slab' from A
    heat        -0.341 vs -0.001   -> take 'heat' from B
FUNCTIONS: same in both: none; different: coefficients; only in A: _geo, facade_u; only in B: _spread
COEFFICIENT VALUES on a validation building, mean over rooms (A vs B); most different first:
    outside      0.02866 vs 0
    wall         0 vs 0.1617
    wall_back    0 vs 0.01449
    wall_out     0 vs 0.01315
    inner        0 vs 0.286
    inner_back   0 vs 0.05415
    slab_back    0.01389 vs 0.025
    slab         0.3317 vs 0.4146
    neighbours   0.1932 vs 0.2195
    air          0.0006831 vs 0.0006831
    heat         0.0006831 vs 0.0006831
```
