# Pre-crossover analysis: A = g000_i0_0002 seed_cf_network, B = g000_i0_0000 seed_cf_history

```
WHERE AND WHEN EACH PARENT IS BETTER (A vs B, rollout RMSE, K)
  SPATIAL
    exterior rooms                               0.732 vs 1.496  -> A
    interior rooms                               0.814 vs 0.926  -> A
    corner rooms (outside wall on 2+ sides)      0.758 vs 1.668  -> A
    outside wall on the top side of the plan     0.704 vs 1.549  -> A
    outside wall on the bottom side of the plan  0.732 vs 1.564  -> A
    outside wall on the left side of the plan    0.739 vs 1.539  -> A
    outside wall on the right side of the plan   0.728 vs 1.515  -> A
    floor 1                                      0.751 vs 1.298  -> A
    floor 2                                      0.765 vs 1.346  -> A
    floor 3                                      0.815 vs 1.522  -> A
    floor 4                                      0.550 vs 1.473  -> A
  TEMPORAL
    target hour 00-04                            0.756 vs 1.392  -> A
    target hour 04-08                            0.870 vs 1.263  -> A
    target hour 08-12                            0.842 vs 1.624  -> A
    target hour 12-16                            0.715 vs 1.585  -> A
    target hour 16-20                            0.649 vs 1.102  -> A
    target hour 20-24                            0.655 vs 1.293  -> A
    occupied (comfort) hours                     0.727 vs 1.507  -> A
    nights and weekends                          0.767 vs 1.309  -> A
    HVAC air on at the target                    0.582 vs 1.018  -> A
    HVAC air off at the target                   0.773 vs 1.434  -> A
    HVAC switched off during the rollout         0.752 vs 1.263  -> A
    HVAC switched on during the rollout          0.651 vs 1.295  -> A
    colder half of outside temperatures          0.852 vs 1.417  -> A
    warmer half of outside temperatures          0.644 vs 1.361  -> A
    first 3 days of the episode                  0.624 vs 1.638  -> A
    after day 3                                  0.763 vs 1.360  -> A
  A is clearly better everywhere: combine by taking A's formulas and only what the TERMS lines favour from the other
  per room: A better in 320 of 338 validation rooms; rmse 0.751 vs 1.388, best-of-both per room 0.738 (the most a combination that keeps each parent's good rooms could reach)
TERMS (one-step residual slope per term; nearer 0 = that term's formula is closer to right)
    air         +0.015 vs -0.187   -> take 'air' from A
    neighbours  -0.742 vs -0.938   -> take 'neighbours' from A
    slab        +0.418 vs +0.173   -> take 'slab' from B
    heat        -0.001 vs -0.341   -> take 'heat' from A
FUNCTIONS: same in both: none; different: coefficients; only in A: _spread; only in B: _geo, facade_u
COEFFICIENT VALUES on a validation building, mean over rooms (A vs B); most different first:
    outside      0 vs 0.02866
    wall         0.1617 vs 0
    wall_back    0.01449 vs 0
    wall_out     0.01315 vs 0
    inner        0.286 vs 0
    inner_back   0.05415 vs 0
    slab_back    0.025 vs 0.01389
    slab         0.4146 vs 0.3317
    neighbours   0.2195 vs 0.1932
    air          0.0006831 vs 0.0006831
    heat         0.0006831 vs 0.0006831
```
