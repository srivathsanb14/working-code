# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 0.  Operator: EXPLORE.

Programs on this island so far (best first):
- seed_cf_network [g000_i0_0002]: score 0.8174; objectives main 0.817, fast 0.544, slow 0.889, exterior 0.732, interior 0.814, passive 0.773, driven 0.582, complexity 772; Starting model: the audited room network, written as coefficient formulas of the fixed state-space function.
- seed_cf_history [g000_i0_0000]: score 1.1724; objectives main 1.172, fast 1.036, slow 1.438, exterior 1.496, interior 0.926, passive 1.434, driven 1.018, complexity 731; Starting model: room air plus a slab store, with the outside-wall U-value computed from the measured history.
- seed_cf_one_node [g000_i0_0004]: score 5.7230; objectives main 5.723, fast 3.282, slow 7.277, exterior 6.460, interior 7.200, passive 6.985, driven 3.265, complexity 156; Starting model: one temperature per room (its air), nominal materials, nothing read from history.

Write 1 candidates with a STRUCTURALLY different hypothesis from all of the above (a different use of the stores, a different way to infer per-building materials from history, different coupling laws), not a re-tuning.
