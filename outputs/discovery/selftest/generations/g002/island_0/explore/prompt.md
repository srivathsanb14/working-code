# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 2, island 0.  Operator: EXPLORE.

Programs on this island so far (best first):
- seed_cf_network [g000_i0_0002]: score 2.2602; objectives main 2.183, fast 1.807, slow 2.384, exterior 2.339, interior 1.253, passive 3.050, driven 0.814, complexity 772; Starting model: the audited room network, written as coefficient formulas of the fixed state-space function.
- seed_cf_history [g000_i0_0000]: score 2.8173; objectives main 2.744, fast 2.591, slow 2.926, exterior 3.098, interior 1.546, passive 3.720, driven 1.685, complexity 731; Starting model: room air plus a slab store, with the outside-wall U-value computed from the measured history.
- mock_mut_g1_0 [g001_i0_0006]: score 2.6443; objectives main 2.629, fast 2.165, slow 3.070, exterior 2.814, interior 2.807, passive 2.923, driven 2.726, complexity 156; [mock mutation, no LLM] rescaled 2 -> 1.6, 25 -> 15 in seed_cf_one_node
- mock_exp_g1_0 [g001_i0_0009]: score 2.6475; objectives main 2.632, fast 2.241, slow 2.755, exterior 2.699, interior 2.405, passive 2.109, driven 2.944, complexity 156; [mock explore, no LLM] rescaled 2.5 -> 3.125, 2 -> 1.2, 25 -> 40 in seed_cf_one_node
- mock_mut_g1_1 [g001_i0_0007]: score 2.8048; objectives main 2.789, fast 2.293, slow 3.436, exterior 3.161, interior 3.028, passive 3.732, driven 2.575, complexity 156; [mock mutation, no LLM] rescaled 2.5 -> 1.5 in seed_cf_one_node
- seed_cf_one_node [g000_i0_0004]: score 2.8151; objectives main 2.799, fast 2.230, slow 3.475, exterior 3.052, interior 3.277, passive 3.879, driven 2.392, complexity 156; Starting model: one temperature per room (its air), nominal materials, nothing read from history.

Write 1 candidates with a STRUCTURALLY different hypothesis from all of the above (a different use of the stores, a different way to infer per-building materials from history, different coupling laws), not a re-tuning.
