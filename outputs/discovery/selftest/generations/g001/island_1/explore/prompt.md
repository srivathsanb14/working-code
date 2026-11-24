# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 1.  Operator: EXPLORE.

Programs on this island so far (best first):
- seed_cf_network [g000_i1_0003]: score 2.2602; objectives main 2.183, fast 1.807, slow 2.384, exterior 2.339, interior 1.253, passive 3.050, driven 0.814, complexity 772; Starting model: the audited room network, written as coefficient formulas of the fixed state-space function.
- seed_cf_history [g000_i1_0001]: score 2.8173; objectives main 2.744, fast 2.591, slow 2.926, exterior 3.098, interior 1.546, passive 3.720, driven 1.685, complexity 731; Starting model: room air plus a slab store, with the outside-wall U-value computed from the measured history.
- seed_cf_one_node [g000_i1_0005]: score 2.8151; objectives main 2.799, fast 2.230, slow 3.475, exterior 3.052, interior 3.277, passive 3.879, driven 2.392, complexity 156; Starting model: one temperature per room (its air), nominal materials, nothing read from history.

Write 1 candidates with a STRUCTURALLY different hypothesis from all of the above (a different use of the stores, a different way to infer per-building materials from history, different coupling laws), not a re-tuning.
