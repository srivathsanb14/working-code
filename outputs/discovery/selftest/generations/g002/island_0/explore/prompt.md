# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 2, island 0.  Operator: EXPLORE.

Programs on this island so far (best first):
- mock_mut_g1_0 [g001_i0_0004]: score 2.4253; [mock mutation, no LLM] rescaled 25 -> 20, 4 -> 2.4 in seed_cf_history_slab
- seed_cf_history_slab [g000_i0_0000]: score 2.5772; Room air plus a slow store summarised from history, and the facade U-value inferred from history.
- mock_cro_g1_0 [g001_i0_0006]: score 2.5772; [mock crossover, no LLM] seed_cf_history_slab with nothing taken from seed_cf_one_node
- mock_mut_g1_1 [g001_i0_0005]: score 2.5791; [mock mutation, no LLM] rescaled 2 -> 1.2 in seed_cf_history_slab
- seed_cf_one_node [g000_i0_0002]: score 3.3729; One temperature per room, nominal materials, no history beyond the last measurement.
- mock_exp_g1_0 [g001_i0_0007]: score 3.8577; [mock explore, no LLM] rescaled 25 -> 40, 2 -> 2.5, 2.5 -> 4, 1e-09 -> 6e-10 in seed_cf_one_node

Write 1 candidates with a STRUCTURALLY different hypothesis from all of the above (a different slow-store summary, a different way to infer per-building materials from history, different coupling laws), not a re-tuning.
