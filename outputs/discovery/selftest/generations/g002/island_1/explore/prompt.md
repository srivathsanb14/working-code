# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 2, island 1.  Operator: EXPLORE.

Programs on this island so far (best first):
- mock_mut_g1_0 [g001_i1_0013]: score 2.4253; [mock mutation, no LLM] rescaled 25 -> 20, 4 -> 2.4 in seed_cf_history_slab
- seed_cf_history_slab [g000_i1_0001]: score 2.5772; Room air plus a slow store summarised from history, and the facade U-value inferred from history.
- mock_mut_g1_1 [g001_i1_0009]: score 2.5782; [mock mutation, no LLM] rescaled 8 -> 6.4, 2 -> 1.6 in seed_cf_history_slab
- mock_exp_g1_0 [g001_i1_0012]: score 2.6539; [mock explore, no LLM] rescaled 4 -> 5, 8 -> 10, 1e-09 -> 8e-10, 25 -> 40 in seed_cf_history_slab
- mock_mut_g1_0 [g001_i1_0008]: score 2.6587; [mock mutation, no LLM] rescaled 2.5 -> 1.5 in seed_cf_history_slab
- mock_cro_g1_0 [g001_i1_0010]: score 2.9850; [mock crossover, no LLM] seed_cf_history_slab with HISTORY taken from seed_cf_one_node

Recent failures on this island (do not repeat):
- mock_cro_g1_1 [rejected]: physics checks failed
    exception during checks: KeyError: 'G_slow'

Write 1 candidates with a STRUCTURALLY different hypothesis from all of the above (a different slow-store summary, a different way to infer per-building materials from history, different coupling laws), not a re-tuning.
