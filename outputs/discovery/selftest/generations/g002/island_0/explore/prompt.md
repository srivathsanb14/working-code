# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 2, island 0.  Operator: EXPLORE.

Programs on this island so far (best first):
- mock_exp_g1_0 [g001_i0_0016]: score 2.2714; [mock explore, no LLM] rescaled 900 -> 540, 0.6 -> 0.36, 50 -> 30, 17.75 -> 22.1875 in seed_cf_network_observer
- mock_mut_g1_0 [g001_i0_0006]: score 2.3663; [mock mutation, no LLM] rescaled 25 -> 20, 4 -> 2.4 in seed_cf_history_slab
- seed_cf_network_observer [g000_i0_0002]: score 2.4331; Stateful seed: the audited room network (air, slab, facade, partitions, doors) with a measurement-corrected state.
- mock_cro_g1_0 [g001_i0_0008]: score 2.4331; [mock crossover, no LLM] seed_cf_network_observer with nothing taken from seed_cf_history_slab
- seed_cf_history_slab [g000_i0_0000]: score 2.5778; Room air plus a slow store summarised from history, and the facade U-value inferred from history.
- mock_mut_g1_1 [g001_i0_0007]: score 2.5799; [mock mutation, no LLM] rescaled 2 -> 1.2 in seed_cf_history_slab

Write 1 candidates with a STRUCTURALLY different hypothesis from all of the above (a different slow-store summary, a different way to infer per-building materials from history, different coupling laws), not a re-tuning.
