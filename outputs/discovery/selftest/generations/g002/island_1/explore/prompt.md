# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 2, island 1.  Operator: EXPLORE.

Programs on this island so far (best first):
- mock_exp_g1_0 [g001_i1_0014]: score 2.2714; [mock explore, no LLM] rescaled 900 -> 540, 0.6 -> 0.36, 50 -> 30, 17.75 -> 22.1875 in seed_cf_network_observer
- mock_mut_g1_0 [g001_i1_0015]: score 2.3663; [mock mutation, no LLM] rescaled 25 -> 20, 4 -> 2.4 in seed_cf_history_slab
- seed_cf_network_observer [g000_i1_0003]: score 2.4331; Stateful seed: the audited room network (air, slab, facade, partitions, doors) with a measurement-corrected state.
- mock_mut_g1_0 [g001_i1_0010]: score 2.4331; [mock mutation, no LLM] rescaled 1 -> 1.25 in seed_cf_network_observer
- seed_cf_history_slab [g000_i1_0001]: score 2.5778; Room air plus a slow store summarised from history, and the facade U-value inferred from history.
- mock_cro_g1_1 [g001_i1_0013]: score 2.5778; [mock crossover, no LLM] seed_cf_history_slab with nothing taken from seed_cf_network_observer

Recent failures on this island (do not repeat):
- mock_mut_g1_1 [screened]: one-step RMSE 1.0871 K is more than 5.0x persistence (0.2169 K) on held-out buildings
- mock_cro_g1_0 [error]: NameError: name 'RC_MASS' is not defined

Write 1 candidates with a STRUCTURALLY different hypothesis from all of the above (a different slow-store summary, a different way to infer per-building materials from history, different coupling laws), not a re-tuning.
