# Discovery log

One line per LLM request and per candidate. Everything is measured on the VALIDATION split (no training split; the test tiers are scored once at the end). Score = validation rollout RMSE (K) at the main horizon + complexity price; selection is by Pareto rank over the objectives unless --selection scalar. Δ = change in score against the best parent.

validation: 10 episodes (the only data the search sees); held out until the end: test_medium 2
ceiling (diagnostic): per-room linear model fitted on the first half of each held-out episode

## Generation 0: seeds (3 programs, copied into each of 2 islands)

- [g000_i0_0000] seed seed_cf_history -> score 2.8173 (1 h 2.591, 3 h 2.808, 6 h 2.926 K)
- [g000_i1_0001] seed seed_cf_history -> score 2.8173 (1 h 2.591, 3 h 2.808, 6 h 2.926 K)
- [g000_i0_0002] seed seed_cf_network -> score 2.2602 (1 h 1.807, 3 h 2.127, 6 h 2.384 K)
- [g000_i1_0003] seed seed_cf_network -> score 2.2602 (1 h 1.807, 3 h 2.127, 6 h 2.384 K)
- [g000_i0_0004] seed seed_cf_one_node -> score 2.8151 (1 h 2.230, 3 h 3.214, 6 h 3.475 K)
- [g000_i1_0005] seed seed_cf_one_node -> score 2.8151 (1 h 2.230, 3 h 3.214, 6 h 3.475 K)

## Generation 1

- island 0 **mutation** of g000_i0_0004: asked for 2, got 2 ([prompt](generations/g001/island_0/mutation/prompt.md), [analysis](generations/g001/island_0/mutation/analysis.md))
- island 0 **crossover** of g000_i0_0004, g000_i0_0000: asked for 2, got 2 ([prompt](generations/g001/island_0/crossover/prompt.md), [analysis](generations/g001/island_0/crossover/analysis.md))
- island 0 **explore**: asked for 1, got 1 ([prompt](generations/g001/island_0/explore/prompt.md))
- island 1 **mutation** of g000_i1_0003: asked for 2, got 2 ([prompt](generations/g001/island_1/mutation/prompt.md), [analysis](generations/g001/island_1/mutation/analysis.md))
- island 1 **crossover** of g000_i1_0005, g000_i1_0003: asked for 2, got 2 ([prompt](generations/g001/island_1/crossover/prompt.md), [analysis](generations/g001/island_1/crossover/analysis.md))
- island 1 **explore**: asked for 1, got 1 ([prompt](generations/g001/island_1/explore/prompt.md))
  - skip duplicate mock_cro_g1_1
  - skip duplicate mock_cro_g1_1
  - [g001_i0_0006] island 0 mutation mock_mut_g1_0 -> score 2.6443 (1 h 2.165, 3 h 2.909, 6 h 3.070 K), Δ -0.1708
  - [g001_i0_0007] island 0 mutation mock_mut_g1_1 -> score 2.8048 (1 h 2.293, 3 h 3.222, 6 h 3.436 K), Δ -0.0103
  - [g001_i0_0008] island 0 crossover mock_cro_g1_0 -> score 2.8151 (1 h 2.230, 3 h 3.214, 6 h 3.475 K), Δ +0.0000
  - [g001_i0_0009] island 0 explore mock_exp_g1_0 -> score 2.6475 (1 h 2.241, 3 h 2.711, 6 h 2.755 K)
  - [g001_i1_0010] island 1 mutation mock_mut_g1_0 -> score 2.4487 (1 h 2.025, 3 h 2.336, 6 h 2.555 K), Δ +0.1885
  - [g001_i1_0011] island 1 mutation mock_mut_g1_1 -> score 2.2760 (1 h 1.831, 3 h 2.147, 6 h 2.383 K), Δ +0.0157
  - [g001_i1_0012] island 1 crossover mock_cro_g1_0 -> REJECTED at stage 0: physics checks failed
  - [g001_i1_0013] island 1 explore mock_exp_g1_0 -> SCREENED at stage 1: one-step RMSE 1.3540 K is more than 5.0x persistence (0.1897 K) on the validation buildings
  (8 candidates evaluated in 7 s with 1 worker(s))

## Generation 2

- island 0 **mutation** of g000_i0_0004: asked for 2, got 2 ([prompt](generations/g002/island_0/mutation/prompt.md), [analysis](generations/g002/island_0/mutation/analysis.md))
- island 0 **crossover** of g001_i0_0007, g000_i0_0002: asked for 2, got 2 ([prompt](generations/g002/island_0/crossover/prompt.md), [analysis](generations/g002/island_0/crossover/analysis.md))
- island 0 **explore**: asked for 1, got 1 ([prompt](generations/g002/island_0/explore/prompt.md))
- island 1 **mutation** of g001_i1_0011: asked for 2, got 2 ([prompt](generations/g002/island_1/mutation/prompt.md), [analysis](generations/g002/island_1/mutation/analysis.md))
- island 1 **crossover** of g000_i1_0005, g000_i1_0003: asked for 2, got 2 ([prompt](generations/g002/island_1/crossover/prompt.md), [analysis](generations/g002/island_1/crossover/analysis.md))
- island 1 **explore**: asked for 1, got 1 ([prompt](generations/g002/island_1/explore/prompt.md))
  - skip duplicate mock_cro_g2_0
  - skip duplicate mock_cro_g2_1
  - skip duplicate mock_cro_g2_0
  - skip duplicate mock_cro_g2_1
  - [g002_i0_0014] island 0 mutation mock_mut_g2_0 -> score 3.1170 (1 h 2.417, 3 h 3.627, 6 h 3.943 K), Δ +0.3020
  - [g002_i0_0015] island 0 mutation mock_mut_g2_1 -> score 2.6675 (1 h 2.226, 3 h 2.744, 6 h 2.793 K), Δ -0.1476
  - [g002_i0_0016] island 0 explore mock_exp_g2_0 -> score 2.8259 (1 h 2.391, 3 h 2.778, 6 h 2.805 K)
  - [g002_i1_0017] island 1 mutation mock_mut_g2_0 -> score 2.2619 (1 h 1.816, 3 h 2.132, 6 h 2.366 K), Δ -0.0140
  - [g002_i1_0018] island 1 mutation mock_mut_g2_1 -> SCREENED at stage 1: one-step RMSE 1.7909 K is more than 5.0x persistence (0.1897 K) on the validation buildings
  - [g002_i1_0019] island 1 explore mock_exp_g2_0 -> score 2.1163 (1 h 1.560, 3 h 1.957, 6 h 2.270 K)
  (6 candidates evaluated in 6 s with 1 worker(s))
- migration: [g002_i1_0019] mock_exp_g2_0 (score 2.1163) island 1 -> island 0 as [g002_i0_0020]
