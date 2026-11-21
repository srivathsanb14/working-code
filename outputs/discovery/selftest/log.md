# Discovery log

One line per LLM request and per candidate. Score = held-out rollout RMSE (K) at the main horizon + complexity price; lower is better. Δ = change against the best parent.

ceiling (diagnostic): per-room linear model fitted on the first half of each held-out episode

## Generation 0: seeds (3 programs, copied into each of 2 islands)

- [g000_i0_0000] seed seed_cf_history_slab -> score 2.5778 (1 h 2.339, 3 h 2.498, 6 h 2.326 K)
- [g000_i1_0001] seed seed_cf_history_slab -> score 2.5778 (1 h 2.339, 3 h 2.498, 6 h 2.326 K)
- [g000_i0_0002] seed seed_cf_network_observer -> score 2.4331 (1 h 1.961, 3 h 2.268, 6 h 2.512 K)
- [g000_i1_0003] seed seed_cf_network_observer -> score 2.4331 (1 h 1.961, 3 h 2.268, 6 h 2.512 K)
- [g000_i0_0004] seed seed_cf_one_node -> score 2.9331 (1 h 2.273, 3 h 3.340, 6 h 3.628 K)
- [g000_i1_0005] seed seed_cf_one_node -> score 2.9331 (1 h 2.273, 3 h 3.340, 6 h 3.628 K)

## Generation 1

- island 0 **mutation** of g000_i0_0000: asked for 2, got 2 ([prompt](generations/g001/island_0/mutation/prompt.md), [analysis](generations/g001/island_0/mutation/analysis.md))
- island 0 **crossover** of g000_i0_0002, g000_i0_0000: asked for 2, got 2 ([prompt](generations/g001/island_0/crossover/prompt.md), [analysis](generations/g001/island_0/crossover/analysis.md))
- island 0 **explore**: asked for 1, got 1 ([prompt](generations/g001/island_0/explore/prompt.md))
- island 1 **mutation** of g000_i1_0003: asked for 2, got 2 ([prompt](generations/g001/island_1/mutation/prompt.md), [analysis](generations/g001/island_1/mutation/analysis.md))
- island 1 **crossover** of g000_i1_0001, g000_i1_0003: asked for 2, got 2 ([prompt](generations/g001/island_1/crossover/prompt.md), [analysis](generations/g001/island_1/crossover/analysis.md))
- island 1 **explore**: asked for 1, got 1 ([prompt](generations/g001/island_1/explore/prompt.md))
  - skip duplicate mock_cro_g1_1
  - [g001_i0_0006] island 0 mutation mock_mut_g1_0 -> score 2.3663 (1 h 2.136, 3 h 2.346, 6 h 2.126 K), Δ -0.2115
  - [g001_i0_0007] island 0 mutation mock_mut_g1_1 -> score 2.5799 (1 h 2.341, 3 h 2.500, 6 h 2.329 K), Δ +0.0021
  - [g001_i0_0008] island 0 crossover mock_cro_g1_0 -> score 2.4331 (1 h 1.961, 3 h 2.268, 6 h 2.512 K), Δ +0.0000
  - [g001_i0_0009] island 0 explore mock_exp_g1_0 -> score 2.7265 (1 h 2.183, 3 h 2.986, 6 h 3.199 K)
  - [g001_i1_0010] island 1 mutation mock_mut_g1_0 -> score 2.4331 (1 h 1.961, 3 h 2.268, 6 h 2.512 K), Δ +0.0000
  - [g001_i1_0011] island 1 mutation mock_mut_g1_1 -> SCREENED at stage 1: one-step RMSE 1.0871 K is more than 5.0x persistence (0.2169 K) on held-out buildings
  - [g001_i1_0012] island 1 crossover mock_cro_g1_0 -> ERROR at stage ?: NameError: name 'RC_MASS' is not defined
  - [g001_i1_0013] island 1 crossover mock_cro_g1_1 -> score 2.5778 (1 h 2.339, 3 h 2.498, 6 h 2.326 K), Δ +0.1447
  - [g001_i1_0014] island 1 explore mock_exp_g1_0 -> score 2.2714 (1 h 1.775, 3 h 2.099, 6 h 2.340 K)
  (9 candidates evaluated in 54 s with 1 worker(s))
- migration: [g001_i0_0006] mock_mut_g1_0 (score 2.3663) island 0 -> island 1 as [g001_i1_0015]
- migration: [g001_i1_0014] mock_exp_g1_0 (score 2.2714) island 1 -> island 0 as [g001_i0_0016]

## Generation 2

- island 0 **mutation** of g001_i0_0008: asked for 2, got 2 ([prompt](generations/g002/island_0/mutation/prompt.md), [analysis](generations/g002/island_0/mutation/analysis.md))
- island 0 **crossover** of g001_i0_0006, g001_i0_0016: asked for 2, got 2 ([prompt](generations/g002/island_0/crossover/prompt.md), [analysis](generations/g002/island_0/crossover/analysis.md))
- island 0 **explore**: asked for 1, got 1 ([prompt](generations/g002/island_0/explore/prompt.md))
- island 1 **mutation** of g001_i1_0010: asked for 2, got 2 ([prompt](generations/g002/island_1/mutation/prompt.md), [analysis](generations/g002/island_1/mutation/analysis.md))
- island 1 **crossover** of g001_i1_0010, g001_i1_0014: asked for 2, got 2 ([prompt](generations/g002/island_1/crossover/prompt.md), [analysis](generations/g002/island_1/crossover/analysis.md))
- island 1 **explore**: asked for 1, got 1 ([prompt](generations/g002/island_1/explore/prompt.md))
  - skip duplicate mock_cro_g2_0
  - skip duplicate mock_cro_g2_1
  - [g002_i0_0017] island 0 mutation mock_mut_g2_0 -> score 2.1245 (1 h 1.658, 3 h 1.930, 6 h 2.130 K), Δ -0.3086
  - [g002_i0_0018] island 0 mutation mock_mut_g2_1 -> score 2.4331 (1 h 1.961, 3 h 2.268, 6 h 2.512 K), Δ +0.0000
  - [g002_i0_0019] island 0 explore mock_exp_g2_0 -> score 2.6515 (1 h 2.415, 3 h 2.555, 6 h 2.392 K)
  - [g002_i1_0020] island 1 mutation mock_mut_g2_0 -> score 2.4202 (1 h 1.949, 3 h 2.254, 6 h 2.495 K), Δ -0.0129
  - [g002_i1_0021] island 1 mutation mock_mut_g2_1 -> score 2.4812 (1 h 2.006, 3 h 2.321, 6 h 2.572 K), Δ +0.0481
  - [g002_i1_0022] island 1 crossover mock_cro_g2_0 -> score 2.3925 (1 h 1.923, 3 h 2.224, 6 h 2.461 K), Δ +0.1212
  - [g002_i1_0023] island 1 crossover mock_cro_g2_1 -> score 2.2701 (1 h 1.777, 3 h 2.097, 6 h 2.336 K), Δ -0.0013
  - [g002_i1_0024] island 1 explore mock_exp_g2_0 -> score 2.2975 (1 h 2.109, 3 h 2.339, 6 h 2.035 K)
  (8 candidates evaluated in 45 s with 1 worker(s))
- migration: [g002_i0_0017] mock_mut_g2_0 (score 2.1245) island 0 -> island 1 as [g002_i1_0025]
