# Discovery log

One line per LLM request and per candidate. Score = held-out rollout RMSE (K) at the main horizon + complexity price; lower is better. Δ = change against the best parent.

ceiling (diagnostic): per-room linear model fitted on the first half of each held-out episode

## Generation 0: seeds (2 programs, copied into each of 2 islands)

- [g000_i0_0000] seed seed_cf_history_slab -> score 2.5772 (1 h 2.339, 3 h 2.498, 6 h 2.326 K)
- [g000_i1_0001] seed seed_cf_history_slab -> score 2.5772 (1 h 2.339, 3 h 2.498, 6 h 2.326 K)
- [g000_i0_0002] seed seed_cf_one_node -> score 3.3729 (1 h 2.273, 3 h 3.340, 6 h 3.628 K)
- [g000_i1_0003] seed seed_cf_one_node -> score 3.3729 (1 h 2.273, 3 h 3.340, 6 h 3.628 K)

## Generation 1

- island 0 **mutation** of g000_i0_0000: asked for 2, got 2 ([prompt](generations/g001/island_0/mutation/prompt.md), [analysis](generations/g001/island_0/mutation/analysis.md))
  - [g001_i0_0004] mock_mut_g1_0 -> score 2.4253 (1 h 2.136, 3 h 2.346, 6 h 2.126 K), Δ -0.1519  [10.8 s]
  - [g001_i0_0005] mock_mut_g1_1 -> score 2.5791 (1 h 2.341, 3 h 2.500, 6 h 2.329 K), Δ +0.0018  [10.8 s]
- island 0 **crossover** of g000_i0_0000, g000_i0_0002: asked for 2, got 2 ([prompt](generations/g001/island_0/crossover/prompt.md), [analysis](generations/g001/island_0/crossover/analysis.md))
  - [g001_i0_0006] mock_cro_g1_0 -> score 2.5772 (1 h 2.339, 3 h 2.498, 6 h 2.326 K), Δ +0.0000  [10.9 s]
  - skip duplicate mock_cro_g1_1
- island 0 **explore**: asked for 1, got 1 ([prompt](generations/g001/island_0/explore/prompt.md))
  - [g001_i0_0007] mock_exp_g1_0 -> score 3.8577 (1 h 2.466, 3 h 3.825, 6 h 4.165 K)  [0.1 s]
- island 1 **mutation** of g000_i1_0001: asked for 2, got 2 ([prompt](generations/g001/island_1/mutation/prompt.md), [analysis](generations/g001/island_1/mutation/analysis.md))
  - [g001_i1_0008] mock_mut_g1_0 -> score 2.6587 (1 h 2.399, 3 h 2.579, 6 h 2.453 K), Δ +0.0815  [10.8 s]
  - [g001_i1_0009] mock_mut_g1_1 -> score 2.5782 (1 h 2.340, 3 h 2.499, 6 h 2.327 K), Δ +0.0009  [10.8 s]
- island 1 **crossover** of g000_i1_0001, g000_i1_0003: asked for 2, got 2 ([prompt](generations/g001/island_1/crossover/prompt.md), [analysis](generations/g001/island_1/crossover/analysis.md))
  - [g001_i1_0010] mock_cro_g1_0 -> score 2.9850 (1 h 1.954, 3 h 2.906, 6 h 3.212 K), Δ +0.4078  [0.2 s]
  - [g001_i1_0011] mock_cro_g1_1 -> REJECTED at stage 0: physics checks failed  [0.0 s]
- island 1 **explore**: asked for 1, got 1 ([prompt](generations/g001/island_1/explore/prompt.md))
  - [g001_i1_0012] mock_exp_g1_0 -> score 2.6539 (1 h 2.429, 3 h 2.575, 6 h 2.422 K)  [10.8 s]
- migration: [g001_i0_0004] mock_mut_g1_0 (score 2.4253) island 0 -> island 1 as [g001_i1_0013]

## Generation 2

- island 0 **mutation** of g001_i0_0004: asked for 2, got 2 ([prompt](generations/g002/island_0/mutation/prompt.md), [analysis](generations/g002/island_0/mutation/analysis.md))
  - [g002_i0_0014] mock_mut_g2_0 -> score 2.4830 (1 h 2.220, 3 h 2.404, 6 h 2.203 K), Δ +0.0577  [10.7 s]
  - [g002_i0_0015] mock_mut_g2_1 -> score 2.4309 (1 h 2.140, 3 h 2.352, 6 h 2.134 K), Δ +0.0056  [10.7 s]
- island 0 **crossover** of g001_i0_0005, g000_i0_0002: asked for 2, got 2 ([prompt](generations/g002/island_0/crossover/prompt.md), [analysis](generations/g002/island_0/crossover/analysis.md))
  - [g002_i0_0016] mock_cro_g2_0 -> REJECTED at stage 0: physics checks failed  [0.0 s]
  - [g002_i0_0017] mock_cro_g2_1 -> REJECTED at stage 0: physics checks failed  [0.0 s]
- island 0 **explore**: asked for 1, got 1 ([prompt](generations/g002/island_0/explore/prompt.md))
  - [g002_i0_0018] mock_exp_g2_0 -> score 2.7190 (1 h 2.492, 3 h 2.640, 6 h 2.477 K)  [11.3 s]
- island 1 **mutation** of g001_i1_0009: asked for 2, got 2 ([prompt](generations/g002/island_1/mutation/prompt.md), [analysis](generations/g002/island_1/mutation/analysis.md))
  - [g002_i1_0019] mock_mut_g2_0 -> score 2.6157 (1 h 2.368, 3 h 2.536, 6 h 2.385 K), Δ +0.0376  [10.8 s]
  - [g002_i1_0020] mock_mut_g2_1 -> score 2.6242 (1 h 2.390, 3 h 2.545, 6 h 2.344 K), Δ +0.0460  [10.9 s]
- island 1 **crossover** of g000_i1_0001, g001_i1_0013: asked for 2, got 2 ([prompt](generations/g002/island_1/crossover/prompt.md), [analysis](generations/g002/island_1/crossover/analysis.md))
  - [g002_i1_0021] mock_cro_g2_0 -> score 2.5831 (1 h 2.344, 3 h 2.504, 6 h 2.335 K), Δ +0.1578  [10.8 s]
  - skip duplicate mock_cro_g2_1
- island 1 **explore**: asked for 1, got 1 ([prompt](generations/g002/island_1/explore/prompt.md))
  - [g002_i1_0022] mock_exp_g2_0 -> score 2.3400 (1 h 2.039, 3 h 2.261, 6 h 2.015 K)  [10.7 s]
- migration: [g002_i1_0022] mock_exp_g2_0 (score 2.3400) island 1 -> island 0 as [g002_i0_0023]
