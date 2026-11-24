# Discovery log

One line per LLM request and per candidate. Everything is measured on the VALIDATION split (no training split; the test tiers are scored once at the end). Score = validation rollout RMSE (K) at the main horizon + complexity price; selection is by Pareto rank over the objectives unless --selection scalar. Δ = change in score against the best parent.

validation: 6 episodes (the only data the search sees); held out until the end: test_medium 4, test_station 2

## Generation 0: seeds (3 programs, copied into each of 2 islands)

- [g000_i0_0000] seed seed_cf_history -> score 1.1724 (1 h 1.036, 3 h 1.235, 6 h 1.438 K)
- [g000_i1_0001] seed seed_cf_history -> score 1.1724 (1 h 1.036, 3 h 1.235, 6 h 1.438 K)
- [g000_i0_0002] seed seed_cf_network -> score 0.8174 (1 h 0.544, 3 h 0.698, 6 h 0.889 K)
- [g000_i1_0003] seed seed_cf_network -> score 0.8174 (1 h 0.544, 3 h 0.698, 6 h 0.889 K)
- [g000_i0_0004] seed seed_cf_one_node -> score 5.7230 (1 h 3.282, 3 h 6.047, 6 h 7.277 K)
- [g000_i1_0005] seed seed_cf_one_node -> score 5.7230 (1 h 3.282, 3 h 6.047, 6 h 7.277 K)

## Generation 1

- island 0 **mutation** of g000_i0_0000: asked for 2, got 2 ([prompt](generations/g001/island_0/mutation/prompt.md), [analysis](generations/g001/island_0/mutation/analysis.md))
- island 0 **crossover** of g000_i0_0002, g000_i0_0000: asked for 2, got 2 ([prompt](generations/g001/island_0/crossover/prompt.md), [analysis](generations/g001/island_0/crossover/analysis.md))
- island 0 **explore**: asked for 1, got 1 ([prompt](generations/g001/island_0/explore/prompt.md))
- island 1 **mutation** of g000_i1_0003: asked for 2, got 2 ([prompt](generations/g001/island_1/mutation/prompt.md), [analysis](generations/g001/island_1/mutation/analysis.md))
- island 1 **crossover** of g000_i1_0001, g000_i1_0003: asked for 2, got 2 ([prompt](generations/g001/island_1/crossover/prompt.md), [analysis](generations/g001/island_1/crossover/analysis.md))
- island 1 **explore**: asked for 1, got 1 ([prompt](generations/g001/island_1/explore/prompt.md))
  - skip duplicate mock_cro_g1_1
  - [g001_i0_0006] island 0 mutation mock_mut_g1_0 -> score 1.5652 (1 h 1.243, 3 h 1.642, 6 h 2.070 K), Δ +0.3928
  - [g001_i0_0007] island 0 mutation mock_mut_g1_1 -> score 1.1733 (1 h 1.037, 3 h 1.236, 6 h 1.439 K), Δ +0.0009
  - [g001_i0_0008] island 0 crossover mock_cro_g1_0 -> score 0.8174 (1 h 0.544, 3 h 0.698, 6 h 0.889 K), Δ +0.0000
  - [g001_i0_0009] island 0 explore mock_exp_g1_0 -> score 5.9478 (1 h 3.160, 3 h 6.218, 6 h 8.125 K)
  - [g001_i1_0010] island 1 mutation mock_mut_g1_0 -> score 0.7854 (1 h 0.535, 3 h 0.679, 6 h 0.857 K), Δ -0.0320
  - [g001_i1_0011] island 1 mutation mock_mut_g1_1 -> score 0.8130 (1 h 0.547, 3 h 0.699, 6 h 0.880 K), Δ -0.0044
  - [g001_i1_0012] island 1 crossover mock_cro_g1_0 -> score 1.0560 (1 h 0.947, 3 h 1.104, 6 h 1.261 K), Δ +0.2386
  - [g001_i1_0013] island 1 crossover mock_cro_g1_1 -> score 1.1327 (1 h 0.979, 3 h 1.146, 6 h 1.329 K), Δ +0.3153
  - [g001_i1_0014] island 1 explore mock_exp_g1_0 -> score 0.7928 (1 h 0.525, 3 h 0.673, 6 h 0.856 K)
  (9 candidates evaluated in 23 s with 4 worker(s))

## Held-out test tiers (scored once; never seen by the search)

| model | test_medium 3h (persistence) | test_station 3h (persistence) |
|---|---|---|
| **mock_mut_g1_0 [g001_i1_0010]** | 0.589 (0.948) | 2.016 (1.274) |
| mock_exp_g1_0 [g001_i1_0014] | 0.509 (0.948) | 2.057 (1.274) |
