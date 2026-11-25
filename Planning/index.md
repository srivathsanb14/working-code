# Planning: what is left before the final solution

Each file below is one pending task, with its goal, the exact commands or code to change, and when it counts as done.
The order is roughly the order to do them in; tasks 13 and 14 do not wait for anything and can start now.

Status as of 25 Nov 2026: the pipeline, scorer, split, LLM loop and submission code are built and tested (closed-form
only: the earlier fitted code has been removed); no data
beyond the kit's 30 shipped scenarios has been generated, and no real API generation has run.

| # | Task | Phase | Priority | Depends on | Status |
|---|---|---|---|---|---|
| 01 | [Stage 1 data: every shipped building in every weather window](01_stage1_kit_data.md) | Data | P0 | — | pending |
| 02 | [Stage 2 data: new buildings from the kit's generator](02_stage2_new_buildings.md) | Data | P1 | 01 | pending |
| 03 | [Build the episodes and audit them](03_dataset_build_and_audit.md) | Dataset | P0 | 01 (02) | pending |
| 04 | [Compute the split on the full data and review it](04_split_review.md) | Dataset | P0 | 03 | pending (code change) |
| 05 | [Review the universal constants once](05_universal_constants_review.md) | Setup | P0 | — | partial |
| 06 | [API access, budget, and a review of the exact prompts](06_api_setup_and_prompt_review.md) | Discovery | P0 | 03, 05 | pending (code change) |
| 07 | [First real API run (small)](07_first_api_run.md) | Discovery | P0 | 04, 06 | pending |
| 08 | [Full discovery run, then the test tiers once](08_full_discovery_run.md) | Discovery | P0 | 07 (09) | pending |
| 09 | [Search improvements, only where the first run shows a need](09_search_improvements.md) | Discovery | P1–P3 | 07 | pending |
| 10 | [Make candidate scoring fast enough for the full dataset](10_scoring_speed.md) | Discovery | P0 | 03 | partial |
| 11 | [Benchmark the controllers on all 30 scenarios and held-out buildings](11_controller_benchmark.md) | Control | P0 | 08, 12 | pending |
| 12 | [Per-step time guard and large buildings](12_time_budget_and_large_buildings.md) | Control | P0 | 02 | partial |
| 13 | [Detect HVAC variants we do not model, and fall back](13_hvac_variants.md) | Control | P0 | — | pending |
| 14 | [Submit the safe pid_water notebook to Kaggle](14_kaggle_baseline_submission.md) | Submission | P0 | — | partial |
| 15 | [Package the discovered model and pass the readiness gate](15_package_and_readiness_gate.md) | Submission | P0 | 08, 11, 12, 13, 14 | partial |
| 16 | [Robustness checks on assumptions](16_robustness_checks.md) | Validation | P2 | 08, 11 | pending |
| 17 | [Choose the two final submissions, and write the paper](17_final_submissions_and_paper.md) | Final | P0 / P1 | 08, 11, 15 | pending |

**Critical path:** 01 → 03 → 04 → 06 → 07 → 08 → 11 → 15 → 17. Tasks 05, 13 and 14 can run alongside it now.

**Deadlines:** entry and team merger 3 Jan 2027; final submission and paper 15 Jan 2027.

This folder is the only task list; the old backlog was removed.
