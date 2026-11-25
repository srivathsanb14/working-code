# Planning: from generating the data to the final submission

Status, 25 Nov 2026: the pipeline, scorer, split, LLM loop, policy runtime and submission package are built and tested
(closed-form only; 55 tests), and the whole chain runs end to end on real kit output. Not done yet: the data beyond the
kit's 30 shipped scenarios, and any real API generation.

**How to use this folder.** Do the steps in order. Each file says what it **needs**, what it **produces**, the exact
commands (they use the paths set in 01 step 1), a checklist, and a **handoff**: the checks that must pass before the next
step starts. Tick the boxes as you go. Installing the repository is in `docs/RUNBOOK.md` §0–1.

## The steps

| # | Step | Needs | Produces | Rough time |
|---|---|---|---|---|
| 01 | [Stage 1 data: 288 scenarios, 1,440 runs, checked against the kit's baseline scores](01_stage1_kit_data.md) | — | `$D/scenarios` (288), `$D/results` (1,440 runs) | 10–20 h of compute |
| 02 | [Build the episodes, compute the split, audit everything](02_episodes_split_audit.md) | 01 | `$D/dataset` with `index.json`, `split.json`; audit report | about 1 h |
| 03 | [A free check of the data, a read of the exact prompts, then the API](03_api_and_prompts.md) | 02 | baseline scores, reviewed prompts, one paid request, budget | half a day |
| 04 | [First real discovery run (small)](04_first_api_run.md) | 03 | `outputs/discovery/g1/` + notes; go / no-go | a few hours |
| 05 | [Stage 2 data: new buildings](05_stage2_new_buildings.md) | 02; footprint decision | about 600 more runs; dataset rebuilt, re-split, re-audited | in the background during 03–04 |
| 06 | [Full discovery run, then the test tiers once](06_full_discovery_run.md) | 04 (go), 05 | `best.py`, `REPORT.md`, test-tier results | many hours |
| 07 | [Search improvements, only where 04 shows a need](07_search_improvements.md) | 04 | targeted fixes | M each |
| 08 | [Per-step time guard and large buildings](08_time_guard_and_large_buildings.md) | 05 | guarded submission policy | 1–2 days |
| 09 | [Detect HVAC variants we do not model, and fall back](09_hvac_variants.md) | — | fallback, documented | 1–2 days |
| 10 | [Benchmark the controllers](10_controller_benchmark.md) | 06, 08 | results table | a day of compute |
| 11 | [Submit the safe pid_water notebook to Kaggle](11_kaggle_baseline_submission.md) | — | a public score | 1 hour |
| 12 | [Package the discovered model and pass the readiness gate](12_package_and_readiness_gate.md) | 06, 08, 09, 10, 11 | model-based submission | half a day |
| 13 | [Robustness checks on assumptions](13_robustness_checks.md) | 06, 10 | degradation table, decisions | M |
| 14 | [Choose the two final submissions, and write the paper](14_final_submissions_and_paper.md) | 06, 10, 12 | final picks, paper | by 15 Jan 2027 |

## How the steps hand off

```
01 stage-1 data ─► 02 episodes + split + audit ─► 03 free check, prompts, API ─► 04 first run ─┐
                   02 done ─► 05 stage-2 data (background) ─► 02 again on combined data ───────┴─► 06 full run
06 ─► 10 benchmark (needs 08) ─► 12 gate (needs 09, 11) ─► 14 final          09 and 11: any time
```

**Where things live:** the kit and this repository side by side; all data under `$D` (`scenarios/`, `results/`,
`logs/`, `buildings/` for stage 2, `dataset/` with `index.json` and `split.json`, `audit/`, `probe/` for throw-away
tests); discovery runs under `working-code/outputs/discovery/<run>/`.

**Rules that hold throughout:** run every kit command from the kit's root and give it an output folder under `$D` (the
kit is never written to); never move `$D/scenarios` after generating it; nothing is fitted; the test tiers are read once.

**Deadlines:** entry and team merger 3 Jan 2027; final submission and paper 15 Jan 2027.
