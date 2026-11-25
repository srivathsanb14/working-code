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
| 01 | [The data: 288 scenarios, 1,440 runs, checked against the kit's baseline scores](01_data_288_scenarios.md) | — | `$D/scenarios` (288), `$D/results` (1,440 runs) | 10–20 h of compute |
| 02 | [Build the episodes, compute the split, audit everything](02_episodes_split_audit.md) | 01 | `$D/dataset` with `index.json` and `split.json`; audit report | about 1 h |
| 03 | [A free check of the data, a read of the exact prompts, then the API](03_api_and_prompts.md) | 02 | baseline scores, reviewed prompts, one paid request, budget | half a day |
| 04 | [First real discovery run (small)](04_first_api_run.md) | 03 | `outputs/discovery/g1/` + notes; go / no-go | a few hours |
| 05 | [Full discovery run, then the test tiers once](05_full_discovery_run.md) | 04 (go) | `best.py`, `REPORT.md`, test-tier results | many hours |
| 06 | [Search improvements, only where 04 shows a need](06_search_improvements.md) | 04 | targeted changes | M each |
| 07 | [Per-step time guard and large buildings](07_time_guard_and_large_buildings.md) | 05 | guarded submission policy | 1–2 days |
| 08 | [Detect HVAC variants we do not model, and fall back](08_hvac_variants.md) | — | fallback, documented | 1–2 days |
| 09 | [Benchmark the controllers](09_controller_benchmark.md) | 05, 07 | results table | a day of compute |
| 10 | [Package the discovered model and pass the readiness gate](10_package_and_readiness_gate.md) | 05, 07, 08, 09 | the Kaggle submission | half a day |
| 11 | [Robustness checks on assumptions](11_robustness_checks.md) | 05, 09 | degradation table, decisions | M |
| 12 | [Choose the two final submissions, and write the paper](12_final_submissions_and_paper.md) | 05, 09, 10 | final picks, paper | by 15 Jan 2027 |

## How the steps hand off

```
01 data ─► 02 episodes + split + audit ─► 03 free check, prompts, API ─► 04 first run ─► 05 full run
05 ─► 07 time guard ─► 09 benchmark ─► 10 gate and submission (needs 08) ─► 12 final          08: any time
```

The data is the six shipped buildings in every weather window; the split holds out the most unusual buildings (two of
the six) and the most different climate, so every test tier comes from these 288 scenarios.

**Where things live:** the kit and this repository side by side; all data under `$D` (`scenarios/`, `results/`,
`logs/`, `dataset/` with `index.json` and `split.json`, `audit/`, `probe/` for throw-away
tests); discovery runs under `working-code/outputs/discovery/<run>/`.

**Rules that hold throughout:** run every kit command from the kit's root and give it an output folder under `$D` (the
kit is never written to); never move `$D/scenarios` after generating it; nothing is fitted; the test tiers are read once.

**Deadlines:** entry and team merger 3 Jan 2027; final submission and paper 15 Jan 2027.
