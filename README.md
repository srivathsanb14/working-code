# working-code: room dynamics for the Smart Buildings competition

Everything here works **beside** the starter kit, never inside it. The kit is only imported from its folder, or its own command-line tools are run with every output pointed into `data/`. `code/setup_check.py` and `code/audit_data.py` verify that `git status` in the kit stays clean.

## Start here

| Document | For |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | **Step by step on a fresh computer**: setup → data made with the kit → audit → build dataset → split → LLM discovery |
| [GUIDE.md](GUIDE.md) | What may / must not / cannot be changed in data generation, and why (Part A). The state-space model's observation, action, state, inputs, outputs and parameters, the timing, and the dataset arrays (Part B) |
| [docs/DISCOVERY_WALKTHROUGH.md](docs/DISCOVERY_WALKTHROUGH.md) | **Closed-form discovery (default)**: where proposals, mutation, crossover and islands happen; what each log and plot is; how to tell whether the LLM can fit; the guidelines checked against the code |
| [kaggle/submission.ipynb](kaggle/submission.ipynb) | The Kaggle notebook (wraps `code/submission/`: vendored pid_water, safety layer, optional discovered-model tracker); see RUNBOOK §9 |
| [docs/SOURCES_AND_FACTS.md](docs/SOURCES_AND_FACTS.md) | Every fact and number behind the deck, with its source (website, Kaggle, kit, simulator data), and corrections |
| [Planning/index.md](Planning/index.md) | **What is left to do**: one file per pending task, with commands and when it counts as done |
| [STATE_SPACE.md](STATE_SPACE.md) | The full analysis: competition, interface, simulator audit, the generic room model and its evidence, the method and plan |

## The pipeline

All steps in order, for a big machine: `./run_pipeline.sh` (settings are environment variables at its top; see `docs/RUNBOOK.md`). Layout: `<root>/smart-buildings-starter-kit` beside `<root>/working-code`; the data is made beforehand with the starter kit's own commands (this repository generates nothing; `docs/RUNBOOK.md` step 2) and read from `working-code/data/` (or `SB_DATA`).

```
 python code/setup_check.py                  0-1  environment and kit checks (writes nothing)
   (data made with the starter kit)         2    the kit's own commands -> scenarios/ and results/ (outside this repo)
 python code/audit_data.py                   3    layout, conventions, coverage, leakage            -> data/audit/
 python code/build_dataset.py --scenarios S --results R   4    read runs, recover commands, room inputs  -> data/dataset/*.npz
 python code/make_splits.py --dataset D     4b   whole episodes -> validation + test tiers      -> D/split.json
 python code/audit_data.py                   5    + replay status, dataset sanity, excitation
 python code/run_discovery.py ...            6    LLM symbolic discovery                            -> outputs/discovery/<run>/
 python code/run_policy.py --model best.py   7    the model inside a submission-style policy, live = offline -> outputs/policy/
```

Discovery: **closed-form formulas, no fitting anywhere**, with islands, mutation, crossover and explore requests to the Claude API (`run_discovery.py`; `--selftest` checks the scorer on a planted truth). `--test` scores the chosen model and the Pareto front once on the held-out tiers at the end; see `docs/DISCOVERY_WALKTHROUGH.md`.

## Layout

```
code/
  common.py              paths (STARTER_KIT, SB_DATA) and safe ways to run / import the kit
  setup_check.py         step 1
  audit_data.py          steps 3 and 5
  build_dataset.py       step 4
  make_splits.py         step 4b (validation + test tiers, computed from the data)
  run_discovery.py       step 6 (discovery/evolve.py)
  run_policy.py          step 7 (closed-form model inside a submission-style policy; --check-replay)
  discovery/             evolve.py (islands, operators, logs), closedform.py (no-fitting scorer, analyses, ceiling),
                         report.py (REPORT.md, plots), selftest.py (planted truth), seeds_cf/, universal_constants.json,
                         starterkit_data.py (replay and room inputs), episodes.py (geometry, episode cache),
                         splits.py, pareto.py, coefform.py, contract.py, prompts.py, proposers.py, dlib.py,
  room_model.py, room_coeff_targets.py
                         room geometry from the floor plan (used by the split and the episodes)
docs/                    RUNBOOK.md, DISCOVERY_WALKTHROUGH.md, SOURCES_AND_FACTS.md; the decks and deck_src/
Planning/                index.md and one file per pending task
data/                    generated data (git-ignored)
tests/                   pytest suite (no starter kit needed); CI in .github/workflows/ci.yml
kaggle/                  submission notebook; code/submission/ holds its policy core
outputs/                 analysis outputs, discovery runs, test reports, geometry/ (per-room CSVs)
requirements.txt         the kit's pinned dependencies (the kit itself is not installed)
requirements-llm.txt     anthropic, for the Claude API proposer
```
