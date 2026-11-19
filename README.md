# working-code: room dynamics for the Smart Buildings competition

Everything here works **beside** the starter kit, never inside it. The kit is only imported from its folder, or its own command-line tools are run with every output pointed into `data/`. `code/setup_check.py` and `code/audit_data.py` verify that `git status` in the kit stays clean.

## Start here

| Document | For |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | **Step by step on a fresh computer**: setup → generate data → audit → build dataset → LLM discovery → test |
| [GUIDE.md](GUIDE.md) | What may / must not / cannot be changed in data generation, and why (Part A). The state-space model's observation, action, state, inputs, outputs and parameters, the timing, and the dataset arrays (Part B) |
| [docs/LLM_DISCOVERY.md](docs/LLM_DISCOVERY.md) | How the LLM is used: what it is sent, what it returns, how to run, steer and judge it |
| [STATE_SPACE.md](STATE_SPACE.md) | The full analysis: competition, interface, simulator audit, the generic room model and its evidence, the method and plan |

## The pipeline

```
 python code/setup_check.py                  0-1  environment and kit checks (writes nothing)
 python code/generate_data.py ...            2    starter-kit CLIs -> data/buildings, scenarios, results, plan.json, split.json
 python code/audit_data.py                   3    layout, conventions, coverage, leakage            -> data/audit/
 python code/build_dataset.py                4    replay policies, check, compute room inputs       -> data/dataset/*.npz
 python code/audit_data.py                   5    + replay status, dataset sanity, excitation
 python code/run_discovery.py ...            6    LLM symbolic discovery                            -> outputs/discovery/<run>/
 python code/test_model.py MODEL.py ...      7    held-out test: open loop + filtered forecasts     -> outputs/test/
```

## Layout

```
code/
  common.py              paths (STARTER_KIT, SB_DATA) and safe ways to run / import the kit
  setup_check.py         step 1
  generate_data.py       step 2
  audit_data.py          steps 3 and 5
  build_dataset.py       step 4
  run_discovery.py       step 6 (launcher for discovery/loop.py)
  test_model.py          step 7
  discovery/             starterkit_data.py (replay and room inputs), episodes.py (geometry, episode cache),
                         contract.py, evaluate.py, diagnostics.py, prompts.py, proposers.py, loop.py, dlib.py,
                         seeds/ (a naive one-node model and the audited slab-wall network)
  room_model*.py, room_inputs.py, room_coeff_*.py, room_spread_fit.py
                         the simulator audit behind STATE_SPACE.md §14 (exact grid reconstruction, coefficient fits)
  *.py (others)          the interface analyses behind STATE_SPACE.md §5-§7
docs/                    RUNBOOK.md, LLM_DISCOVERY.md
data/                    generated data (git-ignored)
outputs/                 analysis outputs, discovery runs, test reports
requirements.txt         the kit's pinned dependencies (the kit itself is not installed)
requirements-llm.txt     anthropic, for --proposer claude
split_example.json       split template (glob patterns)
```
