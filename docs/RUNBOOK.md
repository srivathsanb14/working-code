# Runbook: from generated data to discovered room-dynamics formulas

Run everything from `working-code/`, on the machine that holds the generated data. **Data generation is not part of
this pipeline**: the data is made beforehand with the starter kit (its scenario generator and `evaluate.py`, any
number of buildings, weather windows and controllers). This pipeline only reads it. The starter kit is never written to.

| Step | Command | Reads | Writes |
|---|---|---|---|
| 1. Episodes | `python code/build_dataset.py --results R --scenarios S --out D` | the kit's result and scenario folders | `D/` episodes (one `.npz` per run) + `index.json` |
| 2. Split | `python code/make_splits.py --dataset D` | the episodes | `D/split.json`: validation + test_easy / test_medium / test_hard (+ test_station), whole episodes only |
| 3. Discovery | `python code/run_discovery.py --dataset D --generations 20 --test` | validation episodes (test tiers only at the end) | `outputs/discovery/<run>/` |

All three in one command (`FROM=2` if the episodes are already built):

```bash
ANTHROPIC_API_KEY=... RESULTS=/data/results SCENARIOS=/data/scenarios GENS=20 ./run_pipeline.sh
```

Every candidate is written by the Claude API: one stateless request per operator (the cached system prompt plus that
request), generation 0 included. Nothing is generated, simulated or mocked here. The scorer runs the candidates'
formulas on the validation episodes with numpy; the episodes themselves never leave the machine (the prompts carry the
analysis and a short trajectory excerpt; see docs/DISCOVERY_WALKTHROUGH.md).

Optional afterwards: step 7 (`code/test_model.py`), the policy rehearsal (`code/run_policy.py`) and the Kaggle package
(sections 8–11 below). The sections on generating data (1–5 below) describe how the data machine makes the data
with the kit; they are kept for reference.

**Folder layout.** The only path assumption: one parent folder holds both repositories side by side.

```
<root>/
  smart-buildings-starter-kit/     read only (or set STARTER_KIT)
  working-code/
    data/          buildings, scenarios, results, plan.json, split.json (step 2); audit/ (3, 5); dataset/ (4)
    outputs/       discovery/<run>/ (6); test/ (7); policy/ (8)
```

Put the generated data on another disk with `SB_DATA=/path/to/data`; every script reads it.

Background: `GUIDE.md` covers what may and may not be changed in data generation (Part A) and what the observation, action, state, inputs and outputs of the model are (Part B). `docs/DISCOVERY_WALKTHROUGH.md` covers how the LLM is used; `Planning/index.md` lists what is left to do.

---

## 0. Lay out the folders

```bash
mkdir smart-buildings && cd smart-buildings
```

```bash
git clone https://github.com/smart-buildings-2026/starter-kit smart-buildings-starter-kit
```

```bash
git clone <this repository> working-code
```

The tools find the kit as the sibling folder `../smart-buildings-starter-kit`. If it lives elsewhere:

```bash
export STARTER_KIT=/path/to/smart-buildings-starter-kit
```

To keep the (large) data somewhere other than `working-code/data`:

```bash
export SB_DATA=/big/disk/sb-data
```

## 1. Set up and check

Python 3.12 or newer. The kit is **not** installed (an editable install would write metadata into it); its pinned dependencies are installed instead, and the code imports it from its folder.

```bash
cd working-code
```

```bash
python3.12 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Only for discovery with Claude:

```bash
pip install -r requirements-llm.txt
```

Then check:

```bash
python code/setup_check.py
```

Every line must say `[ok]`. It checks:
- the Python version and the packages;
- that the kit is found and `git status` is clean;
- that one shipped scenario loads through the kit (283 observation fields for `building_train_8`);
- that the kit is still clean afterwards.

## 2. The data: made with the starter kit, not here

This repository generates and simulates nothing. The data is made beforehand, on the machine that will run the
pipeline, with the starter kit's own commands. The plan (exact commands in GUIDE.md A.4):

- **Stage 1:** the six shipped buildings × all 48 weather windows (6 stations × 8) × the 5 shipped controllers, full
  30-day runs: 288 scenarios, 1,440 runs, roughly 100–200 CPU-hours.
- **Stage 2:** new buildings from the kit's generator (seeds from 500; 20 to start), each on one window per station
  with the window index rotated, × 5 controllers: 600 runs for 20 buildings.

```bash
cd smart-buildings-starter-kit
python -m smart_buildings.run_scenario_generator --buildings building_train_1 building_train_2 building_train_4 building_train_5 building_train_7 building_train_8 --output-dir <scenarios>
python -m smart_buildings.evaluate --scenarios-dir <scenarios> --results-dir <results> --policies all_off bang_bang fixed_setpoint pid pid_water
```

What this pipeline needs afterwards is only two folders:
- `<scenarios>/<building>/<weather>/`: `config.gin`, `floor_plan.npy`, `zone_map.npy` (the kit's shipped
  `smart_buildings/training_scenarios` is one such folder);
- `<results>/<building>/<weather>/<controller>/`: `timeseries.csv`, `ahu_log.csv`, `reward_components.csv`, `summary.json`.

Every run must use one of the shipped controllers (their decisions are re-computed from the logs in step 4). The
validation / test split is computed from whatever runs exist (step 4b, `make_splits.py`); nothing is decided here.

## 3. Audit the raw data

```bash
python code/audit_data.py
```

It writes `data/audit/audit.md` and exits with an error if a hard check fails. Read the report:
- **kit untouched:** git status is clean.
- **layout:** every planned run has its four files; row counts reach `--steps`; no NaN; CSV row 0 is the reset state (294 K); every episode starts on a Sunday at 06:00.
- **coverage:** buildings, scenarios and runs per split; building parameters and materials against the generator's own ranges; the station × strip matrix per split.
- **leakage:** no building or scenario in two splits (the split itself comes from `make_splits.py`).

Fix any `[FAIL]` before going on: regenerate the affected run with the kit, or delete the broken run folder.

## 4. Build the dataset

```bash
python code/build_dataset.py --workers 8
```

For every run it:
1. loads the scenario's metadata through the kit (no simulation);
2. rebuilds, from the CSVs, the observations the policy saw;
3. **replays** the unmodified policy to recover the per-room damper and valve commands and the plant setpoints;
4. checks the replay against the logged per-AHU airflow and run commands (expect about 1e-4, which is CSV rounding);
5. computes each room's physical inputs (GUIDE.md B.5);
6. writes `data/dataset/<building>__<weather>__<policy>.npz` (arrays in GUIDE.md B.8), plus `index.json`, `schema.json` and a copy of `split.json`.

Up-to-date files are skipped on reruns.

## 5. Audit again

```bash
python code/audit_data.py
```

It now also reports:
- **replay status per run.** Every run should be `ok`; a `replay mismatch` run is excluded from fitting.
- **dataset sanity:** finite arrays and temperatures within 250–330 K.
- **excitation by policy:** AHU-on fraction, damper and reheat activity, temperature spread. Identification needs movement: `bang_bang` and `all_off` are the informative ones.

Only go on when the audit has no hard failures.

## 6. Discover the dynamics with an LLM

The default mode is **closed-form, with no fitting anywhere**: islands, mutation and crossover, every candidate written
by the Claude API. Every step, file and check is described in [DISCOVERY_WALKTHROUGH.md](DISCOVERY_WALKTHROUGH.md).
Check the scorer first (a planted truth on synthetic inputs must score about 0 K; no API call):

```bash
python code/run_discovery.py --selftest
```

**Split** (whole episodes; no training split, nothing is fitted). The search sees only `val`. `test_easy` (seen
building and station, unseen weather window), `test_medium` (unseen building), `test_hard` (unseen building at the
station farthest from the others in Wasserstein distance) and `test_station` (seen building, that station) are scored
once at the end. Computed from the episodes that exist; a run stops if any episode or scenario would be in two splits:

```bash
python code/make_splits.py --dataset D
```

**Discovery** (needs `pip install -r requirements-llm.txt` and `ANTHROPIC_API_KEY`). Generation 0 is `--n-init`
candidates per island written from the system prompt alone; later generations are mutation, crossover and explore
requests. `--max-usd` stops the run at a budget; `--baselines` also scores the hand-written models as reference rows
(never parents):

```bash
python code/run_discovery.py --dataset D --run main --generations 20 --islands 3 --test --max-usd 50
```

Selection is multi-objective by default (`--selection pareto --objectives main fast slow exterior interior`).
Results land in `outputs/discovery/main/`. Start with `REPORT.md`, then `log.md`, `generations/` (every prompt,
analysis and raw API response), `candidates/`, `plots/`, `database.jsonl`, `summary.json` and `best.py`.

### The earlier fitted modes (comparison only)

Both are implemented; choose with `--deployment` (STATE_SPACE.md §17.8 explains the choice).

| | Option A: offline (default) | Option B: online |
|---|---|---|
| On a new building | no fitting: floor plan + measured history + universal constants | θ re-fitted on its day 1 |
| Fitted on train | PARAMS + one shared θ | PARAMS + one θ per training building |
| Val score | 3 h forecast RMSE from a Kalman-filtered state (needs `network()`) | open-loop RMSE after day 1 |
| Discovery | `--deployment offline` | `--deployment online` |
| Test | `--theta shared` (also `trainmean`, `prior`) | `--theta day1` |

Option A:

```bash
python code/run_discovery.py --run main --deployment offline --generations 0
```

```bash
python code/test_model.py outputs/discovery/main/best.py --params outputs/discovery/main/summary.json --theta shared
```

Option B:

```bash
python code/run_discovery.py --run main --deployment online --generations 0
```

```bash
python code/test_model.py outputs/discovery/main/best.py --params outputs/discovery/main/summary.json --theta day1
```

Both, side by side (writes `outputs/discovery/main_offline/` and `main_online/`, then prints a comparison; with `--proposer claude` the LLM is called for each):

```bash
python code/run_discovery.py --run main --deployment both --generations 0
```

```bash
python code/test_model.py outputs/discovery/main_offline/best.py --params outputs/discovery/main_offline/summary.json --theta all
```

`--theta all` runs shared, trainmean, prior and day1 and writes `outputs/test/<model>_<split>_compare.md`.

## 7. Test the model

Test the best discovered model on the test split, with the constants it was fitted with (default `--theta shared`: the shared θ from `summary.json`, no fitting on test buildings):

```bash
python code/test_model.py outputs/discovery/main/best.py --params outputs/discovery/main/summary.json --on test
```

Any model file works, for example a seed, refitted on train first:

```bash
python code/test_model.py code/discovery/seeds/seed_slab_wall_network.py --on val
```

`--theta` picks how θ reaches a test building: `shared` (default, offline: one universal θ), `trainmean` (geometric mean of per-train-building θ), `prior` (initial values), or `day1` (online: fitted on that building's day 1). Only `day1` fits anything on the test building. Use `--deployment online` in step 6 if you want `summary.json` to carry per-building θ.

The report goes to `outputs/test/<model>_<split>_theta-<mode>/report.md`. Per run, station, policy and building it gives:
- open-loop RMSE on day 1 and on the following days, from the reset state;
- if the model defines `network()`, the 1-, 3- and 6-hour forecast RMSE from a Kalman-filtered state, against persistence. The target for planning is under 0.5 K at 3 h.

These forecasts are fed the true future inputs and weather: they test the model, not a controller (STATE_SPACE.md §17.8).

Look at the **test** numbers once, at the end. Use `--on val` while iterating.

## 8. Rehearse the submission (fitted mode; closed-form: `run_pipeline.sh` step 8 runs `run_policy.py --check-replay`)

The hidden evaluation runs the submitted notebook's Policy class on each unseen (building, weather) scenario: `__init__(building_meta_data, action_spec)` once, then `get_action(obs)` every 5 minutes for 8,640 steps. Step 8 does exactly that with the starter kit's own harness on our held-out split:

```bash
python code/run_policy.py --params outputs/discovery/main_offline/summary.json --on test
```

The policy (`code/runtime/model_policy.py`) builds the model from the metadata alone, feeds it every observation and its own action, and keeps a Kalman-filtered state (`code/runtime/online_model.py`). Control still comes from a base policy (`--base pid_water`), so the score must equal that policy's: a check that the model changes nothing yet. The report gives the score, the model's live 1/3/6 h forecast error and its milliseconds per step. Replacing the base policy's decisions with model-based ones is the controller work (STATE_SPACE.md §18.1).

---

## 9. Build and check the Kaggle submission

The notebook is `kaggle/submission.ipynb`. It wraps `code/submission/transfer_policy.py`:
- **Control:** the kit's `pid_water` baseline, copied into `code/submission/vendor` under Apache-2.0.
- **Safety layer:** clips actions into bounds, repairs NaNs and falls back on any exception.
- **Tracker (optional):** follows the building with the discovered formulas.

1. Rehearse locally through the kit's own harness. With no arguments it runs all 30 shipped scenarios at 8,640 steps, comparable with `baseline_returns.json`:

```bash
python code/rehearse_submission.py --model outputs/discovery/main/best.py
```

2. Build the folder for the Kaggle Dataset. The script checks that it imports without the kit:

```bash
python code/package_submission.py --model outputs/discovery/main/best.py
```

3. Upload `dist/kaggle_dataset/` as a Kaggle Dataset, attach it to the notebook, run the notebook once (it plays one public scenario), then submit.

## 10. Model-based controllers

These live in `code/control/`, all nothing-fitted:
- **Cost estimate** (`costs.py`): the reward evaluated on predictions.
- **Occupancy forecast** (`occupancy.py`).
- **Schedule and future observations** (`schedule.py`).
- **Planner** (`planner.py`): simulates pid_water's closed-loop response for a mode with the discovered model and prices it.
- **Controllers** (`policies.py`):
  - `FeedForwardPolicy`: pre-heat and early set-back decided by the model;
  - `MPCPolicy`: picks schedule, comfort, eco or off every 3 steps.

Compare them with pid_water through the kit's harness:

```bash
python code/rehearse_submission.py --controller mpc --model outputs/discovery/main/best.py
```

```bash
python code/rehearse_submission.py --controller ff --model outputs/discovery/main/best.py
```

```bash
python code/rehearse_submission.py --controller pid_water
```

Check the cost estimate against the kit's logged costs:

```bash
python code/control/validate_costs.py --results <results dir> --runs <building>/<weather>/<policy>
```

## 11. Is it ready to deploy? (task L6)

The design is building- and weather-agnostic: positions come from the metadata, geometry from the floor plan, nothing is fitted, and occupancy is learned online. It has **not yet been shown** to work at the scale of the hidden test.

**Deployable today:** the pid_water-only notebook (no `best.py` attached, task K1). Submit it first to prove the Kaggle interface.

**Before the model-based (MPC) version:**
- [ ] K1: the pid_water notebook has been scored on Kaggle.
- [ ] K7: a per-step time guard. Large generated buildings leave the policy only milliseconds per step.
- [ ] K8: unsupported HVAC variants, such as the kit's heat pump, are detected and fall back to pid_water.
- [ ] I5: a benchmark on the 30 shipped scenarios at 8,640 steps and on generated hold-out buildings (including the largest) on the held-out station. The MPC must beat pid_water on average and in the worst case, within the time budget.
- [ ] The packaged dataset has been rehearsed locally (`rehearse_submission.py`), and the fallback to pid_water verified by injecting faults.

Known gaps that do not block deployment:
- the model is about persistence level at 3 h (tasks G2, G8);
- some simulator constants are assumed equal in the hidden set (I6);
- holidays and the gas price by month are not handled (K9).

## Other tools

- **Geometry of any building** from `floor_plan.npy` and `zone_map.npy`:

  ```bash
  python code/geometry_cli.py <scenario_dir>
  ```

  Use `--all-shipped` for the six shipped buildings.
- **Dataset summary** (after step 4) by split, station, season, policy and building size, written to `docs/DATASET_REPORT.md`:

  ```bash
  python code/dataset_report.py
  ```

- **Rejected seeds:** the kit's generator rejects about one seed in five (a room under 20 m²), so ask it for more seeds than the number of buildings you want.
- **Tests and lint:** install with `pip install -r requirements-dev.txt`, then run:

  ```bash
  pytest -q tests && ruff check code tests
  ```

  CI runs the same checks on every push. The tests need no starter kit.
- **Interrupted discovery runs** resume where they stopped. Every LLM answer is saved (`candidates.json`) before it is evaluated, so re-running the same command never pays for the same request twice.
- **Faster evaluation:** `--workers N` evaluates candidates in N processes, with the same results as serial.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `setup_check` cannot load the scenario | kit path or dependencies | set `STARTER_KIT`; `pip install -r requirements.txt` |
| many `FAILED … 20.0 m2` | normal generator rejections | widen the seed ranges |
| `replay mismatch` | the policy is not deterministic, or reads fields the CSVs do not contain | use the shipped policies (GUIDE.md A.2) |
| `split 'val' matched no episodes` | the split does not match the data | check `data/split.json` and the audit's split table |
| the kit shows changes in `git status` | something ran in the kit without the driver's settings | run kit commands only through these scripts, or with `NUMBA_CACHE_DIR` and `--output-dir`/`--results-dir` outside the kit |
| slow | large generated buildings (100+ rooms) | more `--workers`; `--steps 864` |
