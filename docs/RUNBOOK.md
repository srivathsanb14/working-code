# Runbook: from a fresh computer to a tested room-dynamics model

Every step runs from `working-code/`. The starter kit is only read: imported from its folder, or its own command-line tools run with every output pointed into `working-code/data`. **Nothing is written into the starter kit**; steps 1, 3 and 6 check that.

| Step | Command | Writes |
|---|---|---|
| 0. Lay out the folders | `git clone` both repositories side by side | — |
| 1. Set up and check | `python code/setup_check.py` | nothing |
| 2. Generate data | `python code/generate_data.py …` | `data/buildings`, `data/scenarios`, `data/results`, `data/plan.json`, `data/split.json` |
| 3. Audit the raw data | `python code/audit_data.py` | `data/audit/` |
| 4. Build the dataset | `python code/build_dataset.py` | `data/dataset/` |
| 5. Audit again | `python code/audit_data.py` | `data/audit/` |
| 6. Discover the dynamics with an LLM | `python code/run_discovery.py …` | `outputs/discovery/<run>/` |
| 7. Test the model | `python code/test_model.py …` | `outputs/test/<model>_<split>/` |
| 8. Rehearse the submission | `python code/run_policy.py …` | `outputs/policy/<model>_<base>_<split>/` |

**All of it in one command** (for a big machine; each step resumes or can be started with `FROM=n`):

```bash
./run_pipeline.sh
```

```bash
PROPOSER=claude GENS=20 WORKERS=32 ./run_pipeline.sh
```

The first writes the LLM prompt without calling anything; the second runs real discovery. Settings are environment variables at the top of `run_pipeline.sh` (seed ranges, stations, policies, steps, workers, deployment, proposer, generations).

**Folder layout.** The only path assumption: one parent folder holds both repositories side by side.

```
<root>/
  smart-buildings-starter-kit/     read only (or set STARTER_KIT)
  working-code/
    data/          buildings, scenarios, results, plan.json, split.json (step 2); audit/ (3, 5); dataset/ (4)
    outputs/       discovery/<run>/ (6); test/ (7); policy/ (8)
```

Put the generated data on another disk with `SB_DATA=/path/to/data`; every script reads it.

Background: `GUIDE.md` covers what may and may not be changed in data generation (Part A) and what the observation, action, state, inputs and outputs of the model are (Part B). `docs/LLM_DISCOVERY.md` covers how the LLM is used.

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

## 2. Generate data

`code/generate_data.py` calls the kit's own `run_floor_generator`, `run_scenario_generator` and `evaluate`. The settings follow GUIDE.md A.2–A.3:
- buildings in three disjoint seed batches (train, val, test);
- train and val buildings get one strip from each of Fairbanks, Denver, Atlanta, Phoenix and Singapore, with the season rotated by seed;
- test buildings get Minneapolis only;
- all five shipped policies.

First look at the plan and its size without running anything:

```bash
python code/generate_data.py --train-seeds 500:600 --val-seeds 700:720 --test-seeds 800:820 --steps 2016 --dry-run
```

Then the full set. Generated buildings are large (45–120 rooms), so it takes tens to hundreds of core-hours; it resumes if interrupted:

```bash
python code/generate_data.py --train-seeds 500:600 --val-seeds 700:720 --test-seeds 800:820 --steps 2016 --workers 16
```

| Option | Default | Meaning |
|---|---|---|
| `--train-seeds`, `--val-seeds`, `--test-seeds` | 500:600, 700:720, 800:820 | building seed ranges, start:stop (stop excluded); disjoint, one generator batch each |
| `--stations` | fairbanks denver atlanta phoenix singapore | stations for train and val buildings |
| `--val-station` | phoenix | held out of train; val = val buildings on this station |
| `--test-station` | minneapolis | test buildings only (the shipped station that is in no shipped scenario) |
| `--test-strips` | 3 | strips per test building |
| `--policies` | the five shipped | behaviour policies (must be replayable, GUIDE.md A.2) |
| `--steps` | 2016 | steps per run: 864 = 3 days (minimum), 2016 = 7 days, 8640 = full episode |
| `--include-shipped` | off | also run the 30 shipped scenarios (they join train unless on a held-out station) |
| `--workers` | 4 | parallel processes |
| `--stage` | all | buildings / scenarios / runs, to redo one stage |

Expected messages:
- `building_NNNN: FAILED (… below the 20.0 m2 minimum floor area …)`. The generator rejects that seed by design; the driver marks it `.failed` and skips it. About one seed in four is rejected.
- At the end, `wrote …/plan.json and …/split.json (train N, val N, test N scenarios; F failures)`.

**Rotating the val station (leave-one-station-out).** Train and val buildings have every station, so regenerate only `split.json` with a different val station:

```bash
python code/generate_data.py --val-station denver --stage runs
```

This is quick: nothing new is simulated.

## 3. Audit the raw data

```bash
python code/audit_data.py
```

It writes `data/audit/audit.md` and exits with an error if a hard check fails. Read the report:
- **kit untouched:** git status is clean.
- **layout:** every planned run has its four files; row counts reach `--steps`; no NaN; CSV row 0 is the reset state (294 K); every episode starts on a Sunday at 06:00.
- **coverage:** buildings, scenarios and runs per split; building parameters and materials against the generator's own ranges; the station × strip matrix per split.
- **leakage:** no building in two splits; Minneapolis never in train or val; the val station never in train.

Fix any `[FAIL]` before going on: rerun `generate_data.py`, which resumes, or delete a broken run folder and rerun.

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

The default mode is **closed-form, with no fitting anywhere**. It runs islands, mutation and crossover. Every step, file and check is described in
[DISCOVERY_WALKTHROUGH.md](DISCOVERY_WALKTHROUGH.md). Before anything else, check the machinery on synthetic data (no simulator data, no LLM, about 4 minutes):

```bash
python code/run_discovery.py --selftest
```

Score the seeds and write every prompt of generation 1 without sending anything:

```bash
python code/run_discovery.py --run main --proposer dry --generations 1 --ceiling
```

Exercise islands, mutation and crossover with no LLM (mechanical operators), and read the logs:

```bash
python code/run_discovery.py --run mock --proposer mock --generations 3
```

Run discovery with Claude (needs `requirements-llm.txt` and credentials):

```bash
python code/run_discovery.py --run main --proposer claude --generations 20 --islands 3 --test
```

You can also answer with any other LLM or by hand. Put candidate `.py` files in each request's `generations/gNNN/island_K/<operator>/inbox/`, then run:

```bash
python code/run_discovery.py --run main --proposer dir --generations 1
```

Results land in `outputs/discovery/main/`. Start with `REPORT.md`, then `log.md`, `generations/`, `candidates/`, `plots/`, `database.jsonl`, `summary.json` and `best.py`.

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

- **Rejected seeds:** `run_pipeline.sh` passes `--oversample 1.3` to `generate_data.py`, so each split keeps its planned number of buildings even though the kit rejects about 20% of seeds.
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
