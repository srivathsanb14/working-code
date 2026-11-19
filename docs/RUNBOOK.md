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

Start with a pilot. On an 8-core laptop it took 38 minutes for 6 usable buildings × 5 strips × 5 policies × 864 steps (140 runs); generated buildings are large (45–120 rooms):

```bash
python code/generate_data.py --out data/pilot --train-seeds 500:504 --val-seeds 700:702 --test-seeds 800:802 --steps 864 --workers 8
```

Then the full set (tens of core-hours; it resumes if interrupted):

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

Full details are in `docs/LLM_DISCOVERY.md`. In short:

Score the two seed models (no LLM involved):

```bash
python code/run_discovery.py --run main --generations 0
```

See exactly what the LLM would be sent, without sending it:

```bash
python code/run_discovery.py --run main --proposer dry --generations 1
```

Run discovery with Claude (needs `requirements-llm.txt` and credentials):

```bash
python code/run_discovery.py --run main --proposer claude --generations 20 --n 6
```

Or answer with any other LLM or by hand: put candidate `.py` files in `outputs/discovery/main/inbox/` and run

```bash
python code/run_discovery.py --run main --proposer dir --generations 1
```

Results land in `outputs/discovery/main/`: `summary.json` (leaderboard and fitted constants), `best.py`, `database.jsonl`, `candidates/`, `prompts/`.

## 7. Test the model

Test the best discovered model on the test split, with the global constants it was fitted with:

```bash
python code/test_model.py outputs/discovery/main/best.py --params outputs/discovery/main/summary.json --on test
```

Any model file works, for example a seed, refitted on train first:

```bash
python code/test_model.py code/discovery/seeds/seed_slab_wall_network.py --on val
```

The report goes to `outputs/test/<model>_<split>/report.md`. Per run, station, policy and building it gives:
- open-loop RMSE on day 1 and on the following days, with θ fitted on day 1 only, exactly as a deployed policy would;
- if the model defines `network()`, the 1-, 3- and 6-hour forecast RMSE from a Kalman-filtered state, against persistence. The target for planning is under 0.5 K at 3 h.

Look at the **test** numbers once, at the end. Use `--on val` while iterating.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `setup_check` cannot load the scenario | kit path or dependencies | set `STARTER_KIT`; `pip install -r requirements.txt` |
| many `FAILED … 20.0 m2` | normal generator rejections | widen the seed ranges |
| `replay mismatch` | the policy is not deterministic, or reads fields the CSVs do not contain | use the shipped policies (GUIDE.md A.2) |
| `split 'val' matched no episodes` | the split does not match the data | check `data/split.json` and the audit's split table |
| the kit shows changes in `git status` | something ran in the kit without the driver's settings | run kit commands only through these scripts, or with `NUMBA_CACHE_DIR` and `--output-dir`/`--results-dir` outside the kit |
| slow | large generated buildings (100+ rooms) | more `--workers`; `--steps 864`; fewer buildings for a pilot |
