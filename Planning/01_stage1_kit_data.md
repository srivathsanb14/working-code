# 01 · Stage 1 data: every shipped building in every weather window

**Phase:** Data · **Priority:** P0 · **Size:** L (mostly compute) · **Depends on:** nothing

## Goal

Run the starter kit's own commands to produce the full cross product of the six shipped buildings, all 48 weather
windows and the five shipped controllers, as full 30-day runs. Today only the kit's 30 pre-built scenarios exist (one
window per station per building), and only 12 of those have been simulated locally. This repository generates nothing:
everything below runs inside `smart-buildings-starter-kit`, with every output folder outside the kit.

| | Count |
|---|---|
| Buildings | 6 (`building_train_1, 2, 4, 5, 7, 8`) |
| Weather windows | 48 = 6 stations (Atlanta, Denver, Fairbanks, Minneapolis, Phoenix, Singapore) × 8 start dates |
| Scenarios | 6 × 48 = **288** |
| Controllers | 5: `all_off`, `bang_bang`, `fixed_setpoint`, `pid`, `pid_water` |
| Runs | 288 × 5 = **1,440**, each 8,640 five-minute steps (30 days) |

## Steps

1. **Measure speed first.** On the data machine run one scenario per building for 576 steps and read
   `seconds_per_step` from each `summary.json`. On this Mac the simulator took 0.025–0.057 s per step on the shipped
   buildings, so a 30-day run is about 4–8 min and stage 1 is roughly 100–200 CPU-hours. Record the table in
   `docs/SOURCES_AND_FACTS.md` and size the parallelism from it.
2. **Make the scenarios.** From the kit root (naming only `--buildings` takes every window):
   ```bash
   export NUMBA_CACHE_DIR=../working-code/.numba_cache PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
   D=../working-code/data
   python -m smart_buildings.run_scenario_generator --buildings building_train_1 building_train_2 building_train_4 building_train_5 building_train_7 building_train_8 --output-dir $D/scenarios
   ```
   Expect 288 folders `$D/scenarios/<building>/<window>/` with `config.gin`, `floor_plan.npy`, `zone_map.npy`.
3. **Run the controllers.** `evaluate.py` runs scenarios one after another, so shard it: one process per building (or
   per station) with the kit's `--buildings` / `--weathers` filters, all writing to the same `--results-dir`:
   ```bash
   python -m smart_buildings.evaluate --scenarios-dir $D/scenarios --results-dir $D/results --policies all_off bang_bang fixed_setpoint pid pid_water
   ```
   `--steps` defaults to 8,640. Each run writes `timeseries.csv`, `ahu_log.csv`, `reward_components.csv`, `summary.json`.
4. **Check completeness.** 1,440 `summary.json` files; any missing or failed run listed with its error.
5. **Benchmark check.** For the 30 shipped pairs (`training_set.yaml`), the five controllers' scores must
   match the kit's `baseline_returns.json` within rounding (pid_water ≈ 0.054 mean loss per step). This proves the runs
   are the same simulator the leaderboard uses.
6. **Kit untouched.** `git status` in the kit is clean afterwards (`code/setup_check.py` checks this).

## Done when

- 288 scenarios and 1,440 complete runs exist under `$D/scenarios` and `$D/results`.
- The 30 shipped scenarios reproduce `baseline_returns.json`.
- Seconds per step by building are recorded in `docs/SOURCES_AND_FACTS.md`.
- The kit's git status is clean.

## Notes and risks

- Minneapolis is in no shipped scenario but is a full station here; whether it ends up in validation or a test tier is
  decided by the split (task 04), not now.
- Disk: the 12 local episode files take 47 MB (about 4 MB each), so 1,440 episodes need about 6 GB; the kit's raw
  CSVs are larger. Measure one run's results folder before starting.
