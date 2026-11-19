# Data for the dynamics discovery

All training data is generated with the starter kit's own, unmodified commands. The working code only reads what those commands write; it never steps the simulator itself.

## Assumed layout

```
<SCENARIOS_DIR>/<building>/<weather>/config.gin, config.yaml, floor_plan.npy, zone_map.npy
<RESULTS_DIR>/<building>/<weather>/<policy>/timeseries.csv, ahu_log.csv, reward_components.csv, summary.json
```

The defaults used by `code/run_discovery.py`:
- `--scenarios-dir` is `../smart-buildings-starter-kit/smart_buildings/training_scenarios`, the shipped 30 scenarios.
- `--results-dir` is `data/results`.

For generated buildings, point `--scenarios-dir` at the scenario generator's `--output-dir`.

## Generating it (run from the starter-kit root)

Keep everything the commands write outside the kit:
- pass `--output-dir` and `--results-dir` paths that live in `working-code/data`;
- set `NUMBA_CACHE_DIR`, so numba's compile cache does not land in the kit.

Set up the environment once:

```bash
export NUMBA_CACHE_DIR=../working-code/.numba_cache PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
D=../working-code/data
```

1. Run the shipped scenarios under every shipped behaviour policy:

```bash
python -m smart_buildings.evaluate --policies all_off bang_bang fixed_setpoint pid pid_water --steps 8640 --results-dir $D/results
```

2. Optionally, generate new buildings and scenarios from the same generator and ranges:

```bash
python -m smart_buildings.run_floor_generator --start-seed 500 --count 50 --output-dir $D/buildings --no-pdf
```

```bash
python -m smart_buildings.run_scenario_generator --buildings $D/buildings/*.yaml --weathers atlanta_1 denver_5 fairbanks_1 phoenix_5 singapore_1 minneapolis_1 --output-dir $D/scenarios
```

```bash
python -m smart_buildings.evaluate --scenarios-dir $D/scenarios --policies all_off bang_bang fixed_setpoint pid pid_water --steps 8640 --results-dir $D/results_generated
```

Shorter `--steps` values are fine for model fitting: the discovery loop needs about 3 days, which is 864 steps. Shorter runs are not comparable as competition scores.

## What the code recovers from those files

The CSVs log, per step:
- every room's temperature sensor;
- the outside temperature;
- each AHU's supply temperature, run command and total airflow;
- building occupancy, energy and carbon.

They do not log the per-room damper and reheat commands or the plant setpoints. `code/discovery/starterkit_data.py` recovers them by replay:
- The simulator and the shipped policies are deterministic.
- Those policies read only zone temperatures, the clock, the comfort-band sensors and the outside temperature, all of which can be rebuilt from the files.
- So the unmodified policy class is re-run on the rebuilt observations, and it returns the actions it took.

Every replayed run is checked against the logged per-AHU airflow and run commands. On the sample in `data/results` (6 scenarios × 5 policies, 864 steps) the replay reproduced the logged airflow to within 1.6×10⁻⁴ relative error, which is CSV rounding. Runs that do not reproduce are dropped and listed in `outputs/discovery/<run>/data_report.json`.

A policy you write yourself can be replayed the same way if it is deterministic and reads only those fields. Otherwise, the per-room inputs are not recoverable from the starter kit's outputs.

From the recovered actions plus the logged sensors, `episode_inputs` computes each room's inputs with the simulator's own physics:
- the air-stream conductance and supply temperature;
- reheat watts, using the boiler-tank ramp replayed from the water setpoints;
- internal gain from the comfort schedule;
- building occupancy.

Run the loop with:

```bash
.venv/bin/python code/run_discovery.py --run <name> --proposer dry|dir|claude --generations N --test
```
