# Data for the dynamics discovery

All training data is generated with the starter kit's own, unmodified commands. The working code only reads what those commands write; it never steps the simulator itself. This page says which generation options can and cannot be changed, given how the hidden evaluation is built and how we validate against it.

## 1. What the hidden evaluation varies, and what we can influence

The hidden set is 6 unseen buildings × 5 unseen weather stations, from the same generator and ranges as the shipped data. It runs full 30-day episodes, and our own controller acts in it (STATE_SPACE.md §4.3, §12).

| Hidden-set property | Starter-kit option | Can we influence it? | Setting |
|---|---|---|---|
| **New buildings** from the same generator | `run_floor_generator --start-seed/--count/--seeds` | **Yes**: which and how many buildings | Many distinct buildings, since the building, not the step, is the unit of generalization. Use disjoint seed ranges for train, val and test |
| Footprint, floors, room sizes, AHU count, occupants per zone | `DEFAULT_RANGES` in `run_floor_generator.py` (hard-coded) | **No**, and that is correct | The generator already samples the evaluation ranges. Editing them would train on the wrong distribution |
| Footprint shape | `--shape` | Yes, but don't | Keep the default `composite` (what the shipped and hidden buildings use) |
| Envelope materials (6 properties) | drawn by `run_floor_generator` by Latin hypercube **across each batch** | **Indirectly**: by batch | Generate each split as its own batch (`--count` = split size), so every split covers the material ranges. `--no-materials` never |
| **New weather stations** | `run_scenario_generator --weathers` from the 48 shipped strips (6 stations × 8 dates) | **Partly**: only the 6 shipped stations | **Minneapolis** ships but is in no shipped scenario, so it is the closest stand-in for an unseen station: hold it out for test. Rotate which other station is held out of training (leave-one-station-out) |
| Season of the 30-day window | which strips: `_1` … `_8` per station, about 6 weeks apart | **Yes** | Give each building strips that span seasons and stations, not one strip per station |
| Episode start (always **Sunday 06:00**), comfort schedule, time zone | fixed in every strip and config | No | Matches evaluation. Note that day 1 of every episode, the day θ is identified from, is a Sunday in eco mode |
| Wind film coefficient per station | fixed in each strip YAML | No | It is part of θ (`h`), estimated online |
| Episode length | `evaluate --steps` | **Yes** | ≥ 864 (3 days: the Sunday identification day plus Monday–Tuesday occupied). 2016 (7 days) adds weekday cycles and slow dynamics. Full 8640 on a subset for final checks |
| **Who acts** (our controller at test time) | `evaluate --policies` | **Yes**, among replayable policies | All five shipped policies. `all_off` gives free-float, the cleanest for envelope and slab. `bang_bang` gives strong full-on/off steps, the best excitation. `fixed_setpoint`, `pid` and `pid_water` give realistic operation. `transfer` (all zeros) adds nothing. A policy of your own is usable only if it is deterministic and reads only zone temperatures, the clock, comfort-band sensors and outside air; otherwise its per-room actions cannot be recovered |
| Occupant draws | `RandomizedArrivalDepartureOccupancy` seed (fixed, 17321) | No | Deterministic; total occupancy is logged. The per-room split is the one unobservable input |
| Plant sizing, reward, floors, internal gains, slab and air constants (class B and hard-coded, §12.2) | config / code | **Must not** | Assumed identical in the hidden set; the model hard-codes them |

Beyond the starter kit (optional, flagged): a new weather strip is a four-key YAML (`name`, `weather_csv`, `convection_coefficient`, `start_timestamp`, which should be a Sunday 06:00). A new station needs a public weather CSV (allowed by the rules, §4.3) and a guessed convection coefficient. That is the only way to get more than six stations.

## 2. Validation mirrors the evaluation

| Hidden-set condition | How the split reproduces it |
|---|---|
| Unseen building | `val` and `test` buildings come from seed ranges never used for training |
| Unseen station | `test` uses only Minneapolis; `val` holds out one of the other five stations (rotate it for leave-one-station-out) |
| No per-building fitting offline | Global constants are fitted on `train` only. For each `val`/`test` scenario θ is fitted on **day 1** (the Sunday), then scored open loop on the following days |
| Every regime | Each split covers several seasons and all five policies |

`split_example.json` is a template. Its patterns are globs: with `/` they match `building/weather`, without they match the building, and `!` excludes.

```json
{"train": ["building_05*", "!*/phoenix*", "!*/minneapolis*"],
 "val":   ["building_07[01]*/phoenix*"],
 "test":  ["building_08[01]*/minneapolis*"]}
```

- **Train:** seeds 500–599, on every station except the val station (Phoenix) and the test station (Minneapolis).
- **Val:** seeds 700–719, on Phoenix only, so both the building and the station are unseen.
- **Test:** seeds 800–819, on Minneapolis only.

Because train and val buildings get strips from all five non-Minneapolis stations (§3), the val station can be rotated (leave-one-station-out) by editing only the split, without regenerating. The shipped 30 scenarios (`building_train_*`) can be added to `train`, or kept as a second "competition-like" validation set.

## 3. Commands (run from the starter-kit root)

Keep every output outside the kit, and keep numba's cache out of it:

```bash
export NUMBA_CACHE_DIR=../working-code/.numba_cache PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
D=../working-code/data
```

**Buildings.** Make three batches, one per split, so each batch covers the material ranges. Some seeds produce a room under 20 m², and the scenario generator rejects those; skip them.

```bash
python -m smart_buildings.run_floor_generator --start-seed 500 --count 100 --output-dir $D/buildings/train --no-pdf
```

```bash
python -m smart_buildings.run_floor_generator --start-seed 700 --count 20 --output-dir $D/buildings/val --no-pdf
```

```bash
python -m smart_buildings.run_floor_generator --start-seed 800 --count 20 --output-dir $D/buildings/test --no-pdf
```

**Scenarios.** The generator takes the cross product of `--buildings` and `--weathers`, so pair per building with one call each. Train and val buildings get one strip from each of the five non-Minneapolis stations, which lets the val station rotate; test buildings get Minneapolis strips only.

```bash
for b in $D/buildings/train/*.yaml $D/buildings/val/*.yaml; do python -m smart_buildings.run_scenario_generator --buildings $b --weathers fairbanks_1 denver_3 atlanta_5 phoenix_7 singapore_2 --output-dir $D/scenarios || true; done
```

```bash
for b in $D/buildings/test/*.yaml; do python -m smart_buildings.run_scenario_generator --buildings $b --weathers minneapolis_1 minneapolis_4 minneapolis_7 --output-dir $D/scenarios || true; done
```

Rotate the strip indices between buildings (for example `fairbanks_2 denver_4 …` for the next one) so seasons spread across the set.

**Runs.**

```bash
python -m smart_buildings.evaluate --scenarios-dir $D/scenarios --policies all_off bang_bang fixed_setpoint pid pid_water --steps 2016 --results-dir $D/results
```

Optionally, also run the shipped 30 scenarios:

```bash
python -m smart_buildings.evaluate --policies all_off bang_bang fixed_setpoint pid pid_water --steps 2016 --results-dir $D/results_shipped
```

**Cost.** Roughly 100 steps/s on a small building, slower on large ones; runs are independent, so run them in parallel. For example, 140 buildings × 4 strips × 5 policies × 2016 steps ≈ 5.6 M steps, which is tens of core-hours. Start with fewer buildings and `--steps 864`.

## 4. Discovery and testing

```bash
.venv/bin/python code/run_discovery.py --run main --scenarios-dir data/scenarios --results-dir data/results --split split_example.json --generations 0 --test
```

- Check `outputs/discovery/main/data_report.json`: every run must say `ok`. A replay mismatch means the per-room inputs are unknown, and the run is dropped.
- `--generations N --proposer claude|dir` runs discovery. Models are ranked on `val`.
- With `--test`, `best.py` is scored on `test` at the end (`summary.json → test_rmse_days23`). Look at it only then.

## 5. How the per-room inputs are recovered

The CSVs log, per step:
- every room's temperature sensor;
- the outside temperature;
- each AHU's supply temperature, run command and total airflow;
- building occupancy, energy and carbon.

They do not log per-room dampers and valves or the plant setpoints. `code/discovery/starterkit_data.py` recovers them by replay. The simulator and the shipped policies are deterministic, so the unmodified policy class is re-run on observations rebuilt from the CSVs. Each run is checked against the logged per-AHU airflow and run commands. On the shipped sample and on a generated 120-room, 12-AHU building, the replay reproduced the logged airflow to CSV rounding. The room inputs are then computed with the simulator's own physics: air stream, supply temperature, reheat (with the boiler ramp replayed from the setpoints), internal gain and occupancy.
