# Guide: data generation and the state-space model

A working reference for building the room dynamics model. Part A is what may and may not be changed when generating data. Part B defines the observation, action, state, inputs, outputs and parameters of the model. Background and evidence are in `STATE_SPACE.md` (§5–§7 interface, §12 what varies, §13–§14 the model, §17.7 the discovery loop).

---

## Part A. Data generation

### A.1 Ground rules

1. **Only the starter kit's own commands generate data:** `run_floor_generator`, `run_scenario_generator` and `evaluate`. The working code only reads what they write; it never steps the simulator.
2. **The starter kit is never edited.** Every output goes outside it (`--output-dir`, `--results-dir`), and so does numba's compile cache (`NUMBA_CACHE_DIR`).
3. **Vary what the evaluation generator varies, inside its ranges; hold fixed what it holds fixed.** The hidden test set is 6 unseen buildings × 5 unseen weather stations, drawn from the same generator and ranges and run for 30 days. Data that leaves that distribution trains a model for the wrong world.

### A.2 What can, must not, and cannot be changed

**Can change (and should):**

| Option | Command / flag | Why it matters for the evaluation | Guideline |
|---|---|---|---|
| Which buildings, and how many | `run_floor_generator --start-seed --count` (or `--seeds`) | The hidden buildings are new draws from this generator; the building is the unit of generalization | As many distinct buildings as compute allows. Disjoint seed ranges for train, val and test |
| Batch composition | `--count` per call | Materials are spread evenly across each batch, and a different `--count` redraws them | One batch per split, so each split covers the material ranges |
| Which weather strips | `run_scenario_generator --weathers` (48 shipped strips: 6 stations × 8 dates, about 6 weeks apart) | Hidden stations are unseen; windows can fall in any season | Each building gets strips from several stations and seasons. Keep **Minneapolis** (shipped, but in no shipped scenario) for test only |
| Building–weather pairing | one `run_scenario_generator` call per building (it takes the cross product) | Avoids every building seeing the same weather | Rotate strip indices across buildings |
| Behaviour policies | `evaluate --policies` | Excitation decides what can be identified | All five shipped policies: `all_off` (free-float), `bang_bang` (strong steps), `fixed_setpoint`, `pid`, `pid_water` (realistic) |
| Episode length | `evaluate --steps` | Day 1 identifies θ; later days score the model | ≥ 864 (3 days). 2016 (7 days) recommended. 8640 on a subset for final checks |
| Where outputs go | `--output-dir`, `--scenarios-dir`, `--results-dir` | — | Under `working-code/data/` |

**Must not change** (configurable, but fixed in every shipped scenario and assumed identical in the hidden set; the model hard-codes them):

- Generator sampling ranges (`DEFAULT_RANGES` in `run_floor_generator.py`), footprint shape (`composite`), and `--no-materials`.
- Plant sizing: AHU maximum flow 20 m³/s, VAV maximum flow 2.0 m³/s, fan and pump efficiencies, boiler ramp rates, reheat flow factor.
- Floors and limits: damper ≥ 0.1, static pressure ≥ 200 Pa when running, recirculation ≤ 0.7, discharge air ≤ 308.15 K.
- Schedule and comfort bands, time zone, reward weights and scales, internal gains (10 W/m², standby fraction 0.2), 70 W per occupant.
- Interior material constants (air, slab conductivity and heat capacity), cell size 0.2 m, floor height 3 m, time step 300 s.
- The simulator, reward, environment code, and the gin interface (action ranges, normalizers).
- Simulation start timestamp in a building YAML: it is overridden by the weather strip anyway.

**Cannot change** (fixed by the kit; matches the evaluation):

- Every strip starts on a **Sunday at 06:00**. Day 1, the day θ is identified from, is therefore an unoccupied eco-mode Sunday, exactly as for a deployed policy.
- The wind film coefficient of each station (in the strip YAML) is part of θ and estimated online.
- The occupant random seed is fixed, so runs are deterministic. Total occupancy is logged; which room people are in is not.
- The reset state: every node at 294 K, boiler tank at 340 K.

**Beyond the starter kit (optional; label it if used):**

- A new weather strip is a four-key YAML (`name`, `weather_csv`, `convection_coefficient`, `start_timestamp`, which should be a Sunday 06:00).
- A new station needs a public weather CSV, which the rules allow, and an estimated convection coefficient (8.8–26.7 W/m²K in the shipped set). This is the only way past six stations.

**Policies of your own:** they are usable for data only if they are deterministic and read only zone temperatures, the clock, the comfort-band sensors and outside air. The per-room actions are recovered by replay (A.5); any other policy's actions cannot be recovered from the kit's outputs.

**Known traps:**

- `run_scenario_generator` rejects floor plans with a room under 20 m², and `min_room_size` does not prevent it, so skip failing seeds.
- `num_ahus` is a floor; the realized count can be higher. A generated building reached 12 AHUs and 120 rooms, larger than any shipped one.
- Materials are stratified per batch, so the same command reproduces a batch, but a different `--count` redraws it.
- Shorter `--steps` are fine for fitting but are not comparable competition scores.

### A.3 Validation must mirror the evaluation

| Hidden-set condition | How the split reproduces it |
|---|---|
| Unseen building | val and test buildings come from seed ranges never used in train |
| Unseen station | test uses Minneapolis only; val holds out one other station, rotated (leave-one-station-out) |
| No offline per-building fit | global constants are fitted on train only. On each val/test scenario θ is fitted on **day 1** only, then the model is scored open loop on the following days |
| All regimes | every split spans seasons and all five policies |

Template (`split_example.json`). Patterns are globs: with `/` they match `building/weather`, without they match the building, and `!` excludes.

```json
{"train": ["building_05*", "!*/phoenix*", "!*/minneapolis*"],
 "val":   ["building_07[01]*/phoenix*"],
 "test":  ["building_08[01]*/minneapolis*"]}
```

The test score is looked at once, at the end, never while choosing models.

### A.4 Commands (run from the starter-kit root)

Set up the environment once:

```bash
export NUMBA_CACHE_DIR=../working-code/.numba_cache PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
D=../working-code/data
```

**Buildings:** one batch per split.

```bash
python -m smart_buildings.run_floor_generator --start-seed 500 --count 100 --output-dir $D/buildings/train --no-pdf
```

```bash
python -m smart_buildings.run_floor_generator --start-seed 700 --count 20 --output-dir $D/buildings/val --no-pdf
```

```bash
python -m smart_buildings.run_floor_generator --start-seed 800 --count 20 --output-dir $D/buildings/test --no-pdf
```

**Scenarios:** train and val buildings get one strip from each non-Minneapolis station (rotate the indices between buildings); test buildings get Minneapolis only.

```bash
for b in $D/buildings/train/*.yaml $D/buildings/val/*.yaml; do python -m smart_buildings.run_scenario_generator --buildings $b --weathers fairbanks_1 denver_3 atlanta_5 phoenix_7 singapore_2 --output-dir $D/scenarios || true; done
```

```bash
for b in $D/buildings/test/*.yaml; do python -m smart_buildings.run_scenario_generator --buildings $b --weathers minneapolis_1 minneapolis_4 minneapolis_7 --output-dir $D/scenarios || true; done
```

**Runs:** all five shipped policies.

```bash
python -m smart_buildings.evaluate --scenarios-dir $D/scenarios --policies all_off bang_bang fixed_setpoint pid pid_water --steps 2016 --results-dir $D/results
```

**Cost.** Roughly 100 steps/s on a small building, slower on large ones; runs are independent, so run them in parallel. Start small: fewer buildings and `--steps 864`.

**Resulting layout** (what the working code reads):

```
data/scenarios/<building>/<weather>/config.gin, floor_plan.npy, zone_map.npy
data/results/<building>/<weather>/<policy>/timeseries.csv, ahu_log.csv, reward_components.csv, summary.json
```

### A.5 What the code recovers, and the check to read

The CSVs hold, per step:
- every room's temperature sensor;
- the outside temperature;
- each AHU's supply temperature, run command and total airflow;
- building occupancy, energy and carbon.

They do not hold the per-room damper and valve commands or the plant setpoints. `code/discovery/starterkit_data.py` recovers them by **replaying** the unmodified policy on observations rebuilt from the CSVs. It then checks the replay against the logged per-AHU airflow and run commands.

**Always open `outputs/discovery/<run>/data_report.json` first.** Every run must be `ok`; a "replay mismatch" means that run's inputs are unknown, and it is dropped.

Run the loop:

```bash
.venv/bin/python code/run_discovery.py --run main --scenarios-dir data/scenarios --results-dir data/results --split split_example.json --generations 0 --test
```

---

## Part B. The state-space model

### B.1 Sizes and the time step

- **V** rooms (zones, one VAV box each) and **K** air handlers. In the shipped set V is 30–85 and K is 3–9; generated buildings can be larger.
- One step is 300 s. Step *k* takes the building from the state at time *k* to the state at time *k+1*.

### B.2 The two vectors the environment exchanges

**Observation `o`** (what the policy receives; length `6V + 15K + 58`). Only a few fields feed the model:

| Field (per unit) | Role in the model | Note |
|---|---|---|
| `VAV_n_zone_air_temperature_sensor` | **output y** (room temperature) | normalized `(T − 295)/5`; **one step stale** (B.4) |
| `ahs_ahu_j_outside_air_temperature_sensor` | **disturbance d** (`T_out`) | raw K, the same in every AHU block |
| `num_occupants` | **disturbance d** (building occupants) | raw count, building total only |
| `comfort_mode_now` | **disturbance d** (internal-gain schedule) | also computable from the clock and metadata |
| `hod_*`, `dow_*` | disturbance d (clock) | one sin/cos pair of each suffices |
| `hws_supply_water_temperature_sensor` | known plant state (`T_w`) | also reproducible exactly from the setpoints (B.5) |
| `VAV_n_supply_air_damper/reheat…_command`, AHU and HWS setpoint and run echoes | echo of the control **u** just applied | not needed if the policy stores its own actions |
| heating/cooling setpoint sensors, `comfort_mode_soon` | reward context, not dynamics | what the controller wants, not how the building responds |
| request counts, duplicated run/fan fields, airflow setpoints | ignore | duplicates or constants (STATE_SPACE.md §6) |

**Action `a`** (what the policy returns): `discrete_action` (K+1) and `continuous_action` (3K+2+2V, normalized to [−1, 1]). Always read the index layout from the metadata: `temp_indices`, `pressure_indices` (K+1 entries, the last being the pump), `recirculation_indices`, `water_indices`, `zone_action_indices`. Zero is mid-range, not off.

### B.3 The model's variables

| Symbol | What | Size | Source at run time | Source in training data |
|---|---|---|---|---|
| **x** (state) | room air temperatures `T_i` (observed) **plus hidden heat stores**: slab, facade and partition/door nodes in the §14 structure, or whatever a discovered model declares | V + hidden | filtered estimate | simulated from the reset state |
| `T_w` (plant state) | boiler tank temperature | 1 | obs, or the recursion in B.5 | recursion from replayed setpoints |
| **u** (control) | per AHU *j*: run, static pressure `P_j`, supply setpoint `Tsp_j`, recirculation `r_j`. Hot-water loop: run, water setpoint, pump head `dp`. Per room *i*: damper `d_i`, reheat valve `v_i` | 4K + 3 + 2V | the policy's own previous action | replay of the shipped policy |
| **d** (disturbance) | `T_out`; building occupants `n`; comfort flag; time of day and week | 1 + 1 + 1 + 4 | obs | CSVs (`ahu_log.outside_air`, `reward_components.total_occupancy`), clock |
| **y** (output) | measured room air temperatures | V | obs (stale) | `timeseries.csv` zone columns (VAV order) |
| **G** (geometry, static) | per room: area, facade segments, AHU, floor, bounding box. Per pair: shared-wall cells, door cells, doors | V and V×V | metadata `floor_plan`, `zone_map`, `ahu_zone_assignments` | same |
| **p** (global constants) | shared by every building, such as slab coupling and spreading strength | few | fitted offline on train, shipped with the policy | fitted on train |
| **θ** (per-building constants) | facade conductivity and film, facade, partition and slab capacities, partition conductivity (4–5 identifiable) | ~6 | **estimated online from day 1** | fitted per scenario |

**What is not in the model:**
- Energy, cost, carbon and reward are not observed; they are predicted separately from **u** with the plant laws.
- Per-room occupancy is unobservable; use the building total split equally and treat the rest as process noise.

### B.4 Timing: the alignment that must be right

For step *k*, write `o_k` for the observation the policy sees before choosing `a_k`:

| Quantity used in step *k* | Where it is |
|---|---|
| Room temperatures at the start of step *k* (`x_k`) | `o_{k+1}`, the observation returned **after** step *k*. The zone sensors are one step stale |
| Damper, valve, pressure, recirculation, setpoints, run states during step *k* (`u_k`) | the action `a_k` (and echoed in `o_{k+1}`). Exception: an AHU's very first run command takes effect one step late |
| `T_out`, `T_w`, comfort flag used in step *k* | `o_k`, the observation returned **before** step *k* |
| Occupants during step *k* | `o_{k+1}` (`num_occupants`); in the data, `reward_components.total_occupancy` row *k* |

So an action first shows up in the measured temperatures **two observations later**. The model predicts `x_{k+1}` from `x_k`, `u_k` and `d_k`; the controller sees `x_{k+1}` only in `o_{k+2}`.

In the episode arrays the loop builds:
- `inputs[...][k]` belongs to step *k*;
- `T_meas[k]` is `x_{k+1}`;
- every episode starts from the reset state: every node at 294 K, tank at 340 K.

### B.5 From u, d and x to the physical inputs a room receives

These are deterministic, known from the simulator code, and identical in training and deployment (`episode_inputs` in `starterkit_data.py`, rows 3–5 of STATE_SPACE.md §13.3):

```
P_j    = max(P_cmd_j, 200 Pa) if AHU j runs else 0      r_j = min(r_cmd_j, 0.7)      d_i = max(d_cmd_i, 0.1)
flow_i = max(d_i · 2.0 · sqrt(P_j / 2000), 1e-5) · min(1, 20 / Σ_{rooms of j} unrestricted flow)          [m³/s]
g_i    = 1.2 · 1006 · flow_i                                                                    [W/K]
T_ret,j = area-weighted mean of the start-of-step temperatures of AHU j's rooms
T_sa,j  = Tsp_j if (AHU j runs and r_j T_ret,j + (1 − r_j) T_out > Tsp_j) else r_j T_ret,j + (1 − r_j) T_out
T_w,k+1 = T_w,k + clip(Tw_sp − T_w,k, −0.5, +2.5)  K, from T_w,0 = 340 K
Q_reheat,i = ε_i · g_i · (T_w − T_sa),  ε_i = min(1, water capacity rate / g_i, (308.15 − T_sa)/(T_w − T_sa))
             water flow_i = v_i · 0.108 · sqrt(dp) / 3600 if the hot-water loop runs, else 0
Q_gain,i   = 10 W/m² · area_i · (1 in comfort mode, else 0.2)
Q_occ,i    ≈ 70 W · n / V
```

### B.6 The model equation

```
x_{k+1} = f( x_k, g_k, T_sa,k, Q_k, T_out,k ; G, θ, p )        Q_k = Q_reheat + Q_gain + Q_occ
y_{k+1} = room-air part of x_{k+1}
```

The room air receives the air stream `g_i (T_sa,i − T_i)`, which must be integrated implicitly because `g` reaches about 2400 W/K. It also receives `Q_i`, and exchanges heat with its slab, its facade, the partitions and doors to its neighbours, and (through the facade) the outside. In the §14 structure every coefficient is a geometry count from **G** times a closed form in θ and p, so the same `f` applies to a building of any size. A discovered model may choose different hidden states or coefficient laws, but must keep this interface:

- **Inputs:** G, the B.5 inputs, `T_out`, the clock.
- **Constants:** a few global p and per-building θ.
- **Output:** room temperatures.

### B.7 How a model is judged

1. **Physics checks:** no drift at equilibrium; bounded cooling; heating warms the room and never cools its neighbours; stable under a full air stream (at the initial values and at both bounds of every constant).
2. **Fit:** p and one θ per training scenario, by least squares on open-loop simulation error.
3. **Held-out score:** with p frozen, θ is fitted on day 1 of each val/test scenario, then the score is the open-loop RMSE of room temperature on the following days, plus a small price per AST node.
4. **For the controller:** also check 1-, 3- and 6-hour forecast error from a Kalman-filtered state; the target is under 0.5 K at 3 h (Gate G2, STATE_SPACE.md §19.1). This is not yet ported to starter-kit data.
