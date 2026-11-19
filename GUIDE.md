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

### A.4 Commands

**The driver (recommended).** `code/generate_data.py` runs the starter kit's three tools with exactly the settings above. It writes everything under `working-code/data` (or `$SB_DATA`), creates `plan.json` and `split.json`, resumes after interruption, and records seeds the scenario generator rejects (a `.failed` marker; the building is skipped):

```bash
python code/generate_data.py --train-seeds 500:600 --val-seeds 700:720 --test-seeds 800:820 --steps 2016 --workers 8
```

Add `--dry-run` to print the plan, its size and its cost without running anything. The options are listed in `docs/RUNBOOK.md` step 2.

**By hand (the same thing, run from the starter-kit root).** Set up the environment once:

```bash
export NUMBA_CACHE_DIR=../working-code/.numba_cache PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
D=../working-code/data
```

Buildings, one batch per split (repeat for `--start-seed 700 --count 20 … /val` and `800 … /test`):

```bash
python -m smart_buildings.run_floor_generator --start-seed 500 --count 100 --output-dir $D/buildings/train --no-pdf
```

Scenarios, one call per building (rotate strip indices between buildings; test buildings get Minneapolis strips only):

```bash
for b in $D/buildings/train/*.yaml $D/buildings/val/*.yaml; do python -m smart_buildings.run_scenario_generator --buildings $b --weathers fairbanks_1 denver_3 atlanta_5 phoenix_7 singapore_2 --output-dir $D/scenarios || true; done
```

Runs, with all five shipped policies:

```bash
python -m smart_buildings.evaluate --scenarios-dir $D/scenarios --policies all_off bang_bang fixed_setpoint pid pid_water --steps 2016 --results-dir $D/results
```

By hand you must write `split.json` yourself (A.3) and delete partial scenario folders of rejected seeds.

**Cost.** Roughly 100 steps/s on a small building, slower on large ones (generated buildings reach 120+ rooms); runs are independent, so run them in parallel. Start small.

**Resulting layout** (what the working code reads):

```
data/buildings/{train,val,test}/building_NNNN.yaml
data/scenarios/<building>/<weather>/config.gin, floor_plan.npy, zone_map.npy     (+ <building>/.failed if rejected)
data/results/<building>/<weather>/<policy>/timeseries.csv, ahu_log.csv, reward_components.csv, summary.json
data/plan.json, data/split.json, data/logs/
```

### A.5 Recovering what the files do not hold, and auditing

The CSVs hold, per step:
- every room's temperature sensor;
- the outside temperature;
- each AHU's supply temperature, run command and total airflow;
- building occupancy, energy and carbon.

They do not hold the per-room damper and valve commands or the plant setpoints. `code/build_dataset.py` recovers them by **replaying** the unmodified policy on observations rebuilt from the CSVs. It checks each replay against the logged per-AHU airflow and run commands, then writes the dataset (B.8).

`code/audit_data.py` checks the data before you trust it:
- the kit is untouched;
- layout and step counts are complete;
- the reset state and Sunday start hold;
- coverage against the generator's own ranges and the station × season matrix;
- no leakage between splits;
- after the build, the replay status and how much each policy excites the building.

Run it after generation and again after the build. It exits non-zero on hard failures.

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
- **Optional:** `network(geo, inputs, theta, p)` returning the `dlib.LinearNetwork` and its per-step forcing. This exposes the state, so a Kalman filter can be run and forecasts scored (B.7).

### B.7 How a model is judged

1. **Physics checks:** no drift at equilibrium; bounded cooling; heating warms the room and never cools its neighbours; stable under a full air stream (at the initial values and at both bounds of every constant).
2. **Fit:** p and one θ per training scenario, by least squares on open-loop simulation error.
3. **Held-out score:** with p frozen, θ is fitted on day 1 of each val/test scenario, then the score is the open-loop RMSE of room temperature on the following days, plus a small price per AST node.
4. **For the controller:** also check the 1-, 3- and 6-hour forecast error from a Kalman-filtered state; the target is under 0.5 K at 3 h (Gate G2, STATE_SPACE.md §19.1). `code/test_model.py` does this for any model that defines `network()` (B.6).

### B.8 The dataset on disk

`code/build_dataset.py` writes one `data/dataset/<building>__<weather>__<policy>.npz` per run. It also writes `index.json` (per run: split, replay check, V, K) and `schema.json`. Rooms are in **VAV order** (`VAV_1, VAV_10, VAV_11, …`), the order of the observation and of the CSV zone columns. In the starter kit's `timeseries.csv`, **row k shows `x_k`** (row 0 is the reset state), because the sensors are stale by one step.

| Array | Shape | Symbol (B.3) | Meaning |
|---|---|---|---|
| `T_meas` | (n, V) | y | `x_{k+1}`, room temperatures after step k |
| `y_sensor` | (n+1, V) | y | all sensor rows; row k is `x_k` |
| `obs_lean` | (n, V+8) | o (lean) | before step k: room temperatures `x_k`, `T_w`, `T_out`, occupants, hour sin/cos, weekday sin/cos, comfort (`obs_lean_names`) |
| `u_cont_native`, `u_disc` | (n, 3K+2+2V), (n, K+1) | a | the full action vector the policy sent, in native units (`u_cont_names`, `u_disc_names`) |
| `act__damper`, `act__reheat` | (n, V) | u | realized per-room damper (≥ 0.1) and reheat valve |
| `act__ahu_on`, `act__ahu_pressure`, `act__ahu_sat_setpoint`, `act__ahu_recirculation`, `act__ahu_flow` | (n, K) | u | realized AHU controls and the resulting flow (`ahu_ids` order) |
| `act__hws_on`, `act__hws_water_setpoint`, `act__hws_dp`, `act__T_water` | (n) | u, `T_w` | hot-water loop controls and the tank temperature used in step k |
| `in__g`, `in__T_supply`, `in__Q_reheat`, `in__Q_gain` | (n, V) | B.5 | physical inputs each room receives |
| `in__T_out`, `in__occupants`, `in__comfort`, `in__hour_*`, `in__dow_*` | (n) | d | disturbances of step k |
| `geo__*` | (V), (V,V) | G | geometry (B.3) |
| `meta`, `check` | JSON | — | run identity, `T0` = 294 K; replay check |

Load one with `np.load(path)`, or as an Episode with `discovery.episodes._load_cached(path)`. A model of your own can use any of these arrays during training, but at run time it may only use what B.3 lists as available.

