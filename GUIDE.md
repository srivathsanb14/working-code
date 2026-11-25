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
| Which buildings, and how many | `run_floor_generator --start-seed --count` (or `--seeds`) | The hidden buildings are new draws from this generator; the building is the unit of generalization | The six shipped buildings, then as many new seeds as compute allows (stage 2, A.4) |
| Batch composition | `--count` per call | Materials are spread evenly across each batch, and a different `--count` redraws them | One batch per stage, so it covers the material ranges |
| Which weather strips | `run_scenario_generator --weathers` (48 shipped strips: 6 stations × 8 dates, about 6 weeks apart) | Hidden stations are unseen; windows can fall in any season | Shipped buildings: all 48 (every season at every station, Minneapolis included). New buildings: one per station. Which station is held out is decided by the split (`make_splits.py`), not here |
| Building–weather pairing | naming `--buildings` and/or `--weathers` takes the cross product | Every building seen in every season | Shipped buildings: the full cross product. New buildings: rotate the strip index across buildings |
| Behaviour policies | `evaluate --policies` | Excitation decides what can be identified | All five shipped policies: `all_off` (free-float), `bang_bang` (strong steps), `fixed_setpoint`, `pid`, `pid_water` (realistic) |
| Episode length | `evaluate --steps` | Longer runs give more forecast windows and every phase of the month | 8640 (the full, scored 30 days, the default). Shorter only to save compute |
| Where outputs go | `--output-dir`, `--scenarios-dir`, `--results-dir` | — | Outside the kit, on the data machine (for example `working-code/data/`) |

**Must not change** (configurable, but fixed in every shipped scenario and assumed identical in the hidden set; the model hard-codes them):

- Generator sampling ranges (`DEFAULT_RANGES` in `run_floor_generator.py`), footprint shape (`composite`), and `--no-materials`.
- Plant sizing: AHU maximum flow 20 m³/s, VAV maximum flow 2.0 m³/s, fan and pump efficiencies, boiler ramp rates, reheat flow factor.
- Floors and limits: damper ≥ 0.1, static pressure ≥ 200 Pa when running, recirculation ≤ 0.7, discharge air ≤ 308.15 K.
- Schedule and comfort bands, time zone, reward weights and scales, internal gains (10 W/m², standby fraction 0.2), 70 W per occupant.
- Interior material constants (air, slab conductivity and heat capacity), cell size 0.2 m, floor height 3 m, time step 300 s.
- The simulator, reward, environment code, and the gin interface (action ranges, normalizers).
- Simulation start timestamp in a building YAML: it is overridden by the weather strip anyway.

**Cannot change** (fixed by the kit; matches the evaluation):

- Every strip starts on a **Sunday at 06:00**. Day 1 is therefore an unoccupied eco-mode Sunday, exactly as for a deployed policy (and the only day online identification could use).
- The wind film coefficient of each station (in the strip YAML) is not given to a policy: formulas must infer its effect from the measured history.
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
- Shorter `--steps` save compute but are not comparable competition scores.

### A.3 Validation must mirror the evaluation

There is no training split (nothing is fitted). `code/make_splits.py` computes the split from whatever episodes exist, by
rule; nothing is written by hand:

| Hidden-set condition | How the split reproduces it |
|---|---|
| Unseen building | the most unusual quarter of buildings by floor-plan numbers is held out: all their windows go to test (medium, or hard at the hard station) |
| Unseen station | the station whose outside temperatures are farthest from the rest (Wasserstein-1) is the hard station: held-out buildings there are test_hard, seen buildings there are test_station |
| Unseen month | about one window in four of a seen building at a seen station is test_easy |
| No per-building fit | nothing is fitted anywhere; candidates are scored on validation by forecasts from measured history, as at test time |

Whole scenarios (a building and a window, every controller's run on it) go to one split, and the script stops if any would
be in two. The test tiers are scored once, at the end, never while choosing models.

### A.4 Commands

This repository has no data generator of its own: the data is made with the starter kit's commands, run from the kit root, as below. The pipeline then only reads the finished runs.

**With the kit's commands (run from the starter-kit root).** Set up the environment once:

```bash
export NUMBA_CACHE_DIR=../working-code/.numba_cache PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
D=../working-code/data
```

**Stage 1: every shipped building in every window.** Naming only `--buildings` gives all 48 windows: 6 × 48 = 288
scenarios, × 5 controllers = 1,440 runs of 30 days.

```bash
python -m smart_buildings.run_scenario_generator --buildings building_train_1 building_train_2 building_train_4 building_train_5 building_train_7 building_train_8 --output-dir $D/scenarios
python -m smart_buildings.evaluate --scenarios-dir $D/scenarios --policies all_off bang_bang fixed_setpoint pid pid_water --results-dir $D/results
```

**Stage 2: new buildings.** Start with 20 (ask for about 26: the kit rejects about one seed in five), each on one window
per station with the window index rotated across buildings, × 5 controllers = 600 runs; add more as compute allows.

```bash
python -m smart_buildings.run_floor_generator --start-seed 500 --count 26 --output-dir $D/buildings --no-pdf
k=1; for b in $D/buildings/*.yaml; do python -m smart_buildings.run_scenario_generator --buildings $b --weathers atlanta_$k denver_$k fairbanks_$k minneapolis_$k phoenix_$k singapore_$k --output-dir $D/scenarios || true; k=$((k % 8 + 1)); done
python -m smart_buildings.evaluate --scenarios-dir $D/scenarios --policies all_off bang_bang fixed_setpoint pid pid_water --results-dir $D/results
```

Then, in `working-code`: `SCENARIOS=$D/scenarios RESULTS=$D/results ./run_pipeline.sh` (episodes, split, discovery).
Delete the partial scenario folders of rejected seeds.

**Cost.** The simulator took 0.025–0.057 s per step on the shipped buildings, so a 30-day run is about 4–8 minutes: stage 1 is roughly 100–200 CPU-hours, and generated buildings (up to 120+ rooms) are slower. Runs are independent, so run them in parallel.

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
| **U** (universal constants) | the same for every building and weather: time step, cell size, floor height, air heat capacity, 70 W per person, internal gains | ~10 | shipped with the policy (`discovery/universal_constants.json`) | from the kit's code |
| **materials** (per building) | facade conductivity and film, facade, partition and slab capacities, partition conductivity | ~6, never given | **inferred inside the formulas from the measured history**; nothing is fitted | not used |

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
x_{k+1} = f( x_k, g_k, T_sa,k, Q_k, T_out,k ; G, U, measured history )        Q_k = Q_reheat + Q_gain + Q_occ
y_{k+1} = room-air part of x_{k+1}
```

The room air receives the air stream `g_i (T_sa,i − T_i)`, which must be integrated implicitly because `g` reaches about 2400 W/K. It also receives `Q_i`, and exchanges heat with its slab, its facade, the partitions and doors to its neighbours, and (through the facade) the outside. In the §14 structure every coefficient is a geometry count from **G** times a material property, so the same `f`
applies to a building of any size. Discovery keeps this structure fixed (`prompts.CF_CONTRACT`): per room, the next air
temperature is a weighted average of its heat sources and three hidden stores (slab, outside wall, inner walls), and
the LLM writes a closed-form formula for each of the 11 coefficients from the floor plan, the measured history (up to
288 steps) and universal constants. Nothing is fitted: per-building materials are inferred from the history inside the
formulas (see `docs/DISCOVERY_WALKTHROUGH.md`).

### B.7 How a model is judged

1. **Contract and physics checks:** allowed imports only, no fitted tables, no branching on building size; no drift at
   equilibrium, bounded cooling, heating warms the room and never cools a neighbour, stable under a full air stream.
2. **Screen:** one-step (5-minute) error on the validation episodes from measured history; more than 5× persistence is
   dropped.
3. **Score:** forecasts of 1, 3 and 6 h started every 30 minutes from measured history, with the controller's own future
   commands known and future outside temperature and occupancy held at their last values (as at test time). Main score =
   3 h error + 0.0001 K per syntax-tree node; selection is Pareto over five errors (main, fast, slow, exterior, interior).
4. **Test tiers:** the chosen model and the Pareto front are scored once at the end on test_easy, test_medium, test_hard
   (and test_station); never used for selection.

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

