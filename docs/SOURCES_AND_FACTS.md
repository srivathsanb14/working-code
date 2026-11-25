# Sources and facts behind the deck

Every number in `docs/One_Controller_for_Every_Building.pptx` is traced here. Each slide's footer names its sources
with these tags:

| tag | meaning |
|---|---|
| `[website]` | https://smart-buildings-2026.github.io/competition/ (index, competition, submission, scoring pages) |
| `[Kaggle]` | https://www.kaggle.com/competitions/neurips-2026-smart-buildings (overview, data, leaderboard, rules) |
| `[kit]` | starter-kit code and files (`../smart-buildings-starter-kit`) |
| `[audit]` | our earlier audit of the simulator code: `STATE_SPACE.md`, `outputs/*_annotated.csv` |
| `[sim data]` | real simulator runs: the kept audit files in `outputs/`, or kit `evaluate.py` runs into a scratch folder |
| `[measured]` | numbers measured for the deck (commands below) |
| `[our code]` | `working-code/code` |
| `[synthetic]` | the planted-truth self-test (`code/run_discovery.py --selftest`) |
| `[illustrative]` | a drawing, not data |
| `[decision]` / `[plan]` | agreed in planning / not built yet |

The website and Kaggle pages were read on Nov 19, 2026.

## 1. The competition (website, Kaggle)

| fact | value | source |
|---|---|---|
| Task | one policy, deployed unchanged on unseen buildings and weather | website home, competition |
| Development set | 6 public buildings × 5 weather strips = 30 scenarios | website scoring; Kaggle data |
| Final evaluation | 6 unseen buildings × 5 unseen weather stations = 30 scenarios, same generator and ranges | website scoring; Kaggle data |
| Episode | 8,640 steps of 5 min (30 days); every step scored, including the first | Kaggle evaluation |
| Reward | r = 0.6·comfort − 0.2·energy − 0.2·carbon, never positive; comfort is linear to 5 K outside the band, then quadratic and unbounded, weighted by zone occupancy | Kaggle evaluation; website reward |
| Score | mean over scenarios of \|mean per-step reward\| = −mean(return / 8,640); lower is better | website scoring |
| Determinism | each scenario is run once; returns are reproducible bit for bit | website scoring |
| Leaderboard split | public ≈ 33% of the test data, private 67% | Kaggle leaderboard |
| Leaderboard (Nov 19, 2026) | #1 0.00447, #50 ≈ 0.0059; 845 teams, 9,646 submissions | Kaggle leaderboard, overview |
| Dates | Round-one paper Nov 13, 2026; workshop Dec 13; entry and team merger Jan 3, 2027; final Jan 15, 2027 | Kaggle timeline |
| Prizes | $25,000 leaderboard (8 places) + $15,000 papers | website; Kaggle |

### Submission checklist

| requirement | detail | source |
|---|---|---|
| Form | one Kaggle notebook, nothing else | website submission; Kaggle data |
| Class | subclass `smart_buildings_inference_server.Policy`; `__init__(building_meta_data, action_spec)`; `get_action(obs)`; then `SmartBuildingsInferenceServer(TransferPolicy).run()` | website submission |
| Output | `{"continuous_action": float32 (3K+2+2V,) in [−1, 1], "discrete_action": int32 (K+1,) in {0,1}}` | website; Kaggle data |
| Validity | inside bounds (out of range is rejected); finite (a NaN passes the bounds check and corrupts the episode) | website submission |
| Metadata | action indices (temperature, pressure, recirculation, water, per-zone), observation names and zone-temperature indices, floor plan, zone map, zone→AHU, floors, schedule, comfort and eco windows, time zone, holidays, normalisation | website submission |
| Time | ≤ 9 h notebook run time (CPU or GPU); 15 min for the first step of each scenario, then 60 s per step; the whole submission in one Kaggle session | Kaggle code requirements; website submission |
| Internet | disabled; free, public external data and pre-trained models allowed | Kaggle code requirements; rules |
| Weights | load in `__init__` from a Kaggle Dataset attached to the notebook | website submission |
| State | a fresh policy per scenario; nothing carries between scenarios | website submission |
| Forbidden | hard-coding positions instead of reading the metadata; referencing held-out configurations or weather (disqualifying) | website submission |
| Limits | 5 submissions a day; 2 final submissions | Kaggle rules |
| Licence if winning | code under CC-BY 4.0 | Kaggle rules |

### Time budget (derived)

9 h ÷ (30 × 8,640 steps) ≈ **0.125 s per step**, simulator included, if the gateway runs scenarios one after another.
Whether it runs them in parallel is not documented. The first step of each scenario may take up to 15 min; any later step, at most 60 s.

## 2a. What the kit ships

**Six buildings** (`smart_buildings/buildings/building_train_*.yaml`; rooms and AHUs from `training_scenarios/*/config.gin`):

| building | seed | floors | footprint (YAML units) | rooms | AHUs | people/room | facade k (W/mK), density (kg/m³) |
|---|---|---|---|---|---|---|---|
| building_train_1 | 0 | 3 | 373 × 188 | 39 | 4 | 2 | 0.21, 1,967 |
| building_train_2 | 1 | 2 | 420 × 301 | 85 | 9 | 2 | 0.34, 578 |
| building_train_4 | 3001 | 3 | 430 × 211 | 54 | 6 | 1 | 0.18, 475 |
| building_train_5 | 4007 | 4 | 340 × 199 | 79 | 8 | 2 | 0.30, 1,187 |
| building_train_7 | 8 | 2 | 325 × 218 | 36 | 4 | 1 | 0.55, 415 |
| building_train_8 | 9008 | 2 | 318 × 198 | 30 | 3 | 2 | 0.27, 1,139 |

Their footprints are below the generator's current `DEFAULT_RANGES` (400–700 × 300–500), so generated buildings can be larger than the shipped six.

**30 scenarios** (`training_set.yaml`): each building is paired with one window from each of Atlanta, Denver, Fairbanks, Phoenix and Singapore, and no window is used twice. Minneapolis is shipped but used in no scenario.

**Weather windows** (`weather/<station>_<n>.yaml`): 30 days of recorded hourly 2023 weather, starting at 06:00 on the window's date. The window's start date overrides the building YAML's `start_timestamp`.
- `atlanta_1` starts Jan 22 and `atlanta_3` starts Apr 16: the same city in a different season.
- Window start dates, `_1` to `_8`:
  - Atlanta: Jan 22, Mar 5, Apr 16, May 28, Jul 9, Aug 20, Oct 1, Nov 12
  - Denver: Jan 15, Feb 26, Apr 9, May 21, Jul 2, Aug 13, Sep 24, Nov 5
  - Fairbanks: Jan 1, Feb 12, Mar 26, May 7, Jun 18, Jul 30, Sep 10, Oct 22
  - Minneapolis: Jan 8, Feb 19, Apr 2, May 14, Jun 25, Aug 6, Sep 17, Oct 29
  - Phoenix: Jan 29, Mar 12, Apr 23, Jun 4, Jul 16, Aug 27, Oct 8, Nov 19
  - Singapore: Feb 5, Mar 19, Apr 30, Jun 11, Jul 23, Sep 3, Oct 15, Nov 26

## 2b. What we change to make data, and what we must not

| item | change? | how / why |
|---|---|---|
| building seed | yes, the only building knob | `run_floor_generator --start-seed N --count M` (or `--seeds`) |
| weather window | yes, any of the 48 | `run_scenario_generator --weathers …` |
| driving policy | yes | `evaluate.py --policies …` |
| episode length | yes, with care | 8,640 steps is the scored length, and the plan uses it |
| generator ranges (`DEFAULT_RANGES`) | no | they define the distribution the hidden buildings come from |
| the building YAML, by hand | no | a new seed regenerates the whole YAML; never edit it |
| `config.gin` interface constants (action ranges, normalisers, reward weights) | no | they fix what the gateway hands the policy and how it is scored |
| the kit's code | no | our code only calls the kit's commands |

This repository generates no data: the runs are made with the kit's own commands, which are given only seeds, weather names, controllers and steps.

## 2. The generator (kit code; nothing generated locally)

- **Seeds.** Any integer works. Geometry depends only on the seed (`run_floor_generator.py`).
  - Materials are a Latin-hypercube draw over the whole batch (`materials.py`), so the same seed in a different batch gets different materials.
  - The kit rejects a seed whose floor plan has a room under 20 m² (`scenario_generator._check_generated_zone_areas`). Ask the generator for more seeds than the buildings needed.
- **Ranges** (`DEFAULT_RANGES`):
  - width 400–700 ft, height 300–500 ft, 2–4 floors;
  - rooms 30–60 ft (+11–16 ft span), shape ratio 0.28–0.40, coverage 0.65–0.85;
  - AHUs 3–7, raised to ⌈zones/10⌉; 1–2 occupants per zone.
- **Materials** (`materials.py`):
  - exterior conductivity 0.15–0.55 W/mK (log), exterior density 350–2,400 kg/m³ (log), exterior heat capacity 840–1,000 J/kgK;
  - interior mass density 10–40 kg/m³ (log);
  - partition conductivity 0.45–0.60 W/mK, partition density 110–700 kg/m³ (log);
  - facade U from 1/U = 0.1/k + 0.299.
- **Weather.** 6 stations × 8 windows = 48 strips (`weather/*.yaml`, hourly 2023 data in `weather/csv`).
  - Windows start every six weeks.
  - The outdoor convection coefficient is set per station (8.8–26.7 W/m²K).
  - Minneapolis is in no shipped scenario. The hidden test uses five other stations.
- **Policies.** 5 shipped: `all_off`, `bang_bang`, `fixed_setpoint`, `pid`, `pid_water`.
- **The data plan** (made with the kit's commands; GUIDE.md A.4). Stage 1: the six shipped buildings × all 48 windows × 5 policies, 30-day runs (1,440 runs). Stage 2: new seeds from 500, one window per station each. `run_pipeline.sh` then reads the runs; the split is computed from them.

## 3. Variables

- **Observation**: 6V + 15K + 58. Per zone 6; per AHU 15, which includes outside air repeated in every block; building 58 (boiler 7, clock 48, comfort flags 2, occupants 1).
  - The role of each field type, confirmed in rollouts, is in `outputs/building_train_8_observation_annotated.csv`.
  - Most fields echo a command or follow from one. The dynamics need V room temperatures plus 8 numbers: outside air, boiler water, occupants, comfort flag, and hour and day sin/cos.
- **Actions**: K+1 discrete + 3K+2+2V continuous, normalised to [−1, 1] (`outputs/building_train_8_action_annotated.csv`).
  - Native ranges: supply air 285–305 K, static pressure 0–2,000 Pa, recirculation 0–1, supply water 310–350 K, pump differential pressure 0–20, damper and reheat 0–1.
  - Floors and caps: 200 Pa while running, recirculation ≤ 0.7, damper ≥ 0.1.
- **Comfort band**: 294–297 K during comfort hours and 289–298 K otherwise; morning start 6:00, evening 19:00 (kit `config.gin`).
- **Universal constants** in the simulator:
  - start at 294 K (`config.gin initial_temp`);
  - 70 W per occupant and 10 W/m² internal gains, 0.2 outside comfort hours (audit, STATE_SPACE.md §12.2, §14.2);
  - partition heat capacity 900 J/kgK, not drawn.

## 3a. Timing found by the live tracker

- Live forecasts inside the policy equal the offline scores exactly: one-step error 0.4584 K both ways, 3 h error 2.0203 K both ways, on building_train_8/atlanta_1 over 576 steps (`run_policy.py --check-replay`).
- The `num_occupants` observation runs **two steps behind** the occupancy the kit logs in `reward_components.total_occupancy`. Offline episodes now use the observable value (`starterkit_data.episode_inputs`).

## 3b. The cost estimate checked against the kit

`control/costs.py` estimates electric power (fans, cooling at COP 3.5, pump) and gas power (reheat) from the commands. It then applies the kit's price and carbon tables and the per-AHU normalisation. Checked against `reward_components.csv` on two 30-day pid_water runs:

| run | electric power: corr, estimate/kit | gas power: corr, estimate/kit | normalised energy: corr, estimate/kit |
|---|---|---|---|
| train_2 atlanta_3 | 0.999, 1.03 | 0.977, (mean 156 vs 326 W) | 0.999, 0.97 |
| train_1 fairbanks_1 | 0.999, 0.93 | 0.977, 1.11 | 0.993, 1.03 |

The small gap in mild weather is boiler loss, which the estimate leaves out. The policy cannot read the month, so the gas price is the annual mean.

## 4. How deviations are measured

- **Model** (`closedform.py`):
  - one-step RMSE; free-run rollout RMSE at 1/3/6 h, started every 30 min from measured history;
  - skill against persistence; bias;
  - error by group: exterior/interior/most exposed, HVAC on/off, comfort hours, reheat, cold/warm outside;
  - stability (any rollout outside 230–350 K is rejected);
  - gap to the per-room fitted ceiling.
- **Controller**: exactly as the reward does.
  - Kelvin outside the active band, weighted by occupants, plus energy cost and carbon (the kit's `reward_components.csv` logs each term).
  - Reported per scenario, per climate, worst case and per term (evaluation protocol: STATE_SPACE.md §19).

## 5. Check on real 30-day simulator episodes (nothing fitted)

**Data.** 12 full episodes of 8,640 steps: the 6 shipped buildings × (pid_water, HVAC off), one weather window each (train_1 fairbanks_1, train_2 atlanta_3, train_4 denver_2, train_5 phoenix_1, train_7 singapore_6, train_8 atlanta_1).
- Produced with the kit's own `evaluate.py` into a scratch folder, only for the deck's figures. Nothing went into the repo.
- Replayed into episodes with our `starterkit_data.load_starterkit_episode`, so the inputs are rebuilt from commands, as in deployment.
- Replay check: AHU flow error ≤ 0.02% and run commands 100% matched in all 12.
- pid_water scores per scenario: Fairbanks 0.214, Atlanta-3 0.020, Denver 0.055, Phoenix 0.033, Singapore 0.030, Atlanta-1 0.031.

| model | 5 min | 1 h | 3 h | 6 h |
|---|---|---|---|---|
| persistence | 0.35 | 0.66 | 0.97 | 1.47 |
| `seed_cf_network` (fixed-function coefficients: air + slab, outside-wall and inner-wall stores; mid-range materials) | 0.30 | 0.68 | 0.87 | 1.11 |
| `seed_cf_history` (fixed-function coefficients: air + slab store, outside-wall U from history) | 0.36 | 1.17 | 1.40 | 1.64 |
| `seed_cf_one_node` (fixed-function coefficients, air only, nominal materials; unstable on 1 episode) | 0.52 | 3.64 | 6.61 | 7.94 |
| ceiling: per-room fitted, second half of each episode | — | 0.31 | 0.46 | 0.62 |

RMSE in K.
- **The hand-written starting models are not good enough.** At 3 h the network model loses to persistence in 3 of 12 runs and the history model in 9 of 12 (all six HVAC-off runs). The fitted ceiling beats persistence in all 12. The gap between the starting models and the ceiling is what the LLM search has to close.
- An earlier 3-day check on audit files looked better: 1.47 against 2.40 K at 3 h. Those files used the simulator's own logged heat input instead of inputs rebuilt from commands, and a period with larger swings.
- **Bug found and fixed:** the first version of the ceiling (unconstrained least squares) blew up in free run, reaching 101 K at 3 h. It now uses non-negative weights and the stable end-of-step form.

## 6. Corrections to earlier deck versions

| earlier claim | corrected | source |
|---|---|---|
| "9 hours, so about 0.1 s per step" | 9 h notebook limit (Kaggle). The website adds 15 min for the first step and 60 s per step. 0.125 s per step is derived, and it includes the simulator | Kaggle, website |
| "the generator makes up to about 120 rooms, 3–12 AHUs" | removed: the deck now quotes only the kit's parameter ranges | kit |
| "about 8 material numbers per building" | 6 material properties per building; the outdoor film coefficient is set per weather station | kit `materials.py`, `weather/*.yaml` |
| per-building list included partition specific heat and the outdoor film | partition heat capacity is fixed; the film belongs to the station | kit |
| "a seed always gives the same building" | the same geometry; the materials depend on the batch | kit `run_floor_generator.py` |
| comfort band "21–24 °C" | 294–297 K (20.85–23.85 °C) day, 289–298 K night | kit `config.gin` |
