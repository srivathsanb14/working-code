# LLM discovery: where everything happens, and how to tell whether it works

This is the guide to the default discovery mode: **closed-form formulas with no fitting anywhere**, searched by an
LLM with islands, mutation and crossover. It is the only mode: the earlier fitted modes have been removed.

## 0. What the LLM writes: the coefficient formulas of a fixed state-space function

The dynamics function is fixed (`discovery/coefform.py`, `TermModel`). Per room i, every 5-minute step, with a hidden
slab store S, outside-wall store W and inner-wall store P per room:

```
T_i+ = ( T_i + air_i g_i T_supply_i + outside_i T_out + sum_j neighbours_ij T_j + slab_i S_i + wall_i W_i + inner_i P_i
         + heat_i Q_i ) / ( 1 + air_i g_i + outside_i + sum_j neighbours_ij + slab_i + wall_i + inner_i )
S_i+ = ( S_i + slab_back_i T_i+ ) / ( 1 + slab_back_i )
W_i+ = ( W_i + wall_back_i T_i+ + wall_out_i T_out ) / ( 1 + wall_back_i + wall_out_i )
P_i+ = ( P_i + inner_back_i T_i+ ) / ( 1 + inner_back_i )
```

then the room air is set to the measurement. Each line is the backward-Euler form of `x+ = x + sum_k a_k (source_k - x)`,
stable for any coefficients >= 0; together they are `x(t+1) = A(g) x(t) + B(g) u(t) + E(g) d(t)` with `x = (T, S, W, P)`.

The LLM writes only the **formulas for the coefficients** (`air, heat, outside, neighbours, slab, slab_back, wall,
wall_back, wall_out, inner, inner_back`) and optionally the heat input:

```python
HISTORY = 288; REFRESH = 36
def coefficients(geo, hist, U):   -> {"air": (V,), "heat": (V,), "outside": (V,), "neighbours": (V, V), ...}
def heat(geo, u, U):              -> (B, V) W   # optional
```

Each formula may read only test-time information: geometry from the building metadata, the universal constants, and the
measured history up to now (`hist`, re-evaluated every `REFRESH` steps). The building's configuration file is never
available at test time. Mutation, crossover, exploration and islands evolve these formulas; generated data on unseen
buildings decides which survive. Nothing is fitted. `tests/test_coefform.py` checks that the formulas only see test-time
keys and nothing after the refresh step. A network form (`nodes()` + capacities and conductances) and the older
`step()` / stateful forms still load.

Starting models in this form are in `seeds_cf/` (`seed_cf_network`, `seed_cf_history`, `seed_cf_one_node`); their scores
on the 12 thirty-day runs are in docs/SOURCES_AND_FACTS.md.

## 1. Check the scorer, with no data and no API call

```bash
.venv/bin/python code/run_discovery.py --selftest
```

A **synthetic** check of the scorer: the shipped floor plans with made-up weather and HVAC inputs, and a known formula
(the planted truth) generating the "measured" temperatures. The truth must score about 0 K and beat every hand-written
model. The loop itself is tested with a stand-in for the API client (`tests/test_api_loop.py`): every request is a
single user message under the same system prompt, under 40 KB, with no episode names.

## 2. One generation, step by step (code locations)

| step | what happens | code |
|---|---|---|
| data | episodes from `data/dataset`, split into **validation** (all the search sees) and three held-out **test tiers**: easy (seen building and station, unseen weather window), medium (unseen building), hard (unseen building and the station farthest in Wasserstein distance). No training split: nothing is fitted. The split is computed from the episodes that exist (`splits.split_from_episodes`) unless `split.json` already has the tiers | `evolve.load_split`, `discovery/splits.py`, `code/make_splits.py` |
| generation 0 | per island, one API request (`init`) for `--n-init` first candidates written from the system prompt alone; the hand-written models are not in the population (`--baselines` scores them as reference rows) | `prompts.cf_user_init`, `evolve.run` |
| islands | each island is its own population: its best `--elites` programs by **Pareto rank + crowding** over `--objectives` (default main 3 h score, 1 h, 6 h, exterior rooms, interior rooms; `--selection scalar` for the single score) | `evolve.DB.elites`, `discovery/pareto.py`, `closedform.OBJECTIVES` |
| parent choice | a 2-way tournament inside the island, by Pareto rank then crowding | `evolve.tournament` |
| **pre-mutation analysis** | on the validation episodes: TEMPORAL (error and bias vs horizon against persistence, by hour of day, occupancy, HVAC state and switching during the rollout, outside temperature, episode phase; autocorrelation of the one-step error), SPATIAL (by room class, facade side, floor; correlation of each room's bias with its geometry; whether adjacent rooms err together; worst rooms), TERMS (the one-step residual regressed on each term's exact contribution: which coefficient is too strong or too weak); plus a measured-vs-predicted table | `closedform.analysis`, `temporal_stats`, `spatial_stats`, `term_regression`, `evolve.profile_text` |
| **mutation request** | one LLM request per island: parent source + analysis → `--n-mut` children, each changing one thing | `prompts.cf_user_mutation` |
| **pre-crossover analysis** | for two parents: who wins where (spatial groups) and when (temporal groups), per-room wins and the best-of-both bound, which parent's formula is closer to right for each term, which functions differ, coefficient values side by side | `closedform.crossover_analysis` |
| **crossover request** | one LLM request per island: both parents + that analysis → `--n-cx` children combining templates | `prompts.cf_user_crossover` |
| **explore request** | one LLM request per island: the island's leaderboard → `--n-exp` structurally new candidates | `prompts.cf_user_explore` |
| proposal | the Claude API answers each request: one stateless request (the cached system prompt + this request, no chat history); the answer is saved before evaluation; `--max-usd` caps the run | `proposers.ClaudeProposer`, called in `evolve.run` |
| stage 0 | physics checks on a real floor plan: equilibrium, bounded cooling, heat response, stiff air stream | `closedform.physics_checks` |
| stage 1 | quick screen: one-step (5 min) RMSE on the validation episodes; worse than `--screen` × persistence is screened out | `closedform.evaluate` |
| stage 2 | decisive: free-run rollouts from measured history every `--every` steps, horizons `--horizons` (default 1/3/6 h), stability check; score = `--main-horizon` RMSE + complexity; the objectives vector for Pareto selection | `closedform.episode_scores`, `closedform.rollout` |
| record | one JSON line per candidate: island, operator, parents, scores, analysis | `evolve.record` → `database.jsonl` |
| migration | every `--migrate-every` generations each island's best is copied into the next island (a ring) | end of the generation loop in `evolve.run` |
| report | REPORT.md, plots, summary.json, best.py, rewritten after every generation | `report.write` |
| test tiers | at the end only (`--test`): the chosen model and up to `--test-front` Pareto-front models scored on test_easy / test_medium / test_hard (and test_station); table in log.md, numbers in summary.json | `evolve.score_tests` |

### Added on Nov 20, 2026

- **Parallel evaluation:** `--workers N` evaluates a generation's candidates in forked processes (`evolve.evaluate_many`), with the same results as serial.
- **Resume:**
  - each request's answer is saved to `candidates.json` before evaluation;
  - `in_progress.json` marks an unfinished generation, so the next run finishes it and reuses the saved answers.
- **LLM cost:** tokens and an estimated cost per request (in `log.md` and `candidates.json`) and per run. Edit `ClaudeProposer.PRICE_PER_MTOK` to match the current price list.
- **Robust answers:** code fences, prose around the JSON and duplicate names are accepted. A malformed answer is retried once, with the reason sent back to the model.
- **More checks:**
  - a rejection when a warmer outside makes a room colder;
  - a warning, not a rejection, when shifting every temperature by a constant does not shift the forecast by the same amount;
  - a 24-hour free-run error and drift (`val_24h_mean`, `val_24h_drift_mean`).
- **More inputs:**
  - geometry `facade_top/bottom/left/right`, `facade_sides` (2 or more sides = corner room), `boundary_faces`, `n_neighbours`, `aspect`;
  - the input `T_water`, the boiler supply-water temperature.

  All of them are in the prompt.
- **Lineage:** `plots/lineage.png` in every report.
- **Inside the policy:**
  - `runtime/closedform_tracker.py` runs `best.py` from the metadata, the observations and the policy's own actions;
  - `run_policy.py --check-replay` shows the live forecasts equal the offline scores;
  - this check found that the occupancy observation runs two steps behind the logged occupancy, and the offline episodes now use the observable value.

### Stateful candidates (decision O3)

A candidate may instead keep hidden stores (slab, walls) as a **state corrected by measurements**:
- `init_state` gives the starting state;
- `predict` advances it one step;
- `correct` folds in the measured room temperatures, with a fixed rule and nothing fitted.

The scorer filters the measured history this way, then free-runs from the corrected state (`closedform.filter_states`). The runtime tracker does the same live, and its forecasts equal the offline scores exactly.

`seeds_cf/seed_cf_network.py`, the audited network with mid-range universal materials, was scored on 12 thirty-day simulator episodes:

| model | 1 h | 3 h | 6 h |
|---|---|---|---|
| network observer seed | 0.74 | 0.93 | 1.16 |
| history seed | 0.99 | 1.16 | 1.34 |
| persistence | 0.66 | 0.97 | 1.47 |

RMSE in K. The observer seed is also about 5× faster to score.

## Deployment parity: discovery uses only what a policy has at Kaggle test time

On an unseen building the policy gets the building metadata once, an observation every 5 minutes, and remembers its own actions. Nothing else exists: no YAML, no `config.gin` and no materials. Discovery is held to the same limits.

| what discovery uses | available at test time? | where it comes from at run time |
|---|---|---|
| geometry (`geo`: areas, exterior wall by side, shared walls, doors, AHU, floor, ...) | yes | `floor_plan`, `zone_map`, `ahu_zone_assignments` in the metadata (`episodes.geometry`) |
| measured room temperatures (history buffer) | yes, one step late | observation, de-normalised with the metadata (`runtime/inputs.RoomInputs.measured`) |
| outside temperature (now and history) | yes | observation |
| occupants (building total) | yes, **two steps late** | observation; offline episodes use the same lagged value |
| comfort flag, clock | yes | observation |
| air streams `g`, supply temperature, reheat heat | yes | the policy's own commands through the kit's equipment laws (`runtime/inputs.py`), with return air from measured temperatures |
| boiler water temperature `T_water` | yes | integrated from the policy's own water-setpoint commands (ramp limits) from the 340 K reset |
| internal gains `Q_gain` | assumed | 10 W/m² × area (20% outside comfort hours): a kit-wide constant, not in the metadata (risk tracked in I6) |
| universal constants `U` | assumed | kit-wide values, each citing its kit source (`universal_constants.json`) |
| **future** outside temperature and occupancy | **no** | the score holds them at their last observed values (`closedform.EXOGENOUS`; `--score deploy`, the default) |
| future commands | yes | they are the controller's own plan |
| building identity, materials, weather window, policy name | **no** | never given to candidates; the analysis text the LLM reads names episodes anonymously |

How this is enforced:
1. **Sandbox and contract.** Candidates receive only `geo`, the history buffer, this step's inputs and `U`. They cannot open files or import anything other than numpy, math and dlib. Code that compares a building's size or geometry to fixed numbers (for example `if geo["V"] == 85`) is rejected (`contract._reject_building_specific`).
2. **Test-time score.** Stage 2 scores rollouts with the future outside temperature and occupancy unknown. Rollouts with the true future inputs are reported as a diagnostic (`val_<h>_mean`).
3. **Live parity check.** `run_policy.py --check-replay` runs the model inside a submission-style policy from the Kaggle-style metadata and observations, then compares with the offline episodes:
   - geometry: identical in every field;
   - inputs: identical to rounding;
   - forecasts: equal to the offline scores (0.6014 K both ways at 3 h on building_train_8/atlanta_1).
4. **Held-out evaluation.** Selection uses buildings never seen in training, and the final check uses unseen buildings on an unseen station.
5. **Tests.** `tests/test_deployment.py` covers the allowed keys, the branching rule, the test-time score and the frozen inputs.

On the 12 thirty-day episodes, the stateful seed's 3 h error is 0.959 K at test time vs 0.928 K with true future inputs (persistence 0.973 K).

## 3. Where to look in a run folder (`outputs/discovery/<run>/`)

| file | what it tells you |
|---|---|
| `REPORT.md` | start here: best model, leaderboard, operator statistics, island progress, the best model's family tree, plots, every generation's requests and children |
| `log.md` | the timeline: each request (island, operator, parents, links to its prompt and analysis) and each child (score and Δ against its best parent), migrations |
| `system_prompt.md` | what every request shares: task, physics hypotheses, data dictionary, universal constants, contract, scoring |
| `generations/gNNN/island_K/<operator>/prompt.md` | the exact user message of that request |
| `generations/gNNN/island_K/<operator>/analysis.md` | the pre-mutation or pre-crossover analysis it was given |
| `generations/gNNN/island_K/<operator>/response.json` | the raw LLM response (Claude proposer) |
| `candidates/<id>_<name>.py` | every candidate's source; the id says generation and island (`g002_i1_0022`) |
| `plots/<id>_profile.png` | measured vs predicted temperature (5 min and 3 h ahead) for the most exposed exterior room and the largest interior room |
| `plots/<id>_horizon.png` | held-out error vs horizon, against persistence and the fitted ceiling |
| `plots/progress.png` | best score per island per generation |
| `plots/lineage.png` | the family tree of each island's best |
| `generations/.../candidates.json` | the answer of each request (saved before evaluation; used to resume) |
| `ceiling.json` | `--ceiling`: the per-building fitted reference (diagnostic only) |

## 4. How to tell whether the LLM can fit at all

| check | question it answers | how |
|---|---|---|
| planted truth | is the scorer right? (alignment of history, inputs and targets) | `--selftest`: the truth must score ~0 K and rank first. It does: 0.0 K at every horizon. |
| persistence (floor) | is a model better than "the temperature stays"? | in every leaderboard row and plot |
| fitted ceiling | how well can **any** model with free per-room numbers do on this data? | `--ceiling`: a per-room linear model fitted on the first half of each held-out episode, scored on the second half. A large gap between ceiling and the best formula means the formulas, not the data, are the limit. |
| loop test | do islands, requests, logs and the test tiers work, with stateless requests only? | `tests/test_api_loop.py` (a stand-in API client) |
| operator statistics | do mutation and crossover actually improve on their parents? | REPORT.md, "Operators": "better than best parent" and median Δ |

## 5. The guidelines, checked against the code

| guideline | status | where |
|---|---|---|
| One set of formulas for any building, from layout + measured history; used for MPC | done | contract in `prompts.CF_CONTRACT`; `step(coef, geo, hist, u, U)` |
| No fitting at deployment or beforehand; the only estimation is a closed-form history expression | done (default mode) | `contract.load_closedform` rejects any PARAMS/THETA; `closedform.evaluate` fits nothing |
| Universal constants allowed in formulas; reviewed once before the prompt | done; **review pending** | `code/discovery/universal_constants.json` (`_reviewed: false` until someone checks it) |
| Per-building values inferred from history by formulas | supported; shown in a seed | `seed_cf_history.facade_u` (shrunk least squares over the buffer) |
| Script extracting per-room geometry from floor_plan + zone_map | done | `episodes.geometry` (uses `room_model.grid_geometry`) |
| Conductance = conductivity × area / distance; layers in series | stated in the prompt; used in seeds | `prompts.CF_CONTRACT`, seeds |
| Templates: same for every room; rooms differ by their numbers | stated in the prompt; crossover swaps templates | `prompts.CF_SYSTEM`, `closedform.crossover_analysis` |
| Assembled end-of-step form T⁺ = (T + a·T_s + b)/(1 + a) | helper given; physics enforced by checks, not by the assembler | `dlib.assemble`, `dlib.end_of_step`, `closedform.physics_checks` |
| Hidden states optional; history buffer instead | done | `HISTORY` (1–288 steps), `hist[...]` |
| Discrete-time maps allowed | done | the contract is a one-step map |
| Stage 1 one-step screen (no filtering on noiseless simulator data) | done | `closedform.evaluate` |
| Stage 2 rollouts over the MPC horizon (not yet chosen), stability | done; horizon is a flag | `--horizons`, `--main-horizon` |
| Early-episode error acceptable while the buffer fills | done | `--warmup` steps are not scored; the buffer is padded |
| Few observed inputs: boiler water, outside temperature, comfort flag, time code | partly | T_out, comfort and time go straight to the formulas. Boiler water temperature enters through `Q_reheat`, which a hand-written conversion computes from commands (`starterkit_data.episode_inputs`). |
| Nonlinearity (commands → heat: products, limits, on/off) | **hand-written, not discovered** | `starterkit_data.episode_inputs`, `runtime/inputs.py`; the formulas get heat and air streams |
| Energy not needed for the dynamics | done (not modelled) | the known gaps are listed in the deck |
| Real data need filtering; the simulator does not | not built (simulator only) | — |
| Occupant split between rooms (open question) | seeds split evenly; formulas may do otherwise | seeds |
| Closed-form model inside the submission policy | done | `runtime/closedform_tracker.py`, `run_policy.py --check-replay` |
