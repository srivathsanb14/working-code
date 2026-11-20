# LLM discovery: where everything happens, and how to tell whether it works

This is the guide to the default discovery mode: **closed-form formulas with no fitting anywhere**, searched by an
LLM with islands, mutation and crossover. The earlier fitted modes (`--deployment offline|online|both`) are still
available for comparison; see [LLM_DISCOVERY.md](LLM_DISCOVERY.md).

## 1. Try it in four minutes, with no data and no LLM

```bash
.venv/bin/python code/run_discovery.py --selftest
```

This writes `outputs/discovery/selftest/`. It is a **synthetic** run: the shipped floor plans with made-up weather
and HVAC inputs. A known formula (the planted truth) generates the "measured" temperatures, and the LLM is replaced
by a mechanical proposer. Open `outputs/discovery/selftest/REPORT.md`. Every file a real run writes is there.

## 2. One generation, step by step (code locations)

| step | what happens | code |
|---|---|---|
| data | episodes from `data/dataset` (built by `code/build_dataset.py`), split into train / val / test buildings | `evolve.load_split` |
| seeds | hand-written starting programs, evaluated once and copied into every island | `code/discovery/seeds_cf/*.py`, `evolve.run` (generation 0) |
| islands | each island is its own population: its best `--elites` programs | `evolve.DB.elites(island, k)` |
| parent choice | a 2-way tournament inside the island | `evolve.tournament` |
| **pre-mutation analysis** | the parent's held-out error by room class (exterior, interior, most exposed), HVAC on/off, comfort hours, reheat, cold/warm outside, by horizon vs persistence, worst rooms; plus a measured-vs-predicted table for two rooms | `closedform.analysis`, `evolve.profile_text` |
| **mutation request** | one LLM request per island: parent source + analysis → `--n-mut` children, each changing one thing | `prompts.cf_user_mutation` |
| **pre-crossover analysis** | for two parents: who is better on which group, and which templates (functions, constants) differ | `closedform.crossover_analysis` |
| **crossover request** | one LLM request per island: both parents + that analysis → `--n-cx` children combining templates | `prompts.cf_user_crossover` |
| **explore request** | one LLM request per island: the island's leaderboard → `--n-exp` structurally new candidates | `prompts.cf_user_explore` |
| proposal | the proposer answers each request: `dry` (writes the prompt only), `dir` (reads `.py` files you drop in the request's `inbox/`), `mock` (no LLM; rescales numbers or swaps templates), `claude` | `proposers.py`, called in `evolve.run` |
| stage 0 | physics checks on a real floor plan: equilibrium, bounded cooling, heat response, stiff air stream | `closedform.physics_checks` |
| stage 1 | quick screen: one-step (5 min) RMSE on held-out buildings; worse than `--screen` × persistence is screened out | `closedform.evaluate` |
| stage 2 | decisive: free-run rollouts from measured history every `--every` steps, horizons `--horizons` (default 1/3/6 h), stability check; score = `--main-horizon` RMSE + complexity | `closedform.episode_scores`, `closedform.rollout` |
| record | one JSON line per candidate: island, operator, parents, scores, analysis | `evolve.record` → `database.jsonl` |
| migration | every `--migrate-every` generations each island's best is copied into the next island (a ring) | end of the generation loop in `evolve.run` |
| report | REPORT.md, plots, summary.json, best.py, rewritten after every generation | `report.write` |

## 3. Where to look in a run folder (`outputs/discovery/<run>/`)

| file | what it tells you |
|---|---|
| `REPORT.md` | start here: best model, leaderboard, operator statistics, island progress, the best model's family tree, plots, every generation's requests and children |
| `log.md` | the timeline: each request (island, operator, parents, links to its prompt and analysis) and each child (score and Δ against its best parent), migrations |
| `system_prompt.md` | what every request shares: task, physics hypotheses, data dictionary, universal constants, contract, scoring |
| `generations/gNNN/island_K/<operator>/prompt.md` | the exact user message of that request |
| `generations/gNNN/island_K/<operator>/analysis.md` | the pre-mutation or pre-crossover analysis it was given |
| `generations/gNNN/island_K/<operator>/response.json` | the raw LLM response (Claude proposer) |
| `generations/gNNN/island_K/<operator>/inbox/` | `dir` proposer: put candidate `.py` files here and re-run the same command |
| `candidates/<id>_<name>.py` | every candidate's source; the id says generation and island (`g002_i1_0022`) |
| `plots/<id>_profile.png` | measured vs predicted temperature (5 min and 3 h ahead) for the most exposed exterior room and the largest interior room |
| `plots/<id>_horizon.png` | held-out error vs horizon, against persistence and the fitted ceiling |
| `plots/progress.png` | best score per island per generation |
| `ceiling.json` | `--ceiling`: the per-building fitted reference (diagnostic only) |

## 4. How to tell whether the LLM can fit at all

| check | question it answers | how |
|---|---|---|
| planted truth | is the scorer right? (alignment of history, inputs and targets) | `--selftest`: the truth must score ~0 K and rank first. It does: 0.0 K at every horizon. |
| persistence (floor) | is a model better than "the temperature stays"? | in every leaderboard row and plot |
| fitted ceiling | how well can **any** model with free per-room numbers do on this data? | `--ceiling`: a per-room linear model fitted on the first half of each held-out episode, scored on the second half. A large gap between ceiling and the best formula means the formulas, not the data, are the limit. |
| dry run | is the prompt right before spending money? | `--proposer dry` writes every request; read them |
| mock run | do islands, mutation, crossover, migration and the logs work? | `--proposer mock` (no LLM) |
| operator statistics | do mutation and crossover actually improve on their parents? | REPORT.md, "Operators": "better than best parent" and median Δ |

## 5. The guidelines, checked against the code

| guideline | status | where |
|---|---|---|
| One set of formulas for any building, from layout + measured history; used for MPC | done | contract in `prompts.CF_CONTRACT`; `step(coef, geo, hist, u, U)` |
| No fitting at deployment or beforehand; the only estimation is a closed-form history expression | done (default mode) | `contract.load_closedform` rejects any PARAMS/THETA; `closedform.evaluate` fits nothing |
| Universal constants allowed in formulas; reviewed once before the prompt | done; **review pending** | `code/discovery/universal_constants.json` (`_reviewed: false` until someone checks it) |
| Per-building values inferred from history by formulas | supported; shown in a seed | `seed_cf_history_slab.facade_u` (shrunk least squares over the buffer) |
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
| Nonlinearity (commands → heat: products, limits, on/off) | **hand-written, not discovered** | `starterkit_data.episode_inputs`, `runtime/online_model.py`; the formulas get heat and air streams |
| Energy not needed for the dynamics | done (not modelled) | the known gaps are listed in the deck |
| Real data need filtering; the simulator does not | not built (simulator only) | — |
| Occupant split between rooms (open question) | seeds split evenly; formulas may do otherwise | seeds |
| Closed-form model inside the submission policy | **not built yet** | `run_policy.py` still runs the fitted-mode model |
