# One Controller for Every Building
### The NeurIPS 2026 Smart Buildings competition: the problem, the interface, a state-space model of the building, and our method

*Based on the starter kit `smart-buildings-2026/starter-kit` · Competition page: <https://www.kaggle.com/competitions/neurips-2026-smart-buildings/>*

---

## Summary

Heating, cooling and ventilating an office building is one of the largest energy loads in the world, and it is still run mostly by rules written once and never revisited. The Smart Buildings competition asks for **a single control policy that can run the HVAC system of any commercial building**, with different shapes, sizes, equipment counts, wall constructions and climates, keeping people comfortable while spending as little energy, money and carbon as possible. The policy is a Kaggle notebook, and it is judged on buildings and weather it has never seen.

**Parts I–III** set out the problem and the interface. They cover how a building is heated and cooled, the reward and the score, and what a submission is (§1–§4); every entry of the action vector and of the observation vector, and which quantities are controllable, observable or hidden (§5–§7); and the world being controlled: the simulator's physics, the six training buildings, the five climates, the shipped baselines and why generalization is hard (§8–§12).

**Part IV** is the state-space model. Only a minority of the observation vector carries information a dynamics model needs; the rest echoes the policy's own commands, mirrors setpoints, duplicates other fields or is constant (§13). Room identity and room type (corner, edge, interior) can be rebuilt exactly from the metadata a policy receives, and the type predicts how strongly a room is coupled to the weather, so a fixed-size hidden state indexed by room type is feasible (§14).

**Part V** is our proposed method. Every building obeys the same kind of physics, an energy balance between the air in each room, the walls and slab around it, the weather and the heat from the HVAC plant, and differs only in a handful of numbers. We **generate as much data as possible from the simulator, discover the structure of those equations by symbolic regression, identify each new building's few parameters while the policy runs, and plan with the discovered model inside a control policy** (§15–§19). **Part VI** closes with what has been verified and what has not (§20).

**Reading guide.** Reference tables sit beside the text they support. Statements are tagged where it matters: **[S]** read from source or config; **[D]** computed from the shipped data files (`floor_plan.npy`, `zone_map.npy`, weather, `baseline_returns.json`); **[R]** read from the instantiated environments; **[X]** measured in a rollout (§20.2); **[I]** inferred and still to be checked. §16 onward is proposal and hypothesis, not result. Nothing in this analysis modifies the starter-kit repository; the scripts live in `code/` and write only to `outputs/`.

### Contents

**Part I. The problem**
1. The challenge
2. How a building is heated and cooled
3. The benchmark: episodes, reward and score
4. What is submitted

**Part II. The interface**
5. The action vector
6. The observation vector
7. Controllable, observable and hidden quantities

**Part III. The world being controlled**
8. The physics inside the simulator
9. The six training buildings
10. The five climates
11. A day in the life, and the shipped baselines
12. Why generalization is hard

**Part IV. The state-space model**
13. The dynamics model: what belongs in it
14. A hidden state indexed by room type

**Part V. Our method**
15. What a control policy is, and the ways to build one
16. Our idea: discover the physics, then control with it
17. Stage one: data and discovering the dynamics
18. Stage two: wrapping the model in a controller
19. Testing it honestly, and what could go wrong

**Part VI. Verification**
20. What has been verified and what is still unknown

Appendix A — Files · Appendix B — Profiling run guide for another machine · Appendix C — Using the starter kit · Appendix D — Glossary · Appendix E — Every entry of the `train_8` vectors

---

# Part I. The problem

*What the competition asks, how an office building is heated and cooled, how a run is scored, and what exactly is handed in.*

## 1. The challenge

### 1.1 What is being asked
Imagine being handed the controls of an office building you have never seen. There are a few air handlers on the roof, each feeding a few dozen rooms; a boiler in the basement sends hot water to a small heating coil in every room; and every five minutes you may adjust dozens of dials. Some people are in the building and some are not, the weather outside is whatever it is, and you are scored on how comfortable the rooms stay and how much energy and carbon the building burns. Then you are handed a *different* building — different layout, different number of rooms and air handlers, different walls — in a different climate. The same controller has to cope with that too.

That is the Smart Buildings benchmark. The submission is **one policy**: a program that, given what the building's sensors currently read, returns the next set of commands. It is trained or designed on six sample buildings in five climates (30 scenarios), and judged on held-out buildings and weather drawn from the same generator.

### 1.2 Why it matters, and why it is hard
Commercial buildings use a large share of the world's electricity and gas, and heating, ventilation and air conditioning is usually the biggest piece. Most buildings are run by fixed schedules, fixed setpoints and simple feedback loops (PID controllers), tuned once at commissioning. They ignore weather forecasts, occupancy, the coupling between pieces of equipment, and the building's thermal inertia. A smarter controller could pre-heat a building just in time for the morning, coast through the evening on stored heat, and trade fan energy against reheat energy; the savings are real.

But learned control is hard to deploy:
- **Real buildings are expensive and risky to experiment on**, so learning happens in simulation.
- **Every building differs** in geometry, zone count, equipment, construction and climate. A policy tuned to one building usually fails on the next. Practical deployment needs *transfer*: one policy for a whole fleet, without per-building retraining.
- **The objective mixes competing goals**, and the comfort penalty is unbounded, so one neglected room can wreck an entire episode.

The simulator builds on Google's open-source Smart Buildings control simulator (see `NOTICE`, Apache 2.0). The competition authors (Judah Goldfeder and Ayush Rai, per `pyproject.toml`) added procedural building generation, material sampling, per-room control, real recorded weather and a unified scoring setup.

### 1.3 Our angle
Reinforcement learning would have to rediscover building physics from noisy reward signals, and hand-tuned PID cannot anticipate or weigh cost. We take a third path: every building obeys the same energy-balance structure and differs in a few parameters, so we learn the structure once, in readable form, from simulator data, and estimate each new building's parameters as it runs. §16 develops the idea.

---

## 2. How a building is heated and cooled

### 2.1 The system in the simulator
The simulated building uses the most common commercial-office design, a **variable-air-volume (VAV) system with central air handlers and hot-water reheat**. The photograph below, from the repo, shows the real-world equipment the simulator represents: a rooftop chiller, an air handler, boilers in the plant room, and the meters that record the energy the reward charges for.

![Figure 1 — A building and the plant equipment the simulator models: chiller, air handler, boilers and meters (images/device_locations.png)](images/device_locations.png)

*Figure 1. The real-world counterparts of what is simulated. In the simulator the chiller is implicit (cooling electricity is the thermal load divided by a coefficient of performance of 3.5), the boiler burns natural gas, and the meters correspond to the energy terms in the reward.*

The flow of air and water can be drawn in a few lines:

```
 outside air ─┐
              ├─► [ Air Handler (AHU) ] ──supply air──► [VAV box] ─► Zone (room)
 return air ──┘    mixes, cools, fans        ▲           damper + reheat coil
   (recirc.)                                  │                ▲
                                              │                │ hot water
                        [Boiler + pump: hot-water loop] ───────┘
```

An **air handler** (AHU) mixes air returned from the rooms with fresh outside air, cools the mixture to a target supply temperature, and pushes it through ducts with a fan. Each room (a **zone**) hangs off the duct through a **VAV box**: a *damper* that decides how much of the air that room gets, and a *reheat coil* fed with hot water that can warm that air before it enters the room. The hot water comes from a **boiler** and is circulated by a **pump**. One air handler serves up to ten rooms and picks a single supply temperature for all of them; the dampers and reheat valves are therefore the only place where a decision can be made for an individual room.

Three facts about this design shape everything that follows:

1. **The air handler can only cool.** All heat in the building arrives through the reheat coils from the hot-water loop. What leaves the air handler is `min(mixed-air temperature, supply-air setpoint)`; the setpoint can pull the temperature down but never up.
2. **Fan power rises steeply with pressure.** Fan power is roughly flow × pressure ÷ efficiency, and flow itself grows with the square root of pressure, so running the fan hard is expensive — and running it hard while a room slowly recovers is a classic waste.
3. **There are floors a controller cannot go below.** A running air handler holds a minimum static pressure of 200 Pa and cannot recirculate more than 70 % of its air; every damper stays at least 10 % open. Occupied rooms must keep receiving outside air, so "close everything" is not an available strategy.

### 2.2 Why climate changes the problem
Because the air handler can only cool, its supply-temperature dial goes dead whenever the outside air is so cold that the *mixed* air is already below the coldest setpoint a controller may command (285 K). With recirculation capped at 0.7 and return air near 293 K, that happens below about **−6.8 °C**. The competition's own analysis of the scored windows says the setpoint is inert for 0 % of hours in Singapore, Phoenix and Atlanta, 3.8 % in Denver and **31.5 % in Fairbanks**.

So the same controller faces two different problems. In cold weather the real levers are the **hot-water temperature and the reheat valves**; the air-side supply temperature is "along for the ride." In hot weather the supply temperature, fan pressure and damper allocation are among the strongest levers. Any approach has to know which regime it is in.

---

## 3. The benchmark: episodes, reward and score

### 3.1 An episode
A **scenario** is one building paired with one weather strip. The shipped set has **30 scenarios**: six buildings, each paired with one strip from each of five stations (Fairbanks, Denver, Atlanta, Phoenix and Singapore). `training_set.yaml` defines the pairing; it is deliberately not a cross product, so each building sees one strip per station and no strip is reused.

Each strip replays a different 30-day window of recorded 2023 hourly weather, at a **five-minute timestep**, so an episode is **8,640 steps** (288 per day). Every step counts toward the score; there is no warm-up. The weather is real recorded data, not synthetic.

The building has a **comfort schedule**: on weekdays from 06:00 to 19:00 it is in *comfort mode*, with the zone temperature expected to stay between 294 K and 297 K (about 21–24 °C); otherwise it is in *eco mode*, a wider 289–298 K band that lets the building drift at night and on weekends. A detail worth noticing: **every shipped strip starts on a Sunday at 06:00** [D], so every window contains 8 weekend days and 21 weekday comfort periods. Comfort mode covers 3,274 of the 8,640 steps — **37.9 % in every scenario** [D] — and the remaining 62 % is eco mode, where there is a lot of slack for setback and coasting. (This is also why short smoke-test runs flatter schedule-aware policies: they skip the eco periods.)

### 3.2 The reward
Each five-minute step produces a reward that is **never positive**; zero would be perfect comfort at no energy and no carbon cost, which nothing reaches. It is a weighted trade-off:

```
reward = [ 0.6 · (comfort term) − 0.2 · (energy cost) − 0.2 · (carbon) ] ÷ (0.6 + 0.2 + 0.2)
```

The **comfort term** measures how far occupied zones sit outside their temperature band. Inside the band, occupants are fully productive. Outside, productivity falls *linearly* up to a deviation of 5 K — one kelvin out costs 20 % — and then *quadratically and without bound*: beyond 5 K the shortfall becomes `1 + ((deviation − 5)/5)²`, so productivity can go negative. The unbounded tail is deliberate. An earlier saturating penalty let a do-nothing policy score well by abandoning comfort altogether, because a room 5 K cold and a room 34 K cold looked the same. Now a single badly neglected zone can dominate a whole episode. The productivity scale is also tied to occupancy ($300 per person-hour at most in the shipped configs), so empty rooms cost little.

**Energy cost** is charged for electricity (fans, pumps, the cooling plant) and for natural gas (the boiler); **carbon** is charged for both fuels. Neither is clipped: when consumption exceeds the nominal maximum, the term grows past 1 and the reward falls below −1.

### 3.3 Calibrated so the weights mean what they say
The energy terms are divided by a nominal maximum that scales with the number of air handlers — 80,000 W × K for electricity and 205,000 W × K for gas — so a large building is not penalized per air handler more than a small one. The constants were chosen so that a near-frontier controller loses roughly **60 % of its reward to comfort and 40 % to fuel**, matching the declared weights. (A floor exists: if energy were about three times relatively cheaper, the best "strategy" would be to stop conditioning the building, but the shipped values leave a wide margin.)

In a one-day check, the reward matched `0.6·npr − 0.2·nec − 0.2·nce` to about 2×10⁻⁸ [R], where `npr` is the normalized productivity term and `nec`, `nce` the normalized energy-cost and carbon terms. The reward therefore decomposes exactly into per-step pieces, which will let us report where each baseline loses points.

### 3.4 How the leaderboard score is computed
For each scenario the score is the mean per-step reward over the 8,640 steps. The leaderboard metric is the **mean absolute error against zero** across the 30 scenarios — equally weighted, with no per-building or per-climate grouping, deliberately — so **lower is better**. Episodes of any other length are refused, because shorter ones are less negative and not comparable. The official ranking is computed on Kaggle against held-out buildings and weather strips. The best shipped baseline scores about **0.054**.

### 3.5 The reward as a planner's cost function
For a model-based controller the reward is more than a score: it is the cost function to minimize. Comfort is a piecewise function of each occupied zone's deviation, and energy and carbon are closed-form functions of fan, pump, cooling and boiler power (§8). A controller that predicts the next few hours can therefore *evaluate the true reward on its predictions* rather than learn it from experience. The unbounded comfort tail also tells it how to behave under uncertainty: avoid even small probabilities of large excursions.

---

## 4. What is submitted

### 4.1 The artifact and its contract
A submission is a **Kaggle Notebook** and nothing else. A working example (the "TransferPolicy Notebook") is on the competition's Code tab. The notebook defines one policy class; when the notebook is submitted, the same notebook is re-run unchanged against the hidden test scenarios, and the leaderboard number (§3.4) is computed server-side. The competition's Data page describes the class as subclassing `smart_buildings_inference_server.Policy`; the starter kit's template, and every shipped baseline, needs only two methods:

```python
__init__(self, building_meta_data, action_spec)   # called once per scenario (per building)
get_action(self, obs) -> {"continuous_action": float32[3K+2+2V],
                          "discrete_action":   int32[K+1]}   # called once per 5-minute step
```

The local harness (`evaluate.py`) needs no base class: `initialize_policy` tries `Policy(meta, action_spec)`, then `Policy(meta)`, then `Policy()`, and `get_policy_action` calls `get_action(obs)`. The Kaggle runtime imports `smart_buildings_inference_server`, which is not in the starter kit (it ships in the competition's `kaggle_evaluation` folder, visible only after accepting the rules). So we develop against the two-method contract, keep the class free of local-only imports, and make the final class inherit from `Policy` when the notebook is assembled. The base class's exact behaviour (extra hooks, how metadata is delivered) is unverified (§20.3).

### 4.2 What the harness does with the policy
For each scenario `run_policy` resets the environment and loops 8,640 times: `get_action(obs)`, then `validate_action` (exact shapes; continuous cast to float32, discrete to int32; a wrong shape raises), then `tracker.step(action)`. The policy sees only the observation vector and the metadata (§6.5), never the reward, weather label, month, station or building id, and **hardcoding a reference to a held-out building or weather seed is disqualifying** (template docstring). The policy object is created once per scenario, so it can keep state between calls: the estimator, its previous action, a history.

### 4.3 Limits and rules

| Property | Value | Source |
|---|---|---|
| Notebook runtime | CPU ≤ 9 h or GPU ≤ 9 h per run | Overview page |
| Internet during the run | disabled | Overview page |
| External data and pre-trained models | allowed if free and public; a reasonableness test applies to cost | Overview page, Rules §2.6 |
| Open-source code | only OSI-approved, commercially unrestricted licences | Rules §3.6c |
| Daily submissions | 5; 2 final submissions selected | Rules §2.2 |
| Team size | ≤ 5; mergers allowed | Rules §2.1 |
| Winners | license the winning code CC-BY 4.0 and deliver training and inference code | Rules §2.5, §2.8 |
| Dates | entry and merger deadline 3 Jan 2027; final submission 15 Jan 2027 | Overview page |
| Compute budget per step | not stated; 9 h over 30 scenarios × 8,640 steps ≈ 1.25 s per step, minus simulator time | derived [I] |

What these limits mean for our method is worked out in §17.6.

# Part II. The interface

*What the policy can set and what it can see, entry by entry.*

## 5. The action vector

Every step the policy returns a **hybrid action**: `K+1` discrete on/off entries and `3K+2+2V` continuous setpoints, where **K** is the number of air handlers (AHUs) and **V** the number of zones (one VAV box per zone). That is `4K+3+2V` values, from 75 for the smallest building to 209 for the largest. The container is `{"continuous_action": float32[3K+2+2V], "discrete_action": int32[K+1]}`. Continuous entries travel normalized to [−1, 1] and the environment maps them linearly to the native ranges below; discrete entries are {0, 1}.

### 5.1 Every entry

| # | Entry | Count | Type | Native range | What the simulator does with it |
|---|---|---|---|---|---|
| 1 | `ahu_j_supervisor_run_command` | K | **discrete** {0,1} | off / on | runs fans, cooling coil and ventilation for AHU j |
| 2 | `ahu_j_supply_air_temperature_setpoint` | K | continuous | 285–305 K | supply temperature = min(mixed air, setpoint); inert if the mixed air is already colder |
| 3 | `ahu_j_supply_air_static_pressure_setpoint` | K | continuous | 0–2000 Pa | airflow ∝ √P, fan power ∝ flow·P; floor 200 Pa while running |
| 4 | `ahu_j_recirculation_percentage_command` | K | continuous | 0–1 | share of return air mixed back in; cap 0.7 |
| 5 | hot-water `supervisor_run_command` | 1 | **discrete** {0,1} | off / on | runs boiler and pump |
| 6 | hot-water `supply_water_temperature_setpoint` | 1 | continuous | 310–350 K | boiler target; tank ramps about 0.5 K/min up and 0.1 K/min down |
| 7 | hot-water `differential_pressure_setpoint` | 1 | continuous | 0–20 (config units) | pump head, which sets water flow and pump power |
| 8 | `VAV_n_supply_air_damper_percentage_command` | V | continuous | 0–1 | the zone's share of the AHU's air; floor 0.1 |
| 9 | `VAV_n_reheat_valve_percentage_command` | V | continuous | 0–1 | hot water to the zone's coil; discharge air capped at 308.15 K |

**Every entry is an input to the building's dynamics.** The per-zone entries dominate the dimension (84–86 % of the continuous action), while there are only 11–29 plant-level continuous entries and 4–10 switches; the per-zone pair is the same kind of object in every room. The plant responds on very different timescales: dampers and airflow within a step, boiler water at about 0.5 K/min up and 0.1 K/min down, building mass over hours. Thinking in blocks of 15–60 minutes loses little and avoids chatter. The default action in each scenario's `config.gin` is supply air 293 K, pressure 2000 Pa, recirculation 0.3, water 340 K, differential pressure 20, damper 1.0, reheat 0.0, all units on [S].

### 5.2 Order of the entries [R]

The discrete and continuous parts are separate arrays, each filled in the order of `env._device_action_tuples`: for each AHU j = 1..K the run command (discrete) and then temperature, pressure, recirculation (continuous); then the hot-water loop (run command; water temperature, differential pressure); then every zone in **lexicographic VAV order** (`VAV_1, VAV_10, VAV_11, …, VAV_2, …`) with damper then reheat. For `train_8` (K = 3, V = 30) the discrete array is `[ahu_1, ahu_2, ahu_3, hws]`; continuous slots 0–2 belong to AHU 1, 3–5 to AHU 2, 6–8 to AHU 3, slot 9 is the water temperature, slot 10 the differential pressure, 11–12 are `VAV_1` (damper, reheat), 13–14 `VAV_10`, and so on to slot 70. The metadata provides `temp_indices = [0, 3, 6]`, `pressure_indices = [1, 4, 7, 10]`, `recirculation_indices = [2, 5, 8]`, `water_indices = [9]` and `zone_action_indices` (VAV id → setpoint name → index).

Three traps follow from the layout, and each silently drives the wrong actuator:
- **Zero is not off.** Zero is the *middle* of each normalized range, so a slot never written is a half-open damper or half-shut valve. Write every slot.
- **Never hard-code offsets.** Read the index lists from the metadata. `pressure_indices` has **K+1** entries, not K: the first K are the AHU pressure setpoints and the last is the hot-water differential pressure, so treating the list as one-per-AHU drives the pump instead.
- **Commands override the local thermostat.** Each VAV has a thermostat loop that, left alone, would choose damper and reheat from the zone's comfort band. A command wins over that loop, separately for damper and reheat [S: `vav.py`]. Because the policy writes both every step, the thermostat has no influence on the physics.

Two further facts make the entries more than free variables: a **floor** is applied inside the simulator to the damper (0.1), the pressure (200 Pa while running) and the recirculation (0.7), and the observation reports the realized value (§6).

---

## 6. The observation vector

The observation is a flat float32 vector of length **`6V + 15K + 58`**, from 283 (`train_8`) to 703 (`train_2`) [R]. It is assembled by sorting devices by `device_id` as strings, sorting each device's fields alphabetically, then appending the time block [S: `environment.py`]. The layout is therefore:

| Index range | Content | Size |
|---|---|---|
| 0 … 6V−1 | **VAV blocks**, six fields each, in the order `VAV_1, VAV_10, VAV_11, …, VAV_2, …` (lexicographic, so `VAV_10` precedes `VAV_2`) | 6V |
| next 15K | **AHU blocks** `ahs_ahu_1_*, …, ahs_ahu_K_*`, fifteen fields each | 15K |
| next 7 | **hot-water block** `hws_*` | 7 |
| last 51 | **time and occupancy** (§6.4) | 51 |

The order should always be read from `observation_field_names` in the metadata, never computed. The observation at step *t* reports the state after the action of step *t−1* was applied: the realized and echoed values below equal the action just taken [X].

### 6.1 The six fields of every VAV block

| Field (alphabetical) | Units as delivered | What it is |
|---|---|---|
| `reheat_valve_percentage_command` | (x−0.5)/0.5 | realized reheat position; equals the action exactly [X] |
| `supply_air_damper_percentage_command` | (x−0.5)/0.5 | realized damper position; equals `max(action, 0.1)` [X] |
| `supply_air_flowrate_setpoint` | raw m³/s | the box's maximum airflow; 2.0 at every step [X] |
| `zone_air_cooling_temperature_setpoint_sensor` | raw K | upper comfort bound: 297 K in comfort mode, 298 K in eco mode |
| `zone_air_heating_temperature_setpoint_sensor` | raw K | lower comfort bound: 294 K in comfort mode, 289 K in eco mode |
| `zone_air_temperature_sensor` | (T−295)/5 | zone-average air temperature |

The comfort bounds are reported one step *after* the mode flag changes (3 transitions, 4 mismatched steps in a 601-step rollout [X]).

### 6.2 The fifteen fields of every AHU block

Several of the fifteen are the same underlying attribute under different names or mirror the policy's own commands. A 600-step rollout (§20.2) shows that **only the outside temperature is an independent exogenous signal**.

| Field (alphabetical) | What it is | Relation found [X] |
|---|---|---|
| `cooling_request_count` | zones requesting cooling | constant: always the number of zones the AHU serves (10 in `train_8`) |
| `discharge_fan_speed_percentage_command` | discharge-fan speed | **identical** to `supply_fan_speed_percentage_command` |
| `exhaust_fan_run_command` | exhaust-fan on/off | **identical** to `supervisor_run_command` |
| `outside_air_flowrate_sensor` | ventilation flow, m³/s | follows total flow and recirculation |
| `outside_air_temperature_sensor` | outside temperature, K | **the same value in every AHU block**; the only weather input |
| `recirculation_percentage_command` | realized recirculation | `min(native action, 0.7)` |
| `supervisor_run_command` | AHU on/off | equals the discrete action |
| `supply_air_flowrate_sensor` | total supply airflow, m³/s | derived from dampers and pressure; 0 when off |
| `supply_air_flowrate_setpoint` | airflow target | **identical** to the flow sensor |
| `supply_air_static_pressure_sensor` | duct pressure, Pa | **equals the setpoint** when running (no lag), 0 when off |
| `supply_air_static_pressure_setpoint` | realized pressure setpoint | `max(native action, 200 Pa)` when running, 0 when off |
| `supply_air_temperature_sensor` | supply air temperature, K | **equals the setpoint at every step**; the physical `min(mixed air, setpoint)` is not reported |
| `supply_air_temperature_setpoint` | realized supply-air setpoint | equals the native action |
| `supply_fan_run_command` | supply-fan on/off | **identical** to `supervisor_run_command` |
| `supply_fan_speed_percentage_command` | fan speed fraction | tracks the flow (correlation 1.000) |

### 6.3 The seven hot-water fields

| Field (alphabetical) | What it is | Relation found [X] |
|---|---|---|
| `differential_pressure_sensor` | pump head | **equals the native setpoint** when running, 0 when off (no lag) |
| `differential_pressure_setpoint` | realized pump setpoint | normalized `(x−10)/6` of the native action; 0 (−1.667 normalized) when off |
| `heating_request_count` | zones requesting heat | constant: always the total zone count (30 in `train_8`) |
| `run_status` | whether the loop runs | **identical** to `supervisor_run_command`, in {−1, +1} |
| `supervisor_run_command` | loop on/off | the discrete action in {−1, +1} |
| `supply_water_temperature_sensor` | loop temperature, K | **the only observed plant state with memory**: ramp-limited (+2.5 K per step at most up, −0.5 K per step down), equal to the setpoint 41 % of the time |
| `supply_water_temperature_setpoint` | realized water setpoint | equals the native action |

### 6.4 The 51 time and occupancy fields

| Fields | Count | What they are |
|---|---|---|
| `hod_cos_000..011`, `hod_sin_000..011` | 24 | twelve phase-shifted copies of `cos/sin(θ_hod + 2πi/12)`, θ the hour-of-day angle |
| `dow_cos_000..011`, `dow_sin_000..011` | 24 | the same encoding of the day of week |
| `comfort_mode_now` | 1 | 1 when the schedule is in comfort mode now |
| `comfort_mode_soon` | 1 | the same flag 60 minutes ahead |
| `num_occupants` | 1 | building-total occupant count, raw (0–57 of 60 seen in the rollout); not per zone |

The 48 sinusoids are redundant encodings of two numbers (hour of day, day of week).

### 6.5 Scaling and what the observation does not contain

- **Normalized fields** (matched by exact measurement name in the config) are the zone temperature `(T−295)/5`, the VAV damper and reheat `(x−0.5)/0.5`, the hot-water run command and run status (to {−1, +1}) and the hot-water differential-pressure setpoint `(x−10)/6` [X]. Every AHU field (they carry an `ahu_N_` prefix), the hot-water temperatures and pressure sensor, the comfort bounds and the box airflow are raw. Anything fitted to physical equations must undo the normalization first.
- **Weather is one number**: the current outside temperature. There is no forecast, humidity, wind, solar or dew point, and no weather label, month, station name or building id. A policy must infer the climate from the temperature trace and the time.
- **The policy's own commands are echoed back** as realized positions and setpoints, so a policy can recover what it did last step and see where a floor overrode it.
- **Energy use, cost, carbon and reward are not observed.** The policy has to predict its own cost from a model.
- **Metadata** is delivered once at construction (`evaluate.create_building_meta_data`): the action index lists, `observation_field_names`, `zone_temp_indices`, `ahu_zone_assignments` (AHU → `room_N`, the box being `VAV_N`), `num_floors`, `floor_plan` and `zone_map` arrays, `morning_start_hour` (6) and `evening_start_hour` (19), `comfort_temp_window` (294, 297), `eco_temp_window` (289, 298), `time_zone` (`US/Pacific` everywhere), an empty `holidays` set, and the normalization constants for observations and actions. The six material properties and the grid cell size are **not** given. A policy is a stateful object created once per scenario, so it can store its previous action and history.

---

## 7. Controllable, observable and hidden quantities

### 7.1 Sizes

| Building | K | V | Observation `6V+15K+58` | Discrete `K+1` | Continuous `3K+2+2V` | Total action |
|---|---|---|---|---|---|---|
| train_1 | 4 | 39 | 352 | 5 | 92 | 97 |
| train_2 | 9 | 85 | 703 | 10 | 199 | 209 |
| train_4 | 6 | 54 | 472 | 7 | 128 | 135 |
| train_5 | 8 | 79 | 652 | 9 | 184 | 193 |
| train_7 | 4 | 36 | 334 | 5 | 86 | 91 |
| train_8 | 3 | 30 | 283 | 4 | 71 | 75 |

A typical decision involves roughly 75–209 commands, chosen from 283–703 observed numbers, for a physical system whose full state is about **10⁵-dimensional** (the simulator's thermal grid has 1.2–2.9×10⁵ cells, §9.2). The reduced model used by our method has about two states per zone (air and mass) plus a few plant states, i.e. **70–180 states**, with **5–8 parameters per zone** to estimate (§13, §14).

### 7.2 Who can see and who can set what

| Quantity | Controllable | Observable | Role in the dynamics |
|---|---|---|---|
| AHU on/off, supply-air temperature, static pressure, recirculation | yes | yes (echo) | plant-level input |
| Hot-water on/off, water temperature, differential pressure | yes | yes (echo) | plant-level input |
| Zone damper, reheat valve | yes (command) | yes (realized) | zone-level input |
| Zone air temperature | no | yes | **output and primary state** |
| Supply-water temperature | no | yes | plant state with memory |
| Supply-air and pressure sensors, flows, fan speeds | no | yes | mirror the commands or derive from them (§6) |
| Comfort bounds, comfort-mode flags | no | yes | schedule context; enters the reward |
| Outside-air temperature | no | yes | **disturbance** |
| Total occupancy | no | yes | **disturbance** (also enters the reward) |
| Slab, wall and envelope temperatures | no | no | **latent state** |
| Per-room occupancy and internal gains | no | no | **latent disturbance** |
| Boiler tank temperature | no | partly (through the water sensor) | latent, lagged state |
| Building parameters (R, C, UA) | no | no | **unknown parameters to identify** |
| Energy, cost, carbon, reward | no | no | outcomes; must be predicted |

The **hidden quantities** are the ones that make the problem hard: slab and wall temperatures, per-room occupancy and heat gains, the boiler tank and, above all, the building's thermal parameters. A controller that does well has to act as if it knew them, which means estimating them from how the visible quantities respond. Room temperature is observed, not hidden; what is hidden is what sits behind it.

# Part III. The world being controlled

*The physics, the six training buildings, the five climates, the baselines to beat, and why a controller must transfer.*

## 8. The physics inside the simulator

The simulator is a composition of subsystems of very different mathematical character. Reading its source (`smart_buildings/smart_control/`), the plant breaks naturally into six blocks. These equations matter because they are what our method tries to rediscover.

### 8.1 How the simulator is organized
The building is a 2D grid of **control volumes** — cells labelled exterior, interior wall, interior mass (slab and furniture) or air. Each step, a backward-Euler heat-diffusion solve advances the temperature field, factorized once per episode with SciPy (it replaced an older truncated iterative solver that added an artificial, weather-dependent thermal lag). The equipment models (`air_handler.py`, `vav.py`, `boiler.py`, `hot_water_system.py`, `pump.py`) turn commands into heat and energy; `stochastic_occupancy.py` makes occupants arrive, leave and take lunch at random; `thermostat.py` runs a per-zone hysteretic Heat/Cool/Off/Passive-Cool state machine; `setpoint_schedule.py` provides the comfort and eco bands; `weather_controller.py` replays the recorded station data (linearly interpolated from hourly to five-minute values [S]); and the `reward/` package computes the score. Floors and the discharge-temperature limit are enforced inside the simulator, so a policy sits *on top of* these inner loops rather than replacing them.

### 8.2 The six blocks

**A. Zone thermal dynamics — the hard, slow part.** The grid is well captured, zone by zone, by a reduced-order resistor–capacitor (RC) network:

```
C_z · dT_z/dt =  Σ_{j ∈ nbrs(z)} (T_j − T_z)/R_zj          (conduction through partitions)
               + (T_out − T_z)/R_env,z                       (envelope loss or gain)
               + (T_mass,z − T_z)/R_mass                     (exchange with slab and furniture)
               + Q_vav,z                                     (HVAC heat delivered to the zone)
               + Q_internal,z(occupancy, time)               (people, lighting, equipment)
C_mass · dT_mass/dt = (T_z − T_mass)/R_mass
```

The capacitances and resistances are exactly what the six sampled material properties and the floor geometry determine. The simulator also includes **internal gains** — a configurable heat per square metre, a standby fraction, and 70 W of sensible heat per person — so occupied rooms run warmer. Only the zone-*average* air temperature is observed; the slab and wall temperatures are hidden.

**B. VAV box and reheat coil — fast, algebraic, nonlinear.** The airflow a box demands scales with its damper opening and the square root of the fan's static pressure (`V̇ ≈ damper · V̇_max · √(P / 2000 Pa)`), scaled down if the air handler cannot supply everyone. The reheat coil's effectiveness is `ε = min(1, water capacity-rate ÷ air capacity-rate)`, further capped so discharge air does not exceed 308.15 K. Discharge temperature is `T_sa + ε (T_water − T_sa)` and the heat delivered to the zone is `c_air ρ_air V̇ (T_disch − T_zone)`. Hot-water flow to each coil is the pump's total flow shared in proportion to the valve openings. This block is mostly algebra with `min` saturations.

**C. Air handler — algebraic.** Mixed air is `r · T_return + (1 − r) · T_out` with recirculation `r ≤ 0.7`; supply air is `min(mixed, setpoint)` while running. Fan power is flow × pressure ÷ efficiency for the supply fan at full flow and the exhaust fan at flow × (1 − r). Cooling electricity is the thermal load divided by a COP of 3.5.

**D. Hot-water system — slower and stateful.** The boiler's tank temperature rises at a finite rate and falls at a finite rate; gas power is the heat carried by the water flow plus dissipation plus the tank's own heating, floored at zero; pump power is flow × density × gravity × head ÷ efficiency. (An air-source heat-pump variant exists but the shipped scenarios use the boiler.)

**E. Disturbances.** Outside temperature from recorded data with a per-station convection coefficient; stochastic occupancy per person; and the deterministic comfort schedule.

**F. The reward** is a closed-form function of state and action (§3).

### 8.3 What this means for identification
Most of the inputs to these equations are visible: zone temperature, damper and valve positions, outside temperature, air-handler supply temperature and pressure, water supply temperature. What is missing is the slab and wall temperatures, per-room occupancy and heat gains, and the parameters. **That gap — between what is visible and what the equations need — is the identification problem our method addresses.**

---

## 9. The six training buildings

### 9.1 What a building is
Each shipped building is a multi-storey office block with an irregular ("composite") outline, divided into rectangular rooms. A **zone** is one room served by exactly one VAV box; every zone belongs to exactly one air handler, up to ten per air handler. The generator also sets how many people occupy each zone — one or two — and a set of envelope properties.

Figure 2, from the repo, shows three of them floor by floor, with each room colored by the air handler that serves it.

![Figure 2 — Floor plans of building_train_8, building_train_1 and building_train_2, rooms colored by air handler (images/floorplan_variability.png)](images/floorplan_variability.png)

*Figure 2. Three buildings of increasing size. Rooms tile an irregular outline; an air handler's rooms are not always contiguous and can span two floors (air handler 3 in `building_train_1` serves rooms on the second and third floors). Notched outlines mean perimeter exposure differs room to room: corner and perimeter rooms feel the weather, interior rooms do not. The figure labels `building_train_8` "28 zones", but the shipped config and the runtime check both give 30 [R], so the title is out of date; the labels for `building_train_1` (39) and `building_train_2` (85) match.*

A policy therefore cannot assume that the rooms of one air handler form a compact block, nor that two buildings have anything in common beyond the rules of physics.

### 9.2 Size: from compact to sprawling
The six buildings span a wide range. The smallest, `building_train_8`, has 30 zones, 3 air handlers and 2 floors. `building_train_7` has 36 zones, `building_train_1` 39 (three floors), `building_train_4` 54 (six air handlers), `building_train_5` 79 across four floors — the tallest — and `building_train_2` 85 zones, 9 air handlers on two wide floors with many small rooms. More zones mean a longer action vector (up to 209 values), more simultaneous decisions, more coupling through the shared hot-water loop, and slower simulation. The table below gives each building's structure and §7.1 the vector sizes.


| Building | AHUs K | Zones V | Zones per AHU | Floors | Footprint W×H | Occupants / zone | Total occupants [S] | Thermal grid cells [D] |
|---|---|---|---|---|---|---|---|---|
| train_1 | 4 | 39 | 10, 10, 9, 10 | 3 | 373×188 | 2 | 78 | 606×339 = 205,434 |
| train_2 | 9 | 85 | 6, 9, 10×7 | 2 | 420×301 | 2 | 170 | 624×422 = 263,328 |
| train_4 | 6 | 54 | 6, 10, 9, 9, 10, 10 | 3 | 430×211 | 1 | 54 | 664×432 = 286,848 |
| train_5 | 8 | 79 | 9, 10×7 | 4 | 340×199 | 2 | 158 | 858×342 = 293,436 |
| train_7 | 4 | 36 | 7, 10, 10, 9 | 2 | 325×218 | 1 | 36 | 458×279 = 127,782 |
| train_8 | 3 | 30 | 10, 10, 10 | 2 | 318×198 | 2 | 60 | 386×320 = 123,520 |


### 9.3 Character: how leaky, how heavy
The most important differences are invisible in the floor plan. Each building's envelope is described by six sampled numbers (table below), which reduce to a few physical ideas: *how leaky* the facade is (its U-value, derived from conductivity through the generator's relation `1/U = 0.1/k + 0.299`), *how heavy* it is (density × heat capacity, which sets thermal inertia), how much **slab mass** the interior holds (which governs overnight coasting), how strongly the **partitions couple** neighbouring rooms, and how many occupants each zone has (which sets internal heat gain and the weight of comfort in the reward). The derived U-values and heaviness figures below are computed from the shipped YAMLs, not simulator outputs.

- **`building_train_1` — tight and heavy.** Low conductivity (U ≈ 1.30) and a very heavy facade (≈ 1.87 MJ/m³K). It is slow to warm up or cool down and coasts for hours, so it rewards patient *pre*-conditioning and punishes late reaction.
- **`building_train_2` — light and quick, and the biggest.** U ≈ 1.68, a light facade (≈ 0.52) and the lowest slab mass among the large buildings. It responds fast and stores little; with 85 zones and 9 air handlers it also demands the most coordination.
- **`building_train_4` — well insulated but light.** The lowest U of all (≈ 1.18) but a light facade (≈ 0.43), and only one person per zone, so internal gains are low. It holds heat well yet has little thermal mass to ride on.
- **`building_train_5` — in the middle, and tall.** U ≈ 1.59, a fairly heavy facade (≈ 1.03) and heavy partitions; moderate time constants across four floors.
- **`building_train_7` — the leakiest and lightest.** U ≈ 2.08 and the lightest facade (≈ 0.40): weather gets in fast and the envelope stores almost nothing, though the interior slab is heavy. One person per zone, so low gains.
- **`building_train_8` — heavy facade, almost no slab, strongly coupled rooms.** U ≈ 1.49, a heavy facade (≈ 0.96), the *lowest* interior mass and the *highest* partition conductivity, so zones drag their neighbours the most.

A heavy, tight building rewards anticipation. A light, leaky one punishes lateness and rewards fast feedback. A controller that works on all six has to handle all of these regimes, which is precisely why we want to treat them as different *parameter values* of one underlying structure (§17.3). Notably, the per-zone plant sizing (fan efficiency 0.9, VAV maximum airflow 2.0 m³/s, reheat flow factor, boiler heating rate, pump head) is the **same in every scenario** [S]: between buildings, what varies is topology, size, occupancy and envelope, not the equipment per room.

The six sampled envelope properties [S], with two derived columns computed from `1/U = 0.1/k + 0.299` and exterior density × heat capacity:

| Building | ext. conductivity k | ext. density | int.-mass density | int.-wall conductivity | int.-wall density | ext. heat capacity | ≈ facade U | ≈ ext. ρ·cp (MJ/m³K) |
|---|---|---|---|---|---|---|---|---|
| train_1 | 0.212 | 1967 | 32.3 | 0.536 | 415 | 951 | 1.30 | 1.87 |
| train_2 | 0.339 | 578 | 14.4 | 0.515 | 180 | 906 | 1.68 | 0.52 |
| train_4 | 0.182 | 475 | 23.4 | 0.464 | 484 | 897 | 1.18 | 0.43 |
| train_5 | 0.304 | 1187 | 28.0 | 0.497 | 643 | 866 | 1.59 | 1.03 |
| train_7 | 0.548 | 415 | 35.7 | 0.478 | 155 | 972 | 2.08 | 0.40 |
| train_8 | 0.270 | 1139 | 11.5 | 0.581 | 336 | 843 | 1.49 | 0.96 |

### 9.4 How the geometry is stored [D]

- **Codes.** `floor_plan.npy` and `zone_map.npy` both use 0 = interior space, 1 = wall, 2 = exterior, on a 20 cm grid, with the shape of the building's thermal grid. `floor_plan` has **doors as open cells**; `zone_map` has the same doors filled in as walls, so **rooms are exactly the connected components of `zone_map == 0`**. The two files differ only at door cells (2 cells per door: 40 cells in `train_7` and `train_8`, 72 in `train_1`, 102 in `train_4`, 118 in `train_5`, 166 in `train_2`). Because the thermal solver runs on `floor_plan`, doors are paths for heat between neighbouring rooms. For example `train_1` has 141,779 interior-space, 6,946 wall and 56,709 exterior cells. Grids are identical across a building's five weather strips.
- **Room numbers.** The simulator names the rooms `room_1 … room_V` by connected-component label in raster order, and the box of `room_N` is `VAV_N`. §14.1 shows that the same numbering is reproduced from the metadata arrays alone.
- **Floors are separate pieces of the grid.** Each floor is its own copy of a plan, stacked in one image with exterior space between them (`train_8` has two floors of 15 rooms, `train_1` three floors). Within the grid there is no thermal contact between floors, so no vertical heat exchange is modelled (inferred from the layout; the solver should be checked, §20.3).
- **AHUs follow room numbering, not geography.** Rooms are assigned to AHUs in consecutive blocks of up to ten, so several AHUs straddle two floors.

| Building | Floors | Rooms per floor | AHU → rooms on each floor |
|---|---|---|---|
| train_1 | 3 | 13 / 14 / 12 | ahu_1: 10 on f1 · ahu_2: 3 f1 + 7 f2 · ahu_3: 7 f2 + 2 f3 · ahu_4: 10 f3 |
| train_2 | 2 | 42 / 43 | ahu_5 straddles (7 f1 + 3 f2); the other eight serve a single floor |
| train_4 | 3 | 17 / 20 / 17 | ahu_3 (1+8) and ahu_5 (3+7) straddle |
| train_5 | 4 | 19 / 19 / 22 / 19 | ahu_4 (9+1) and ahu_7 (1+9) straddle |
| train_7 | 2 | 19 / 17 | ahu_3 straddles (2+8) |
| train_8 | 2 | 15 / 15 | ahu_2 straddles (5+5) |

---

## 10. The five climates

### 10.1 Five stations, one year
The weather is a full year (2023) of recorded hourly data from five airport stations — and a sixth, Minneapolis, that ships as generator input but is not used in the 30 scenarios. Each scenario replays one 30-day window. The stations span almost the full range of conditions an HVAC system sees:

- **Fairbanks** (subarctic) is the most extreme. Over the year it runs from −42 °C to +29 °C. Winter windows are pure heating problems: the envelope loses heat fast, hot water is the only effective lever, and the air-side supply temperature is inert about a third of the time. Summer windows are mild and cheap. The air is calm (about 3–10 km/h) and the convection coefficient is the lowest (8.8 W/m²K).
- **Denver** (high-altitude continental) is cold in winter and hot and dry in summer, with large swings *within* a window — for example −12.8 to +26.7 °C during one November window — which makes anticipation valuable. It is windy (15–20 km/h) with the highest convection coefficient (26.7), so the facade exchanges heat quickly.
- **Atlanta** (humid subtropical) has mild winters (down to −3.9 °C in the windows) and hot humid summers (up to 35 °C). Many conditions are shoulder-like, where economizing and avoiding simultaneous cool-and-reheat matter more than raw capacity.
- **Phoenix** (hot desert) reaches 46 °C in summer, is very dry (15–47 % humidity) and has a large day–night swing, so summer windows are cooling-dominated with the largest air-side and fan loads; its winters are mild (minimum 0.4 °C).
- **Singapore** (equatorial) hardly changes: 23–34 °C all year, with 76–85 % humidity. There is no heating at all, only a steady cooling load whose pattern repeats every day.

### 10.2 Season matters as much as station
The shipped windows start on different dates, so the *season* of a window matters as much as its station. Taking each window's mean temperature against the 21–24 °C comfort band as a rough guide (a rough classification, not a simulator output):

- **8 windows are heating-dominated** (mean below about 10 °C): four Fairbanks (`fairbanks_1, 2, 7, 8`), two Denver (`denver_2, 8`) and two Atlanta (`atlanta_1, 8`).
- **7 are shoulder** (about 10–20 °C): `fairbanks_4, 5`, `denver_3, 7`, `atlanta_2` and `phoenix_1, 8`.
- **15 are cooling-dominated** (above about 20 °C): `denver_5, 6`, `atlanta_3, 5, 6`, `phoenix_3, 4, 5, 6` and all six Singapore windows.

So half the scenarios are cooling problems, about a quarter are heating problems and the rest sit between — the climate label alone does not tell the regime; the window does (Fairbanks means alone run from −21 °C to +19 °C). The extremes are instructive: `fairbanks_1` (1 January, −42 to +1 °C) is the hardest heating case; `phoenix_5` (mid-July, 25–46 °C) is the hardest cooling case; the Singapore windows are the easiest; and `fairbanks_5` (mid-June, 11–29 °C) is the easiest of the cold-climate set. §10.4 lists all 30 windows.

### 10.3 What the policy actually experiences
Of all this, the policy sees only **one number**: the current outside-air temperature. Humidity, wind and dew point exist in the data files but are not exposed, and the convection coefficient is a fixed per-station constant. Weather therefore enters the problem through three channels: the heat the envelope gains or loses, the mixed-air temperature at the air handler, and whether the economizer (using outside air for cooling) helps. A subtle wrinkle is that all 30 scenarios configure a **US/Pacific** schedule time zone [D], even for Singapore or Atlanta strips whose weather timestamps are station-local; whether that matters is unverified.

### 10.4 The weather data

Annual 2023 hourly records (8,760 rows each, no missing values) [D]:

| Station | Annual min / mean / max (°C) | Convection coeff. (W/m²K) | Humidity / wind in scored windows |
|---|---|---|---|
| Fairbanks | −42.0 / −0.9 / 28.9 | 8.8 | 52–82 % / 3–10 km/h |
| Denver | −21.1 / 11.4 / 37.2 | 26.7 | 44–50 % / 15–20 km/h |
| Atlanta | −10.0 / 17.9 / 35.0 | 19.1 | 62–71 % / 10–16 km/h |
| Phoenix | 0.4 / 24.6 / 46.1 | 19.5 | 15–47 % / 8–15 km/h |
| Singapore | 22.9 / 28.0 / 34.0 | 17.6 | 76–85 % / 7–15 km/h |
| Minneapolis (shipped, unused) | −24.4 / 8.9 / 35.0 | — | — |

The 30 scored windows (720 hourly rows each; temperatures in °C) [D]:

| Strip | Start | Min | Mean | Max | | Strip | Start | Min | Mean | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| fairbanks_1 | 01-01 | −42.0 | −21.4 | 1.1 | | atlanta_1 | 01-22 | −1.7 | 9.6 | 22.2 |
| fairbanks_2 | 02-12 | −25.6 | −8.2 | 4.4 | | atlanta_2 | 03-05 | −3.9 | 14.5 | 29.4 |
| fairbanks_4 | 05-07 | 1.6 | 12.5 | 22.8 | | atlanta_3 | 04-16 | 8.0 | 20.8 | 30.6 |
| fairbanks_5 | 06-18 | 10.6 | 18.9 | 28.9 | | atlanta_5 | 07-09 | 20.0 | 27.0 | 35.0 |
| fairbanks_7 | 09-10 | −2.8 | 6.4 | 16.7 | | atlanta_6 | 08-20 | 13.3 | 25.4 | 33.9 |
| fairbanks_8 | 10-22 | −23.9 | −10.6 | 2.2 | | atlanta_8 | 11-12 | −3.3 | 9.7 | 22.2 |
| denver_2 | 02-26 | −10.6 | 5.0 | 20.0 | | phoenix_1 | 01-29 | 0.4 | 13.9 | 26.1 |
| denver_3 | 04-09 | −2.8 | 11.5 | 28.9 | | phoenix_3 | 04-23 | 15.0 | 27.0 | 39.4 |
| denver_5 | 07-02 | 12.8 | 24.2 | 37.2 | | phoenix_4 | 06-04 | 23.3 | 34.4 | 43.3 |
| denver_6 | 08-13 | 10.6 | 22.1 | 36.1 | | phoenix_5 | 07-16 | 25.0 | 36.8 | 46.1 |
| denver_7 | 09-24 | 0.2 | 12.9 | 28.3 | | phoenix_6 | 08-27 | 23.9 | 33.1 | 42.8 |
| denver_8 | 11-05 | −12.8 | 6.1 | 26.7 | | phoenix_8 | 11-19 | 2.5 | 14.4 | 27.8 |

| Strip | Start | Min | Mean | Max | | Strip | Start | Min | Mean | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| singapore_1 | 02-05 | 23.5 | 27.3 | 33.0 | | singapore_6 | 09-03 | 23.0 | 28.1 | 33.0 |
| singapore_3 | 04-30 | 23.0 | 28.6 | 33.7 | | singapore_7 | 10-15 | 24.0 | 27.5 | 33.0 |
| singapore_4 | 06-11 | 23.1 | 28.7 | 33.7 | | singapore_8 | 11-26 | 23.0 | 26.9 | 33.1 |

---

## 11. A day in the life, and the shipped baselines

### 11.1 What a day looks like
Figure 3 is a two-day window from the repo (starting 10 July 06:00) for a building with six air handlers. It shows a baseline controller doing its job, and it makes concrete several things we have so far described only in words.

![Figure 3 — Two simulated days: zone temperatures against the comfort and eco bands, per-step reward, and occupancy (images/overview.png)](images/overview.png)

*Figure 3. Top: zone temperatures colored by air handler against the comfort band (294–297 K) and eco band (289–298 K). Middle: per-step reward. Bottom: occupancy. The outside air is drawn as a smooth sinusoid of 32–36 °C and the start is a Monday, so this illustrates an earlier configuration, not one of the 30 shipped scenarios (which use recorded weather and Sunday starts).*

Reading it from the top:

- **Temperatures** hug the lower edge of the comfort band during the working day, drift toward the bottom of the wider eco band overnight (to about 291–292 K) and jump back up when the building returns to comfort mode at 06:00. The saw-tooth traces are the controller's on/off behaviour.
- **The reward** settles near a plateau of about −0.18 when nothing dramatic is happening — the unavoidable cost of running equipment and holding zones in band — and **dips sharply, to roughly −0.6 and −0.95, at the 06:00 switch to comfort mode**. At that moment zones that drifted down in the eco band are suddenly below the tighter comfort band. This is the price of *reactive* control, and it is the clearest opportunity for a controller that knows the morning is coming.
- **Occupancy** ramps from about 06:00, peaks near midday at 30–33 people, and falls to a lingering 7–8 overnight, averaging 16.7.

### 11.2 What the shipped baselines are
The repo includes five reference controllers (plus an unused extended PID in `pid_plus.py`) in `smart_buildings/baselines/`; there is no learned baseline ("bring your own"):

- **`bang_bang`** switches each air handler on when its zones' mean temperature leaves the band and off once it returns, at full-blast setpoints.
- **`fixed_setpoint`** holds one comfort band around the clock, with no eco setback.
- **`pid`** runs a proportional–integral–derivative loop per air handler on its zones' mean temperature error, and scales fan pressure with control effort (zero effort → minimum airflow, full effort → full), because fan power is flow × pressure. Underneath, each VAV's damper and reheat are set from its own zone's temperature, so a comfortable room throttles to its ventilation minimum while neighbours are still being conditioned.
- **`pid_water`** adds modulating reheat and a demand-driven hot-water temperature reset. It is the strongest of the four.
- **`all_off`** never runs anything, as a reference for what an unconditioned building costs; it is not part of the scored set.

Their docstrings are candid about the lessons: gentle PID can *use more energy* than bang-bang if it holds fan pressure high through a long recovery, and the per-zone damper and reheat layer is "where the headroom over the shipped baselines mostly is." None of them uses a model, a forecast or a cost-aware optimization. That is the gap our method targets.

### 11.3 The headline numbers
Using `baseline_returns.json` (total reward over 8,640 steps, converted to a per-step mean; closer to 0 is better) [D]:

| Baseline | Mean per-step reward | Best scenario | Worst scenario |
|---|---|---|---|
| `bang_bang` | −0.225 | −0.077 | −0.308 |
| `fixed_setpoint` | −0.066 | −0.017 | −0.200 |
| `pid` | −0.067 | −0.017 | −0.186 |
| `pid_water` | **−0.054** | −0.017 | −0.212 |

The bar to beat is therefore about **0.054**. Breakdowns by climate and building are in §11.6.

### 11.4 No baseline wins everywhere
The most interesting finding is that **no single baseline dominates.** `pid_water` is best in 18 of the 30 scenarios and the plain `fixed_setpoint` is best in the other 12; `pid_water` is actually *worse* than `fixed_setpoint` in 12. The simple fixed band wins in Singapore and in buildings 1, 4 and 5; the modulating controllers win in Denver and Phoenix and in buildings 2 and 7. The worst baseline differs by climate too: `bang_bang` is worst in the cold and `fixed_setpoint` is worst in the heat (Phoenix, −0.120), so a policy that beats the floor everywhere has to be good in both. In Singapore, the fixed band beats both PID variants, which is worth testing as a symptom of PID gains or fan-pressure scaling being mismatched to sustained cooling load [I].

A sobering and useful number follows from this: an **oracle that picks the best of the four baselines for each scenario** would average about **−0.047**, against −0.054 for the best single baseline — roughly 13 % better, from *choosing the right behaviour* alone. That is a rough measure of what regime-aware control can gain before any new idea.

### 11.5 The extremes
The worst case is `building_train_1 / fairbanks_1` — a big, heavy, tight building through −42 °C: `pid_water` −0.212, `pid` −0.151, `fixed_setpoint` −0.147, `bang_bang` −0.287. The best cases are mild windows, such as `building_train_5 / fairbanks_5` (a mild Fairbanks summer), where three of the four baselines reach −0.017 — plausibly close to a floor set by unavoidable ventilation and occupied-hour energy [I]. The headroom, in other words, is in the extremes and in the transitions (mornings, evenings, seasons), not in the easy middle.

### 11.6 Detail by climate and building

Per-step mean reward from `baseline_returns.json` (closer to 0 is better) [D]:

| Mean by climate | bang_bang | fixed_setpoint | pid | pid_water |
|---|---|---|---|---|
| Atlanta | −0.232 | −0.033 | −0.042 | **−0.030** |
| Denver | −0.184 | −0.077 | −0.050 | **−0.037** |
| Fairbanks | −0.200 | −0.074 | **−0.076** | −0.079 |
| Phoenix | −0.238 | −0.120 | −0.095 | **−0.078** |
| Singapore | −0.272 | **−0.025** | −0.075 | −0.047 |

| Mean by building | bang_bang | fixed_setpoint | pid | pid_water |
|---|---|---|---|---|
| train_1 | −0.211 | **−0.072** | −0.079 | −0.086 |
| train_2 | −0.243 | −0.080 | −0.066 | **−0.047** |
| train_4 | −0.189 | **−0.035** | −0.039 | −0.035 |
| train_5 | −0.223 | **−0.043** | −0.072 | −0.046 |
| train_7 | −0.235 | −0.092 | −0.058 | **−0.043** |
| train_8 | −0.250 | −0.073 | −0.090 | **−0.068** |

Mean total reward over 8,640 steps: bang_bang −1945.7, fixed_setpoint −569.5, pid −582.0, pid_water −467.8.

---

## 12. Why generalization is hard

The competition is designed so that memorizing is impossible. Five things vary or are hidden:

1. **Topology.** Zone count, air-handler count (the generator raises `num_ahus` to `ceil(zones/10)`), floor-plan shape and the room-to-air-handler assignment all change, and with them the observation and action sizes. Held-out buildings can be larger than any shipped one.
2. **Construction.** Each building draws its six envelope properties *independently* by Latin hypercube sampling (`materials.py`): exterior conductivity 0.15–0.55 (log scale), exterior density 350–2400, interior-mass density 10–40, interior-wall conductivity 0.45–0.60, interior-wall density 110–700, exterior heat capacity 840–1000. The authors explain why: choosing among *named* constructions collapses to a one-dimensional "how heavy is the building" line, because the properties are almost perfectly correlated. Independent draws fill a volume that includes real-world corners — a leaky heavy masonry warehouse and a tight light panel build are both possible.
3. **Climate and season** (§10): −42 °C to +46 °C, with different 30-day windows per scenario.
4. **A hidden test set** from the same generator and ranges, with weather strips that do not ship.
5. **No identity features.** No building id, month or station name appears in the observation, and hard-coding references to held-out configs or weather is disqualifying.

The conclusion is that the policy must **infer both building physics and climate from live signals**. It is as much a system-identification problem as a control problem — which is exactly the bet in §16.

# Part IV. The state-space model

*Which of the interface's numbers belong in a model of the building, and a fixed-size hidden state built from room types.*

## 13. The dynamics model: what belongs in it

### 13.1 Sorting every entry by role

The observation is the output of the data-generating process, and it mixes six different kinds of entry. Only some belong in a dynamics model. This regroups the role table of §7.2 by what the dynamics model needs. The roles of every individual index are listed in Appendix E and, for all six buildings, in `outputs/building_train_N_observation_annotated.csv` (filter on `use_in_dynamics_model`).

| Group | Entries | In the dynamics model? |
|---|---|---|
| **A. Outputs with memory** | zone air temperatures (V); supply-water temperature (1) | yes: the states to predict |
| **B. Derived signals** | supply airflow per AHU (K), outside-air flow (K), fan speed (K) | optional intermediates, computed from dampers, pressure and recirculation |
| **C. Inputs** | all actions (K+1 discrete, 3K+2+2V continuous); the realized damper is `max(action, 0.1)` and the realized reheat equals the action | yes |
| **D. Disturbances** | outside temperature (1 value), total occupancy, hour-of-day phase (2 numbers), weekday phase (2), comfort-mode flag | yes |
| **E. Context for the reward** | per-zone heating and cooling bounds, `comfort_mode_soon` | no: they say what the policy wants, not how the building responds |
| **F. Echoes, duplicates, constants** | setpoint echoes, run-command and fan copies, sensors that mirror setpoints, request counts, box maximum airflow | no (the constants are parameters) |

Of the six fields per zone, three are model signals (temperature, damper, reheat), one is a constant and two are reward context. Of the fifteen fields per AHU, one (outside temperature) is exogenous and at most three are derived signals. The pressure and temperature "sensors" are not independent measurements of lagged equipment: they mirror the commands. Only the zone temperatures and the hot-water temperature carry memory in the observation.

### 13.2 Removing the setpoints from the observation

The observation vector itself cannot be shortened: its shape is fixed by the environment, and the evaluation server builds it. What can be removed is *use*: a policy selects entries by name through `observation_field_names` and ignores the rest, at no cost. The question is whether the entries it drops can be recovered, and for the setpoint-type entries they can, because each falls into one of two classes:

- **Action setpoints** (AHU temperature, pressure and recirculation setpoints, water temperature, differential pressure, all run commands, and the realized damper and reheat): each is a deterministic function of the action just applied and the unit's run state, with the known floors (`max(a, 200 Pa)` when running and 0 when off, `min(a, 0.7)`, `max(a, 0.1)`) [X]. A stateful policy that stores its previous action reproduces them exactly, so they add no information. The one exception is the first step, whose observation reflects the reset state (for example a supply setpoint of 298 K and a pressure of 1000 Pa) rather than an action; the default action of §5.1 is applied from there.
- **Comfort setpoints** (the per-zone heating and cooling bounds): they are a function of the clock, the schedule and the metadata windows (`comfort_temp_window`, `eco_temp_window`, `morning_start_hour`, `evening_start_hour`), and all zones carry the same bounds in the shipped scenarios [X]. They trail `comfort_mode_now` by one step, which the schedule reproduces.

Sensors that mirror setpoints (AHU pressure and supply temperature, hot-water differential pressure) fall in the first class, and the duplicate and constant fields need no recovery. Dropping all of these leaves the **lean observation**: zone temperatures (V), supply-water temperature (1), outside temperature (1), occupants (1), one cos/sin pair for the hour (2), one for the weekday (2) and the comfort-mode flag (1), i.e. **V + 8** values, optionally plus the airflow and outside-air-flow signals (2K).

| Building | Observation | Lean observation V+8 | with flow signals V+8+2K | Actions (inputs) |
|---|---|---|---|---|
| train_1 | 352 | 47 | 55 | 97 |
| train_2 | 703 | 93 | 111 | 209 |
| train_4 | 472 | 62 | 74 | 135 |
| train_5 | 652 | 87 | 103 | 193 |
| train_7 | 334 | 44 | 52 | 91 |
| train_8 | 283 | 38 | 44 | 75 |

What is lost: nothing the dynamics need, provided the policy records its own actions. What stays unobserved is the physical supply temperature `min(mixed air, setpoint)`, because the sensor reports the setpoint; the model must compute it from the outside temperature, recirculation and the return temperature.

### 13.3 The state-space model

```
x_t = [ zone air temperatures T_z (observed, V),  hidden mass temperatures M_z (V),
        supply-water temperature (observed, 1),   boiler tank (lagged),
        per-room occupancy and internal gains (hidden) ]
u_t = [ u_d (K+1),  u_c (3K+2+2V) ]                         commanded
d_t = [ T_out, occupants, comfort flag, hour phase, weekday phase ]
x_{t+1} = f(x_t, u_t, d_t ; θ)                              one 5-minute step
y_t     = [ T_z, T_water ]                                  what the model must reproduce
```

The identifiable reduced form per zone is the resistor–capacitor balance of §8.2 (block A), driven by the VAV heat of block B: `Q_vav,z = ρ c_p V̇_z(damper_z, P_ahu) (T_disch,z − T_z)` with `T_disch,z = T_sa + ε(reheat_z, water flow)(T_water − T_sa)`.

Per zone the observed quantities are `T_z`, the realized damper and reheat, and the schedule bounds; the AHU contributes `T_sa` and `P` (from the actions), the loop `T_water`, and `T_out` is the disturbance. Hidden per zone: `M_z`, `q_z` and the parameters `(C_z, R_env,z, R_m, R_zj)`.

---

## 14. A hidden state indexed by room type

The number of rooms varies from 30 to 85, so a model that follows the observation changes size with the building. The alternative is a **fixed-size hidden state** indexed by room *type* (corner, edge, interior) rather than by room, with an encoder that pools the zone observations into it and a decoder that returns a temperature for every room. This section establishes from the code and from experiments that each ingredient is available.

### 14.1 Room identity and type can be rebuilt from the metadata [X]

`code/metadata_only_check.py` rebuilds everything from the dict returned by `create_building_meta_data` and compares it with the simulator's own room dictionary:

| Check | Result |
|---|---|
| Rooms = connected components of `zone_map == 0`, numbered in raster order (`scipy.ndimage.label`), equal to the simulator's `room_N` cell sets | **323 of 323 rooms** identical, all six buildings |
| The `VAV_n` ids in the observation names, in the action index map and in `ahu_zone_assignments` equal the room numbers 1…V | yes, all six buildings |
| Floors from the floor plan alone (connected pieces of the non-exterior footprint) | 3, 2, 3, 4, 2, 2 floors, matching `num_floors` |
| Doors between rooms from `floor_plan` versus `zone_map` | available (cells open in one and closed in the other) |
| Room type from the metadata arrays versus from the simulator's masks | **318 of 323** rooms get the same type; the five differences are boundary cases of the exterior-reach threshold |

So every entry of the observation and action vectors can be assigned to a room, and every room to a type, a floor, an AHU and a set of neighbours, using only what the policy is given. Cell size (0.2 m) is not in the metadata, but only relative areas are needed.

### 14.2 The types

For each room the code counts room-boundary cells facing an exterior wall within 1.2 m, separately for N, S, E and W, and calls a side a *face* if at least 5 cells (1 m) face it. Zero faces is **interior**, one **edge**, two or more **corner**.

| Building | Rooms | Corner | Edge | Interior | Area range (m²) | Doors |
|---|---|---|---|---|---|---|
| train_1 | 39 | 25 | 9 | 5 | 59.6 – 345.8 | 36 |
| train_2 | 85 | 8 | 37 | 40 | 52.0 – 328.0 | 83 |
| train_4 | 54 | 26 | 20 | 8 | 39.0 – 361.8 | 51 |
| train_5 | 79 | 40 | 23 | 16 | 34.6 – 207.5 | 59 |
| train_7 | 36 | 14 | 11 | 11 | 50.9 – 176.4 | 20 |
| train_8 | 30 | 13 | 12 | 5 | 23.6 – 203.0 | 20 |

`outputs/building_train_N_rooms.png` draws each building with rooms coloured by type. Because the footprints are composite and notched, many irregular rooms touch the exterior on two or more sides and count as corner (25 of 39 in `train_1`); the three-way split is **threshold-dependent**, and the mix differs strongly between buildings (`train_2` is mostly edge and interior, `train_5` mostly corner). The continuous quantities behind the label (exterior contact per side, exterior cells per m², neighbours through walls and through doors) are in `outputs/building_train_N_rooms.csv` and are the better inputs.

### 14.3 Types predict how strongly a room follows the weather [X]

`code/all_off_response.py` switches every AHU and the boiler off for 864 steps (three days, `train_8` / `atlanta_1` and `train_2` / `atlanta_3`) and regresses each room's one-step change on the outside-minus-room temperature, the neighbour-mean-minus-room temperature, total occupancy and a constant: `T[t+1] − T[t] = a·(T_out − T) + n·(T̄_nbr − T) + g·occ + c`. The envelope coefficient `a` (per 5-minute step) by type:

| Building | Type | Rooms | Mean `a` | Std | Implied time constant 1/a (median) |
|---|---|---|---|---|---|
| train_8 | corner | 13 | 0.0033 | 0.0010 | 22 h |
| train_8 | edge | 12 | 0.0011 | 0.0007 | 66 h |
| train_8 | interior | 5 | −0.0001 | 0.0001 | none (≈ 0) |
| train_2 | corner | 8 | 0.0080 | 0.0012 | 10 h |
| train_2 | edge | 37 | 0.0041 | 0.0015 | 21 h |
| train_2 | interior | 40 | 0.0008 | 0.0007 | 83 h |

The ordering corner > edge > interior holds in both buildings, and across all rooms the coefficient correlates with exterior cells per m² at **r = 0.81** (`train_8`) and **0.97** (`train_2`). The median R² of the one-step regressions is 0.26 and 0.33 (one-step differences are small and noisy), so the coefficients are reliable as group differences rather than as per-room values. The all-off drift also contains internal gains, which the occupancy term only partly absorbs.

### 14.4 What the model contains

```
per zone z (set by metadata, same for every building)
    static features   s_z = [ log area, exterior cells per m² (N, S, E, W), n wall neighbours, n door neighbours, type ]
    dynamic inputs    w_z,t = [ T_z, realized damper, realized reheat ]                      (from the lean observation)

encoder   h_c,t = Pool_{z : type(z) = c} φ( w_z,t , s_z )        c ∈ {corner, edge, interior}; φ shared by all zones
          h_p,t = ψ( T_water, AHU on/off, pressure, supply setpoint, recirculation )          (plant block; from the actions)
          h_t   = [ h_corner, h_edge, h_interior, h_p ]                                          size independent of V and K

dynamics  h_{t+1} = F( h_t, ū_t, d_t ; building descriptor )     ū_t = per-type mean damper and reheat, plant commands
                                                                   d_t = outside temperature, occupants, comfort flag, clock
decoder   T̂_z,t+1 = G( h_type(z),t+1 , s_z )                      one temperature per room

action    u_z,t = π( h_type(z),t , s_z ) + δ_z                    a per-type plan expanded to every room
```

With three components per type (air-like, mass-like, gain-like) and a plant block of about three numbers, the hidden state has **about 12–15 values for every building**, against a physical state of 10⁵ and an observation of 283–703. The building descriptor is computed once from the arrays: counts and areas per type, exterior fraction, door and wall adjacency counts between types, zones per AHU. The number of trainable parameters does not depend on V or K, and the model applies unchanged to a held-out building of any size.

### 14.5 Limits

- **Plant couplings are not type-local.** Zones on one AHU compete for its air (the simulator scales all boxes down if the fan cannot supply them), all coils share one hot-water loop, and AHUs straddle floors. The plant block and the AHU assignment must enter as explicit inputs.
- **The three types are coarse.** Within a type the coefficient still varies (corner `a` in `train_8` ranges 0.0014–0.0043), so the continuous exposure features belong in the decoder and the encoder, not only the label.
- **Neighbours matter beyond type.** An interior room beside a corner room behaves differently from one surrounded by interior rooms; neighbour-type counts or one round of message passing over the door and wall adjacency capture this at small cost.
- **Per-room hidden state remains.** The mass temperature and internal gains of each room are not observed; the pooled hidden state represents their type-level behaviour.
- **Floors need no index** in the thermal model (they are independent in the grid); a floor index matters only through the serving AHU.
- **The classification and regression use one weather strip per building.** Whether the type predicts the response in all regimes (hot, mixed, occupied) needs data from more scenarios.

# Part V. Our method

*Control-policy background, then the proposal: generate data from the simulator, discover a symbolic state-space model, and put a controller on top.*

## 15. What a control policy is, and the ways to build one

### 15.1 The idea of a policy
A **policy** is a function from what you can observe to what you will do:

```
a_t = π(o_t, history, building_meta_data)
```

Every five minutes the environment hands over an observation and expects an action. The policy is the *only* thing submitted: the physics, weather, occupants and reward all belong to the environment. A good policy is **closed-loop** — it reacts to measurements — rather than an open-loop schedule ("run the fans 06:00–19:00"), because weather and occupancy are not known in advance and the building's response is only partly predictable.

A little vocabulary will be used throughout. The **state** is everything that determines the future — including the hidden slab and wall temperatures and the full thermal field — and is never fully observed. A **disturbance** is something that drives the system but cannot be commanded: outside temperature, occupants, internal gains. The **plant** is the building plus its equipment (the simulator). A **dynamics model** is a controller's *internal* approximation of how the plant evolves, `x_{t+1} = f(x_t, a_t, d_t)`; the competition does not require one, but ours is built around one. The **horizon** is how far ahead a policy reasons: useful decisions such as pre-heating or coasting span roughly 1–6 hours, i.e. 12–72 steps.

### 15.2 A partially observed problem across a family of buildings
Formally this is a **partially observable Markov decision process** with a hybrid action space, and with a *family* of environments: the transition function depends on a hidden **context** — geometry, materials, equipment sizes, weather regime — that the policy never sees. The objective is therefore meta-level: do well *on average across the generator's distribution*, which is what the leaderboard samples. Because the context is hidden but *partially revealed by behaviour*, a good policy does implicit or explicit system identification: it watches how zone temperatures respond to what it did and adapts.

Partial observability bites in three ways here. The observation gives a zone *average*, not the slab, walls or individual cells, and thermal mass stores heat that the sensor cannot see but that returns later. Occupancy is a building-level count, while internal gains and the reward are per zone. And weather is only the current temperature, with no forecast — whether a useful short-term predictor can be built from the time of day and recent history is itself a design question.

### 15.3 The layers of a building controller
It helps to think of control in layers; the shipped baselines and any good solution implement some version of this hierarchy:

```
 Layer 4  Scheduling / supervisory     "occupied from 06:00?", pre-heat start time, setback depth
 Layer 3  Plant-level setpoints         AHU supply-air temp, fan static pressure, recirculation,
                                        hot-water supply temp, pump differential pressure
 Layer 2  Zone-level allocation         per-VAV damper and reheat-valve positions
 Layer 1  Equipment on/off              AHU run commands, hot-water run command
 ----------------------------------------------------------------------------------------
 Layer 0  Built-in loops (inside sim)   thermostat mode machine, fan/valve actuators,
                                        ventilation floors, discharge-temp cap
```

Layer 0 is not yours: the simulator already contains per-zone thermostats, floors and discharge limits. A policy sits on top and commands Layers 1–4. Knowing what Layer 0 does prevents you from "controlling" something the simulator overrides.

### 15.4 Families of approach, and how they fare here
**Rule-based and bang-bang** control switches equipment on and off when a variable leaves a band. It is simple and robust but wasteful — full-blast setpoints and constant oscillation.

**Feedback control (PID)** sets the command from the error between target and measurement: proportional to the present error, plus an integral term that removes steady offsets, plus a derivative that damps overshoot. It needs no model, which is why it is the industry default. Its weaknesses all appear in this task: it is *reactive* (it acts after comfort has been lost, so it cannot pre-heat before the morning ramp or coast through the evening); it is *local* (an air handler's PID does not know its neighbour is also drawing hot water); gains tuned for one building are wrong for another (`pid` hard-codes `kp=0.5, ki=0.02, kd=0.1`, a compromise across the shipped six); and it is *cost-blind* (it does not know that gas and electricity are priced differently or that fan power climbs steeply with pressure).

**Feed-forward plus feedback** adds a model-based term that anticipates what the plant will need — "outside is −25 °C, so expect the envelope to draw this much power; raise the water temperature now" — and lets PID trim the residual. It is classical, low-risk and a strong upgrade. It *requires a model of the building's heat balance*, which is exactly what symbolic discovery would supply.

**Model predictive control (MPC)** solves, at every step, a small optimization over the next few hours using a dynamics model, applies only the first action and re-solves:

```
minimize   Σ_{k=0}^{H-1}  [ comfort_penalty(x_{t+k}) + λ_E·energy(a_{t+k}) + λ_C·carbon(a_{t+k}) ]
subject to x_{t+k+1} = f(x_{t+k}, a_{t+k}, d̂_{t+k}),   a_{t+k} within the allowed set (with floors)
```

It is the textbook way to exploit thermal mass and to trade fan, cooling and reheat energy against comfort. It needs a model, a state estimate, a disturbance forecast, and solve times that fit the evaluation budget (8,640 steps × 30+ scenarios).

**Reinforcement learning** learns the policy by trial and error in simulation. Model-free methods (PPO, SAC, TD3, hybrid-action variants) need very many simulator steps, and this simulator slows with building size. RL must also solve the variable-shape problem (§15.6). It can discover non-obvious trade-offs but is brittle off-distribution, and with an unbounded comfort penalty one bad exploration can produce enormous negative returns.

**Model-based RL** learns a dynamics model from data and plans or trains inside it. It is sample-efficient, but black-box models can compound error over a horizon and hide physical inconsistencies — the repo's own comments describe simulator bugs where energy was created from nothing, which physics-regression tests now guard against. An unphysical model gets exploited.

**Hybrids** — a structured baseline plus a learned residual, an MPC "teacher" distilled into a fast network, RL fine-tuning of a physics-informed controller — are, in practice, where most strong systems end up. Ours belongs to the feed-forward / MPC / model-based family, with one distinguishing twist: **the model is symbolic and discovered from data**, rather than hand-derived or a neural network.

### 15.5 What good control looks like physically
Consider a winter morning in Denver, with the building empty overnight and occupants due at 06:00 on a weekday. A *reactive* controller waits until the comfort band activates — zones below 294 K — and then slams the boiler to maximum. The zones are cold for the first hours (comfort loss, linear to 5 K, then quadratic) and the boiler runs at peak (energy and carbon): exactly the dip in Figure 3. A *model-based* controller instead estimates how long the building takes to warm from its current temperature (the thermal time constant set by the envelope and interior mass); starts reheat *before* 06:00 at moderate water temperature and low fan pressure, arriving at the band edge precisely at 06:00; uses minimum fan pressure (flow ∝ √P, so power grows roughly as P^1.5) and leaves dampers at their floors for rooms that are already warm; and backs off in the evening and lets the slab carry the building through the eco band. Every one of those decisions is a statement about the building's heat balance, which is the case for learning that balance explicitly.

The levers by regime, summarized: in **heat**, the supply-air temperature, fan pressure, damper allocation and recirculation (economizer off); in **cold**, hot-water temperature and pump pressure and the per-zone reheat valves, with the air-side supply temperature largely dead; in **shoulder weather**, the economizer and minimal fans while avoiding cooling and reheating at once; and in **time**, pre-conditioning before `morning_start_hour`, coasting before `evening_start_hour`, and deep weekend setback, using slab mass to shift load. Only occupied zones carry the comfort term, so empty rooms can sit near the band edge.

### 15.6 The shape problem
Because K and V vary, a policy needs an *architecture* that produces the right number of outputs for any building. Four ideas recur: **index tables** from the metadata, built once in `__init__` (what all baselines do); **weight sharing** across zones and air handlers, where one small function maps each zone's local features plus its air handler's features to that zone's damper and reheat; **set or graph encoders**, with zones as nodes, air handlers as hubs and adjacency from the floor plan; and **pooling** for plant-level actions, since air-handler and boiler setpoints depend on aggregates (mean or maximum demand) over their zones. A symbolic model gets this property for free if it is written *per zone and per equipment unit* with building-specific *parameters*: the equation is the same everywhere; only the coefficients and the connectivity differ (§17.3).

### 15.7 Where things go wrong
Several failure modes deserve a mention because they come up repeatedly:
- The **unbounded comfort penalty**: a failure invisible in the mean — a stuck valve, an air handler left off in a cold snap — can dominate a scenario. Guard rails help: if an occupied zone approaches the band edge, fall back to a conservative baseline-like action.
- **Zero is not off** and **floors are silent**: write every slot, and read realized positions from the observation, not the command (§5, §6).
- **Dead actuators**: tuning the supply-air temperature in cold climates wastes effort and mis-attributes effects.
- **Chatter**: each VAV's thermostat has hysteresis (heating continues to the band midpoint), and toggling equipment faster than that loop causes oscillation.
- **Short-episode flattery**: one-day runs skip eco periods, so never compare returns across step counts.
- **Overfitting**: the shipped set is only six buildings. Sample many more with the generators (Appendix C) and split by building.
- **Compute**: simulation cost scales with building size, and `get_action` runs 8,640 times per scenario; do index-table work once, in `__init__`.

The submission contract itself is minimal (§4.1).

### 15.8 Candidate directions
Putting this together, five directions look promising, and they combine well. (1) A **structured, physics-informed heuristic** that improves on `pid_water` with feed-forward from outside temperature and schedule, per-zone allocation, fan-pressure minimization, a hot-water reset keyed to the *maximum* zone demand, and pre-heat/pre-cool — cheap, interpretable and transfers by construction. (2) **MPC with an identified model**: fit a low-order RC model online from the first days and plan over a short horizon. (3) **Set- or graph-structured RL**, a shared per-zone and per-air-handler network (DeepSets, a GNN or a transformer) trained on the 30 scenarios plus many generated ones. (4) A **learned residual on a heuristic**, which stabilizes training and bounds the damage of the unbounded penalty. (5) **Imitation followed by RL fine-tuning**, or domain-randomized RL with a context encoder that infers a latent "building + climate" vector from recent history. **Our method combines directions 1 and 2**: a physics-structured controller around an *identified* model.

---

## 16. Our idea: discover the physics, then control with it

> *Status: §16–§19 describe the team's proposed approach. It is a design, not an implementation. Statements about the simulator come from its source; statements about what the method will achieve are hypotheses to be tested.*

### 16.1 The idea
Instead of training a black-box policy or hand-tuning PIDs, **discover compact, interpretable equations that describe how the building responds to HVAC commands and weather**, using symbolic regression or sparse identification on data collected from the simulator; then **wrap those equations in a control policy** — feed-forward with feedback, or MPC — that plans by simulating the discovered model. Because the equations have the same *form* in every building and only a handful of *parameters* change, the policy can generalize to unseen buildings by identifying those parameters online, in the first hours or days of the episode.

### 16.2 Why it fits this competition
- **Held-out buildings.** Physics has universal structure — an energy balance. Discover it once; fit only the resistances, capacitances and gains per building.
- **Variable shapes.** The equations are written per zone and per unit with a shared form; the policy composes them using the room-to-air-handler assignment and the floor plan.
- **The unbounded comfort penalty.** A model lets the controller *predict* an excursion before it happens and veto risky actions; interpretable equations make that check auditable.
- **Energy and carbon.** Fan, pump, coil, boiler and cooling models are close to closed-form in the simulator, so a symbolic model can price actions exactly, where RL must learn costs from noisy reward.
- **Compute.** Small closed-form models are cheap; thousands of rollouts per step are affordable inside MPC.
- **Sample efficiency.** A few simulated weeks of excitation data from generated buildings may suffice, whereas RL typically needs orders of magnitude more simulation.

### 16.3 What "symbolic discovery" means
**Symbolic regression** searches the space of mathematical expressions — for example `a·(T_out − T_z) + b·ṁ·(T_s − T_z)` — for ones that fit data well while staying *simple*, in contrast to a neural network's millions of opaque coefficients. Several families of method are relevant:

- **SINDy** (sparse identification of nonlinear dynamics, with its control extension) picks a library of candidate terms and solves a sparse regression `ẋ ≈ Θ(x,u)·Ξ` so only a few survive. It is fast and works well when the library is right, and our physics supplies a strong library. Its **weak-form** variant integrates against test functions rather than differentiating, which is robust to noise and coarse sampling — relevant to five-minute zone averages.
- **Genetic-programming symbolic regression** (PySR, gplearn, Operon) evolves expression trees under a complexity penalty and returns a Pareto front of accuracy versus size. It can find structures outside the library, such as √-pressure laws, but is slower; it is good for the nonlinear equipment maps.
- **Physics-guided methods** (AI Feynman and relatives) use dimensional analysis, symmetry and separability to decompose the search and enforce units and monotonicity.
- **Neural- or LLM-guided search** uses a learned prior to propose candidate expressions; useful as an accelerator, with the final model kept symbolic.
- **Grey-box hybrids** pair known physics with a small learned residual, later distilled symbolically — a pragmatic middle path when a pure symbolic model leaves structured error.

The governing principle is **parsimony**: among models of similar accuracy, prefer the simplest, because it generalizes better to unseen buildings and is easier to wrap in an optimizer.

### 16.4 What to discover, and what to leave known
Using the blocks of §8.2, we divide the work so that the search space stays small and the result physical. The **zone RC dynamics** (block A) and the **inter-zone coupling** (restricted to floor-plan neighbours, since only adjacent rooms couple) are the core, building-dependent, hardest part: we *discover the structure and fit the parameters*. The **occupancy gain** is discovered as a low-order function of occupancy, schedule flags and hour, and the **weather-to-envelope** link as an effective loss coefficient plus convection coupling. The **VAV airflow and coil heat** (block B) are seeded with their known form and refined — symbolic regression confirms the saturations and fits the constants. The **air handler** (block C) and **boiler and pump** (block D) are algebraic and well documented, so we keep their known form and fit only COP, efficiency, dissipation and tank constants; gas is about half of the energy reward, so these matter.

Where an equation is already known exactly — the fan and pump laws — we do not spend search budget rediscovering it, but we *do use the discovery pipeline to verify it*: if symbolic regression on simulator data recovers `√P` and `V̇·ΔP/η`, that validates the toolchain before we trust it on the harder zone dynamics.

### 16.5 The pipeline
Six stages take us from simulator data to a scored policy:

```
 ┌───────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │ 1. DATA           │   │ 2. DISCOVERY         │   │ 3. PARAMETRIZE &     │
 │ Excite simulator  │──►│ Symbolic regression  │──►│    VALIDATE          │
 │ across many       │   │ / sparse ID on       │   │ Fit per-building     │
 │ generated         │   │ pooled data →        │   │ params; multi-step   │
 │ buildings+weather │   │ structure (shared)   │   │ rollout error;       │
 └───────────────────┘   └──────────────────────┘   │ held-out buildings   │
                                                    └──────────┬───────────┘
                                                               ▼
 ┌───────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │ 6. EVALUATE       │◄──│ 5. CONTROL POLICY    │◄──│ 4. ONLINE ADAPTATION │
 │ evaluate.py,      │   │ MPC / FF+PID wrapper │   │ State estimator +    │
 │ score_leaderboard │   │ around the model     │   │ recursive parameter  │
 │ 30 + generated    │   │ + safety fallback    │   │ estimation at run    │
 └───────────────────┘   └──────────────────────┘   │ time                 │
                                                    └──────────────────────┘
```

Stages 1–3 are the subject of §17, stages 4–5 of §18, and stage 6 of §19.

---

## 17. Stage one: data and discovering the dynamics

### 17.1 Generating the data
There is no logged dataset (Appendix A), so the simulator is the data source, and the repo can generate arbitrarily many buildings and scenarios. Our angle is to **generate as much data as possible, then fit one symbolic state-space model on all of it, then put a controller on top.** Identification needs **persistent excitation**: the inputs must vary enough to reveal the dynamics.

**How data is produced.** Buildings come from `run_floor_generator` (seeds from 500 upward; materials are stratified across a batch and reproducible from the seed list), scenarios from `run_scenario_generator` (naming `--buildings` or `--weathers` takes the cross product; no arguments rebuilds the shipped 30 pairs), and the commands are in Appendix C. The shipped manifest pairs each building with one strip per station, but the generator inputs for the other stations and buildings are shipped too ("nothing here limits what you may generate"). Our collection loop lives outside the starter kit and drives the environment through `reset` and `step` using the `evaluate.py` helpers (`load_hybrid_environment`, `create_building_meta_data`, `SimulationTracker`), as `code/rollout_checks.py` already does, with a behaviour policy in place of random actions.

**What to vary.**
- **Buildings.** Many seeds, so geometry, K, V and independently drawn materials span the generator's range. This is what lets the discovered *structure* generalize to hidden buildings; the number of distinct buildings, not the number of steps, limits transfer.
- **Weather.** All five climates and as many strips of each as exist (`list_recorded_weather_presets()`), covering heating-, cooling- and mixed seasons.
- **Behaviour policies (excitation).** Random-step or pseudo-random sequences on dampers and reheat valves, ramps on hot-water temperature and fan pressure, air-handler and pump on/off cycling, with the actions held for 6–24 steps so the thermal response shows through the 5-minute step; plus the four shipped baselines (`bang_bang`, `fixed_setpoint`, `pid`, `pid_water`) for realistic operating regimes, and `all_off` for pure envelope response (§14.3). Pure excitation visits unrealistic corners; baseline data alone is poorly excited because the baselines often sit on actuator floors, so both are needed. Later rounds add data collected under the fitted controller itself (DAgger-style), to cover the states it will actually visit.
- **Regimes.** Occupied and eco periods, coils saturated and idle, floors active and inactive, full 30-day episodes (shorter runs skip the eco-setback periods and flatter schedule-aware behaviour).

**What to log per step.** The full observation vector and the action actually sent; the realized damper and reheat values (already in the observation, §6.1); and, for the grey-box route, the simulator's *internal* quantities from the reward-info structures (heat applied to each zone, fan and gas power, electricity and carbon terms). The internal quantities identify blocks B–D in isolation with near-noise-free data, then compose them; the policy will not see them at submission time, so the final model may only *need* observable inputs. Store the metadata JSON beside every trajectory, so room types, floors and AHU assignments (§14) can be rebuilt for each building.

**Data budget.** One 8,640-step episode on the smallest building takes about 90 s [X] (≈100 steps/s); larger buildings are slower and unmeasured (§20.3), so the numbers below are for the smallest building and scale up by the measured slowdown for the others. One episode logs 8,640 rows of `V+8` to `6V+15K+58` values; for the 30 shipped scenarios that is 259,200 rows, about 45 minutes of simulation. Episodes are independent, so generation parallelizes across CPU processes with no coordination. The practical limits are the number of CPU cores and disk, not the method.

**Fitting-side rules.**
- **Targets.** Use the 300-second step with finite differences `(T_{t+1} − T_t)/Δt` as derivative targets, or fit a discrete-time one-step map directly, which avoids differentiating noise (the simulator is noise-free, §20.3).
- **Split by building**, never by time within a building, otherwise we measure interpolation rather than transfer.
- **Mind the scales.** The observation is mixed-scale (§6.5); undo the normalization before fitting physical equations.
- **Use the lean model inputs.** The model's inputs and outputs are the ones in §13.2–13.3 (zone temperatures, water temperature, outside air, occupants, time and comfort-mode features, plus the previous action), not the full 6V+15K+58 vector.

### 17.2 Designing the library
Whether we use SINDy or genetic-programming regression, we choose the *inputs, operators and constraints* — this is where physical knowledge is injected. The variables include the zone temperature, the outside temperature, neighbouring zone temperatures (or their mean), supply-air and water temperatures, the damper, valve, pressure and recirculation settings, occupancy, schedule flags, and sine/cosine of time. The **derived physical features** do much of the work: airflow `V̇ ∝ d·√P`, the air capacity rate `c_a ρ_a V̇`, the temperature differences `(T_sa − T_z)`, `(T_w − T_z)`, `(T_out − T_z)`, and the mixed-air temperature `r·T_ret + (1−r)·T_out`. The operators are `+ − × ÷ √ min max clip`, possibly `tanh` for smooth saturation, with `sin/cos` allowed only for time and `exp` avoided for equipment maps. The **constraints** enforce dimensional consistency, non-negative gains, sign and monotonicity (more reheat means more heat into the zone; higher outside temperature means more heat gain), and conservation — the coil's heat to the air must equal the heat charged to the boiler, a property the simulator's own regression tests check. A complexity penalty is tuned so the final model is read from the *elbow* of the Pareto front, not from its most accurate end.

### 17.3 The generalization trick: structure versus parameters
This is the heart of the method. We write each model as

```
dT_z/dt = F( x_z, u_z, d_t ; θ_building )
```

where **F — the symbolic structure — is discovered once on pooled data from many buildings**, and **θ — a small vector of lumped capacitances, conductances and gains — is specific to each building and estimated online.** This differs from "fit a separate model per building offline," which is impossible for held-out buildings, and from "one global model with no parameters," which cannot represent different constructions. In this light, the six buildings of §9 are simply six values of θ, and what separates `building_train_1` from `building_train_7` is a handful of numbers, not a different kind of physics.

There are two ways to obtain θ for a new building. **Online estimation** from the first hours or days of the episode (recursive least squares, an extended Kalman filter or moving-horizon estimation) is fast and well posed because RC models are linear in their parameters. **Priors from metadata** use the floor-plan and zone-map arrays, which give zone area, perimeter and adjacency and so predict capacitance and exterior exposure; we can learn a symbolic map `θ = g(metadata)` as a starting point and refine it online. The material properties are not exposed (§6.5), so θ must otherwise be inferred from behaviour alone. §14 shows that room type, exterior exposure and adjacency can be rebuilt exactly from those arrays and that exposure predicts the envelope coupling, so the prior has a concrete form.

### 17.4 Noise, hidden states and identifiability
Several difficulties are predictable:
- **Hidden states.** A one-state-per-zone model will show systematic hour-scale error because the slab and walls store heat the sensor cannot see. Remedies: add one or two latent states per zone and estimate them; use lagged temperatures or moving averages as a proxy for mass temperature; or fit a two-time-constant structure and let sparsity prune it.
- **Identifiability.** When nothing is excited — the building sits in band with floors active — some parameters are unobservable; use regularized estimation with priors and freeze parameters when excitation is low.
- **Aggregation error.** The observation is a zone *average* of a spatial field; treat the residual as process noise and make the controller robust to it.
- **Regime switching.** Floors, the `min()` in supply temperature, the coil-effectiveness cap and thermostat hysteresis make behaviour piecewise. Include explicit `min/max/clip` operators, learn a regime classifier, or fit local models per regime gated by symbolic switching conditions.
- **Stochastic occupancy** is best modelled as a disturbance with a learned expected profile by hour and day plus uncertainty, not predicted exactly.

### 17.5 When is a model good enough to control with?
One-step accuracy is misleading — nearly any model predicts `T_{t+1} ≈ T_t`. We will judge models by **multi-step rollout error** at 1, 3 and 6 hours *under the actions the controller would actually pick*; by performance on **held-out buildings and climates**, stratified by climate, season and occupied versus eco; and, ultimately, by **closed-loop value** — plug the model into the controller and measure the leaderboard-style score, since a less accurate but better-conditioned model often wins. We also require **physical sanity** (heat balance closes, no heat from nothing, correct signs, bounded extrapolation), **robustness** (a ±20 % parameter perturbation should not wreck the policy) and low **cost per evaluation**, since the model is called many times inside MPC.

### 17.6 Shipping the fitted model and the runtime budget
The constraints of §4.3 shape what can be submitted.
- **Everything the policy needs at run time travels inside the notebook or an attached Kaggle dataset.** There is no internet, so no downloads and no installs. The discovered equations, fitted constants and per-type priors are written as plain Python/NumPy or a small JSON or `.npz` file, with versions pinned. Prefer NumPy at run time over a symbolic-regression toolchain: PySR needs Julia, and the availability of SINDy and CasADi in the Kaggle image is unverified (§20.3). Do the discovery off-line and ship only the result.
- **The policy is built once per building and adapts online.** `__init__` does the per-building setup (index tables, parameter priors from the floor plan and zone map, §14); the object then carries the estimator state and its previous action between calls (§13.2 relies on this to drop the setpoints from the model's inputs). The notebook is deterministic and re-run on unseen buildings, so there is no per-building offline fit.
- **Runtime.** The 8,640 × 30 evaluation must finish inside the notebook limit, and the simulator's own time counts: about 90 s per scenario on the smallest building [X], more on larger ones (unmeasured). A closed-form model with tens of terms per zone and per type evaluates in microseconds, so even a sampling MPC over hundreds of candidate action sequences is small next to the simulator.
- **Reproducibility.** Winners must deliver training and inference code (Rules §2.8), so the data-generation scripts, seeds, the discovery configuration and the notebook builder are versioned together; trajectories are regenerable from seeds rather than stored.
- **Fallbacks.** The notebook includes the PID-like fallback of §18.4 in the same file, so a bad model cannot take the policy off the leaderboard.

---

## 18. Stage two: wrapping the model in a controller

### 18.1 Three wrappers, built incrementally
The same model can power three increasingly ambitious controllers.

**Wrapper 1 — model-based feed-forward on the existing structure (lowest risk).** Keep a `pid_water`-like architecture but replace hand-set choices with model outputs: required heat per zone from the discovered balance becomes the damper and reheat commands; hot-water temperature is set to the minimum that satisfies the *largest* zone demand; fan pressure to the minimum that delivers the required airflow; and the pre-heat start time follows from the identified time constant. Feedback trims what is left. This can only be modestly worse than the baseline if the model is wrong, because the model computes setpoints rather than being trusted blindly.

**Wrapper 2 — model predictive control (the main target).** At each step: (1) **estimate** the state and update θ; (2) **forecast** disturbances — outside temperature over the horizon (persistence plus a diurnal pattern fitted from the day so far, or the last 24 hours) and occupancy from the schedule flags and time-of-day profile; (3) **optimize** the action sequence over a horizon of 2–6 hours, with coarser 15–30-minute blocks beyond the first hour to keep it cheap; (4) **apply** the first action. Solver choice depends on the model's smoothness: gradient-based methods (CasADi/IPOPT, or JAX with L-BFGS) if it is smooth; **sampling-based MPC** (MPPI, CEM, random shooting) if it has `min/max`, thresholds or hysteresis — robust, easily parallel with NumPy/Numba, and able to handle discrete on/off directly; or a hierarchy with continuous optimization for setpoints and enumeration or rules for equipment on/off per time block. The cost is the competition reward evaluated on predicted trajectories, plus small regularizers for action smoothness and terminal terms (the slab should end the horizon in a good state). Because the comfort term is unbounded, we add a **chance margin**: aim at the band centre plus a buffer proportional to predicted uncertainty.

**Wrapper 3 — distillation or a learned residual (optional, later).** Use MPC as a teacher to train a small, fast policy network, or learn an RL residual on top of the MPC action. Useful if solve time is a constraint or systematic model bias can be corrected from reward feedback.

### 18.2 The decisions it has to make
Layer by layer (§15.3): for **equipment on/off**, run an air handler only when its zones need conditioning *or* ventilation is required during occupied hours, switch off in eco mode when the building can coast (the model predicts the coast time before the eco band is violated), and avoid short cycling. For **per-zone dampers and reheat**, deliver each zone's need at minimum airflow, using the damper floor for satisfied zones and avoiding cooling at the air handler while reheating at the VAV unless required. For the **air-handler supply temperature**, treat it as a lever only when the mixed air is above the setpoint; when the model predicts the mixed air below the coldest commandable setpoint (cold climates), hold a nominal value and use the water side — the model makes the inert regime explicit instead of letting it be discovered by ablation. Set **static pressure** to the lowest value that meets the served zones' flow demand; use **recirculation** for free cooling when `T_out` is below return temperature and cooling is needed, limit it in cold weather, and respect the 70 % cap; set the **hot-water temperature and pump head** to the lowest values that satisfy the most demanding coil; and time **pre-conditioning and setback** so the building reaches the comfort band at `morning_start_hour` and drifts to the eco band at `evening_start_hour`.

### 18.3 Making an unobserved model usable
The controller needs the hidden states and the building parameters. In rough order of simplicity: **recursive least squares** on the linear-in-parameter form, with a simple observer for hidden states; an **augmented-state Kalman filter** over zone temperatures, slab temperatures and parameters, with the symbolic model as the process model; **moving-horizon estimation**, which fits state and parameters to the last N steps and is more robust to nonlinearity at higher cost; and a **warm-start phase** in which the first hours run safe baseline-like behaviour with small deliberate dithering to learn θ before switching to MPC. There is no warm-up exemption — every step is scored — so the dithering must stay gentle and band-respecting.

### 18.4 Handling model error safely
Models are wrong, and the comfort penalty is unforgiving, so safety is layered: an **integral-like correction** on the one-step prediction error (offset-free MPC) cancels persistent bias; **uncertainty-aware margins** widen the comfort buffer when parameter variance is high or the regime is poorly covered by training data; a **fallback controller** takes over if an occupied zone leaves a safe inner band or recent prediction error exceeds a threshold, handing control to a robust baseline-like policy (for instance `pid_water` logic) until the model recovers; the known **floors** (200 Pa, 0.7 recirculation, 10 % damper) are enforced inside the optimizer so planned actions are realizable; and **rate limits** protect against the thermostat's hysteretic inner loop and equipment transients.

### 18.5 What the submitted policy looks like
It conforms to the contract of §4.1 and, in the assembled notebook, inherits from `Policy`. This is a design sketch, not code in the repo:

```python
class SymbolicMPCPolicy:
    def __init__(self, building_meta_data, action_spec):
        # 1. Build index tables from metadata (temp_indices, pressure_indices[:K] and [K],
        #    water_indices, recirculation_indices, zone_action_indices, observation_field_names).
        # 2. Instantiate the discovered symbolic model structure with per-zone / per-unit
        #    parameters initialised from priors derived from the floor plan and zone map.
        # 3. Initialise the estimator (state + parameters) and the safety fallback policy.
    def get_action(self, obs):
        # a. Parse obs -> measurements (undo normalization); update estimator (state, theta).
        # b. Forecast disturbances (outside temp, occupancy, schedule flags).
        # c. Solve the receding-horizon problem with the symbolic model -> action sequence.
        # d. Safety layer: floors, rate limits, fallback if in danger.
        # e. Scatter into continuous_action[3K+2+2V] and discrete_action[K+1] via the index
        #    tables; write every slot explicitly (zero is not off).
        # return {"continuous_action": ..., "discrete_action": ...}
```

`get_action` runs 8,640 times per scenario and 30 times over on the leaderboard, so model evaluation and any optimization must be budgeted against the 9-hour notebook limit (§4.3, §17.6). The workflow around it: develop and score locally with `python -m smart_buildings.evaluate --policies <key>` and `score_leaderboard.py` (register the class in `POLICY_MAP`, or keep it outside the starter kit and call `run_policy` directly), then paste the same class into the Kaggle notebook.

---

## 19. Testing it honestly, and what could go wrong

### 19.1 The questions we want answered
1. **Does the discovered structure transfer?** Fit on N buildings and test the rollout error on unseen buildings and climates.
2. **How much does symbolic structure help?** Compare against a hand-derived RC model with identical parameters, black-box MLP/GRU dynamics, and a linear ARX model.
3. **Does online identification close the gap** between the pooled model and a per-building oracle whose parameters are read from the config?
4. **What does the model buy the controller?** Ablate in steps: PID baseline → + feed-forward → + water reset → + pre-conditioning → full MPC.
5. **Where is the headroom per climate?** Break results down by station and season. We expect the cold to reward water-side and reheat management and pre-heat, the heat to reward supply temperature, fan pressure and damper allocation, and the mild middle to reward the economizer.

### 19.2 The protocol
We evaluate on the **shipped 30 scenarios** at 8,640 steps — directly comparable with `baseline_returns.json` (the reference is about −0.054 for `pid_water`, and about −0.047 for the per-scenario best-of-baselines oracle of §11.4) — and on a **generated hold-out set** of many new buildings (seeds from 500) paired with station strips, including buildings larger than the shipped six. We report the leaderboard metric, a per-climate and per-season breakdown, the **worst-scenario score** (tail risk), the comfort/energy/carbon decomposition of §3.3, and wall-clock time per step. We never compare across step counts, never tune on the hold-out set, and run `check_scales.py` on generated scenarios so the reward scale is right. The baselines to beat and to include in ablations are `pid_water`, `pid`, `fixed_setpoint` and `bang_bang`; a learned RL controller would be a useful external comparison since the repo ships none.

### 19.3 The main risks
- **The symbolic model is too simple for hidden-state dynamics**, giving multi-hour prediction errors and poor pre-conditioning. *Mitigation:* a latent mass state, online estimation, validation on multi-step rollouts.
- **The search returns non-physical equations** that fit but do not extrapolate. *Mitigation:* physics-constrained libraries, units, monotonicity, a complexity penalty, held-out-building validation.
- **Piecewise or hysteretic behaviour defeats smooth optimizers.** *Mitigation:* sampling-based MPC, explicit regime gating, a smoothed cost.
- **The identification period is expensive** because every step is scored. *Mitigation:* priors from metadata, gentle dithering, a baseline-like safe mode.
- **The unbounded comfort penalty** means one zone failure dominates. *Mitigation:* safety margins, a fallback controller, worst-case reporting.
- **Compute budget**: MPC may be too slow for 30+ scenarios in a session. *Mitigation:* short horizons, coarse blocks, vectorized sampling, distillation.
- **Overfitting to the six shipped buildings.** *Mitigation:* train and validate on generated buildings, split by building.
- **Reward-hacking of model quirks**, where the model claims "free heat" and the simulator disagrees. *Mitigation:* energy-balance checks and closed-loop evaluation in the true simulator.
- **Rule compliance**: the policy may use only the observation and the metadata; no station or building identifiers.
- **Unverified Kaggle runtime details** — the `Policy` base class and which libraries the image contains — which could invalidate a heavy dependency (§20.3). *Mitigation:* ship plain NumPy; test the assembled notebook on Kaggle early.

### 19.4 A staged roadmap
1. **Instrument and understand.** Run the baselines (Appendix B), log per-step internals, reproduce `baseline_returns.json` for a few scenarios and map the observation fields to model variables.
2. **Validate the pipeline on known physics.** Use symbolic regression or SINDy to recover VAV airflow (`∝ d·√P`), the fan and pump laws and the coil heat — a unit test for the toolchain.
3. **Discover the zone thermal structure** from pooled generated-building data, and compare against a hand-derived RC reference.
4. **Online identification.** Show that θ estimated from the first day on an unseen building gives multi-hour rollout errors near the oracle.
5. **Wrapper 1.** Feed-forward, water reset and pre-conditioning; target beating `pid_water` on the shipped set.
6. **Wrapper 2.** MPC with a safety fallback; ablate horizon, cost margins and solver.
7. **Hold-out validation** on many generated buildings, with stress tests (a cold snap, a heat wave, larger K).
8. **Package** into the policy class (§4.1, §17.6), verify runtime, submit.

# Part VI. Verification

*What was checked by running code, and what remains open.*

## 20. What has been verified and what is still unknown

### 20.1 How this report was checked

Most numbers come from reading configs and source and from computing over the shipped data files. Everything tagged [R] and [X] comes from running code in `code/`, which imports the starter kit, calls only `reset` and `step` on the environments, and writes only into `outputs/` (the starter-kit checkout stayed unchanged on `main` throughout). The scripts are `dump_vectors.py` (field and action names for all six buildings), `rollout_checks.py` (600 random-action steps, `train_8`), `explore_zone_map.py`, `floors_and_ahus.py` and `plot_rooms.py` (room geometry), `metadata_only_check.py` (§14.1) and `all_off_response.py` (§14.3), with `annotate_vectors.py` and `build_appendix.py` for the listings.

### 20.2 Confirmed

**By instantiating the environments [R].** The vector lengths, specs, index layout and metadata keys of §5–§6 were read from all six buildings (a docstring's 271 for the smallest observation is stale; it is 283). Beyond those:
- A freshly reset building is finite and quiet: zones at 294 K, dampers at the 0.1 floor, reheat closed, airflow 0, hot-water loop at 340 K, no occupants (the strip starts on a Sunday), comfort flags 0. `env.step(action)` returns a tuple whose first two entries are the observation and the reward.
- With a zero continuous action and every unit on, one simulated day gave a per-step reward of −0.130 to −0.116, worse than any tuned baseline mean, consistent with "zero is not off" (§5.2).
- The smallest building simulates at roughly 100 steps per second, so a full episode takes about 90 seconds there; larger buildings are slower.

**By the 600-step rollout [X].** `rollout_checks.py` runs `train_8` / `atlanta_1` for about 50 simulated hours, crossing the Monday 06:00 switch to comfort mode, with piecewise-random actions resampled every 12 steps and each AHU and the hot-water loop switched off about 15 % of the time. It is the source of every "relation found [X]" in §6.1–§6.3 (which fields are duplicates, mirrors or constants, which echoes are floored, which signals have memory); results are in `outputs/rollout_train_8_checks.json`. Under these random actions zone temperatures ranged over 283.0–304.0 K, and occupants were nonzero on 30 % of steps (0–57 of 60; comfort mode on 30.2 %).

**Geometry and hidden-state checks [X].** §14.1 (room numbering, types and floors from metadata alone) and §14.3 (envelope coupling by type).

Caveats: the rollouts use one or two buildings, one weather strip each and non-tuned actions, so they do not reach every regime; the airflow signals were not tested for memory.

### 20.3 Still unknown
1. The **value ranges and variability over a full episode** under each baseline: zone-temperature spread, actuator saturation, how often the supply-air setpoint is inert, the occupancy profile.
2. The **reward decomposition per baseline and climate** (comfort versus electricity, gas and carbon), which would show where the headroom is.
3. Whether the **US/Pacific schedule on non-Pacific weather** has any effect.
4. **Sensor noise or delay**, if any; initial readings and the rollouts look noise-free.
5. **Simulation speed across all building sizes**, which bounds how much data generation and in-simulation MPC is affordable.
6. The exact **internal-gain parameters** in the scenario configs and whether they vary by building.
7. Whether **solver libraries** (CasADi, JAX, PySR with Julia) are available in the Kaggle runtime image. The notebook limit (9 h) is known; the exact `smart_buildings_inference_server.Policy` base class and the `kaggle_evaluation` files were not visible without accepting the rules, so whether the runtime passes the same metadata and observation as `evaluate.py` is unverified.
8. Whether the **airflow and outside-air-flow signals have memory**, and whether the physical supply temperature `min(mixed air, setpoint)` is observable anywhere (the sensor reports the setpoint).
9. Whether **room type predicts the response in all regimes** (§14.5) and whether the exterior-reach threshold of §14.2 is the right one; the all-off rollouts cover two buildings.
10. Whether the **solver couples floors** (§9.4); the layout suggests not.

Items 1, 2 and 5 can be closed with the repo's existing commands; Appendix B is a recipe for doing that on another machine.

---

# Appendices

## Appendix A. Files

### A.1 The shipped files

| Asset | Count / size | Contents |
|---|---|---|
| Building definitions `buildings/*.yaml` | 6 (`building_train_{1,2,4,5,7,8}`) | seed, footprint, floors, room-size range, AHU count, occupants per zone, six material properties |
| Scenarios `training_scenarios/<b>/<w>/` | 30 (≈ 100 MB) | `config.gin` (541 lines: simulator, reward, normalization), `config.yaml` (readable summary), `floor_plan.npy`, `zone_map.npy` |
| Weather stations `weather/csv/*.csv` | 6 files × 8,760 rows | `TempC, DewPointC, WindspeedKmph, Humidity, TempF`, station name |
| Weather strips `weather/*.yaml` | 48 (6 stations × 8) | which CSV, convection coefficient, 30-day start |
| `training_set.yaml` | 30 pairs | the shipped scenario set |
| `baseline_returns.json` | 30 scenarios × 4 baselines | total reward at 8,640 steps |

**No pre-recorded trajectory dataset ships.** There are no logged observations, actions or rewards; all dynamics data must be generated by running the simulator. The shipped set amounts to 259,200 control decisions (30 × 8,640), about 13.95 million zone-steps (323 zones across the six buildings × 5 climates × 8,640), and roughly 121 million observation values and 34.6 million action values — a lot of data per building, but only six distinct buildings; for generalization the unit that matters is the building, not the step.

### A.2 The analysis files

| Path | Contents |
|---|---|
| `STATE_SPACE.md` | this document |
| `images/` | the three figures of the starter kit, copied here so the figures render |
| `outputs/building_train_N_observation.csv`, `…_action.csv` | every observation and action entry with its index (and initial value) |
| `outputs/building_train_N_observation_annotated.csv`, `…_action_annotated.csv` | the same with role, use in the dynamics model, units and meaning |
| `outputs/building_train_N_rooms.csv`, `…_rooms.png` | per-room area, exterior contact per side, type, neighbours, AHU and floor |
| `outputs/building_train_N_meta.json`, `zone_map_summary.json`, `floors_and_ahus.json` | metadata and geometry summaries |
| `outputs/rollout_train_8.npz`, `rollout_train_8_checks.json` | raw data and results of the 600-step rollout |
| `outputs/metadata_only_check.json`, `building_train_N_alloff_response.csv`, `alloff_response_summary.json` | results of §14.1 and §14.3 |
| `code/` | `dump_vectors.py`, `annotate_vectors.py`, `explore_zone_map.py`, `plot_rooms.py`, `floors_and_ahus.py`, `rollout_checks.py`, `metadata_only_check.py`, `all_off_response.py`, `build_appendix.py` |

The scripts are run from the starter-kit root (the gin configs use relative paths) with the environment in `.venv/` (Python 3.12, the dependencies pinned in the starter kit's `pyproject.toml`), for example `cd smart-buildings-starter-kit && PYTHONDONTWRITEBYTECODE=1 ../working-code/.venv/bin/python ../working-code/code/rollout_checks.py 600`.

---

## Appendix B. Profiling run guide for another machine

*Description only; nothing here has been run.* This is a recipe for pulling the repo onto another computer and producing the data that closes the unknowns of §20.3 (items 1, 2 and 5), using only commands and files already in the starter kit. **No GPU is required**: the simulator is CPU-bound (NumPy, SciPy sparse solves, Numba); a multi-core CPU helps because scenarios are independent and can run in parallel.

1. **Set up (once).** Python ≥ 3.12. Any of: `pip install -e .`; the provided `environment.yml` (conda) or `uv.lock` (uv); or the `Dockerfile`.
2. **Smoke test (about a minute).** `python -m smart_buildings.run_environment --building building_train_8 --weather atlanta_1` confirms the install and reproduces the dimensions of §6.
3. **Profiling runs.** Five scenarios that span the extremes of §11, each under the four baselines for the full 8,640 steps (the `--steps` default; shorter runs are not comparable):

   | Scenario | Why |
   |---|---|
   | `building_train_1 / fairbanks_1` | Worst case for all baselines (−42 °C to +1 °C); the supply-air setpoint is inert most often |
   | `building_train_1 / phoenix_5` | Hottest window (to 46 °C); same building, for a clean climate contrast |
   | `building_train_8 / singapore_7` | Smallest building, near-constant tropical weather; the fixed band tends to win |
   | `building_train_5 / fairbanks_5` | Mild Fairbanks summer; three baselines near −0.017, the apparent floor |
   | `building_train_7 / denver_5` | Leakiest and lightest envelope, hot summer; modulating controllers beat the fixed band |

   One command per scenario: `python -m smart_buildings.evaluate --policies bang_bang fixed_setpoint pid pid_water --buildings building_train_1 --weathers fairbanks_1`. Run the five as parallel processes if cores allow. Rough cost: about 100 steps per second on the smallest building, so minutes for small buildings and longer for large ones. Omit `--buildings` and `--weathers` to cover all 30. Optionally add `all_off` to see what an unconditioned building costs.
4. **Output** lands in `smart_buildings/results/<building>/<weather>/<policy>/` (inside the repo tree, so treat it as scratch and copy it out):

   | File | Contents |
   |---|---|
   | `summary.json` | Total and average reward, average occupancy, average zone temperature, the three reward weights, and the *summed* normalized productivity regret, energy cost and carbon terms |
   | `timeseries.csv` | Per step: timestamp, reward, occupancy, and one column per zone temperature (`zone_<i>_temp`, kelvin) |
   | `reward_components.csv` | Per step: agent reward, productivity reward, electricity cost, gas cost, carbon emitted, normalized productivity regret, normalized energy cost, normalized carbon, total occupancy |
   | `ahu_log.csv` | Per step: outside air temperature, mean recirculation temperature, and per-AHU supply temperature, run command and flow rate |
   | `overview.png`, `rewards.png`, `ahu.png` | Charts of temperatures vs. band, reward and AHU behaviour (the first resembles Figure 3) |

   `python score_leaderboard.py` turns a results tree into the leaderboard number beside the shipped baselines; it needs all 30 scenarios at 8,640 steps.
5. **What to compute (for example with pandas).**
   - *Reward decomposition:* per step `reward = 0.6·npr − 0.2·nec − 0.2·nce` (§3.3). Compare the three contributions per scenario and baseline; raw electricity and gas costs are in `reward_components.csv`.
   - *Comfort:* compare each `zone_<i>_temp` with the active band (comfort 294–297 K on weekdays 06:00–19:00, eco 289–298 K otherwise) and compute the fraction of zone-steps outside by more than 0, 1 and 5 K, split by mode, plus the worst zone.
   - *Ranges:* zone-temperature percentiles and spread, supply temperatures and flows, occupancy profile, and mean reward by hour of day (the morning dips of Figure 3).
   - *Air-handler behaviour:* run fraction and on/off switches from the run-command columns; supply temperature against the mixed-air limit for the inert-setpoint question.
6. **What the CLI does not save.** Actions, the raw observation and the realized damper and reheat positions are not written by `evaluate.py`, so damper/valve saturation, chatter and commanded-setpoint statistics need your own logging driver, built on the same helpers (`load_hybrid_environment`, `create_building_meta_data`, `POLICY_MAP`, and the `SimulationTracker` in `smart_control/utils/scenario_generator.py`). Record the observation vector, the applied continuous and discrete actions, and the realized damper and reheat positions (normalized as `(x−0.5)/0.5`).

What the results feed: the reward split shows whether the headroom is mostly comfort (prediction and pre-conditioning) or energy (setpoints and fans); the air-handler and comfort tables show where baselines fail per regime; and the ranges set scaling and excitation levels for §17.1.

---

## Appendix C. Using the starter kit

```bash
cd smart-buildings-starter-kit
pip install -e .            # Python >= 3.12; deps pinned to Kaggle's image

# Inspect one scenario: spaces + a smoke rollout
python -m smart_buildings.run_environment --building building_train_1 --weather fairbanks_1

# Evaluate your policy (edit smart_buildings/baselines/transfer_policy.py)
python -m smart_buildings.evaluate --policies transfer \
    --buildings building_train_1 --weathers fairbanks_1      # one scenario, 30 days
python -m smart_buildings.evaluate --policies transfer       # all 30

# Compare with baselines, get the leaderboard number
python -m smart_buildings.evaluate --policies pid_water pid fixed_setpoint bang_bang
python score_leaderboard.py          # reads baseline_returns.json by default

# Generate extra training data (writes building YAMLs plus PDF floor-plan previews)
python -m smart_buildings.run_floor_generator --start-seed 500 --count 10
python -m smart_buildings.run_scenario_generator --buildings building_0500 --weathers fairbanks_1 phoenix_6
python -m smart_buildings.run_scenario_generator --plot building_train_1 fairbanks_1   # plot a floor plan and zone map
```

Results land in `smart_buildings/results/<building>/<weather>/<policy>/`. A `Dockerfile`, `environment.yml` and `uv.lock` are provided for reproducible environments. Naming `--buildings` or `--weathers` takes the cross product; no arguments rebuilds the shipped 30 pairs. The repo contains no PDFs of its own — only the three PNGs in `images/` — but the floor generator writes PDF previews.

```
README.md                   official readme
baseline_returns.json       baseline total_reward per scenario (8,640 steps)
check_scales.py             asserts reward denominators = c × K per scenario
score_leaderboard.py        results tree -> leaderboard number + baselines
smart_buildings/
  evaluate.py               env loading, rollout, POLICY_MAP, metadata construction
  run_environment.py        inspect / smoke-test one scenario
  run_floor_generator.py    sample buildings
  run_scenario_generator.py building × weather -> scenarios (config.gin)
  training_set.yaml         the 30 shipped pairs
  baselines/                transfer (your policy), bang_bang, fixed_setpoint, pid, pid_plus, all_off
  buildings/                6 building YAMLs (geometry, AHUs, materials)
  weather/                  per-station CSVs + 48 strip YAMLs
  training_scenarios/       30 generated scenarios (config.gin each)
  smart_control/            simulator, environment, reward, utils, protos
images/                     figures used in the README (and in this report)
```

---

## Appendix D. Glossary

- **AHU** — air handler unit; the central unit that conditions and moves air to several zones.
- **VAV** — variable air volume box; the per-zone damper plus reheat coil.
- **SAT** — supply-air temperature.
- **Reheat** — warming already-cooled supply air at the zone with a hot-water coil.
- **Economizer / recirculation** — mixing outside air into the supply for free cooling.
- **Static pressure** — the duct pressure the fan maintains; it drives airflow and fan power.
- **Comfort / eco mode** — occupied-hours versus setback temperature bands.
- **Thermal mass** — heat stored in slab, walls and furniture; it enables pre-conditioning and coasting.
- **Hybrid action** — a mix of discrete (on/off) and continuous (setpoint) components.
- **Control volume (CV)** — one cell of the finite-difference thermal grid.
- **Policy, POMDP, MPC** — see §15. **RC model, SINDy, symbolic regression** — see §8.2 and §16.3.
- **Held-out set** — evaluation buildings and weather not shipped, drawn from the same generator.

---

*Provenance.* §1–§4 and §11 come from the repo's README, docstrings and files and the Kaggle pages; §8 and parts of §15 come from reading the simulator source (`vav.py`, `air_handler.py`, `boiler.py`, `hot_water_system.py`, `pump.py`, `thermostat.py`, `simulator.py`). §9–§12 and the appendices are computed from the shipped YAML and `config.gin` files, the `.npy` grids, the weather CSVs and `baseline_returns.json`; §5–§7 and §13–§14 come from the environment's own field names, the rollouts and the geometry experiments listed in §20.1. The control-theory framing in §15 and everything from §16 on are our synthesis and proposal; claims about the method are hypotheses, not results. Baseline per-step numbers, derived facade U-values and heat capacities are computed from the shipped files.

## Appendix E. Every entry of the train_8 vectors

Generated by `code/build_appendix.py` from the environment's own field names and `outputs/building_train_8_*_annotated.csv`. `idx` is the position in the vector.

### E.1 Observation vector (283 entries)

| idx | field | role | in dynamics model? | units as delivered | meaning |
|---|---|---|---|---|---|
| 0 | `VAV_1_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 1 | `VAV_1_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 2 | `VAV_1_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 3 | `VAV_1_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 4 | `VAV_1_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 5 | `VAV_1_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 6 | `VAV_10_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 7 | `VAV_10_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 8 | `VAV_10_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 9 | `VAV_10_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 10 | `VAV_10_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 11 | `VAV_10_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 12 | `VAV_11_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 13 | `VAV_11_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 14 | `VAV_11_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 15 | `VAV_11_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 16 | `VAV_11_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 17 | `VAV_11_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 18 | `VAV_12_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 19 | `VAV_12_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 20 | `VAV_12_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 21 | `VAV_12_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 22 | `VAV_12_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 23 | `VAV_12_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 24 | `VAV_13_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 25 | `VAV_13_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 26 | `VAV_13_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 27 | `VAV_13_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 28 | `VAV_13_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 29 | `VAV_13_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 30 | `VAV_14_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 31 | `VAV_14_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 32 | `VAV_14_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 33 | `VAV_14_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 34 | `VAV_14_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 35 | `VAV_14_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 36 | `VAV_15_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 37 | `VAV_15_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 38 | `VAV_15_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 39 | `VAV_15_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 40 | `VAV_15_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 41 | `VAV_15_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 42 | `VAV_16_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 43 | `VAV_16_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 44 | `VAV_16_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 45 | `VAV_16_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 46 | `VAV_16_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 47 | `VAV_16_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 48 | `VAV_17_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 49 | `VAV_17_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 50 | `VAV_17_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 51 | `VAV_17_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 52 | `VAV_17_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 53 | `VAV_17_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 54 | `VAV_18_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 55 | `VAV_18_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 56 | `VAV_18_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 57 | `VAV_18_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 58 | `VAV_18_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 59 | `VAV_18_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 60 | `VAV_19_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 61 | `VAV_19_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 62 | `VAV_19_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 63 | `VAV_19_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 64 | `VAV_19_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 65 | `VAV_19_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 66 | `VAV_2_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 67 | `VAV_2_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 68 | `VAV_2_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 69 | `VAV_2_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 70 | `VAV_2_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 71 | `VAV_2_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 72 | `VAV_20_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 73 | `VAV_20_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 74 | `VAV_20_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 75 | `VAV_20_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 76 | `VAV_20_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 77 | `VAV_20_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 78 | `VAV_21_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 79 | `VAV_21_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 80 | `VAV_21_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 81 | `VAV_21_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 82 | `VAV_21_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 83 | `VAV_21_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 84 | `VAV_22_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 85 | `VAV_22_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 86 | `VAV_22_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 87 | `VAV_22_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 88 | `VAV_22_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 89 | `VAV_22_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 90 | `VAV_23_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 91 | `VAV_23_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 92 | `VAV_23_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 93 | `VAV_23_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 94 | `VAV_23_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 95 | `VAV_23_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 96 | `VAV_24_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 97 | `VAV_24_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 98 | `VAV_24_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 99 | `VAV_24_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 100 | `VAV_24_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 101 | `VAV_24_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 102 | `VAV_25_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 103 | `VAV_25_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 104 | `VAV_25_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 105 | `VAV_25_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 106 | `VAV_25_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 107 | `VAV_25_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 108 | `VAV_26_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 109 | `VAV_26_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 110 | `VAV_26_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 111 | `VAV_26_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 112 | `VAV_26_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 113 | `VAV_26_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 114 | `VAV_27_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 115 | `VAV_27_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 116 | `VAV_27_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 117 | `VAV_27_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 118 | `VAV_27_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 119 | `VAV_27_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 120 | `VAV_28_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 121 | `VAV_28_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 122 | `VAV_28_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 123 | `VAV_28_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 124 | `VAV_28_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 125 | `VAV_28_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 126 | `VAV_29_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 127 | `VAV_29_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 128 | `VAV_29_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 129 | `VAV_29_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 130 | `VAV_29_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 131 | `VAV_29_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 132 | `VAV_3_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 133 | `VAV_3_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 134 | `VAV_3_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 135 | `VAV_3_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 136 | `VAV_3_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 137 | `VAV_3_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 138 | `VAV_30_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 139 | `VAV_30_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 140 | `VAV_30_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 141 | `VAV_30_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 142 | `VAV_30_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 143 | `VAV_30_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 144 | `VAV_4_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 145 | `VAV_4_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 146 | `VAV_4_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 147 | `VAV_4_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 148 | `VAV_4_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 149 | `VAV_4_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 150 | `VAV_5_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 151 | `VAV_5_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 152 | `VAV_5_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 153 | `VAV_5_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 154 | `VAV_5_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 155 | `VAV_5_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 156 | `VAV_6_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 157 | `VAV_6_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 158 | `VAV_6_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 159 | `VAV_6_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 160 | `VAV_6_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 161 | `VAV_6_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 162 | `VAV_7_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 163 | `VAV_7_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 164 | `VAV_7_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 165 | `VAV_7_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 166 | `VAV_7_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 167 | `VAV_7_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 168 | `VAV_8_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 169 | `VAV_8_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 170 | `VAV_8_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 171 | `VAV_8_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 172 | `VAV_8_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 173 | `VAV_8_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 174 | `VAV_9_reheat_valve_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | reheat valve position = the action exactly (rollout-confirmed) |
| 175 | `VAV_9_supply_air_damper_percentage_command` | INPUT (realized) | YES - input | (x-0.5)/0.5 -> [0,1] | damper position the box actually took = max(action, 0.1) (rollout-confirmed) |
| 176 | `VAV_9_supply_air_flowrate_setpoint` | CONSTANT (box parameter) | parameter only | raw m3/s | box max airflow; 2.0 at every step (rollout-confirmed) |
| 177 | `VAV_9_zone_air_cooling_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | upper comfort bound (297 K comfort / 298 K eco); reported one step AFTER the mode flag changes |
| 178 | `VAV_9_zone_air_heating_temperature_setpoint_sensor` | CONTEXT (comfort band) | NO - reward/policy only | raw K | lower comfort bound (294 K comfort / 289 K eco); reported one step AFTER the mode flag changes |
| 179 | `VAV_9_zone_air_temperature_sensor` | OUTPUT / STATE | YES - target | (T-295)/5 -> K | zone-average air temperature |
| 180 | `ahs_ahu_1_cooling_request_count` | CONSTANT (rollout-confirmed) | NO | raw count | = number of zones the AHU serves (10 for train_8) at every step of a 600-step random-action rollout |
| 181 | `ahs_ahu_1_discharge_fan_speed_percentage_command` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw [0,1] | identical to supply_fan_speed_percentage_command at every step |
| 182 | `ahs_ahu_1_exhaust_fan_run_command` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw {0,1} | identical to supervisor_run_command at every step |
| 183 | `ahs_ahu_1_outside_air_flowrate_sensor` | ALGEBRAIC (derived) | optional | raw m3/s | outside-air (ventilation) flow, follows total flow and recirculation |
| 184 | `ahs_ahu_1_outside_air_temperature_sensor` | EXOGENOUS (disturbance) | YES - disturbance | raw K | outside air temperature; same number in every AHU block |
| 185 | `ahs_ahu_1_recirculation_percentage_command` | INPUT (echo) | NO - use the action | raw [0,0.7] | = min(native action, 0.7) (rollout-confirmed) |
| 186 | `ahs_ahu_1_supervisor_run_command` | INPUT (echo) | NO - use the action | raw {0,1} | AHU on/off; equals the discrete action exactly |
| 187 | `ahs_ahu_1_supply_air_flowrate_sensor` | ALGEBRAIC (derived from dampers + pressure) | YES - intermediate signal | raw m3/s | total supply airflow; 0 when the unit is off |
| 188 | `ahs_ahu_1_supply_air_flowrate_setpoint` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw m3/s | identical to supply_air_flowrate_sensor at every step |
| 189 | `ahs_ahu_1_supply_air_static_pressure_sensor` | ALGEBRAIC (= setpoint) | NO - function of the action | raw Pa | equals max(action, 200 Pa) when running and 0 when off, with no lag |
| 190 | `ahs_ahu_1_supply_air_static_pressure_setpoint` | INPUT (echo) | NO - use the action | raw Pa | = max(native action, 200 Pa) when running, 0 when off (rollout-confirmed) |
| 191 | `ahs_ahu_1_supply_air_temperature_sensor` | ALGEBRAIC (= setpoint) | NO - function of the action | raw K | mirrors the setpoint exactly in every step; the physical min(mixed air, setpoint) is NOT what is reported |
| 192 | `ahs_ahu_1_supply_air_temperature_setpoint` | INPUT (echo) | NO - use the action | raw K | equals the native action exactly (rollout-confirmed) |
| 193 | `ahs_ahu_1_supply_fan_run_command` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw {0,1} | identical to supervisor_run_command at every step |
| 194 | `ahs_ahu_1_supply_fan_speed_percentage_command` | ALGEBRAIC (tracks flow) | optional | raw [0,1] | fan speed fraction; correlation 1.000 with supply airflow |
| 195 | `ahs_ahu_2_cooling_request_count` | CONSTANT (rollout-confirmed) | NO | raw count | = number of zones the AHU serves (10 for train_8) at every step of a 600-step random-action rollout |
| 196 | `ahs_ahu_2_discharge_fan_speed_percentage_command` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw [0,1] | identical to supply_fan_speed_percentage_command at every step |
| 197 | `ahs_ahu_2_exhaust_fan_run_command` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw {0,1} | identical to supervisor_run_command at every step |
| 198 | `ahs_ahu_2_outside_air_flowrate_sensor` | ALGEBRAIC (derived) | optional | raw m3/s | outside-air (ventilation) flow, follows total flow and recirculation |
| 199 | `ahs_ahu_2_outside_air_temperature_sensor` | EXOGENOUS (disturbance) | YES - disturbance | raw K | outside air temperature; same number in every AHU block |
| 200 | `ahs_ahu_2_recirculation_percentage_command` | INPUT (echo) | NO - use the action | raw [0,0.7] | = min(native action, 0.7) (rollout-confirmed) |
| 201 | `ahs_ahu_2_supervisor_run_command` | INPUT (echo) | NO - use the action | raw {0,1} | AHU on/off; equals the discrete action exactly |
| 202 | `ahs_ahu_2_supply_air_flowrate_sensor` | ALGEBRAIC (derived from dampers + pressure) | YES - intermediate signal | raw m3/s | total supply airflow; 0 when the unit is off |
| 203 | `ahs_ahu_2_supply_air_flowrate_setpoint` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw m3/s | identical to supply_air_flowrate_sensor at every step |
| 204 | `ahs_ahu_2_supply_air_static_pressure_sensor` | ALGEBRAIC (= setpoint) | NO - function of the action | raw Pa | equals max(action, 200 Pa) when running and 0 when off, with no lag |
| 205 | `ahs_ahu_2_supply_air_static_pressure_setpoint` | INPUT (echo) | NO - use the action | raw Pa | = max(native action, 200 Pa) when running, 0 when off (rollout-confirmed) |
| 206 | `ahs_ahu_2_supply_air_temperature_sensor` | ALGEBRAIC (= setpoint) | NO - function of the action | raw K | mirrors the setpoint exactly in every step; the physical min(mixed air, setpoint) is NOT what is reported |
| 207 | `ahs_ahu_2_supply_air_temperature_setpoint` | INPUT (echo) | NO - use the action | raw K | equals the native action exactly (rollout-confirmed) |
| 208 | `ahs_ahu_2_supply_fan_run_command` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw {0,1} | identical to supervisor_run_command at every step |
| 209 | `ahs_ahu_2_supply_fan_speed_percentage_command` | ALGEBRAIC (tracks flow) | optional | raw [0,1] | fan speed fraction; correlation 1.000 with supply airflow |
| 210 | `ahs_ahu_3_cooling_request_count` | CONSTANT (rollout-confirmed) | NO | raw count | = number of zones the AHU serves (10 for train_8) at every step of a 600-step random-action rollout |
| 211 | `ahs_ahu_3_discharge_fan_speed_percentage_command` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw [0,1] | identical to supply_fan_speed_percentage_command at every step |
| 212 | `ahs_ahu_3_exhaust_fan_run_command` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw {0,1} | identical to supervisor_run_command at every step |
| 213 | `ahs_ahu_3_outside_air_flowrate_sensor` | ALGEBRAIC (derived) | optional | raw m3/s | outside-air (ventilation) flow, follows total flow and recirculation |
| 214 | `ahs_ahu_3_outside_air_temperature_sensor` | EXOGENOUS (disturbance) | YES - disturbance | raw K | outside air temperature; same number in every AHU block |
| 215 | `ahs_ahu_3_recirculation_percentage_command` | INPUT (echo) | NO - use the action | raw [0,0.7] | = min(native action, 0.7) (rollout-confirmed) |
| 216 | `ahs_ahu_3_supervisor_run_command` | INPUT (echo) | NO - use the action | raw {0,1} | AHU on/off; equals the discrete action exactly |
| 217 | `ahs_ahu_3_supply_air_flowrate_sensor` | ALGEBRAIC (derived from dampers + pressure) | YES - intermediate signal | raw m3/s | total supply airflow; 0 when the unit is off |
| 218 | `ahs_ahu_3_supply_air_flowrate_setpoint` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw m3/s | identical to supply_air_flowrate_sensor at every step |
| 219 | `ahs_ahu_3_supply_air_static_pressure_sensor` | ALGEBRAIC (= setpoint) | NO - function of the action | raw Pa | equals max(action, 200 Pa) when running and 0 when off, with no lag |
| 220 | `ahs_ahu_3_supply_air_static_pressure_setpoint` | INPUT (echo) | NO - use the action | raw Pa | = max(native action, 200 Pa) when running, 0 when off (rollout-confirmed) |
| 221 | `ahs_ahu_3_supply_air_temperature_sensor` | ALGEBRAIC (= setpoint) | NO - function of the action | raw K | mirrors the setpoint exactly in every step; the physical min(mixed air, setpoint) is NOT what is reported |
| 222 | `ahs_ahu_3_supply_air_temperature_setpoint` | INPUT (echo) | NO - use the action | raw K | equals the native action exactly (rollout-confirmed) |
| 223 | `ahs_ahu_3_supply_fan_run_command` | DUPLICATE (rollout-confirmed) | NO (redundant) | raw {0,1} | identical to supervisor_run_command at every step |
| 224 | `ahs_ahu_3_supply_fan_speed_percentage_command` | ALGEBRAIC (tracks flow) | optional | raw [0,1] | fan speed fraction; correlation 1.000 with supply airflow |
| 225 | `hws_differential_pressure_sensor` | ALGEBRAIC (= setpoint) | NO - function of the action | raw | = native pump setpoint when running, 0 when off, no lag (rollout-confirmed) |
| 226 | `hws_differential_pressure_setpoint` | INPUT (echo) | NO - use the action | normalized (x-10)/6 | native pump setpoint when running; reads 0 (-1.667 normalized) when off (rollout-confirmed) |
| 227 | `hws_heating_request_count` | CONSTANT (rollout-confirmed) | NO | raw count | = total zone count (30 for train_8) at every step |
| 228 | `hws_run_status` | INPUT (echo) | NO (equals the hot-water run action) | {-1,+1} normalized | identical to hws_supervisor_run_command at every step |
| 229 | `hws_supervisor_run_command` | INPUT (echo) | NO - use the action | {-1,+1} normalized | hot-water on/off: -1 = off, +1 = on |
| 230 | `hws_supply_water_temperature_sensor` | PLANT STATE (true memory) | YES - the one observed plant state | raw K | ramp-limited: +2.5 K/step max up, -0.5 K/step max down; equals the setpoint only 41% of the time |
| 231 | `hws_supply_water_temperature_setpoint` | INPUT (echo) | NO - use the action | raw K | equals the native action exactly (rollout-confirmed) |
| 232 | `hod_cos_000` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 233 | `hod_cos_001` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 234 | `hod_cos_002` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 235 | `hod_cos_003` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 236 | `hod_cos_004` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 237 | `hod_cos_005` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 238 | `hod_cos_006` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 239 | `hod_cos_007` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 240 | `hod_cos_008` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 241 | `hod_cos_009` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 242 | `hod_cos_010` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 243 | `hod_cos_011` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 244 | `hod_sin_000` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 245 | `hod_sin_001` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 246 | `hod_sin_002` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 247 | `hod_sin_003` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 248 | `hod_sin_004` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 249 | `hod_sin_005` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 250 | `hod_sin_006` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 251 | `hod_sin_007` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 252 | `hod_sin_008` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 253 | `hod_sin_009` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 254 | `hod_sin_010` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 255 | `hod_sin_011` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 256 | `dow_cos_000` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 257 | `dow_cos_001` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 258 | `dow_cos_002` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 259 | `dow_cos_003` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 260 | `dow_cos_004` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 261 | `dow_cos_005` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 262 | `dow_cos_006` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 263 | `dow_cos_007` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 264 | `dow_cos_008` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 265 | `dow_cos_009` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 266 | `dow_cos_010` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 267 | `dow_cos_011` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 268 | `dow_sin_000` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 269 | `dow_sin_001` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 270 | `dow_sin_002` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 271 | `dow_sin_003` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 272 | `dow_sin_004` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 273 | `dow_sin_005` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 274 | `dow_sin_006` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 275 | `dow_sin_007` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 276 | `dow_sin_008` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 277 | `dow_sin_009` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 278 | `dow_sin_010` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 279 | `dow_sin_011` | EXOGENOUS (clock) | YES - as 2 phases | [-1,1] | phase-shifted cos/sin of hour-of-day or day-of-week (12 copies, redundant) |
| 280 | `comfort_mode_now` | CONTEXT (schedule) | YES - cheap known disturbance proxy | {0,1} | building in comfort mode now |
| 281 | `comfort_mode_soon` | CONTEXT (schedule lookahead) | NO - policy only | {0,1} | comfort mode 60 min ahead |
| 282 | `num_occupants` | EXOGENOUS (disturbance) | YES - disturbance | raw count (normalization constant is 0) | building-total occupant count, not per zone (0-57 of 60 seen in the rollout) |

### E.2 Action vector (75 entries; discrete part and continuous part are separate arrays, `idx` is within each part)

| part | idx | device | setpoint | role | meaning |
|---|---|---|---|---|---|
| discrete | 0 | ahs | `ahu_1_supervisor_run_command` | DISCRETE INPUT | on/off for AHU |
| continuous | 0 | ahs | `ahu_1_supply_air_temperature_setpoint` | CONTINUOUS INPUT | supply-air temperature target 285-305 K |
| continuous | 1 | ahs | `ahu_1_supply_air_static_pressure_setpoint` | CONTINUOUS INPUT | fan static-pressure target 0-2000 Pa (floor 200 Pa) |
| continuous | 2 | ahs | `ahu_1_recirculation_percentage_command` | CONTINUOUS INPUT | return-air share 0-1 (cap 0.7) |
| discrete | 1 | ahs | `ahu_2_supervisor_run_command` | DISCRETE INPUT | on/off for AHU |
| continuous | 3 | ahs | `ahu_2_supply_air_temperature_setpoint` | CONTINUOUS INPUT | supply-air temperature target 285-305 K |
| continuous | 4 | ahs | `ahu_2_supply_air_static_pressure_setpoint` | CONTINUOUS INPUT | fan static-pressure target 0-2000 Pa (floor 200 Pa) |
| continuous | 5 | ahs | `ahu_2_recirculation_percentage_command` | CONTINUOUS INPUT | return-air share 0-1 (cap 0.7) |
| discrete | 2 | ahs | `ahu_3_supervisor_run_command` | DISCRETE INPUT | on/off for AHU |
| continuous | 6 | ahs | `ahu_3_supply_air_temperature_setpoint` | CONTINUOUS INPUT | supply-air temperature target 285-305 K |
| continuous | 7 | ahs | `ahu_3_supply_air_static_pressure_setpoint` | CONTINUOUS INPUT | fan static-pressure target 0-2000 Pa (floor 200 Pa) |
| continuous | 8 | ahs | `ahu_3_recirculation_percentage_command` | CONTINUOUS INPUT | return-air share 0-1 (cap 0.7) |
| discrete | 3 | hws | `supervisor_run_command` | DISCRETE INPUT | on/off for hot-water loop |
| continuous | 9 | hws | `supply_water_temperature_setpoint` | CONTINUOUS INPUT | loop water temperature target 310-350 K |
| continuous | 10 | hws | `differential_pressure_setpoint` | CONTINUOUS INPUT | pump head target 0-20 |
| continuous | 11 | VAV_1 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 12 | VAV_1 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 13 | VAV_10 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 14 | VAV_10 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 15 | VAV_11 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 16 | VAV_11 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 17 | VAV_12 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 18 | VAV_12 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 19 | VAV_13 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 20 | VAV_13 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 21 | VAV_14 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 22 | VAV_14 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 23 | VAV_15 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 24 | VAV_15 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 25 | VAV_16 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 26 | VAV_16 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 27 | VAV_17 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 28 | VAV_17 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 29 | VAV_18 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 30 | VAV_18 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 31 | VAV_19 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 32 | VAV_19 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 33 | VAV_2 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 34 | VAV_2 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 35 | VAV_20 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 36 | VAV_20 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 37 | VAV_21 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 38 | VAV_21 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 39 | VAV_22 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 40 | VAV_22 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 41 | VAV_23 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 42 | VAV_23 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 43 | VAV_24 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 44 | VAV_24 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 45 | VAV_25 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 46 | VAV_25 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 47 | VAV_26 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 48 | VAV_26 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 49 | VAV_27 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 50 | VAV_27 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 51 | VAV_28 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 52 | VAV_28 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 53 | VAV_29 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 54 | VAV_29 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 55 | VAV_3 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 56 | VAV_3 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 57 | VAV_30 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 58 | VAV_30 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 59 | VAV_4 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 60 | VAV_4 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 61 | VAV_5 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 62 | VAV_5 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 63 | VAV_6 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 64 | VAV_6 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 65 | VAV_7 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 66 | VAV_7 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 67 | VAV_8 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 68 | VAV_8 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
| continuous | 69 | VAV_9 | `supply_air_damper_percentage_command` | CONTINUOUS INPUT | VAV damper 0-1 (floor 0.1) |
| continuous | 70 | VAV_9 | `reheat_valve_percentage_command` | CONTINUOUS INPUT | VAV reheat valve 0-1 |
