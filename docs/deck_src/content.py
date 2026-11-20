"""The storyline.  Every slide carries `src`: where each claim on it comes from.

Source tags:  [website] smart-buildings-2026.github.io/competition   [Kaggle] the competition's Kaggle pages
  [kit] starter-kit code or files   [audit] our audit of the simulator code (STATE_SPACE.md, outputs/*annotated.csv)
  [sim data] simulator runs (kept audit files in outputs/, or kit evaluate.py runs into a scratch folder)
  [measured] numbers we measured for this deck   [our code] working-code/code   [synthetic] the planted-truth self-test
  [illustrative] a drawing, not data   [decision] agreed in planning   [plan] not built yet
"""
S, SECTIONS = [], []


def add(kind, id, **kw):
    S.append(dict(kind=kind, id=id, **kw))


def section(id, num, title, question, notes, desc):
    SECTIONS.append((id, desc))
    add("section", id, num=num, title=title, question=question, notes=notes)


# =========================================================================== opening
add("statement", "cover", eyebrow="NeurIPS 2026 · Smart Buildings Challenge", title="One controller for every building",
    sub="Predicting every room’s temperature in any building from its floor plan and its own measured history, so a controller can plan ahead",
    foot="Working session deck · every slide names its sources", notes="From zero: the task, why it is hard, the usual approaches, our approach, how it works, the evidence, what is next.")
add("points", "questions", eyebrow="The story in seven questions", title="Each part answers one question",
    points=["1 · What is the problem, and what exactly must we submit?", "2 · Why is it hard? What varies between buildings?",
            "3 · What do people usually do, and why not here?", "4 · What do we do instead, and why should it work?",
            "5 · How does it work: the physics, the variables, the search", "6 · How do we measure deviations, and what do the data say so far?",
            "7 · What is built, what is not, and what comes next?"],
    src="Every slide ends with a source line: [website] [Kaggle] [kit] [audit] [sim data] [measured] [our code] [synthetic] [illustrative] [decision] [plan]",
    notes="The source line says whether a claim comes from the official pages, the starter kit, real simulator data, our measurements, our code, or is a drawing or a plan.")

# =========================================================================== 1 problem
section("s1", "1", "The problem", "What are we asked to do, what exactly must we submit, and how is it scored?",
        "Motivation, the plant, the official task, the submission rules, the score.", "1 · The problem")
add("stats", "why", eyebrow="Why it matters", title="Buildings use a large share of the world’s energy, and HVAC is the biggest part",
    stats=[("~40%", "of U.S. carbon emissions come from buildings"), ("40–60%", "of a building’s emissions come from heating, ventilation and air conditioning"),
           ("~15%", "of global energy consumption goes to HVAC")],
    callout="Most building controllers only work for the building they were developed for. The competition asks for one policy that transfers.",
    src="[website] competition home page; [Kaggle] overview", notes="Winning solutions may get to test on three real buildings (Kaggle overview).")
add("image", "plant", eyebrow="The plant", title="What is being controlled: air handlers, VAV boxes, a hot-water system",
    img="kit_device_locations.png", side="right",
    points=["Air handlers (AHUs) cool and mix air and send it to at most 10 zones each", "Each zone has a VAV box: a damper (its share of air) and a reheat valve",
            "One hot-water system: a boiler and a pump feed the reheat coils", "Meters record the energy and carbon the reward charges for"],
    caption="Image: starter kit images/device_locations.png", src="[website] competition.html, Building components; [kit] image",
    notes="Training buildings have 3 to 9 AHUs and 30 to 85 rooms (kit configs; the website says 28 to 85). Generated buildings can be much larger (part 2).")
add("table", "evaluation", eyebrow="The task", title="One policy, run unchanged on buildings and weather it has never seen",
    rows=[["", ""], ["Development set", "6 public buildings × 5 weather windows = 30 scenarios (shipped with the kit)"],
          ["Final evaluation", "6 unseen buildings × 5 unseen weather stations = 30 scenarios, same generator and ranges"],
          ["One scenario", "8,640 steps of 5 minutes = 30 days; every step is scored, including the first"],
          ["Each step", "the policy gets the observation vector; returns discrete on/off commands and continuous setpoints"],
          ["Determinism", "same policy + same scenario = same return, bit for bit; each scenario is run once"]],
    colw=[0.24, 0.76], header=False,
    callout="The public leaderboard uses about 33% of the test data; the final ranking uses the other 67%.",
    src="[website] scoring.html, competition.html; [Kaggle] overview, data, leaderboard", notes="Test buildings and weather are never released; the metadata arrives only at run time inside the sandbox.")
add("table", "submission", eyebrow="The submission", title="What a Kaggle submission must be",
    rows=[["Requirement", "Detail"],
          ["Form", "one Kaggle notebook; nothing else is submitted"],
          ["Class", "subclass smart_buildings_inference_server.Policy; __init__(building_meta_data, action_spec); get_action(obs)"],
          ["Return", "{continuous_action: float32 [3K+2+2V] in [−1, 1], discrete_action: int32 [K+1] in {0, 1}}; finite, in bounds"],
          ["Metadata", "action and observation indices, field names, floor plan, zone map, zone→AHU, floors, schedule, normalisation"],
          ["Runtime", "≤ 9 h per notebook; no internet; 15 min for the first step of a scenario, then 60 s per step"],
          ["Weights", "loaded in __init__ from a Kaggle Dataset attached to the notebook"],
          ["Limits", "5 submissions a day; 2 final submissions; deadline Jan 15, 2027 (entry and team merger Jan 3)"],
          ["Forbidden", "hard-coding positions instead of reading the metadata; referencing held-out buildings or weather"]],
    colw=[0.16, 0.84], callout=None,
    src="[website] submission.html; [Kaggle] overview (code requirements, timeline), rules (submission limits)",
    notes="A fresh policy is built for every scenario: state carries between steps but never between scenarios. A NaN passes the bounds check and corrupts the rest of the episode. Winners license their code under CC-BY 4.0.")
add("table", "budget", eyebrow="The clock", title="The time budget is tight: about 0.125 s per step for everything",
    rows=[["Quantity", "Value", "Source"],
          ["Notebook limit", "9 h, no internet", "[Kaggle] code requirements"],
          ["Steps to run", "30 scenarios × 8,640 = 259,200", "[website] scoring"],
          ["Per step, if scenarios run one after another", "≈ 0.125 s, simulator included", "arithmetic"],
          ["First step of each scenario", "up to 15 min (load models, set up)", "[website] submission"],
          ["Any later step", "at most 60 s", "[website] submission"]],
    colw=[0.42, 0.33, 0.25],
    callout="The simulator itself uses part of that budget, so whatever the policy computes each step must be cheap. This is one reason the model is a small closed-form formula.",
    src="[Kaggle] overview: code requirements; [website] submission.html, scoring.html", notes="Whether the gateway runs scenarios in parallel is not documented.")
add("image", "scoring", eyebrow="The score", title="Mean per-step loss over 30 scenarios: comfort 0.6, energy 0.2, carbon 0.2",
    img="kit_overview.png", side="right",
    points=["r = 0.6 · comfort regret − 0.2 · energy cost − 0.2 · carbon; never positive", "Comfort: 0 inside the band; linear up to 5 K outside, then quadratic, no bound",
            "Comfort is weighted by each zone’s occupants, not its area", "Score = mean over scenarios of −(return / 8,640); lower is better; 0 is unreachable"],
    caption="Image: starter kit images/overview.png: room temperatures, reward and occupancy over two days",
    src="[website] scoring.html, competition.html Reward; [Kaggle] evaluation; [kit] image",
    notes="The comfort band is 294–297 K in comfort hours and 289–298 K in eco hours (kit config.gin). Comfort is scored only against the active band.")
add("table", "leaderboard", eyebrow="Where the bar is", title="Baselines on the 30 public scenarios, and the current leaderboard",
    rows=[["Policy", "Score (lower is better)", "Set"],
          ["bang_bang", "0.225", "30 public scenarios (kit)"], ["fixed_setpoint", "0.066", "30 public scenarios (kit)"],
          ["pid", "0.067", "30 public scenarios (kit)"], ["pid_water (best baseline)", "0.054", "30 public scenarios (kit)"],
          ["best baseline per scenario (oracle)", "0.047", "30 public scenarios (kit)"],
          ["Leaderboard #1 (Nov 19, 2026)", "0.0045", "hidden test, public 33% (Kaggle)"], ["Leaderboard #50 (Nov 19, 2026)", "≈ 0.0059", "hidden test, public 33% (Kaggle)"]],
    colw=[0.4, 0.25, 0.35],
    callout="The sets differ, so the numbers are not strictly comparable, but leading teams are far below the baselines: the reward rewards much better control than PID.",
    src="[kit] baseline_returns.json (score = −return / 8,640, averaged); [Kaggle] leaderboard page, read Nov 19, 2026",
    notes="845 teams, 9,646 submissions at the time of reading.")
add("statement", "objective", eyebrow="The objective, in one sentence",
    title="One policy that, dropped into any unseen building, keeps rooms comfortable at the lowest energy and carbon",
    sub="using only that building’s metadata and what it measures while it runs, within about 0.125 s per step",
    src="[website] [Kaggle] task statement; [measured] time budget", notes="Everything that follows serves this sentence.")

# =========================================================================== 2 hard
section("s2", "2", "Why it is hard", "What varies between buildings, and what do rooms actually do?",
        "The generator, the weather, then real temperature profiles.", "2 · Why it is hard")
add("table", "generator", eyebrow="What the generator varies", title="Every hidden building is a fresh draw from the same generator",
    rows=[["Varied", "Range", "Per"],
          ["Floors; footprint", "2–4; 400–700 × 300–500 ft, composite shapes", "building (seed)"],
          ["Room size", "30–60 ft (+11–16 ft span)", "building (seed)"],
          ["Air handlers", "3–7, raised to ⌈zones / 10⌉", "building (seed)"],
          ["Occupants per zone", "1–2", "building (seed)"],
          ["Facade conductivity, density, heat capacity", "0.15–0.55 W/mK; 350–2,400 kg/m³; 840–1,000 J/kgK", "building (batch draw)"],
          ["Slab/furniture density", "10–40 kg/m³", "building (batch draw)"],
          ["Partition conductivity, density", "0.45–0.60 W/mK; 110–700 kg/m³", "building (batch draw)"],
          ["Weather: station, 30-day window, outdoor film", "6 stations × 8 windows; film 8.8–26.7 W/m²K per station", "scenario"]],
    colw=[0.38, 0.44, 0.18],
    callout="Any integer is a seed: buildings are unlimited (the kit rejects seeds that make a room under 20 m²). Weather is limited to 48 windows. Geometry depends only on the seed; the six materials are drawn over the whole batch.",
    src="[kit] run_floor_generator.py DEFAULT_RANGES, smart_control/utils/materials.py PROPERTIES, weather/*.yaml", notes="16 building parameters + 3 weather parameters vary; everything else is constant across buildings (audit, STATE_SPACE.md §12).")
add("image", "layouts", eyebrow="Every building is different", title="Different layouts, room counts and air-handler groupings",
    img="kit_floorplan_variability.png", side="bottom",
    points=["Three of the six: 30, 39 and 85 rooms; 3, 4 and 9 AHUs (the figure labels train_8 “28 zones”; its config has 30 rooms)", "No small list of room types: rooms differ in continuous numbers (area, outside wall, shared walls, doors)"],
    caption="Image: starter kit images/floorplan_variability.png", src="[website] Figure 3; [kit] image", notes="The hidden buildings come from the same generator.")
add("image", "climates", eyebrow="Weather over one episode", title="Six 30-day windows: from below −40 °C in Fairbanks to tropical Singapore",
    img="ep30_weather.png", side="bottom",
    points=["The kit has 6 stations × 8 windows of 30 days (hourly, 2023); the hidden test uses five other stations", "Each scenario is one building in one 30-day window: 8,640 steps"],
    caption="[sim data] outside air in the six 30-day episodes used on the next slides", src="[kit] weather/*.yaml, weather/csv; [sim data] kit evaluate.py, 8,640 steps (scratch folder)",
    notes="Weather is the scarce resource: buildings are unlimited, weather is 48 windows.")
add("image", "profiles", eyebrow="What a 30-day episode looks like", title="All six shipped buildings, every room, 30 days, the best baseline",
    img="ep30_profiles.png", side="right",
    points=["Orange: rooms with an outside wall; blue: interior rooms; green: the active comfort band", "Weekdays: the baseline pulls rooms into the band; nights and weekends: rooms drift",
            "The drift depends on the room and the climate: small in Phoenix, large in Fairbanks", "This is what the model must predict, room by room, for a whole month"],
    caption="[sim data] pid_water, 8,640 steps, one weather window per building",
    src="[sim data] kit evaluate.py runs (scratch folder); replayed with our starterkit_data.load_starterkit_episode", notes="pid_water scores ≈ 0.21 in fairbanks_1 and 0.020 in atlanta_3: the climate dominates the score.")
add("image", "week", eyebrow="Zooming into one week", title="One week of the same episode: day cycles, weekends, room by room",
    img="ep30_week.png", side="bottom",
    points=["Mornings: rooms are pulled up into the band; evenings: they coast down at different speeds", "Exterior rooms swing more than interior rooms"],
    caption="[sim data] building_train_2, Atlanta window 3, pid_water, days 7–14", src="[sim data] kit evaluate.py (scratch folder)", notes="")
add("image", "geometry", eyebrow="But the difference is predictable", title="How fast a room cools follows from the floor plan",
    img="ep30_geometry.png", side="right",
    points=["HVAC off: more outside wall per m² of floor → faster loss", "That number comes straight from the floor plan",
            "The spread between buildings comes from materials, which differ and are not given", "So: geometry is visible; materials must be read from behaviour"],
    caption="[sim data] HVAC off, first 12 h of the episode, six shipped buildings", src="[sim data] kit evaluate.py all_off (scratch folder); geometry from our episodes.geometry()",
    notes="This picture motivates the approach.")
add("image", "control", eyebrow="Control matters", title="The same building, HVAC off vs the best baseline, over 30 days",
    img="ep30_policies.png", side="bottom",
    points=["atlanta_3: off, rooms run warm and wide; pid_water holds them near the band", "fairbanks_1: off, rooms fall far below the band; pid_water heats, at a large energy cost (score ≈ 0.21)"],
    caption="[sim data] mean of all rooms, band = 10–90 %; 8,640 steps", src="[sim data] kit evaluate.py (scratch folder)",
    notes="Doing better than PID means anticipating, which needs a model that predicts hours ahead.")
section("s3", "3", "The usual approaches", "How is this normally done, and why does it fall short here?",
        "Rules, reinforcement learning, black-box models, per-building fitted physics.", "3 · The usual approaches")
add("image", "rules", eyebrow="Option 1 · rules", title="Feedback rules react to the error now; they cannot plan",
    img="baselines.png", side="right",
    points=["Fixed setpoints, bang-bang, PID: the kit’s baselines", "No forecast: they heat after the room is already cold",
            "pid_water is best on average, yet on building_train_1 in fairbanks_1 it scores worse than a fixed setpoint (0.212 vs 0.147)", "Still useful: a safe first submission"],
    caption="[kit] baseline_returns.json, 30 public scenarios", src="[kit] baseline_returns.json", notes="Rules do not adapt to the building.")
add("cards", "smarter", eyebrow="Options 2–4", title="Three smarter approaches, and where each one breaks here",
    cards=[("Reinforcement learning", "Learns a policy by trial and error.\n\nNeeds many episodes; every test building is new, observation and action sizes change per building, and nothing trains during evaluation.", "breaks: unseen buildings", "accent1"),
           ("Black-box learned model", "A neural network of the building, used to plan.\n\nNeeds data from that building; on day 0 there is none, and layouts change.", "breaks: no data on day 0", "accent2"),
           ("Physics fitted per building", "An RC model whose constants are fitted to the building’s data.\n\nNeeds an identification period on the new building and compute inside the 0.125 s budget.", "breaks: fitting at deployment", "accent3")],
    src="[website] variable-dimensional hybrid action space; [measured] time budget; reasoning, not experiments", notes="These are arguments, not measured comparisons; leaderboard teams may use any of them.")
add("table", "comparison", eyebrow="Side by side", title="What each approach needs, and what it gives",
    rows=[["Approach", "Data from the new building?", "Transfers to new layouts?", "Plans ahead?", "Readable?"],
          ["Rules (PID)", "no", "yes, but untuned", "no", "yes"], ["Reinforcement learning", "much", "poorly", "implicitly", "no"],
          ["Black-box model + MPC", "much", "poorly", "yes", "no"], ["Physics fitted per building + MPC", "a day or more", "yes", "yes", "yes"],
          ["Ours: universal formulas + MPC", "only what it measures as it runs", "by construction", "yes", "yes"]],
    colw=[0.3, 0.2, 0.18, 0.14, 0.18], src="[decision] our assessment (not measured)", notes="The last row is the target.")
add("image", "tried", eyebrow="What we tried first", title="Fitting constants works, but costs a day on every new building",
    img="horizon.png", side="right",
    points=["Online: re-fit materials on each building’s first day", "Offline: fit one set of constants on training buildings",
            "Both run in code and stay available for comparison", "With materials fitted on day 1 the hand-built network forecasts well: the structure is right"],
    caption="[sim data] earlier audit, 3-day episodes: persistence vs hand-built network, materials fitted on day 1, filtered state",
    src="[our code] discovery/loop.py (--deployment offline|online); [sim data] outputs/room_model_forecast.json", notes="The question became: can formulas read the materials from history, so nothing is fitted?")

# =========================================================================== 4 ours
section("s4", "4", "Our approach", "What do we do instead, and why should it work?",
        "Write the physics once; geometry from the floor plan; materials from the measured history; fit nothing.", "4 · Our approach")
add("compare", "observation", eyebrow="The key observation", title="What changes between buildings, and what does not",
    left=("Changes", ["Geometry: visible in the floor plan", "Six materials: hidden, but they show in how rooms warm and cool", "Weather and people: measured as they happen"]),
    right=("Does not change", ["The simulator’s physics and equipment laws", "Air and water properties, the 0.2 m grid, the 3 m height, 5-minute step",
                               "Start at 294 K; 70 W per occupant; 10 W/m² internal gains"]),
    callout="So write the physics once, fill in the geometry from the plan, and read the materials from the measured history.",
    src="[kit] config.gin (initial_temp 294, schedule), materials.py; [audit] STATE_SPACE.md §12.2 constants (70 W, 10 W/m²)", notes="Universal values in the simulator are generator choices; on real buildings we first try the same constants.")
add("statement", "goal", eyebrow="Our objective", title="One set of closed-form formulas that predicts every room’s temperature in any building",
    sub="from the floor plan, the recent measured history and universal constants. Nothing is fitted: not on the new building, not beforehand.",
    src="[decision] planning notes", notes="The only estimation allowed is a closed-form expression of the history buffer inside the formulas.")
add("image", "whympc", eyebrow="Why a model", title="A model lets the controller plan ahead: model predictive control",
    img="mpc_idea.png", side="right",
    points=["Predict each room’s temperature for candidate command plans", "Choose the plan with the best comfort, energy and carbon",
            "Apply the first step; plan again 5 minutes later", "Comfort comes from temperatures, so the model must get those right; energy enters as a control cost"],
    caption="[illustrative] two heating plans before the comfort band starts", src="[illustrative] drawing; [plan] the controller is not built",
    notes="Known gaps of an energy proxy: fan power ∝ damper × pressure^1.5; cooling and gas depend on temperatures; the cost reference scales with the number of AHUs; comfort is weighted by unobserved occupants per room.")
add("cards", "whyllm", eyebrow="Why an LLM", title="An LLM writes the formulas; data decides which are right",
    cards=[("The space is huge", "Which heat stores, which couplings, how coefficients follow from geometry, how to read materials from history.", "", "accent1"),
           ("LLMs write physics code", "It proposes readable programs, each a hypothesis, and reacts to diagnostics.", "", "accent2"),
           ("The scorer decides", "Every candidate is checked and scored on unseen buildings. The LLM never supplies a verdict or a fitted number.", "", "accent3")],
    callout="The result is readable physics that can be checked and tried on real buildings.",
    src="[our code] discovery/evolve.py, closedform.py; [plan] no real LLM generation has run yet", notes="The machinery is tested with a mechanical stand-in for the LLM.")
add("compare", "nofit", eyebrow="What no fitting buys, and costs", title="Why we insist on no fitting at all",
    left=("Buys", ["Works from step 1: no identification period", "A bad score means the formula is wrong, not the optimizer", "Cheap: 0.23 ms per step on 85 rooms"]),
    right=("Costs", ["Some error in the first hours while the history buffer fills (accepted)", "Materials must be inferable from history"]),
    src="[decision] planning notes; [measured] step time", notes="Decisions: no fitting, universal constants allowed, early error accepted, discrete-time maps allowed.")

# =========================================================================== 5 how
section("s5", "5", "How the LLM discovery works", "What problem does the LLM solve, what does it read and write, and what comes out?",
        "The problem, one generation step by step, the physics and variables it is given, real excerpts of a prompt, an analysis and a candidate, the output, and how it runs on the big machine.", "5 · LLM discovery, end to end")
add("eq", "discproblem", eyebrow="The problem the LLM solves", title="Find one program that predicts the next temperature of every room in every building",
    eq="T_room(t + 5 min)  =  step( geometry, measured history, this step’s inputs, universal constants )",
    points=["The same step() must work for a 30-room and an 85-room building and larger ones, in Fairbanks and in Phoenix", "No number may be fitted: per-building materials must be read from the history inside the formula",
            "Too many possible structures to write by hand; and fitting is ruled out, so we search over programs", "The LLM proposes programs; an automatic scorer on unseen buildings decides; an evolutionary loop keeps the best"],
    src="[decision] planning notes; [our code] contract.load_closedform, prompts.CF_CONTRACT", notes="This is symbolic discovery: the output is code a person can read, not a black box.")
add("flow", "genloop", eyebrow="One generation, step by step", title="What happens on each island in each generation",
    steps=[("1 Pick parents", "tournament on the island"), ("2 Analyse", "where each parent is wrong"), ("3 Ask the LLM", "mutate · cross · explore"),
           ("4 Check, screen", "physics + one-step"), ("5 Roll out", "1/3/6 h on unseen buildings"), ("6 Keep", "record; migrate best")],
    points=["Each operator is one LLM request; every request, analysis, answer and candidate is saved to disk", "The score is the 3 h rollout error on held-out buildings; nothing is fitted at any stage",
            "Every few generations each island sends its best program to the next island"],
    src="[our code] discovery/evolve.py run(), closedform.evaluate", notes="Seeds start every island in generation 0.")
add("image", "roomsees", eyebrow="The physics · one room", title="What one room exchanges heat with",
    img="room_sees.png", side="right",
    points=["Its own air: the fast part", "A slab and furniture node: the slow part", "Outside walls, then the outside air through a film",
            "Shared walls and doors, then neighbours", "Supply air, reheat, people, lights and equipment"],
    caption="[illustrative] schematic of the audited structure", src="[audit] STATE_SPACE.md §14 (structure read from the simulator code)", notes="How many stores need their own state is a question for the data.")
add("image", "fastslow", eyebrow="The physics · heat storage", title="A room cools fast, then slowly",
    img="fast_slow.png", side="right",
    points=["Fast: the air follows the walls within minutes", "Slow: heavy materials release heat over hours", "One temperature per room is not enough",
            "Instead of hidden states, formulas may use averages of the measured history"],
    caption="[sim data] building_train_1, Fairbanks, heating stops Monday 19:00", src="[sim data] outputs/ (kept audit files)", notes="Averaging a linear system gives a linear system.")
add("image", "nonlinear", eyebrow="The physics · nonlinearity", title="Linear in temperatures, nonlinear in commands",
    img="airflow_law.png", side="right",
    points=["Heat flow between rooms, to outside and into the slab is linear in temperatures", "Commands become heat through products, limits and on/off switches",
            "That conversion is hand-written (from the audit) and done before the formulas"],
    caption="[audit] the simulator’s airflow law", src="[audit] STATE_SPACE.md; [our code] starterkit_data.episode_inputs", notes="Discovering the conversion is possible later; today it is not discovered.")
add("image", "obsgroups", eyebrow="Every observed variable", title="All 6V + 15K + 58 observation fields, by group, over three days",
    img="obs_groups.png", side="right",
    points=["Per zone 6, per AHU 15, building 58 (boiler 7, clock 48, comfort 2, occupants 1)", "Outside air arrives inside every AHU block (same value K times)"],
    caption="[sim data] building_train_2, Atlanta, pid_water, as delivered to the policy (normalised)", src="[sim data] outputs/roomdata_building_train_2_atlanta_3_pidwater.npz",
    notes="For 85 rooms and 9 AHUs: 703 numbers per step.")
add("table", "obsroles", eyebrow="What each observed field is", title="Most fields echo a command or follow from one",
    rows=[["Field type (count)", "What it is", "Model needs it?"],
          ["zone air temperature (V)", "the target: zone-average air temperature", "yes: target"],
          ["damper, reheat command (2V)", "the commanded position (damper ≥ 0.1)", "input (= our action)"],
          ["heating / cooling setpoint (2V)", "the comfort band, 294–297 K day, 289–298 K night", "no: reward only"],
          ["box airflow setpoint (V)", "constant 2.0 m³/s", "no"],
          ["AHU supply temp, pressure, recirculation, run (≈ 8 per AHU)", "echo or equal the commands", "no: use the action"],
          ["AHU supply airflow, fan speed, outside-air flow", "follow from dampers and pressure", "intermediate"],
          ["AHU duplicates and request counts (≈ 5)", "identical copies; constants", "no"],
          ["outside air temperature (K copies)", "weather, now", "yes: disturbance"],
          ["boiler supply-water temperature", "ramp-limited: the one observed plant state", "yes: input"],
          ["occupants, comfort now / soon, clock (12 phases ×4)", "building total; schedule; time", "yes (occupants, comfort, 4 clock)"]],
    colw=[0.36, 0.42, 0.22], src="[audit] outputs/building_train_8_observation_annotated.csv (each role confirmed in rollouts)",
    notes="So the dynamics need V room temperatures + 8 numbers: outside air, boiler water, occupants, comfort flag, hour sin/cos, day sin/cos.")
add("table", "actions", eyebrow="Every command", title="What the policy sends: K + 1 switches and 3K + 2 + 2V setpoints",
    rows=[["Command", "Per", "Range (native)", "Simulator applies"],
          ["run on/off", "AHU, hot-water system", "{0, 1}", "first AHU start takes effect one step late"],
          ["supply-air temperature", "AHU", "285–305 K", "the coil cools to it; cannot heat above mixed air"],
          ["static pressure", "AHU", "0–2,000 Pa", "floor 200 Pa while running"],
          ["recirculation", "AHU", "0–1", "capped at 0.7"],
          ["supply-water temperature", "hot-water system", "310–350 K", "boiler ramps +2.5 / −0.5 K per step"],
          ["pump differential pressure", "hot-water system", "0–20", "0 when off"],
          ["damper", "zone", "0–1", "at least 0.1 open"], ["reheat valve", "zone", "0–1", "flow 0.108 √Δp; discharge ≤ 308.15 K"]],
    colw=[0.24, 0.2, 0.18, 0.38], callout="All continuous commands arrive normalised to [−1, 1]; 0 is the middle of the range, not off.",
    src="[audit] outputs/building_train_8_action_annotated.csv, STATE_SPACE.md §6; [website] ventilation floors; [Kaggle] data tab", notes="Indices must be read from the metadata.")
add("image", "geomscript", eyebrow="The model · geometry", title="A script reads every room’s numbers from the floor plan",
    img="rooms_classes.png", side="right",
    points=["Input: floor_plan and zone_map from the metadata, nothing else", "Per room: area, outside-wall length, bounding box, floor, AHU",
            "Per pair: shared wall length, door width, number of doors", "Corner, edge, interior fall out of these numbers"],
    caption="[audit] building_train_2 room classes (outputs/building_train_2_rooms.png)", src="[our code] discovery/episodes.py geometry()", notes="The same function runs inside the policy.")
add("cards", "coefficients", eyebrow="The model · coefficients", title="Every coefficient is geometry × material",
    cards=[("Between two cells", "conductance = conductivity × contact area / distance", "the law", "accent2"),
           ("Layers in series", "room → one-cell wall → room;  room → facade → outside film: resistances add", "walls", "accent1"),
           ("Geometry", "areas, wall lengths, door widths: counted from the floor plan", "visible", "accent3"),
           ("Material", "universal constants, or read from history by a formula", "hidden", "accent1")],
    src="[audit] STATE_SPACE.md §14 (finite-volume solver)", notes="Facade U follows 1/U = 0.1/k + 0.299 in the kit (materials.py).")
add("table", "constants", eyebrow="The model · constants", title="Universal constants, and the values that differ per building",
    rows=[["Universal (may be written into formulas)", "Differs (must be read from history)"],
          ["Air density and specific heat; water density and specific heat", "Facade conductivity, density, heat capacity (building)"],
          ["Grid cell 0.2 m, height 3 m, step 300 s", "Slab/furniture density (building)"],
          ["Start temperature 294 K", "Partition conductivity and density (building)"],
          ["70 W per occupant; 10 W/m² gains (0.2 off-hours)", "Occupants per zone, 1–2 (building); outdoor film (weather station)"]],
    colw=[0.5, 0.5], callout="The universal list is reviewed once before it goes into the prompt (code/discovery/universal_constants.json).",
    src="[kit] materials.py, weather/*.yaml, config.gin; [audit] STATE_SPACE.md §12.2", notes="Partition heat capacity is fixed (900 J/kgK), not drawn.")
add("cards", "templates", eyebrow="The model · templates", title="A room’s equation is the sum of its filled templates",
    cards=[("Building physics", "air, slow store, outside wall, shared walls, doors", "passive", "accent2"), ("HVAC", "supply air, reheat", "controls", "accent1"),
           ("Disturbances", "people, lights and equipment, weather", "outside the controller", "accent3"), ("Bookkeeping", "history summaries; per-building values read from history", "glue", "accent3")],
    callout="Every room uses the same ingredients in different amounts: a corner room gets an outside-wall term, an interior room does not.",
    src="[decision] planning notes; [our code] prompts.CF_SYSTEM", notes="One physical question per template, so crossover can recombine them.")
add("eq", "assemble", eyebrow="The model · one step", title="The same end-of-step form assembles every room",
    eq="T_next = ( T  +  Σ_k w_k · T_k  +  b ) / ( 1 + Σ_k w_k )",
    points=["w_k = 300 s × conductance / capacity for each source: supply air, neighbours, outside, slow store", "b = 300 s × direct heat / capacity",
            "A weighted average plus heat: stable for any airflow", "Inputs: geometry, history buffer (≤ 1 day), this step’s inputs, universal constants"],
    src="[our code] dlib.assemble, contract.load_closedform", notes="The supply air is implicit because it can turn a room’s air over faster than one step.")
add("flow", "pipeline", eyebrow="The search · end to end", title="From generated data to a controller",
    steps=[("Generate", "kit tools only"), ("Geometry", "floor-plan script"), ("Propose", "LLM writes formulas"), ("Score", "unseen buildings, no fitting"), ("Keep best", "islands"), ("Control", "MPC")],
    points=["Steps 1–5 are built and tested; step 6 (the controller) is next", "One command: ./run_pipeline.sh"],
    src="[our code] run_pipeline.sh; [plan] controller", notes="Nothing touches the kit.")
add("cards", "llmio", eyebrow="The search · the LLM", title="What goes into the LLM, and what comes out",
    cards=[("In", "Task and physics hypotheses\nData dictionary: geometry, inputs, history buffer\nUniversal constants\nTemplate contract and end-of-step form\nOperator: mutate, cross or explore\nParent code, scores and analysis\nRecent failures with reasons", "", "accent2"),
           ("Out", "Candidates: name, hypothesis, Python module\n\nprepare(geometry); step(history, inputs) → next temperatures\n\nNo free constants.", "", "accent1")],
    src="[our code] prompts.py (cf_system_prompt, cf_user_*), proposers.py", notes="Claude by default, any LLM via an inbox folder.")
add("image", "islands", eyebrow="The search · evolution", title="Islands, mutation, crossover and exploration",
    img="lineage.png", side="bottom",
    points=["Each island evolves on its own", "Mutation fixes one parent; crossover combines two; explore tries something new", "Every few generations an island’s best moves to the next"],
    caption="[illustrative] family tree", src="[our code] discovery/evolve.py", notes="Each operator is a separate LLM request per island per generation.")
add("cards", "analysis", eyebrow="The search · before each request", title="The LLM is told where the parents are wrong",
    cards=[("Before a mutation", "Error on unseen buildings by:\nexterior vs interior, most exposed rooms\nHVAC on/off, comfort hours/nights\nreheat on, cold/warm outside\ngrowth with horizon, worst rooms\n+ measured vs predicted table", "pre-mutation", "accent1"),
           ("Before a crossover", "Both parents side by side:\nwho wins where\nwhich templates differ\nso the child takes each template from the parent that does it better", "pre-crossover", "accent2")],
    callout="Written to analysis.md next to each prompt: you can read exactly what the LLM was told.",
    src="[our code] closedform.analysis, crossover_analysis", notes="Example in the self-test logs.")

# =========================================================================== 6 deviations + evidence
add("code", "llmreads", eyebrow="What the LLM reads (excerpt of the real prompt)", title="The contract: what a candidate must be",
    code="""Write one Python module (numpy, math and dlib only; deterministic; no files):

    HISTORY = 36          # steps of measured history your formulas read (1..288)

    def prepare(geo, U):                  # optional: geometry-only templates, once per building
        return coef

    def step(coef, geo, hist, u, U):      # one 5-minute step for a BATCH of B moments
        return T_next                     # (B, V) room air temperatures at the end of this step

  hist["T"]  (B, HISTORY, V) measured room temperatures, oldest first
  u[k]       this step's inputs: per-room (B, V), building-wide (B,)
  U          universal constants (table below)
  In rollouts your own predictions are fed back into hist["T"] (free run), so the map must stay stable.""",
    points=["Around it: the task, physics hypotheses, data dictionary, universal constants, scoring rules", "Then the operator-specific part: parents, their analysis, recent failures"],
    src="[our code] outputs/discovery/selftest/system_prompt.md (generated by prompts.cf_system_prompt)", notes="The system prompt is the same every generation, so it is cached.")
add("code", "llmanalysis", eyebrow="What the LLM is told about a parent (real excerpt)", title="A pre-mutation analysis: where the parent is wrong",
    code="""held-out rollout RMSE by horizon (model / persistence), K: 1h 2.339/1.464, 3h 2.498/3.005, 6h 2.326/4.722
3h rollout error by group (rmse, bias), K:
  exterior rooms                         2.739  +1.250
  interior rooms                         1.645  +0.786
  most exposed third of exterior rooms   3.026  +1.402
  HVAC air on                            1.520  +0.434
  HVAC air off                           3.391  +2.057
  colder half of outside temps           3.489  +2.612
  warmer half                            1.288  -0.019
skill vs persistence by horizon (ratio <1 is better): 1h 1.60, 3h 0.83, 6h 0.49""",
    points=["Read: too warm when the HVAC is off and it is cold outside, worst in the most exposed rooms", "So the obvious mutation: more heat loss through the facade, or a different slow store",
            "Plus a measured-vs-predicted table for two rooms"],
    src="[synthetic] outputs/discovery/selftest/generations/g001/island_0/mutation/analysis.md (closedform.analysis)", notes="Self-test data (planted truth), so the numbers are synthetic; the format is what the LLM gets on real data.")
add("code", "llmwrites", eyebrow="What the LLM writes back (the history seed, abridged)", title="A candidate: templates assembled into one step",
    code="""HISTORY = 288                                  # one day of 5-minute steps

def step(coef, geo, hist, u, U):
    T = hist["T"][:, -1]                       # rooms now
    terms, b = _known(coef, geo, T, u, U, _slow(hist["T"]))   # supply air, neighbours,
                                               # slow store = weighted average of history
    Uf = facade_u(coef, geo, hist, U)          # facade U-value READ FROM HISTORY:
                                               # shrunk least squares over the buffer
    terms.append((coef["A_fac"] * Uf[:, None] * U["DT"] / coef["C"], u["T_out"][:, None]))
    return dlib.assemble(T, terms, b)          # (T + sum w*T_k + b) / (1 + sum w)""",
    points=["Every number is universal, geometric, or computed from the history buffer", "The answer comes as JSON: name, one-paragraph hypothesis, code"],
    src="[our code] code/discovery/seeds_cf/seed_cf_history_slab.py (a hand-written seed in the format the LLM must use)", notes="No LLM generation has run yet; this is a seed written in the same contract.")
add("cards", "discoutput", eyebrow="What comes out", title="The output of discovery, and where it goes",
    cards=[("best.py", "The winning formulas: prepare() and step(). Readable, a few hundred lines at most, runs in under a millisecond per step.", "the model", "accent1"),
           ("REPORT.md, log.md", "How we got there: leaderboard, family tree, operator success, every prompt and answer, measured-vs-predicted plots.", "the evidence", "accent2"),
           ("Into the policy", "The policy builds geometry from the metadata, keeps a history buffer, calls step() to predict hours ahead, and MPC chooses commands.", "next step", "accent3")],
    src="[our code] evolve.py, report.py (outputs); [plan] the policy and MPC are not built", notes="summary.json holds the scores and the test-split result.")
add("table", "bigmachine", eyebrow="Running it on the big machine", title="One command runs the whole pipeline; nothing here was generated locally",
    rows=[["Step", "What it does", "Command (inside ./run_pipeline.sh)"],
          ["1", "check the setup and the kit", "code/setup_check.py"], ["2", "generate buildings, scenarios and runs with the kit’s own tools", "code/generate_data.py"],
          ["3, 5", "audit the raw data and the dataset", "code/audit_data.py"], ["4", "build the dataset (one file per building × weather × policy)", "code/build_dataset.py"],
          ["6", "LLM discovery: islands, mutation, crossover; scores the best on test", "code/run_discovery.py --proposer claude --test"],
          ["7", "fitted ceiling, as a diagnostic", "code/run_discovery.py --ceiling"]],
    colw=[0.08, 0.5, 0.42], mono_first=False,
    callout="Before spending on the LLM: run_discovery.py --selftest (4 minutes, no data) and --proposer dry (writes every prompt, calls nothing).",
    src="[our code] run_pipeline.sh, docs/RUNBOOK.md", notes="Defaults: train seeds 500–599, val 700–719, test 800–819; settings are environment variables at the top of run_pipeline.sh.")
section("s6", "6", "Measuring deviations", "How do we measure how wrong a model or a controller is, and what do the data say so far?",
        "Model deviations, control deviations, then the real-data check and the self-test.", "6 · Deviations and evidence")
add("table", "devmodel", eyebrow="Model deviations", title="How we measure how wrong a model is",
    rows=[["Measure", "Definition", "Why"],
          ["One-step error", "RMSE of the 5-minute prediction from measured history", "quick screen (stage 1)"],
          ["Rollout error at h", "RMSE of free-run forecasts 1, 3, 6 h ahead, started every 30 min", "what MPC uses; decides (stage 2)"],
          ["Skill", "rollout error ÷ persistence error at the same horizon", "below 1 = better than ‘temperature stays’"],
          ["Bias", "mean signed error", "a wrong steady-state gain shows here"],
          ["By group", "exterior / interior / most exposed; HVAC on/off; comfort hours; reheat; cold/warm", "says where to fix (pre-mutation)"],
          ["Stability", "any rollout leaving 230–350 K", "rejects the candidate"],
          ["Gap to ceiling", "candidate vs per-room fitted reference", "is the formula or the data the limit?"]],
    colw=[0.17, 0.53, 0.3], src="[our code] closedform.episode_scores, analysis, ceiling", notes="All held-out: buildings never used for the LLM’s feedback selection.")
add("image", "devcontrol", eyebrow="Control deviations", title="Measured exactly as the reward does: comfort, energy, carbon, every step",
    img="ep30_reward.png", side="bottom",
    points=["Daily mean loss per step, split into its three terms, for all six 30-day episodes", "Weekends cost little: no one is there to be uncomfortable; cold climates are dominated by comfort and energy"],
    caption="[sim data] pid_water, kit reward_components.csv; score = mean loss per step over the episode",
    src="[website] reward definition; [sim data] kit evaluate.py (scratch folder)", notes="Reported per scenario, per climate, worst case and per term.")
add("image", "comfortdev", eyebrow="Control deviations", title="Comfort deviation per day: kelvin outside the band in comfort hours",
    img="ep30_comfort.png", side="bottom",
    points=["The baseline leaves rooms outside the band every weekday morning", "This is the term a model-based controller can cut by pre-heating"],
    caption="[sim data] six 30-day pid_water episodes; band from the schedule (294–297 K comfort)", src="[sim data] kit evaluate.py (scratch folder); [kit] config.gin schedule",
    notes="The reward also weights by occupants; this plot is unweighted.")
add("image", "realhorizon", eyebrow="Real data, 30-day episodes, nothing fitted", title="The hand-written seeds are not good enough yet: this is the job for the search",
    img="ep30_horizon.png", side="right",
    points=["History seed: 0.99 / 1.16 / 1.34 K at 1 / 3 / 6 h", "Persistence: 0.66 / 0.97 / 1.47 K: the seed only wins at 6 h",
            "One-node seed: 3.6 / 6.6 / 7.9 K, and unstable on one episode", "Fitted ceiling: 0.31 / 0.46 / 0.62 K: formulas can do far better"],
    caption="[sim data] 12 episodes × 8,640 steps: 6 shipped buildings × (pid_water, HVAC off); inputs rebuilt from commands, as in deployment",
    src="[measured] closedform.evaluate on kit evaluate.py runs replayed by our starterkit_data (flow error ≤ 0.02%, run commands 100% matched)",
    notes="An earlier 3-day check on audit files (simulator's own heat input) looked better; the 30-day check with deployment-like inputs is the honest one.")
add("image", "byepisode", eyebrow="Real data, per episode", title="Where the seed loses: almost everywhere at 3 h, worst under HVAC off",
    img="ep30_by_episode.png", side="bottom",
    points=["The ceiling (per-room fitted, diagnostic) beats persistence in 11 of 12 episodes", "The gap between seed and ceiling is what the LLM search has to close"],
    caption="[sim data] 3 h rollout RMSE per 30-day episode; b1 = building_train_1, pid = pid_water, off = HVAC off", src="[measured] closedform.evaluate, closedform.ceiling", notes="")
add("image", "realprofile", eyebrow="Real data · measured vs predicted", title="Week 2 of a 30-day episode: what the seed gets right, and where it is wrong",
    img="ep30_profile_pred.png", side="right",
    points=["Daily shape: right", "Under the baseline’s rapid switching the 3 h forecast over-swings", "Exterior room: level off at night and at start-up",
            "Exactly what the pre-mutation analysis tells the LLM to fix"],
    caption="[sim data] building_train_2, Atlanta window 3, pid_water, days 7–14; seed_cf_history_slab, nothing fitted",
    src="[measured] closedform.profile on the replayed 30-day episode", notes="")
add("cards", "canfit", eyebrow="Three reference points", title="Is a bad score the LLM’s fault or the data’s?",
    cards=[("Floor: persistence", "‘The temperature stays.’ Every candidate must beat it.", "floor", "accent3"),
           ("Ceiling: fitted per room", "Non-negative per-room weights fitted on half of each episode; stable by construction. Never deployed.", "diagnostic only", "accent2"),
           ("Planted truth", "Data from a known formula: the scorer must give it zero and rank it first.", "self-test", "accent1")],
    callout="On real data the first ceiling version blew up in free run (101 K at 3 h): a good one-step fit is not a good rollout. It is now stable.",
    src="[our code] closedform.ceiling, selftest.py; [measured]", notes="That failure is the guideline’s warning, seen in our own code.")
add("image", "selftest", eyebrow="Self-test", title="The machinery works end to end",
    img="selftest_profile.png", side="right",
    points=["Planted truth: 0.000 K at every horizon (history, inputs, targets aligned)", "Seeds 2.5 / 3.3 K, persistence 3.0 K, best after 2 generations 2.3 K",
            "22 candidates, 2 islands: mutations, crossovers (3 failed checks), migrations", "Every prompt, analysis and candidate is on disk"],
    caption="[synthetic] shipped floor plans, made-up inputs, temperatures from a known formula, mechanical stand-in for the LLM",
    src="[synthetic] outputs/discovery/selftest/REPORT.md", notes="python code/run_discovery.py --selftest (about 4 minutes).")
add("table", "wheretolook", eyebrow="Where to look", title="Every proposal, mutation and crossover leaves a file",
    rows=[["File (outputs/discovery/<run>/)", "What it shows"], ["REPORT.md", "best model, leaderboard, operator success, family tree, plots"],
          ["log.md", "each request and each child, with its change against the parent"], ["generations/gNNN/island_K/<operator>/prompt.md", "exactly what the LLM was asked"],
          ["…/analysis.md · …/response.json", "what it was told · what it answered"], ["candidates/<id>_<name>.py", "every candidate’s code"],
          ["plots/<id>_profile.png · _horizon.png", "measured vs predicted · error vs horizon"], ["database.jsonl", "island, operator, parents, scores per candidate"]],
    colw=[0.45, 0.55], mono_first=True, src="[our code] evolve.py, report.py; docs/DISCOVERY_WALKTHROUGH.md", notes="")

# =========================================================================== 7 next
section("s7", "7", "Where we are", "What is built, what is not, and what comes next?", "Honest status, open decisions, roadmap.", "7 · Next")
add("table", "status", eyebrow="Honest status", title="What is built, what is partial, what is not",
    rows=[["Piece", "Status"], ["Data generation with kit tools; dataset; audits", "built; full run not done"],
          ["Geometry from floor plan and zone map", "built"], ["Closed-form contract, no fitting; constants file", "built; constants to review"],
          ["Scoring, analyses, floor, ceiling, self-test", "built and tested"], ["Islands, operators, logs, report, plots", "built and tested (mock LLM)"],
          ["Real-data check of the seeds", "12 × 30-day episodes: seed loses to persistence at 1–3 h"], ["Command-to-heat conversion", "hand-written, not discovered"],
          ["Real LLM generations", "not run"], ["Closed-form model inside the policy; MPC; a submission", "not built"]],
    colw=[0.6, 0.4], src="[our code]; [measured]", notes="")
add("table", "decisions", eyebrow="Decisions", title="Settled, and still open",
    rows=[["Open question", "Options"], ["Goal", "best Kaggle score, or a generic discovered model"], ["Simulator structure in the prompt", "graph only, or the full physics chain"],
          ["MPC horizon", "not chosen; 1, 3, 6 h all reported"], ["History summary", "raw lags, differences, window averages, LLM-proposed features"],
          ["Occupants per room", "by area, equally, or a time-of-day rule"], ["Weather for training", "48 windows only: how to hold out stations fairly"]],
    colw=[0.38, 0.62], callout="Settled: no fitting · universal constants allowed · reviewed once · real buildings first try the same constants · early error accepted · discrete-time maps allowed.",
    src="[decision] planning notes", notes="")
add("flow", "roadmap", eyebrow="Roadmap", title="From here to a submission, with a fallback at every stage",
    steps=[("Full data", "big machine"), ("LLM generations", "real search"), ("Pick horizon", "for MPC"), ("Model in policy", "tracker"), ("MPC", "≤ 0.125 s/step")],
    points=["Submit a pid_water-like baseline early (5 submissions a day)", "Then feed-forward: model-computed setpoints with PID trims", "Then MPC; deadline January 15, 2027"],
    src="[plan] full backlog: docs/next_tasks.csv (68 tasks); [Kaggle] timeline, limits", notes="Each stage is a complete submission.")
add("statement", "close", eyebrow="In one line", title="Formulas we can read, written once, run on any building",
    sub="Geometry from the floor plan · materials from the measured history · everything else universal · nothing fitted", src="", notes="Questions?")
