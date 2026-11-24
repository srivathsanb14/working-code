"""Clarity pass on parts 1-5 (slides 2-47): fewer words per slide, one idea per slide, physics told in plain terms,
the time budget and the score explained from first principles, the seed described as a building's ID number."""
import content4 as C4

B, S, SECTIONS = C4.B, C4.S, C4.SECTIONS


def put(sid, kind, **kw):
    """Replace a slide's content; keep its id, and its sources and notes unless new ones are given."""
    s = B[sid]
    keep = {k: s[k] for k in ("id", "src", "notes") if k in s}
    s.clear(); s.update(keep); s["kind"] = kind; s.update(kw)


# ============================================================ opening
put("questions", "table", eyebrow="The story", title="Twelve parts in four acts",
    rows=[["Act", "Parts", "The question it answers"],
          ["The setting", "1 The problem · 2 The data · 3 What the data look like",
           "What must we build, what data do we have, and how do rooms behave over a month?"],
          ["The idea", "4 The physics · 5 The usual approaches · 6 Our approach",
           "How does heat move, why do standard methods not fit, and what do we do instead?"],
          ["The machinery", "7 LLM discovery · 8 Evidence · 9 Inside the policy",
           "How are formulas found, how well do they forecast, and do they behave the same in a live policy?"],
          ["Using it", "10 Controllers · 11 The Kaggle submission · 12 Where we are",
           "How does a forecast become control, how is it submitted, and what is left to do?"]],
    colw=[0.16, 0.36, 0.48], size=26,
    src="this deck")

put("legend", "cards", eyebrow="How to read this deck", title="Every slide says where its facts come from",
    cards=[["From the organisers", "[website]  [Kaggle]\nthe official competition pages, read Nov 19, 2026\n\n[kit]\nthe starter kit's code, files and images",
            "official", "accent2"],
           ["From our own work", "[audit]  our reading of the simulator code\n\n[sim data]  simulator runs made with the kit's evaluate.py\n\n[measured]  numbers we computed from those runs\n\n[our code]  working-code/code",
            "ours", "accent1"],
           ["Not data", "[synthetic]  made-up data with a known answer, used to test our code\n\n[DRAWN]  a schematic, not data\n\n[decision] · [plan]  a choice we made · not built yet",
            "careful", "accent3"]],
    callout="The tags appear at the start of the source line at the bottom of each slide.",
    src="this deck")

put("decmap", "table", eyebrow="Every decision, and where it is explained", title="The main decisions, by area",
    rows=[["Area", "Main decisions", "Part"],
          ["Data generation", "use only the kit's tools · new buildings from new seeds · hold out whole weather stations · all 5 controllers", "2"],
          ["Dataset", "recover commands by replay · use only what the policy can see · heat flows as inputs", "2"],
          ["Model form", "a state-space model · formulas, not backprop · nothing fitted · universal constants", "6"],
          ["Scoring", "physics checks · forecast horizons 1/3/6 h · test-time inputs · a price per formula size", "7"],
          ["Search", "islands of candidates · LLM edit operators · parent choice · retries and resume", "7"],
          ["Runtime", "same timing live as offline · parity check · safety layer and fallback", "9"],
          ["Controllers", "build on pid_water · cost = the reward · occupancy forecast · replanning", "10"],
          ["Submission", "one notebook · bundled code · rehearsal · readiness gate", "11"]],
    colw=[0.18, 0.72, 0.10], size=24,
    src="this deck; docs/DISCOVERY_WALKTHROUGH.md, docs/SOURCES_AND_FACTS.md")

# ============================================================ part 1
put("submission", "code", eyebrow="The submission", title="What we submit: one notebook with one Python class",
    code=("from kaggle_evaluation import smart_buildings_inference_server as sbi\n"
          "import numpy as np\n\n"
          "class TransferPolicy(sbi.Policy):\n"
          "    def __init__(self, building_meta_data, action_spec):\n"
          "        # called once per scenario, with this building's\n"
          "        # floor plan, zone map, zone->AHU map, schedule,\n"
          "        # and where each command sits in the action vector\n"
          "        super().__init__(building_meta_data, action_spec)\n"
          "        self.K1 = building_meta_data[\"n_discrete_action\"]    # K + 1\n"
          "        self.C = building_meta_data[\"n_continuous_action\"]   # 3K + 2 + 2V\n\n"
          "    def get_action(self, obs):\n"
          "        # called every 5 simulated minutes, 8,640 times\n"
          "        return {\"continuous_action\": np.zeros(self.C, np.float32),\n"
          "                \"discrete_action\":   np.zeros(self.K1, np.int32)}\n\n"
          "sbi.SmartBuildingsInferenceServer(TransferPolicy).run()"),
    points=["K = air handlers, V = rooms; both change from building to building, so sizes are read from the metadata",
            "Setpoints in [−1, 1], switches in {0, 1}; out-of-range is rejected, and a NaN ruins the episode",
            "No scenario list, no loop: Kaggle's gateway owns the simulator and calls get_action",
            "No internet; model files come from an attached Kaggle Dataset. 5 submissions a day, 2 final, deadline Jan 15, 2027",
            "Forbidden: hard-coding positions, or referring to held-out buildings or weather"],
    src="[website] submission.html (interface, guidelines); [Kaggle] overview (code requirements, timeline), rules (submission limits)")

put("budget", "flow", eyebrow="The clock", title="Why about 0.125 s per step: one 9-hour run must play all 30 hidden scenarios",
    steps=[("You press Submit", "Kaggle re-runs the same notebook once, offline"),
           ("The gateway plays 30 scenarios", "6 hidden buildings × 5 hidden stations, a fresh policy for each"),
           ("8,640 steps each", "30 days of 5-minute steps: 259,200 calls to get_action"),
           ("All in one 9-hour session", "32,400 s ÷ 259,200 ≈ 0.125 s per step")],
    points=["When we run the notebook ourselves it plays one public example; only the scored re-run plays all 30",
            "The 0.125 s includes the simulator: on our Mac it alone takes 0.025–0.057 s per step, leaving roughly 0.07–0.10 s for the policy",
            "Whether the gateway runs scenarios in parallel is not documented, so we plan for one after another",
            "The per-call caps (15 min for the first step, 60 s for later ones) are far looser: with 259,200 calls, the 9 h total is what binds"],
    src="[website] submission.html ('Kaggle re-runs the same notebook against the hidden test scenarios'; time limits); [Kaggle] code requirements (9 h); [sim data] kit summary.json seconds_per_step, 12 runs")

put("scoring", "eq", eyebrow="The score", title="The score is the average loss per 5-minute step, over the 30 scenarios",
    eq="loss per step  =  0.6 × comfort  +  0.2 × energy  +  0.2 × carbon\nscore  =  mean over scenarios of (loss summed over 8,640 steps ÷ 8,640)",
    points=["Comfort: the share of occupants' productivity lost. 0 inside the band; a room d K outside loses d/5 up to 5 K, then 1 + ((d − 5)/5)², with no cap; rooms are weighted by occupants, and an empty building costs 0",
            "Energy: this step's electricity and gas cost ÷ the cost at a reference load of 80 kW electric + 205 kW gas per air handler; it can exceed 1",
            "Carbon: the same ratio for emissions",
            "0.6 / 0.2 / 0.2 are the reward weights fixed in the kit's config. The kit's reward is −loss, so it is never positive; lower score is better and 0 (perfect comfort for free) is unreachable"],
    src="[website] scoring.html (metric); [kit] smart_control/reward/setpoint_energy_carbon_regret.py, config.gin (weights, reference loads)")

put("leaderboard", "table", eyebrow="Where the bar is", title="The kit's best controller scores 0.054; the leaderboard leader scores 0.0045",
    rows=[["Controller", "What it does", "Score"],
          ["bang_bang", "each air handler blasts full heat or cold whenever its rooms leave the band", "0.225"],
          ["pid", "each air handler turns its supply temperature up or down in proportion to the rooms' error", "0.067"],
          ["fixed_setpoint", "pid, but one comfort band all day: no night or weekend setback", "0.066"],
          ["pid_water (best baseline)", "pid on the worst room, plus free cooling, a modulating reheat valve and boiler water set by demand", "0.054"],
          ["best of the four, picked per scenario afterwards", "an upper bound on what choosing among them could give", "0.047"],
          ["Leaderboard #50 / #1 (Nov 19, 2026)", "other teams' policies, on the hidden test", "≈ 0.0059 / 0.0045"]],
    colw=[0.27, 0.58, 0.15], size=23,
    callout="Baselines are scored on the 30 public scenarios, the leaderboard on hidden ones, so the comparison is rough. Still, the leaders lose about a tenth of what pid_water loses per step: there is a lot of room above rule-based control.",
    src="[kit] baseline_returns.json (score = −return / 8,640, averaged), baselines/*.py docstrings; [Kaggle] leaderboard, read Nov 19, 2026")

# ============================================================ part 2
B["shipped"]["rows"][0][1] = "Seed (ID)"
B["shipped"].update(
    title="Six training buildings, each built by the kit's generator from its own seed number",
    callout=("A seed is simply the ID number the generator used to build that building, like a serial number. We never change the seed "
             "of these six (that would be a different building). To get more buildings, we ask the generator for new, unused numbers."))
B["shipped"]["rows"][0][7] = "Wall: conductivity, density"

put("seed", "cards", eyebrow="What a seed is", title="A seed is a building's ID number: same number, same building",
    cards=[["The seed fixes the shape", "Floors, footprint, rooms and their layout, number of air handlers, people per room.", "visible in the metadata", "accent2"],
           ["The batch fixes the materials", "Wall, slab and partition materials are drawn for a whole batch of seeds at once.", "hidden", "accent1"],
           ["Each run picks the rest", "Which weather window, which controller drives the building, how many steps.", "chosen per run", "accent3"]],
    callout="We never edit a seed or a building. New data = new seed numbers: 500–599 for training, 700–719 for validation, 800–819 for test.")

put("change", "compare", eyebrow="Making more data", title="Three things we vary, and what we never touch",
    left=("We vary", ["Which building: a new seed number gives a new building",
                      "Which weather: one of the kit's 48 windows",
                      "Which controller drives it: the 5 shipped ones",
                      "How long: 7-day runs (the score uses 30 days)"]),
    right=("We never touch", ["The generator's ranges: they define where the hidden buildings come from",
                              "A building's file: editing it moves the building off that distribution",
                              "Action ranges, normalisers, reward weights: they fix what the gateway sends and scores",
                              "The kit's code: we only call its commands"]),
    callout="So every training building is one the hidden test could have drawn.")

put("datarecipe", "flow", eyebrow="How more data is made", title="Training data = new buildings × weather windows × controllers",
    steps=[("100 buildings", "seeds 500–599, all new"),
           ("× 4 weather windows", "Fairbanks, Denver, Atlanta, Singapore"),
           ("× 5 controllers", "all_off, bang_bang, fixed_setpoint, pid, pid_water"),
           ("= 2,000 runs", "7 days each (2,016 steps)")],
    points=["Validation: 20 other buildings (700–719) × Phoenix × 5 controllers = 100 runs",
            "Test: 20 more (800–819) × 3 Minneapolis windows × 5 controllers = 300 runs",
            "Whole stations are held out, because the hidden test uses stations we never see",
            "The window number rotates with the seed, so every season appears; the 30 shipped scenarios can be added (except Phoenix)"],
    src="[our code] generate_data.py defaults (make_plan, strip_for, split_json), run_pipeline.sh; [kit] run_floor_generator.py, run_scenario_generator.py")

put("generator", "cards", eyebrow="What the generator varies", title="Every hidden building is a fresh draw from these same ranges",
    cards=[["Shape: set by the seed", "Floors 2–4\nFootprint 400–700 × 300–500 ft\nRooms 30–60 ft across\nAir handlers 3–7 (at least one per 10 rooms)\n1–2 people per room", "visible", "accent2"],
           ["Materials: drawn per batch", "Wall conductivity 0.15–0.55 W/mK\nWall density 350–2,400 kg/m³\nSlab and furniture 10–40 kg/m³\nInside walls 0.45–0.60 W/mK", "hidden", "accent1"],
           ["Weather: chosen per run", "6 stations × 8 windows of 30 days\nWind film 8.8–26.7 W/m²K, one value per station", "chosen", "accent3"]],
    callout="Shapes reach the policy through the metadata; materials never do, so they must be inferred from how rooms behave.",
    src="[kit] run_floor_generator.py DEFAULT_RANGES, smart_control/utils/materials.py PROPERTIES, weather/*.yaml")

B["layouts"].update(side="right", points=[
    "Three of the six: 30, 39 and 85 rooms; 3, 4 and 9 air handlers",
    "Rooms are not a few standard types: each has its own area, outside wall, shared walls and doors",
    "So the model must be written for any floor plan, not a fixed list of rooms",
    "(The figure labels train_8 “28 zones”; its config has 30 rooms)"])

put("decdatagen", "table", eyebrow="Decisions · data generation", title="How the training data is generated, and why",
    rows=[["Question", "Our choice", "Why"],
          ["Which tools?", "only the kit's own commands", "the data must come from the simulator the hidden test uses"],
          ["How to get new buildings?", "new seed numbers", "a seed rebuilds the whole building from the generator's ranges"],
          ["How to split?", "different seeds for train, validation, test", "test buildings must be unseen in training"],
          ["Which weather?", "hold out whole stations (Phoenix, Minneapolis)", "the hidden test uses unseen stations"],
          ["Which controllers?", "all five shipped ones", "rooms must be seen drifting, swinging and held steady"],
          ["How long?", "7 days per run", "covers weekdays and a weekend at a quarter of the cost"],
          ["Rejected seeds?", "draw 30% extra and keep the planned count", "the kit rejects about 1 seed in 5 (a room under 20 m²)"]],
    colw=[0.24, 0.36, 0.40], size=23,
    src="[our code] generate_data.py, run_pipeline.sh; [kit] run_floor_generator.py, scenario_generator.py")

put("decdataset", "table", eyebrow="Decisions · dataset", title="How simulator runs become training examples, and why",
    rows=[["Question", "Our choice", "Why"],
          ["The kit's logs lack per-room commands", "re-run the same controller on the logged observations and record its commands", "it is deterministic; checked: airflow matches the log to 0.02%"],
          ["When is a temperature seen?", "the observation at step t shows step t − 1", "use exactly what the policy sees, live and offline"],
          ["Occupancy", "the head count the policy sees (two steps behind the log)", "only what is observable may be used"],
          ["Model inputs", "heat flows computed from commands with the kit's equipment equations", "the command-to-heat part is known and nonlinear"],
          ["Boiler water", "computed from the setpoint with the boiler's ramp limits", "the one equipment state the policy can predict"],
          ["Geometry", "from floor plan, zone map and air-handler map only", "exactly what a new building's metadata gives"]],
    colw=[0.24, 0.42, 0.34], size=22,
    src="[our code] discovery/starterkit_data.py, build_dataset.py, runtime/inputs.py; [audit] STATE_SPACE.md")

# ============================================================ part 3
B["s3"]["question"] = "What happens to every room over 30 days, and where does the score come from?"
B["profiles"].update(side="bottom",
    title="30 days, six buildings: rooms are held in the band on working days and drift at night and on weekends",
    points=["Line = middle room of its kind; shading = 80% of rooms; green = comfort band (occupied hours); grey = weekends",
            "Cold climates (Fairbanks, Denver): rooms with outside walls fall far below the band overnight",
            "Mild climates (Atlanta, Singapore): interior rooms sit at or above the top of the band",
            "This, room by room for a month, is what the model must forecast"],
    caption="[sim data] pid_water, 8,640 steps, 30-minute averages; one weather window per building")
B["week"].update(title="One working week up close: 06:00, 19:00 and the weekend",
    points=["06:00: comfort hours start and pid_water pulls rooms into the band (top edge 23.85 °C)",
            "19:00: the system sets back; interior rooms warm, rooms with outside walls cool toward the outside air",
            "Weekend: nobody in, so rooms drift freely",
            "Each room drifts at its own speed: that speed is what a forecast must get right"],
    caption="[sim data] building_train_2, Atlanta window 3 (April), pid_water, days 7–14, 30-minute averages")
B["control"].update(title="Why control matters: the same building with the HVAC off and with the best baseline",
    points=["Atlanta, April, HVAC off: rooms reach 35–45 °C; with no air supplied, heat from people, lights and equipment stays in",
            "Fairbanks, January, HVAC off: rooms fall to about −15 °C; pid_water holds 15–20 °C at a large energy cost (score 0.214)",
            "The HVAC dominates room temperature, so a model must take the commands as inputs"],
    caption="[sim data] line = median room; shading = middle 80% of rooms; green = comfort band; 8,640 steps")
B["geometry"].update(title="How fast a room loses heat can be read from the floor plan",
    points=["Each dot: one room with an outside wall, HVAC off, first 12 hours",
            "x: outside-wall length per m² of floor, computed from the floor plan alone",
            "y: how much of the indoor–outdoor gap the room closes per hour",
            "Within a building the dots lie on a line (r = 0.79–0.97): more outside wall, faster loss",
            "Between buildings the slope differs about fivefold: wall materials, which the metadata does not give"],
    caption="[sim data] HVAC off, six shipped buildings; static figure made by our script from the kit runs",
    notes="All part 3 figures are static images produced by docs/deck_src/make_figs7.py from twelve 30-day kit runs (6 buildings × pid_water / HVAC off).")
B["devcontrol"].update(title="Where the score comes from: comfort, except in the coldest climate",
    points=["Left: each scenario's score split into its three terms. Comfort is the largest in five of six",
            "Fairbanks: heating cost (energy 0.055 + carbon 0.089) outweighs comfort (0.070)",
            "Right, Denver day by day: weekends cost almost nothing (nobody there to be uncomfortable); working days cost 0.04–0.11 per step"],
    caption="[sim data] pid_water, kit reward_components.csv; score = mean loss per step over the episode")
B["comfortdev"].update(title="Comfort losses: cold mornings in cold climates, slightly warm rooms everywhere else",
    points=["Too cold: Fairbanks rooms start the working day 3.8 K below the band and stay about 1.4 K below; Denver and Atlanta-January also start cold and recover slowly",
            "Too warm: everywhere except Fairbanks, rooms sit about 0.2–0.5 K above the band all day",
            "A controller that forecasts can pre-heat before 06:00 and stop short of overshooting: the gap a model-based controller targets"],
    caption="[sim data] six 30-day pid_water episodes; band 294–297 K (20.85–23.85 °C); mean over rooms, not weighted by occupants")

# ============================================================ part 4: physics in plain terms
B["s4"].update(title="How heat moves", question="What happens physically inside a room, and which of it can we confirm from the simulator's data?")
B["roomsees"].update(title="One room is a pot of air, connected to everything around it",
    points=["The room's air changes temperature fast (minutes): it holds little heat",
            "The floor slab and furniture store much more heat and change slowly (hours)",
            "Outside walls leak heat to the outside air; the wind outside (the 'wind film') speeds this up",
            "Inside walls and doors trade heat with the neighbouring rooms",
            "The HVAC adds or removes heat with supply air and reheat; people, lights and equipment add heat"])
B["linear"].update(title="Heat flows in proportion to the temperature gap: twice the gap, twice the flow",
    points=["HVAC off, at night: each hour's temperature change against the indoor–outdoor gap",
            "One straight line per building (r = 0.71–1.00): Newton's law of cooling",
            "The slope differs by building because the walls differ; crosses (19:00, lights dimmed to 20%) are an extra input, not a break in the law"])
B["fastslowreal"].update(title="Rooms have two speeds: the air in minutes, the building's mass in hours",
    points=["Fairbanks, after the 19:00 setback: 1.9 K lost in 30 minutes, then only 0.14 K per hour (Denver, Phoenix, Atlanta-Jan alike)",
            "Fast drop: the air loses its little heat; slow tail: the slab slowly gives back what it stored. In April Atlanta and Singapore rooms warm instead",
            "The slab is never measured, so a model must infer its temperature from the recent history"])
B["commands"].update(title="Commands do not turn into heat in a straight line",
    points=["Air into a room = damper × √(duct pressure) × a constant; the damper never closes below 10%, and the flow is zero when the air handler is off",
            "Reheat heat is limited by the airflow, the hot-water temperature and a 35 °C cap on the air leaving the coil",
            "So we convert commands into heat flows with the kit's own equipment equations; in heat flows, the physics is linear again"])
B["obsgroups"].update(title="What the policy sees each step: 6 numbers per room, 15 per air handler, 58 for the building",
    points=["Per room: its temperature, its damper and reheat commands, two setpoints, an airflow setpoint",
            "Per air handler: supply temperature, pressure, flows, fan, and a copy of the outside temperature",
            "Building: boiler (7), clock (48), comfort flags (2), people in the building (1)"])
B["obsroles"].update(title="Of all those numbers, the model needs room temperatures and eight more",
    rows=[["Field (count)", "What it is", "Used by the model?"],
          ["room temperature (V)", "what we forecast", "yes: the target"],
          ["damper, reheat command (2V)", "our own last commands", "as inputs (we know them anyway)"],
          ["heating, cooling setpoint (2V)", "the comfort band: 294–297 K by day, 289–298 K at night", "no: only the score uses it"],
          ["other air-handler fields (14 each)", "copies of our commands, or follow from them", "no: use the commands"],
          ["outside air temperature (K copies)", "the weather now", "yes"],
          ["boiler water temperature", "changes slowly; the one plant state we observe", "yes"],
          ["people, comfort flag, clock", "building head count; schedule; time", "yes: people, comfort flag, time of day, weekday"]],
    colw=[0.3, 0.42, 0.28], size=23, callout=None)
B["actions"].update(title="What the policy controls, and what the plant actually does with it")
B["actions"]["rows"][0][3] = "What actually happens"
put("constants", "compare", eyebrow="The model · constants", title="Some numbers are the same in every building; others must be inferred",
    left=("Same everywhere: written into formulas", ["Density and specific heat of air and water",
                                                     "Grid cell 0.2 m, ceiling 3 m, step 5 minutes",
                                                     "Every room starts at 294 K (20.85 °C)",
                                                     "70 W per person; 10 W/m² from lights and equipment (20% after hours)"]),
    right=("Differs per building: inferred from measurements", ["Wall conductivity, density, heat capacity",
                                                                 "Slab and furniture density",
                                                                 "Inside-wall conductivity and density",
                                                                 "People per room (1–2); the wind film (set by the weather station)"]),
    callout="The 'same everywhere' list is reviewed once before it goes into the LLM prompt (code/discovery/universal_constants.json).",
    src="[kit] materials.py, weather/*.yaml, config.gin; [audit] STATE_SPACE.md §12.2")
B["coefficients"].update(title="Every heat path = contact size × how well the material conducts",
    cards=[["The law", "Heat flow per kelvin = conductivity × contact area ÷ thickness", "physics", "accent2"],
           ["Layers add up", "Air → wall → outside air works like resistors in series: the resistances add", "walls", "accent1"],
           ["From the floor plan", "Wall lengths, areas, door widths: counted from the metadata", "visible", "accent3"],
           ["From behaviour", "Material values: universal constants, or computed from recent measurements by the formula", "hidden", "accent1"]],
    callout="So a building's whole thermal model is its floor plan plus a handful of material numbers.")

# ============================================================ part 5
B["s5"]["question"] = "How is building control normally done, and why does it not fit this competition?"
B["rules"].update(title="Option 1, rules: react to the error now, never plan ahead",
    points=["Fixed setpoints, bang-bang, PID (the kit's baselines) look only at how far rooms are from the band right now",
            "No forecast: heating starts after a room is already cold, so every working day starts outside the band (part 3)",
            "Tuned once for all buildings: pid_water is best on average, yet on train_1 in Fairbanks a fixed setpoint beats it (0.147 vs 0.212)",
            "Safe and simple: the base our own controller builds on"])
B["smarter"].update(title="Options 2–4 learn or fit a model, and each needs something we do not have",
    cards=[["Reinforcement learning", "Learns a policy by trial and error over many episodes.\n\nHere every test building is new, its size changes, and nothing can be trained during the evaluation.", "needs: training on the building", "accent1"],
           ["A neural-network model", "Learns the building's behaviour, then plans with it.\n\nHere there is no data on the new building at the start, and layouts differ.", "needs: the building's data", "accent2"],
           ["Physics, fitted per building", "A resistor-capacitor (RC) heat model whose constants are fitted to the building.\n\nHere there is no day of data to fit on, and only 0.125 s per step.", "needs: a fitting period", "accent3"]])
B["comparison"].update(title="Side by side: only a universal formula needs no data from the new building",
    callout="The bottom row is what this deck builds: physics structure so it transfers, nothing fitted so it works from the first step, and MPC so it plans ahead.")
B["tried"].update(title="What we tried first: fitting the constants works, but needs a day of each building",
    points=["We wrote a heat network by hand and fitted its material numbers on day 1 of each building",
            "It then forecasts far better than persistence ('the temperature stays where it is'): the structure is right",
            "But the hidden test gives no day 1 to fit on, and fitting must fit in 0.125 s per step",
            "So: keep the structure, and compute the material numbers from the measurements with a formula. That is part 6"])

# wide figures read better beside the text than above it
for sid in ("profiles", "week", "control", "devcontrol", "comfortdev", "linear", "fastslowreal", "commands", "climates"):
    B[sid]["side"] = "right"
B["change"]["title"] = "What we vary, and what we never touch"
