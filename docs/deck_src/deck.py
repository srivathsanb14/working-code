"""The full working deck: every slide, in order, as data.  story.py renders it to HTML and to story.json (for the
PowerPoint); content_short.py reuses it as the meeting deck's appendix.  Edit the slides here directly.

Each slide is a dict with an id, a kind (statement, section, table, cards, compare, flow, figure, image, code, eq,
points, stats) and that kind's fields; "src" names the sources shown at the bottom of the slide."""

S = [
 {
  "kind": "statement",
  "id": "cover",
  "eyebrow": "NeurIPS 2026 · Smart Buildings Challenge",
  "title": "One controller for every building",
  "sub": "Predicting every room’s temperature in any building from its floor plan and its own measured history, so a controller can plan ahead",
  "foot": "Working session deck · every slide names its sources",
  "notes": "From zero: the task, why it is hard, the usual approaches, our approach, how it works, the evidence, what is next."
 },
 {
  "id": "questions",
  "src": "this deck",
  "notes": "The source line on every slide says whether a claim comes from the official pages, the kit, simulator runs, our measurements or our code.",
  "kind": "table",
  "eyebrow": "The story",
  "title": "Twelve parts in four acts",
  "rows": [
   [
    "Act",
    "Parts",
    "The question it answers"
   ],
   [
    "The setting",
    "1 The problem · 2 The data · 3 What the data look like",
    "What must we build, what data do we have, and how do rooms behave over a month?"
   ],
   [
    "The idea",
    "4 The physics, from cell to state space · 5 The usual approaches · 6 Our approach",
    "How does heat move, how is it written as a state-space model, what is unknown, and how do we fill it?"
   ],
   [
    "The machinery",
    "7 LLM discovery · 8 Evidence · 9 Inside the policy",
    "How does an LLM find the formulas, how are they scored and tested on held-out data, and do they behave the same in a live policy?"
   ],
   [
    "Using it",
    "10 After discovery: control · 11 The Kaggle submission · 12 Where we are",
    "Once the model exists, how could a controller use it, how is it submitted, and what is left to do?"
   ]
  ],
  "colw": [
   0.16,
   0.36,
   0.48
  ],
  "size": 26
 },
 {
  "id": "legend",
  "src": "this deck",
  "notes": "Every figure is data, a kit image, or computed from a real floor plan.",
  "kind": "cards",
  "eyebrow": "How to read this deck",
  "title": "Every slide says where its facts come from",
  "cards": [
   [
    "From the organisers",
    "[website]  [Kaggle]\nthe official competition pages, read Nov 19, 2026\n\n[kit]\nthe starter kit's code, files and images",
    "official",
    "accent2"
   ],
   [
    "From the simulator: its runs and its code",
    "[sim data]  runs made with the kit's evaluate.py\n\n[measured]  numbers we computed from those runs\n\n[audit]  our line-by-line reading of the simulator's code",
    "data",
    "accent1"
   ],
   [
    "From our code and plans",
    "[our code]  working-code/code\n\n[decision]  a choice we made, with the reason\n\n[plan]  not built yet",
    "ours",
    "accent3"
   ]
  ],
  "callout": "Every figure is simulator data, a kit image, or computed from a real building's floor plan."
 },
 {
  "id": "decmap",
  "src": "this deck; docs/DISCOVERY_WALKTHROUGH.md, docs/SOURCES_AND_FACTS.md",
  "notes": "Each area has a decision table with the reason and the alternative or evidence.",
  "kind": "table",
  "eyebrow": "Every decision, and where it is explained",
  "title": "The main decisions, by area",
  "rows": [
   [
    "Area",
    "Main decisions",
    "Part"
   ],
   [
    "Data generation",
    "made beforehand with the kit's own tools, outside this pipeline (not yet run) · all 5 controllers · results so far: the six shipped buildings",
    "2"
   ],
   [
    "Dataset",
    "recover commands by replay · use only what the policy can see · heat flows as inputs · split whole episodes into validation and three test tiers, computed from the data",
    "2"
   ],
   [
    "Physics",
    "derive from one cell to rooms · backward Euler like the simulator · one linear state-space model per building",
    "4"
   ],
   [
    "Model form",
    "state-space form · no backpropagation, no fitting (enforced in code) · universal constants · history formulas",
    "6"
   ],
   [
    "Scoring",
    "physics checks · forecast horizons 1/3/6 h · test-time inputs · five objectives (Pareto) · a price per formula size",
    "7"
   ],
   [
    "Search",
    "Claude API only, generation 0 included · islands · mutation, crossover, explore · analysis by place, time and term · retries and resume",
    "7"
   ],
   [
    "Runtime",
    "same timing live as offline · parity check · safety layer and fallback",
    "9"
   ],
   [
    "Controllers",
    "build on pid_water · cost = the reward · occupancy forecast · replanning",
    "10"
   ],
   [
    "Submission",
    "one notebook · bundled code · rehearsal · readiness gate",
    "11"
   ]
  ],
  "colw": [
   0.18,
   0.72,
   0.1
  ],
  "size": 24,
  "font": 11
 },
 {
  "kind": "section",
  "id": "s1",
  "num": "1",
  "title": "The problem",
  "question": "What are we asked to do, what must we submit, and how is it scored?",
  "notes": ""
 },
 {
  "kind": "stats",
  "id": "why",
  "eyebrow": "Why it matters",
  "title": "Buildings use a large share of the world’s energy, and HVAC is the biggest part",
  "stats": [
   [
    "~40%",
    "of U.S. carbon emissions come from buildings"
   ],
   [
    "40–60%",
    "of a building’s emissions come from heating, ventilation and air conditioning"
   ],
   [
    "~15%",
    "of global energy consumption goes to HVAC"
   ]
  ],
  "callout": "Most building controllers only work for the building they were developed for. The competition asks for one policy that transfers.",
  "src": "[website] competition home page; [Kaggle] overview",
  "notes": "Winning solutions may get to test on three real buildings (Kaggle overview)."
 },
 {
  "kind": "image",
  "id": "plant",
  "eyebrow": "The equipment",
  "title": "What is being controlled: air handlers, VAV boxes, a hot-water system",
  "img": "kit_device_locations.png",
  "side": "right",
  "points": [
   "Air handlers (AHUs) mix outside and return air, cool it, and send it to at most 10 zones each",
   "Each zone (a room) has a variable-air-volume (VAV) box: a damper sets its share of air, a reheat valve warms it",
   "One hot-water system: a boiler and a pump feed the reheat coils",
   "Meters record the energy and carbon the reward charges for"
  ],
  "caption": "Image: starter kit images/device_locations.png",
  "src": "[website] competition.html, Building components; [kit] image",
  "notes": "The six shipped buildings have 3 to 9 AHUs and 30 to 85 rooms (kit configs; the website says 28 to 85). Generated buildings can be much larger (part 2)."
 },
 {
  "kind": "table",
  "id": "evaluation",
  "eyebrow": "The task",
  "title": "One policy, run unchanged on buildings and weather it has never seen",
  "rows": [
   [
    "",
    ""
   ],
   [
    "Development set",
    "6 public buildings × 5 weather windows = 30 scenarios (shipped with the kit)"
   ],
   [
    "Final evaluation",
    "6 unseen buildings × 5 unseen weather stations = 30 scenarios, same generator and ranges"
   ],
   [
    "One scenario",
    "8,640 steps of 5 minutes = 30 days; every step is scored, including the first"
   ],
   [
    "Each step",
    "the policy gets the observation vector; returns discrete on/off commands and continuous setpoints"
   ],
   [
    "Determinism",
    "same policy + same scenario = same return, bit for bit; each scenario is run once"
   ]
  ],
  "colw": [
   0.24,
   0.76
  ],
  "header": False,
  "callout": "The public leaderboard uses about 33% of the test data; the final ranking uses the other 67%.",
  "src": "[website] scoring.html, competition.html; [Kaggle] overview, data, leaderboard",
  "notes": "Test buildings and weather are never released; the metadata arrives only at run time inside the sandbox."
 },
 {
  "id": "submission",
  "src": "[website] submission.html (interface, guidelines); [Kaggle] overview (code requirements, timeline), rules (submission limits)",
  "notes": "A fresh policy is built for every scenario: state carries between steps but never between scenarios. A NaN passes the bounds check and corrupts the rest of the episode. Winners license their code under CC-BY 4.0.",
  "kind": "code",
  "eyebrow": "The submission",
  "title": "What we submit: one notebook with one Python class",
  "code": "from kaggle_evaluation import smart_buildings_inference_server as sbi\nimport numpy as np\n\nclass TransferPolicy(sbi.Policy):\n    def __init__(self, building_meta_data, action_spec):\n        # called once per scenario, with this building's\n        # floor plan, zone map, zone->AHU map, schedule,\n        # and where each command sits in the action vector\n        super().__init__(building_meta_data, action_spec)\n        self.K1 = building_meta_data[\"n_discrete_action\"]    # K + 1\n        self.C = building_meta_data[\"n_continuous_action\"]   # 3K + 2 + 2V\n\n    def get_action(self, obs):\n        # called every 5 simulated minutes, 8,640 times\n        return {\"continuous_action\": np.zeros(self.C, np.float32),\n                \"discrete_action\":   np.zeros(self.K1, np.int32)}\n\nsbi.SmartBuildingsInferenceServer(TransferPolicy).run()",
  "points": [
   "K = air handlers, V = rooms; both change from building to building, so sizes are read from the metadata",
   "Setpoints in [−1, 1], switches in {0, 1}; out-of-range is rejected, and a NaN ruins the episode",
   "No scenario list, no loop: Kaggle's gateway owns the simulator and calls get_action",
   "No internet; model files come from an attached Kaggle Dataset. 5 submissions a day, 2 final, deadline Jan 15, 2027",
   "Forbidden: hard-coding positions, or referring to held-out buildings or weather"
  ]
 },
 {
  "kind": "image",
  "id": "obsgroups",
  "eyebrow": "The interface · what the policy sees",
  "title": "What the policy sees each step: 6 numbers per room, 15 per air handler, 58 for the building",
  "img": "obs_groups.png",
  "side": "right",
  "points": [
   "Per room: its temperature, its damper and reheat commands, two setpoints, an airflow setpoint",
   "Per air handler: supply temperature, pressure, flows, fan, and a copy of the outside temperature",
   "Building: boiler (7), clock (48), comfort flags (2), people in the building (1)"
  ],
  "caption": "[sim data] first 3 days of an episode: building_train_2, Atlanta, pid_water, as delivered to the policy (normalised)",
  "src": "[sim data] outputs/roomdata_building_train_2_atlanta_3_pidwater.npz",
  "notes": "For 85 rooms and 9 AHUs: 703 numbers per step."
 },
 {
  "kind": "table",
  "id": "obsroles",
  "eyebrow": "The interface · which fields matter",
  "title": "Most of those numbers echo our own commands; the new information is room temperatures and eight more",
  "rows": [
   [
    "Field (count)",
    "What it is",
    "Used by the model?"
   ],
   [
    "room temperature (V)",
    "what we forecast",
    "yes: the target"
   ],
   [
    "damper, reheat command (2V)",
    "our own last commands",
    "as inputs (we know them anyway)"
   ],
   [
    "heating, cooling setpoint (2V)",
    "the comfort band: 294–297 K by day, 289–298 K at night",
    "no: only the score uses it"
   ],
   [
    "other air-handler fields (14 each)",
    "copies of our commands, or follow from them",
    "no: use the commands"
   ],
   [
    "outside air temperature (K copies)",
    "the weather now",
    "yes"
   ],
   [
    "boiler water temperature",
    "changes slowly; the one plant state we observe",
    "yes"
   ],
   [
    "people, comfort flag, clock",
    "building head count; schedule; time",
    "yes: people, comfort flag, time of day, weekday"
   ]
  ],
  "colw": [
   0.3,
   0.42,
   0.28
  ],
  "src": "[audit] outputs/building_train_8_observation_annotated.csv (each role confirmed in simulator runs)",
  "notes": "So the dynamics need V room temperatures + 8 numbers: outside air, boiler water, occupants, comfort flag, hour sin/cos, day sin/cos.",
  "size": 23,
  "callout": None
 },
 {
  "kind": "table",
  "id": "actions",
  "eyebrow": "The interface · what the policy sends",
  "title": "What the policy controls, and what the plant actually does with it",
  "rows": [
   [
    "Command",
    "Per",
    "Range (native)",
    "What actually happens"
   ],
   [
    "run on/off",
    "AHU, hot-water system",
    "{0, 1}",
    "first AHU start takes effect one step late"
   ],
   [
    "supply-air temperature",
    "AHU",
    "285–305 K",
    "the coil cools to it; cannot heat above mixed air"
   ],
   [
    "static pressure",
    "AHU",
    "0–2,000 Pa",
    "floor 200 Pa while running"
   ],
   [
    "recirculation",
    "AHU",
    "0–1",
    "capped at 0.7"
   ],
   [
    "supply-water temperature",
    "hot-water system",
    "310–350 K",
    "boiler ramps +2.5 / −0.5 K per step"
   ],
   [
    "pump differential pressure",
    "hot-water system",
    "0–20",
    "0 when off"
   ],
   [
    "damper",
    "zone",
    "0–1",
    "at least 0.1 open"
   ],
   [
    "reheat valve",
    "zone",
    "0–1",
    "flow 0.108 √Δp; discharge ≤ 308.15 K"
   ]
  ],
  "colw": [
   0.24,
   0.2,
   0.18,
   0.38
  ],
  "callout": "All continuous commands arrive normalised to [−1, 1]; 0 is the middle of the range, not off.",
  "src": "[audit] outputs/building_train_8_action_annotated.csv, STATE_SPACE.md §6; [website] ventilation floors; [Kaggle] data tab",
  "notes": "Indices must be read from the metadata."
 },
 {
  "id": "budget",
  "src": "[website] submission.html ('Kaggle re-runs the same notebook against the hidden test scenarios'; time limits); [Kaggle] code requirements (9 h); [sim data] kit summary.json seconds_per_step, 12 runs",
  "notes": "Whether the gateway runs scenarios in parallel is not documented.",
  "kind": "flow",
  "eyebrow": "The clock",
  "title": "Why about 0.125 s per step: one 9-hour run must play all 30 hidden scenarios",
  "steps": [
   [
    "You press Submit",
    "Kaggle re-runs the same notebook once, offline"
   ],
   [
    "The gateway plays 30 scenarios",
    "6 hidden buildings × 5 hidden stations, a fresh policy for each"
   ],
   [
    "8,640 steps each",
    "30 days of 5-minute steps: 259,200 calls to get_action"
   ],
   [
    "All in one 9-hour session",
    "32,400 s ÷ 259,200 ≈ 0.125 s per step"
   ]
  ],
  "points": [
   "When we run the notebook ourselves it plays one public example; only the scored re-run plays all 30",
   "The 0.125 s includes the simulator: on our Mac it alone takes 0.025–0.057 s per step, leaving roughly 0.07–0.10 s for the policy",
   "Whether the gateway runs scenarios in parallel is not documented, so we plan for one after another",
   "The per-call caps (15 min for the first step, 60 s for later ones) are far looser: with 259,200 calls, the 9 h total is what binds"
  ]
 },
 {
  "id": "scoring",
  "src": "[website] scoring.html (metric); [kit] smart_control/reward/setpoint_energy_carbon_regret.py, config.gin (weights, reference loads)",
  "notes": "The comfort band is 294–297 K in comfort hours and 289–298 K in eco hours (kit config.gin). Comfort is scored only against the active band.",
  "kind": "eq",
  "eyebrow": "The score",
  "title": "The score is the average loss per 5-minute step, over the 30 scenarios",
  "eq": "loss per step  =  0.6 × comfort  +  0.2 × energy  +  0.2 × carbon\nscore  =  mean over scenarios of (loss summed over 8,640 steps ÷ 8,640)",
  "points": [
   "Comfort: the share of occupants' productivity lost. 0 inside the band; a room d K outside loses d/5 up to 5 K, then 1 + ((d − 5)/5)², with no cap; rooms are weighted by occupants, and an empty building costs 0",
   "Energy: this step's electricity and gas cost ÷ the cost at a reference load of 80 kW electric + 205 kW gas per air handler; it can exceed 1",
   "Carbon: the same ratio for emissions",
   "0.6 / 0.2 / 0.2 are the reward weights fixed in the kit's config. The kit's reward is −loss, so it is never positive; lower score is better and 0 (perfect comfort for free) is unreachable"
  ]
 },
 {
  "id": "leaderboard",
  "src": "[kit] baseline_returns.json (score = −return / 8,640, averaged), baselines/*.py docstrings; [Kaggle] leaderboard, read Nov 19, 2026",
  "notes": "845 teams, 9,646 submissions at the time of reading.",
  "kind": "table",
  "eyebrow": "Where the bar is",
  "title": "The kit's best controller scores 0.054; the leaderboard leader scores 0.0045",
  "rows": [
   [
    "Controller",
    "What it does",
    "Score"
   ],
   [
    "bang_bang",
    "each air handler blasts full heat or cold whenever its rooms leave the band",
    "0.225"
   ],
   [
    "pid",
    "each air handler turns its supply temperature up or down in proportion to the rooms' error",
    "0.067"
   ],
   [
    "fixed_setpoint",
    "pid, but one comfort band all day: no night or weekend setback",
    "0.066"
   ],
   [
    "pid_water (best baseline)",
    "pid on the worst room, plus free cooling, a modulating reheat valve and boiler water set by demand",
    "0.054"
   ],
   [
    "best of the four, picked per scenario afterwards",
    "an upper bound on what choosing among them could give",
    "0.047"
   ],
   [
    "Leaderboard #50 / #1 (Nov 19, 2026)",
    "other teams' policies, on the hidden test",
    "≈ 0.0059 / 0.0045"
   ]
  ],
  "colw": [
   0.27,
   0.58,
   0.15
  ],
  "size": 23,
  "callout": "Baselines are scored on the 30 public scenarios, the leaderboard on hidden ones, so the comparison is rough. Still, the leaders lose about a tenth of what pid_water loses per step: there is a lot of room above rule-based control."
 },
 {
  "kind": "statement",
  "id": "objective",
  "eyebrow": "The objective, in one sentence",
  "title": "One policy that, dropped into any unseen building, keeps rooms comfortable at the lowest energy and carbon",
  "sub": "using only that building’s metadata and what it measures while it runs, within about 0.125 s per step",
  "src": "[website] [Kaggle] task statement; [measured] time budget",
  "notes": "The rest of the deck works toward this."
 },
 {
  "kind": "section",
  "id": "s2",
  "num": "2",
  "title": "The data",
  "question": "What does the kit ship (six buildings, and the generator that made them), what is a weather window, and how would we make more data?",
  "notes": ""
 },
 {
  "kind": "table",
  "id": "shipped",
  "eyebrow": "What the kit ships · buildings",
  "title": "The six buildings we are given are for development; the test uses six other buildings",
  "rows": [
   [
    "Building",
    "Seed",
    "Floors",
    "Footprint (ft)",
    "Rooms",
    "AHUs",
    "People / room",
    "Wall: conductivity, density"
   ],
   [
    "building_train_1",
    "0",
    "3",
    "373 × 188",
    "39",
    "4",
    "2",
    "0.21, 1,967"
   ],
   [
    "building_train_2",
    "1",
    "2",
    "420 × 301",
    "85",
    "9",
    "2",
    "0.34, 578"
   ],
   [
    "building_train_4",
    "3001",
    "3",
    "430 × 211",
    "54",
    "6",
    "1",
    "0.18, 475"
   ],
   [
    "building_train_5",
    "4007",
    "4",
    "340 × 199",
    "79",
    "8",
    "2",
    "0.30, 1,187"
   ],
   [
    "building_train_7",
    "8",
    "2",
    "325 × 218",
    "36",
    "4",
    "1",
    "0.55, 415"
   ],
   [
    "building_train_8",
    "9008",
    "2",
    "318 × 198",
    "30",
    "3",
    "2",
    "0.27, 1,139"
   ]
  ],
  "colw": [
   0.17,
   0.08,
   0.08,
   0.13,
   0.08,
   0.08,
   0.13,
   0.25
  ],
  "callout": "Six buildings ship pre-built, as examples; the kit's generator makes as many more as we like (next slides). Final scores come only from 6 other buildings at 5 other weather stations, never released.",
  "src": "[kit] smart_buildings/buildings/building_train_*.yaml, training_scenarios/*/config.gin; [website] scoring.html; [Kaggle] data page (read Nov 23, 2026)",
  "notes": "Footprint units as written in the YAML. Facade k in W/mK, density in kg/m³."
 },
 {
  "id": "floorplans",
  "kind": "figure",
  "eyebrow": "What the kit ships · buildings, drawn",
  "title": "The six pre-built buildings at one scale: 30 to 85 rooms",
  "img": "buildings.png",
  "h": 520,
  "callout": "Floor 1 of each, from the kit's own floor-plan files. Any new seed gives the generator another building from the same family.",
  "src": "[kit] training_scenarios/*/zone_map.npy; room types: outputs/building_*_rooms.csv; make_figs11.py"
 },
 {
  "kind": "table",
  "id": "scenarios30",
  "eyebrow": "What the kit ships · scenarios",
  "title": "Each building is paired with one window from each of five stations",
  "rows": [
   [
    "Building",
    "Atlanta",
    "Denver",
    "Fairbanks",
    "Phoenix",
    "Singapore"
   ],
   [
    "building_train_1",
    "atlanta_2",
    "denver_7",
    "fairbanks_1",
    "phoenix_5",
    "singapore_8"
   ],
   [
    "building_train_2",
    "atlanta_3",
    "denver_8",
    "fairbanks_2",
    "phoenix_6",
    "singapore_1"
   ],
   [
    "building_train_4",
    "atlanta_5",
    "denver_2",
    "fairbanks_4",
    "phoenix_8",
    "singapore_3"
   ],
   [
    "building_train_5",
    "atlanta_6",
    "denver_3",
    "fairbanks_5",
    "phoenix_1",
    "singapore_4"
   ],
   [
    "building_train_7",
    "atlanta_8",
    "denver_5",
    "fairbanks_7",
    "phoenix_3",
    "singapore_6"
   ],
   [
    "building_train_8",
    "atlanta_1",
    "denver_6",
    "fairbanks_8",
    "phoenix_4",
    "singapore_7"
   ]
  ],
  "colw": [
   0.2,
   0.16,
   0.16,
   0.16,
   0.16,
   0.16
  ],
  "callout": "6 buildings × 5 stations = 30 scenarios; no window is used twice. Minneapolis ships with the kit but is in no scenario. The hidden test uses 6 other buildings and 5 other stations.",
  "src": "[kit] smart_buildings/training_set.yaml",
  "notes": "Running run_scenario_generator with explicit --buildings/--weathers gives the cross product instead."
 },
 {
  "kind": "table",
  "id": "windows",
  "eyebrow": "What the kit ships · weather",
  "title": "atlanta_1 and atlanta_3: the same city, different 30-day stretches of 2023",
  "rows": [
   [
    "Station (wind film, W/m²K)",
    "_1",
    "_2",
    "_3",
    "_4",
    "_5",
    "_6",
    "_7",
    "_8"
   ],
   [
    "Atlanta (19.1)",
    "Jan 22",
    "Mar 5",
    "Apr 16",
    "May 28",
    "Jul 9",
    "Aug 20",
    "Oct 1",
    "Nov 12"
   ],
   [
    "Denver (26.7)",
    "Jan 15",
    "Feb 26",
    "Apr 9",
    "May 21",
    "Jul 2",
    "Aug 13",
    "Sep 24",
    "Nov 5"
   ],
   [
    "Fairbanks (8.8)",
    "Jan 1",
    "Feb 12",
    "Mar 26",
    "May 7",
    "Jun 18",
    "Jul 30",
    "Sep 10",
    "Oct 22"
   ],
   [
    "Minneapolis (21.5)",
    "Jan 8",
    "Feb 19",
    "Apr 2",
    "May 14",
    "Jun 25",
    "Aug 6",
    "Sep 17",
    "Oct 29"
   ],
   [
    "Phoenix (19.5)",
    "Jan 29",
    "Mar 12",
    "Apr 23",
    "Jun 4",
    "Jul 16",
    "Aug 27",
    "Oct 8",
    "Nov 19"
   ],
   [
    "Singapore (17.6)",
    "Feb 5",
    "Mar 19",
    "Apr 30",
    "Jun 11",
    "Jul 23",
    "Sep 3",
    "Oct 15",
    "Nov 26"
   ]
  ],
  "colw": [
   0.24,
   0.095,
   0.095,
   0.095,
   0.095,
   0.095,
   0.095,
   0.095,
   0.095
  ],
  "callout": "A window = 30 days of recorded hourly weather from 06:00 on that date. The weather files hold all of 2023; the kit fixes 48 windows (6 × 8), and only the 30 in shipped scenarios have ever been simulated. Wind film = how easily an outside wall loses heat at that station.",
  "src": "[kit] smart_buildings/weather/<station>_<n>.yaml (start_timestamp, convection_coefficient) and weather/csv",
  "notes": "The window's start date overrides the building YAML's start_timestamp (scenario_generator: episode_start = weather.start_timestamp or sim.start_timestamp)."
 },
 {
  "id": "weather30",
  "kind": "figure",
  "eyebrow": "What the kit ships · weather, drawn",
  "title": "Only 30 episodes ever simulated: −42 °C to +46 °C",
  "img": "weather.png",
  "h": 520,
  "callout": "Orange: the 30 shipped episodes, all we have. Grey: the rest of 2023, in the weather file but in no run. The test uses 5 other stations, never released.",
  "src": "[kit] weather/csv/*.csv (2023, hourly), weather/<scenario>.yaml start dates, training_set.yaml; [website] scoring.html; make_figs11.py"
 },
 {
  "id": "seed",
  "src": "[kit] run_floor_generator.py (--start-seed, --count, --seeds; build_config, draw_materials); [website] scoring.html ('the generator supports unlimited scenario creation')",
  "notes": "",
  "kind": "cards",
  "eyebrow": "What the kit ships · the generator",
  "title": "Give the generator a number (the 'seed') and it builds a whole building",
  "cards": [
   [
    "You give it",
    "One integer, for example 500.\n\nCommand: run_floor_generator --start-seed 500 --count 100 builds buildings 500, 501, … 599.",
    "input",
    "accent3"
   ],
   [
    "It builds",
    "A floor plan, rooms and doors, air handlers, people per room, and the wall, slab and partition materials.",
    "output",
    "accent2"
   ],
   [
    "Same number, same building",
    "The generator is deterministic: a number always gives the same layout. Any number not used before gives a building nobody has seen.",
    "why it matters",
    "accent1"
   ]
  ],
  "callout": "The six shipped buildings list the numbers they were made from (0, 1, 3001, 4007, 8, 9008), but they are smaller than the generator's footprint ranges, so running it with those numbers gives different, larger buildings. Read the six as examples from the same family."
 },
 {
  "id": "generator",
  "src": "[kit] run_floor_generator.py DEFAULT_RANGES, smart_control/utils/materials.py PROPERTIES, weather/*.yaml",
  "notes": "16 building parameters + 3 weather parameters vary; everything else is constant across buildings (audit, STATE_SPACE.md §12).",
  "kind": "cards",
  "eyebrow": "What the generator varies",
  "title": "What the generator varies: the ranges it draws every new building from",
  "cards": [
   [
    "Shape: set by the seed",
    "Floors 2–4\nFootprint 400–700 × 300–500 ft\nSmallest room side 30–60 ft\nAir handlers 3–7 (more if needed: at most 10 rooms each)\n1–2 people per room",
    "visible",
    "accent2"
   ],
   [
    "Materials: drawn per batch",
    "Wall conductivity 0.15–0.55 W/mK\nWall density 350–2,400 kg/m³\nSlab and furniture 10–40 kg/m³\nInside walls 0.45–0.60 W/mK",
    "hidden",
    "accent1"
   ],
   [
    "Weather: chosen per run",
    "48 fixed 30-day windows (6 stations × 8); 30 of them in shipped scenarios\nThe files hold all of 2023, hourly\nWind film 8.8–26.7 W/m²K, one value per station",
    "chosen",
    "accent3"
   ]
  ],
  "callout": "Footprints: the generator draws 400–700 × 300–500 ft; the six shipped buildings are 318–430 × 188–301 ft. The organisers say the test uses the same generator and ranges as the shipped buildings, so which size range the test uses is not certain."
 },
 {
  "kind": "image",
  "id": "layouts",
  "eyebrow": "Every building is different",
  "title": "Different layouts, room counts and air-handler groupings",
  "img": "kit_floorplan_variability.png",
  "side": "right",
  "points": [
   "Three of the six: 30, 39 and 85 rooms; 3, 4 and 9 air handlers",
   "Rooms are not a few standard types: each has its own area, outside wall, shared walls and doors",
   "So the model must be written for any floor plan, not a fixed list of rooms",
   "(The figure labels train_8 “28 zones”; its config has 30 rooms)"
  ],
  "caption": "Image: starter kit images/floorplan_variability.png",
  "src": "[website] Figure 3; [kit] image",
  "notes": "Shapes vary continuously; the model has to take any floor plan."
 },
 {
  "id": "official",
  "kind": "table",
  "eyebrow": "Development and test buildings, from the organisers' pages",
  "title": "Which buildings are for development, which are for the test, and what we may generate",
  "rows": [
   [
    "Question",
    "Answer",
    "Source"
   ],
   [
    "Are the six shipped buildings the test set?",
    "No. They are the public development set: 6 buildings × 5 stations = 30 scenarios",
    "[website] scoring; [Kaggle] data"
   ],
   [
    "What is the test set?",
    "6 other buildings × 5 other weather stations, from the same generator and parameter ranges",
    "[website] scoring; [Kaggle] data"
   ],
   [
    "Do we ever see the test buildings or their seeds?",
    "No. Never released; the policy gets each one's metadata only at run time, inside Kaggle's sandbox",
    "[Kaggle] data"
   ],
   [
    "May we generate more buildings?",
    "Yes, as many buildings and scenarios as we like. We fit nothing, so they become validation and test data",
    "[Kaggle] data; [website] scoring"
   ],
   [
    "What do we do with the six shipped buildings?",
    "They are data like any other: a rule (next slides) decides which runs the search sees and which are held out",
    "[decision]; make_splits.py"
   ]
  ],
  "colw": [
   0.3,
   0.5,
   0.2
  ],
  "size": 23,
  "font": 11,
  "callout": "Whatever data exists (the six shipped buildings now, generated ones later) is split by one rule into a validation set the search sees and three test tiers it never sees.",
  "src": "[website] scoring.html, competition.html; [Kaggle] data tab, read Nov 23, 2026; [our code] discovery/splits.py, make_splits.py",
  "notes": "A limit: our hand-written models and the physics audit were developed while looking at the six shipped buildings, so for those models the shipped test tiers are not a perfectly clean test."
 },
 {
  "id": "change",
  "src": "[kit] README and run_floor_generator.py arguments (the kit's commands take seeds, weather names, controllers, steps); [audit] STATE_SPACE.md §12 (interface constants)",
  "notes": "The shipped buildings' footprints (318–430 × 188–301 ft) are below the generator's ranges (400–700 × 300–500 ft), so generated buildings can be larger.",
  "kind": "compare",
  "eyebrow": "Making more data, with the kit's own commands",
  "title": "What we vary, and what we never touch",
  "left": [
   "We vary",
   [
    "Which building: run the generator with a new number",
    "Which weather: all 48 windows for the shipped buildings, one per station for new ones",
    "Which controller drives it: the 5 shipped ones",
    "How long: full 30-day runs, as scored"
   ]
  ],
  "right": [
   "We never touch",
   [
    "The generator's ranges: they define where the hidden buildings come from",
    "A building's file: editing it moves the building off that distribution",
    "Action ranges, normalisers, reward weights: they fix what the gateway sends and scores",
    "The kit's code: we only call its commands"
   ]
  ],
  "callout": "So every generated building would be one the hidden test could have drawn. Not generated yet: every result in this deck uses only the six shipped buildings."
 },
 {
  "id": "datarecipe",
  "src": "[kit] run_floor_generator.py, run_scenario_generator.py (naming --buildings takes the cross product), evaluate.py; GUIDE.md A.4; [measured] 0.025–0.057 s per step",
  "notes": "Run with the kit's own commands on the data machine; this repository generates and simulates nothing.",
  "kind": "flow",
  "eyebrow": "How the data is made (on the data machine, before this pipeline)",
  "title": "The plan: every shipped building in every window, then new buildings",
  "steps": [
   [
    "Stage 1: 6 shipped buildings",
    "× all 48 windows (6 stations × 8) = 288 scenarios"
   ],
   [
    "× 5 controllers",
    "all_off, bang_bang, fixed_setpoint, pid, pid_water"
   ],
   [
    "= 1,440 runs",
    "30 days each (8,640 steps); about 100–200 CPU-hours"
   ],
   [
    "Stage 2: new buildings",
    "kit generator from seed 500; 20 to start, one window per station each: 600 runs"
   ]
  ],
  "points": [
   "Made beforehand with the starter kit's own commands; this pipeline only reads the finished runs",
   "Every run becomes one episode: one building, one 30-day weather window, one controller",
   "Stage 1 gives every building every season at every station; stage 2 adds new floor plans, which is what the hidden test varies",
   "The split (next slides) is computed from whatever episodes exist. Not generated yet: every result in this deck uses the 30 shipped scenarios"
  ]
 },
 {
  "id": "decdatagen",
  "src": "[kit] run_floor_generator.py, run_scenario_generator.py, evaluate.py, README; [our code] run_pipeline.sh (reads the runs)",
  "notes": "",
  "kind": "table",
  "eyebrow": "Decisions · data generation",
  "title": "How more data will be made with the kit, and why",
  "rows": [
   [
    "Question",
    "Our choice",
    "Why"
   ],
   [
    "Which tools?",
    "only the kit's own commands",
    "the data must come from the simulator the hidden test uses"
   ],
   [
    "How to get new buildings?",
    "run the kit's generator with new numbers",
    "it builds whole buildings from the same ranges as the hidden test"
   ],
   [
    "How to split?",
    "no training split; validation + easy / medium / hard test tiers, computed from the data",
    "nothing is fitted, so nothing needs training data; the tiers measure how far the formulas generalize"
   ],
   [
    "The six shipped buildings?",
    "split by the same rule as any other data",
    "no special case: the rule decides from the data"
   ],
   [
    "Which weather?",
    "shipped buildings: all 48 windows; new buildings: one window per station, rotated",
    "every season at every station; the split then holds out the farthest station (Wasserstein)"
   ],
   [
    "Which controllers?",
    "all five shipped ones",
    "rooms must be seen drifting, swinging and held steady"
   ],
   [
    "How long?",
    "30 days per run (8,640 steps)",
    "the scored length; every phase of the month"
   ],
   [
    "Rejected seeds?",
    "ask the kit's generator for ~30% extra seeds",
    "the kit rejects about 1 seed in 5 (a room under 20 m²)"
   ]
  ],
  "colw": [
   0.24,
   0.36,
   0.4
  ],
  "size": 23,
  "callout": "Not generated yet: every result in this deck uses only the six shipped buildings."
 },
 {
  "id": "decdataset",
  "src": "[our code] discovery/starterkit_data.py, build_dataset.py, runtime/inputs.py; [audit] STATE_SPACE.md",
  "notes": "",
  "kind": "table",
  "eyebrow": "Decisions · dataset",
  "title": "How simulator runs become episodes, and why",
  "rows": [
   [
    "Question",
    "Our choice",
    "Why"
   ],
   [
    "The kit's logs lack per-room commands",
    "re-run the same controller on the logged observations and record its commands",
    "it is deterministic; checked: airflow matches the log to 0.02%"
   ],
   [
    "When is a temperature seen?",
    "the observation at step t shows step t − 1",
    "use exactly what the policy sees, live and offline"
   ],
   [
    "Occupancy",
    "the head count the policy sees (two steps behind the log)",
    "only what is observable may be used"
   ],
   [
    "Model inputs",
    "heat flows computed from commands with the kit's equipment equations",
    "the command-to-heat part is known and nonlinear"
   ],
   [
    "Boiler water",
    "computed from the setpoint with the boiler's ramp limits",
    "the one equipment state the policy can predict"
   ],
   [
    "Geometry",
    "from floor plan, zone map and air-handler map only",
    "exactly what a new building's metadata gives"
   ]
  ],
  "colw": [
   0.24,
   0.42,
   0.34
  ],
  "size": 22
 },
 {
  "id": "splits",
  "kind": "table",
  "eyebrow": "How the data is split · no training set",
  "title": "The search sees only validation runs; three test tiers stay hidden until the end",
  "rows": [
   [
    "Split",
    "What is held out",
    "On the six shipped buildings (computed)",
    "What it tests"
   ],
   [
    "Validation",
    "nothing: every candidate is scored here and every analysis comes from here",
    "buildings 1, 4, 5, 7 at Atlanta, Denver, Phoenix, Singapore: 12 scenarios",
    "what the search optimizes"
   ],
   [
    "Test · easy",
    "the weather window: seen building, seen station, different 30 days",
    "1/atlanta_2, 4/denver_2, 5/phoenix_1, 7/singapore_6",
    "a new month in a known building"
   ],
   [
    "Test · medium",
    "the building",
    "buildings 2 and 8 at the four seen stations: 8 scenarios",
    "a building never scored on"
   ],
   [
    "Test · hard",
    "the building and the climate",
    "buildings 2 and 8 at Fairbanks: 2 scenarios",
    "a new building in a new climate"
   ],
   [
    "Extra",
    "the climate only",
    "buildings 1, 4, 5, 7 at Fairbanks: 4 scenarios (kept out so Fairbanks stays unseen)",
    "a diagnostic"
   ]
  ],
  "colw": [
   0.13,
   0.3,
   0.35,
   0.22
  ],
  "size": 21,
  "font": 10.5,
  "callout": "Whole episodes only: both controllers' runs of a scenario always go to the same split, and a run stops if any episode or scenario would be in two splits. No training split: nothing is fitted.",
  "src": "[our code] discovery/splits.py (assign, split_from_episodes, make_split), make_splits.py; [measured] make_splits.py --from-scenarios --only-shipped (outputs/split_shipped.json); splits.check_whole_episodes",
  "notes": "Easy windows are chosen so that their building and their station both stay in validation."
 },
 {
  "id": "whichheld",
  "kind": "eq",
  "eyebrow": "How the held-out station and buildings are chosen",
  "title": "Nothing is hard-coded: the farthest climate and the most unusual floor plans are held out",
  "eq": "W₁(station, rest) = ∫ | F_station(x) − F_rest(x) | dx\n(F = share of hours colder than x: the average distance each hourly temperature must move to match the others)",
  "eqsize": 22,
  "eqpt": 11,
  "points": [
   "Wasserstein-1 distance of each station's hourly outside temperatures to the other stations pooled (K): Fairbanks 21.9, Singapore 14.6, Phoenix 11.9, Denver 7.5, Atlanta 5.1. The farthest, Fairbanks, is the hard station",
   "Buildings: four numbers from the floor plan (floor area, rooms, mean room area, outside wall per floor area), standardized; the farthest from the average and then from each other are held out: building 2 (2.9 standard deviations), then building 8 (2.0)",
   "One building in four is held out (rounded up); one validation window in four becomes an easy test",
   "The same rule runs on whatever data exists. On the 12 runs we have locally it again picks Fairbanks and buildings 2 and 8"
  ],
  "src": "[our code] splits.station_distances, atypical_buildings, split_from_episodes; [kit] weather/csv/*.csv, scenario config.yaml windows; [measured] make_splits.py output (_why)"
 },
 {
  "kind": "section",
  "id": "s3",
  "num": "3",
  "title": "What the data look like",
  "question": "What happens to every room over 30 days, and where does the score come from?",
  "notes": ""
 },
 {
  "kind": "image",
  "id": "climates",
  "eyebrow": "The runs in this part",
  "title": "The 12 runs we simulated: six buildings, one of their 30-day windows each, −40 °C to 30 °C",
  "img": "ep30_weather.png",
  "side": "right",
  "points": [
   "Each line is one 30-day window: fairbanks_1 (Jan), atlanta_3 (Apr), denver_2 (Feb), phoenix_1 (Jan), singapore_6 (Oct), atlanta_1 (Jan)",
   "Each building is run twice in that window: with the best baseline (pid_water) and with the HVAC off: 12 runs in all",
   "These 12 runs are the data behind parts 3, 4 and 8"
  ],
  "caption": "[sim data] outside air in the six 30-day windows; kit evaluate.py, 8,640 steps",
  "src": "[kit] weather/*.yaml, weather/csv; [sim data] kit evaluate.py, 8,640 steps",
  "notes": "Weather is the scarce resource: buildings are unlimited, weather is 48 windows."
 },
 {
  "kind": "image",
  "id": "profiles",
  "eyebrow": "What a 30-day episode looks like",
  "title": "30 days, six buildings: rooms are held in the band on working days and drift at night and on weekends",
  "img": "ep30_profiles.png",
  "side": "right",
  "points": [
   "Line = middle room of its kind; shading = 80% of rooms; green = comfort band (occupied hours); grey = weekends",
   "Cold climates (Fairbanks, Denver): rooms with outside walls fall far below the band overnight",
   "Mild climates (Atlanta, Singapore): interior rooms sit at or above the top of the band",
   "This, room by room for a month, is what the model must forecast"
  ],
  "caption": "[sim data] pid_water, 8,640 steps, 30-minute averages; one weather window per building",
  "src": "[sim data] kit evaluate.py runs; replayed with our starterkit_data.load_starterkit_episode",
  "notes": "pid_water scores ≈ 0.21 in fairbanks_1 and 0.020 in atlanta_3: the climate dominates the score."
 },
 {
  "kind": "image",
  "id": "week",
  "eyebrow": "Zooming into one week",
  "title": "One working week up close: 06:00, 19:00 and the weekend",
  "img": "ep30_week.png",
  "side": "right",
  "points": [
   "06:00: comfort hours start and pid_water pulls rooms into the band (top edge 23.85 °C)",
   "19:00: the system sets back; interior rooms warm, rooms with outside walls cool toward the outside air",
   "Weekend: nobody in, so rooms drift freely",
   "Each room drifts at its own speed: that speed is what a forecast must get right"
  ],
  "caption": "[sim data] building_train_2, Atlanta window 3 (April), pid_water, days 7–14, 30-minute averages",
  "src": "[sim data] kit evaluate.py",
  "notes": ""
 },
 {
  "kind": "image",
  "id": "control",
  "eyebrow": "Control matters",
  "title": "Why control matters: the same building with the HVAC off and with the best baseline",
  "img": "ep30_policies.png",
  "side": "right",
  "points": [
   "Atlanta, April, HVAC off: rooms reach 35–45 °C; with no air supplied, heat from people, lights and equipment stays in",
   "Fairbanks, January, HVAC off: rooms fall to about −15 °C; pid_water holds 15–20 °C at a large energy cost (score 0.214)",
   "The HVAC dominates room temperature, so a model must take the commands as inputs"
  ],
  "caption": "[sim data] line = median room; shading = middle 80% of rooms; green = comfort band; 8,640 steps",
  "src": "[sim data] kit evaluate.py",
  "notes": "Doing better than PID means anticipating, which needs a model that predicts hours ahead."
 },
 {
  "kind": "image",
  "id": "devcontrol",
  "eyebrow": "Where the loss comes from",
  "title": "Where the score comes from: comfort, except in the coldest climate",
  "img": "ep30_reward.png",
  "side": "right",
  "points": [
   "Left: each scenario's score split into its three terms. Comfort is the largest in five of six",
   "Fairbanks: heating cost (energy 0.055 + carbon 0.089) outweighs comfort (0.070)",
   "Right, Denver day by day: weekends cost almost nothing (nobody there to be uncomfortable); working days cost 0.04–0.11 per step"
  ],
  "caption": "[sim data] pid_water, kit reward_components.csv; score = mean loss per step over the episode",
  "src": "[website] reward definition; [sim data] kit evaluate.py",
  "notes": "Reported per scenario, per climate, worst case and per term."
 },
 {
  "kind": "image",
  "id": "comfortdev",
  "eyebrow": "Comfort deviation",
  "title": "Comfort losses: cold mornings in cold climates, slightly warm rooms everywhere else",
  "img": "ep30_comfort.png",
  "side": "right",
  "points": [
   "Too cold: Fairbanks rooms start the working day 3.8 K below the band and stay about 1.4 K below; Denver and Atlanta-January also start cold and recover slowly",
   "Too warm: everywhere except Fairbanks, rooms sit about 0.2–0.5 K above the band all day",
   "A controller that forecasts can pre-heat before 06:00 and stop short of overshooting: the gap a model-based controller targets"
  ],
  "caption": "[sim data] six 30-day pid_water episodes; band 294–297 K (20.85–23.85 °C); mean over rooms, not weighted by occupants",
  "src": "[sim data] kit evaluate.py; [kit] config.gin schedule",
  "notes": "The reward also weights by occupants; this plot is unweighted."
 },
 {
  "kind": "image",
  "id": "geometry",
  "eyebrow": "But the difference is predictable",
  "title": "How fast a room loses heat can be read from the floor plan",
  "img": "ep30_geometry.png",
  "side": "right",
  "points": [
   "Each dot: one room with an outside wall, HVAC off, first 12 hours",
   "x: outside-wall length per m² of floor, computed from the floor plan alone",
   "y: how much of the indoor–outdoor gap the room closes per hour",
   "Within a building the dots lie on a line (r = 0.79–0.97): more outside wall, faster loss",
   "Between buildings the slope differs about fivefold: wall materials, which the metadata does not give"
  ],
  "caption": "[sim data] HVAC off, six shipped buildings; static figure made by our script from the kit runs",
  "src": "[sim data] kit evaluate.py all_off; geometry from our episodes.geometry()",
  "notes": "All part 3 figures are static images produced by docs/deck_src/make_figs7.py from twelve 30-day kit runs (6 buildings × pid_water / HVAC off)."
 },
 {
  "kind": "section",
  "id": "s4",
  "num": "4",
  "title": "The physics, from one cell to a state-space model",
  "question": "How does heat move in the simulator, how do we write a whole building as one state-space model, and which numbers in it are unknown?",
  "notes": ""
 },
 {
  "id": "roadmap4",
  "kind": "flow",
  "eyebrow": "The plan for this part",
  "title": "Five steps from the simulator's grid to the model we forecast with",
  "steps": [
   [
    "1 Cells",
    "the simulator's world: 20 cm squares"
   ],
   [
    "2 Energy balance",
    "heat in − heat out = storage"
   ],
   [
    "3 Rooms",
    "lump the cells of a room into a few nodes"
   ],
   [
    "4 State space",
    "state, control, disturbance, output; every equation"
   ],
   [
    "5 Unknowns",
    "which numbers in A, B, E we do not know"
   ]
  ],
  "points": [
   "Each step is checked against the simulator's code (our audit) or its data (part 3)",
   "A real room of building_train_8 is worked through with numbers",
   "The end point is the question the rest of the deck answers: how to fill the unknown numbers without fitting"
  ],
  "src": "[audit] STATE_SPACE.md §8, §14 (read from building.py, finite_difference_solver.py, vav.py)"
 },
 {
  "id": "cells",
  "kind": "image",
  "eyebrow": "Step 1 · the simulator's world",
  "title": "A building is a grid of 20 cm cells, 3 m tall",
  "img": "phys_cells.png",
  "side": "right",
  "points": [
   "Every cell is outside air, facade, partition wall, room air or door; floor_plan and zone_map in the metadata say which",
   "Each floor is a separate grid: no heat between floors, no roof, no ground, no sun",
   "Every 5 minutes the simulator updates every cell's temperature",
   "From the metadata alone we rebuild every cell exactly: 323 of 323 rooms and 0 mismatched cells over the six buildings"
  ],
  "caption": "[kit] building_train_8 floor_plan.npy and zone_map.npy; cell classes computed by our room_model.grid_geometry",
  "src": "[audit] STATE_SPACE.md §8.1, §14.1-14.2; [our code] room_model.py; [measured] outputs/metadata_only_check.json, room_model_validate.json"
 },
 {
  "id": "balance",
  "kind": "eq",
  "eyebrow": "Step 2 · conservation of energy",
  "title": "One cell: the heat that flows in is stored as a temperature rise",
  "eq": "C_i · dT_i/dt  =  Σ_j G_ij · (T_j − T_i)  +  Q_i",
  "points": [
   "C_i, heat capacity (J/K) = density × specific heat × volume. One air cell: 1,206 J/m³K × 0.2 × 0.2 × 3 m = 145 J/K",
   "G_ij, conductance (W/K) = conductivity × contact area ÷ distance (Fourier's law). One cell face is 0.2 m × 3 m = 0.6 m²",
   "Q_i, heat put in directly (W): supply air, reheat, people, lights and equipment",
   "Every term is linear in temperature: twice the temperature gap, twice the flow. The next slide checks this on data"
  ],
  "src": "[audit] STATE_SPACE.md §14.2; [kit] config.gin (interior_cv_density 1.2, heat_capacity 1005, control_volume_cm 20, floor_height_cm 300)"
 },
 {
  "kind": "image",
  "id": "linear",
  "eyebrow": "Checked on data · linear in temperature",
  "title": "Heat flows in proportion to the temperature gap: twice the gap, twice the flow",
  "img": "real_linear.png",
  "side": "right",
  "points": [
   "HVAC off, at night: each hour's temperature change against the indoor–outdoor gap",
   "One straight line per building (r = 0.71–1.00): Newton's law of cooling",
   "The slope differs by building because the walls differ; crosses (19:00, lights dimmed to 20%) are an extra input, not a break in the law"
  ],
  "caption": "[sim data] six 30-day HVAC-off episodes; mean room temperature, hourly changes at night",
  "src": "[sim data] kit evaluate.py all_off; [measured] straight-line fit per building",
  "notes": "The code says the solver is linear in temperatures; the data agree."
 },
 {
  "id": "layers",
  "kind": "table",
  "eyebrow": "Step 2 · walls are layers in series",
  "title": "Through a wall, resistances add: 1/G = Σ thickness ÷ (conductivity × area)",
  "rows": [
   [
    "Path, per 0.6 m² face",
    "Layers it crosses",
    "Conductance G (W/K)",
    "At mid-range materials"
   ],
   [
    "room → outside",
    "half an air cell + facade cell + outside film",
    "0.6 / (0.002 + 0.1/k_ext + 1/h)",
    "1.48 W/K  (U ≈ 2.5 W/m²K)"
   ],
   [
    "room → room through a wall",
    "half an air cell + partition cell + half an air cell",
    "0.6 / (0.004 + 0.2/k_iw)",
    "1.56 W/K"
   ],
   [
    "room → room through a door",
    "two air cells (a door is open air)",
    "0.6 / (2 × 0.2/50)",
    "75 W/K"
   ],
   [
    "room air → its slab and furniture",
    "a fixed coupling per m² of floor",
    "5 W/m²K × floor area",
    "297 W/K for a 59 m² room"
   ]
  ],
  "colw": [
   0.22,
   0.32,
   0.26,
   0.2
  ],
  "size": 24,
  "callout": "Room air has an effective conductivity of 50 W/mK (0.1 m / 50 = 0.002). A door face carries as much heat as about 48 faces of wall, so a 2-cell door ≈ 19 m of partition: doors dominate room-to-room exchange.",
  "src": "[audit] STATE_SPACE.md §14.2 (per-face formulas); mid-range k_ext 0.29, k_iw 0.525 W/mK, h 17.75 W/m²K (network seed)",
  "font": 11
 },
 {
  "id": "sources",
  "kind": "cards",
  "eyebrow": "Step 2 · the heat sources",
  "title": "Four ways heat enters a room, each computed from commands or known constants",
  "cards": [
   [
    "Supply air",
    "g · (T_supply − T)\n\ng = air density × specific heat × airflow, in W/K. At the box maximum of 2.0 m³/s, g = 2,412 W/K",
    "HVAC",
    "accent2"
   ],
   [
    "Reheat",
    "q, in W: hot water warms the supply air at the box; limited by airflow, water temperature and a 35 °C discharge cap",
    "HVAC",
    "accent2"
   ],
   [
    "Lights and equipment",
    "10 W per m² of floor during comfort hours, 20% of that otherwise",
    "known schedule",
    "accent3"
   ],
   [
    "People",
    "70 W per person in the room. Only the building total is observed (two steps late)",
    "measured",
    "accent1"
   ]
  ],
  "callout": "The commands turn into g, T_supply and q through the kit's own equipment equations (next slide); after that, everything is linear again.",
  "src": "[audit] STATE_SPACE.md §8.2 B, §14.2; [kit] config.gin (vav_max_air_flowrate 2.0), vav.py"
 },
 {
  "kind": "image",
  "id": "commands",
  "eyebrow": "Checked on data · nonlinear in commands",
  "title": "Commands do not turn into heat in a straight line",
  "img": "real_commands.png",
  "side": "right",
  "points": [
   "Air into a room = damper × √(duct pressure) × a constant; the damper never closes below 10%, and the flow is zero when the air handler is off",
   "Reheat heat is limited by the airflow, the hot-water temperature and a 35 °C cap on the air leaving the coil",
   "So we convert commands into heat flows with the kit's own equipment equations; in heat flows, the physics is linear again"
  ],
  "caption": "[sim data] commands replayed from 30-day pid_water episodes (flow matches the logged AHU flow to 0.02%)",
  "src": "[sim data] kit evaluate.py; [our code] starterkit_data.episode_inputs (the kit’s equipment laws)",
  "notes": "This conversion is hand-written from the simulator audit; the formulas receive its outputs."
 },
 {
  "id": "roomeq",
  "kind": "eq",
  "eyebrow": "Step 3 · from cells to rooms",
  "title": "Lump each room's cells into a few nodes: the same equations for every room",
  "eq": "C_i   dT_i/dt  = G_m,i (M_i − T_i) + κW_i (W_i − T_i) + Σ_j κP_ij (P_ij − T_i) + Σ_j κD_ij (T_j − T_i)\n                 + g_i (T_s − T_i) + q_i + Q_gain,i + 70·n_i                       room air\nC_m,i dM_i/dt = G_m,i (T_i − M_i)                                  slab and furniture\nC_W,i dW_i/dt = κW_i (T_i − W_i) + γ_i (T_out − W_i)                outside wall\nC_P   dP_ij/dt = κP_ij (T_i − P_ij) + κP_ji (T_j − P_ij)            wall shared with room j",
  "eqsize": 22,
  "eqpt": 11,
  "points": [
   "T_i: room air (measured). M_i, W_i, P_ij: slab, outside wall, shared walls (never measured)",
   "Doors are open air, so they link two rooms' air directly (κD)",
   "This is the cell equation summed over a room's cells: same physics, fewer temperatures"
  ],
  "src": "[audit] STATE_SPACE.md §14.3 (room equation); [our code] seeds_cf/seed_cf_network.py (the same network, as coefficient formulas)"
 },
 {
  "id": "coefftable",
  "kind": "table",
  "eyebrow": "Step 3 · every coefficient",
  "title": "Every number in the room equations = a count from the floor plan × a material property",
  "rows": [
   [
    "Coefficient",
    "Formula",
    "From the floor plan",
    "Material (hidden)"
   ],
   [
    "C_i, air",
    "1,206 J/m³K × 3 m × A_i",
    "floor area A_i",
    "none"
   ],
   [
    "C_m,i, G_m,i, slab",
    "ρc_mass × 3 m × A_i;  5 W/m²K × A_i",
    "floor area",
    "ρc_mass"
   ],
   [
    "κW_i, air → outside wall",
    "n_f × 0.6 / (0.002 + 0.05/k_ext), in series with a spreading resistance",
    "facade faces n_f",
    "k_ext"
   ],
   [
    "γ_i, outside wall → outside air",
    "n_f × 0.6 / (0.05/k_ext + 1/h)",
    "facade faces n_f",
    "k_ext, h"
   ],
   [
    "C_W,i, outside wall",
    "ρc_ext × facade cell volume",
    "facade cells",
    "ρc_ext"
   ],
   [
    "κP_ij, C_P, shared wall",
    "n × 0.6 / (0.002 + 0.1/k_iw) + spreading;  ρc_iw × wall volume",
    "shared wall faces n",
    "k_iw, ρc_iw"
   ],
   [
    "κD_ij, door",
    "n × 75 W/K, in series with spreading on both sides",
    "door faces n",
    "none"
   ]
  ],
  "colw": [
   0.22,
   0.43,
   0.17,
   0.18
  ],
  "size": 22,
  "callout": "Spreading resistance: heat entering through a short opening must spread through the room's air. It is a universal formula in area and opening length, the same in every building.",
  "src": "[audit] STATE_SPACE.md §14.3 (coefficient table, spreading resistance R_s(A, L)); [our code] seeds_cf/seed_cf_network.py"
 },
 {
  "id": "worked",
  "kind": "image",
  "eyebrow": "Step 3 · a worked example",
  "title": "Room 5 of building_train_8, with numbers",
  "img": "phys_room.png",
  "side": "right",
  "points": [
   "Air: C = 1,206 × 3 × 59.4 m² = 215 kJ/K. Slab: C_m = 3.56 MJ/K (17× the air), coupled by G_m = 297 W/K",
   "17 m of outside wall: air → wall 255 W/K, wall → outside 221 W/K; in series 118 W/K (≈ 2.3 W/m²K over 51 m²)",
   "To room 4: 9.8 m of wall (137 W/K to the wall node) and a 2-cell door (68 W/K after spreading, 150 before)",
   "To room 12: 5.2 m of wall (74 W/K to the wall node)"
  ],
  "caption": "[our code] network seed formulas at mid-range materials, applied to the real floor plan",
  "src": "[kit] building_train_8 floor_plan.npy, zone_map.npy; [our code] episodes.geometry, seeds_cf/seed_cf_network.py"
 },
 {
  "id": "timescales",
  "kind": "stats",
  "eyebrow": "Step 3 · what those numbers predict",
  "title": "Three time scales follow from capacity ÷ conductance",
  "stats": [
   [
    "89 s",
    "full supply air resets the room: 215 kJ/K ÷ 2,412 W/K, under one 5-minute step"
   ],
   [
    "12 min",
    "room air and slab even out: 215 kJ/K ÷ 297 W/K"
   ],
   [
    "≈ 9 h",
    "air + slab lose heat outside: 3.78 MJ/K ÷ 118 W/K"
   ]
  ],
  "callout": "Fast, then slow: the two speeds seen in the data on the next slide. (A rough single-room estimate: neighbours and the wall's own storage are ignored.)",
  "src": "[measured] from the worked example's numbers; [kit] vav_max_air_flowrate 2.0 m³/s"
 },
 {
  "kind": "image",
  "id": "fastslowreal",
  "eyebrow": "Checked on data · two speeds",
  "title": "Rooms have two speeds: the air in minutes, the building's mass in hours",
  "img": "real_fastslow.png",
  "side": "right",
  "points": [
   "Fairbanks, after the 19:00 setback: 1.9 K lost in 30 minutes, then only 0.14 K per hour (Denver, Phoenix, Atlanta-Jan alike)",
   "Fast drop: the air loses its little heat; slow tail: the slab slowly gives back what it stored. In April Atlanta and Singapore rooms warm instead",
   "The slab is never measured, so a model must infer its temperature from the recent history"
  ],
  "caption": "[sim data] six 30-day pid_water episodes; mean room temperature after every weekday 19:00 (21 evenings each)",
  "src": "[sim data] kit evaluate.py; [measured] averages over evenings",
  "notes": ""
 },
 {
  "id": "euler",
  "kind": "eq",
  "eyebrow": "Step 4 · stepping in time",
  "title": "Step forward with backward Euler, the same scheme the simulator uses",
  "eq": "C (x⁺ − x) / Δt        =  −L x⁺ + G_out T_out + g ⊙ (T_s − x⁺) + Q\n(C/Δt + L + diag g) x⁺  =  (C/Δt) x + G_out T_out + g ⊙ T_s + Q",
  "eqsize": 30,
  "eqpt": 15,
  "points": [
   "Heat flows are evaluated at the END of the step. x: all node temperatures; L: the conductance matrix (L_ii = Σ_j G_ij + G_out,i, L_ij = −G_ij); Δt = 300 s. The next slide writes this out node by node",
   "Why the end of the step: supply air can replace a room's heat in 89 s. A forward step multiplies the error by 1 − 300/89 = −2.4 each step and blows up",
   "For one room with only supply air this gives T⁺ = (T + a·T_s)/(1 + a), a = g·Δt/C = 3.4: always between T and T_s, stable for any airflow"
  ],
  "src": "[audit] STATE_SPACE.md §8.1, §14.3 (backward Euler at 300 s); [our code] dlib.LinearNetwork, dlib.end_of_step"
 },
 {
  "id": "sseqfull",
  "kind": "eq",
  "eyebrow": "Step 4 · every equation, discretised",
  "title": "One backward-Euler step, written out node by node (+ = end of the step)",
  "eq": "air i:    (C_i/Δt + G_m,i + κW_i + Σ_j κP_ij + Σ_j κD_ij + g_i) T_i⁺ − G_m,i M_i⁺ − κW_i W_i⁺\n                  − Σ_j κP_ij P_ij⁺ − Σ_j κD_ij T_j⁺  =  (C_i/Δt) T_i + g_i T_s + q_i + Q_gain,i + 70 n_i\nslab i:   (C_m,i/Δt + G_m,i) M_i⁺ − G_m,i T_i⁺            =  (C_m,i/Δt) M_i\nwall i:   (C_W,i/Δt + κW_i + γ_i) W_i⁺ − κW_i T_i⁺        =  (C_W,i/Δt) W_i + γ_i T_out\nwall ij:  (C_P/Δt + κP_ij + κP_ji) P_ij⁺ − κP_ij T_i⁺ − κP_ji T_j⁺  =  (C_P/Δt) P_ij",
  "eqsize": 20,
  "eqpt": 10,
  "points": [
   "Left: every unknown temperature at the end of the step. Right: what is known now (the current state, the commands, the disturbances)",
   "One equation per node, N equations in N unknowns: a single linear solve per step, nothing iterative",
   "Each coefficient is one entry of the coefficient table: a floor-plan count × a material number"
  ],
  "src": "[audit] STATE_SPACE.md §14.3 (room equations); [our code] dlib.LinearNetwork, dlib.network_predict (the same solve)"
 },
 {
  "id": "ssmatrix",
  "kind": "eq",
  "eyebrow": "Step 4 · the state-space model",
  "title": "Stack every node: x(t+1) = A(g) x(t) + B(g) u(t) + E(g) d(t), one model for the whole building",
  "eq": "x(t+1)  =  A(g) x(t)  +  B(g) u(t)  +  E(g) d(t)          y(t)  =  H x(t)\n\nM(g) = D + L + diag(g, 0)     D = diag(C)/Δt     A = M⁻¹ D     B = M⁻¹ B₀     E = M⁻¹ E₀     H = [ I_V  0 ]",
  "eqsize": 24,
  "eqpt": 12,
  "points": [
   "x = (T, M, W, P), N numbers;  u = (g_i T_s, q_i), 2V numbers;  d = (T_out, Q_gain,i + 70 n_i);  y = the V room temperatures",
   "L: the conductance matrix (row sums on the diagonal, −G off it). B₀ puts u on the air rows; E₀ puts T_out on the wall rows (γ_i) and gains on the air rows",
   "The commands enter twice: through u, and through g on the diagonal of M. So the model is linear in the state but the matrices depend on the airflow"
  ],
  "src": "[audit] STATE_SPACE.md §13.2, §14.3; [our code] dlib.LinearNetwork (A0inv = (D + L)⁻¹, Woodbury update for diag g)"
 },
 {
  "id": "ssvars",
  "kind": "table",
  "eyebrow": "Step 4 · what is what",
  "title": "State, control, disturbance and output of the building model",
  "rows": [
   [
    "Role",
    "Symbol",
    "What it is",
    "Count (building_train_8)",
    "Known how?"
   ],
   [
    "State (observed)",
    "T_i",
    "air temperature of room i",
    "V = 30",
    "measured every 5 min, one step late"
   ],
   [
    "State (hidden)",
    "M_i",
    "slab and furniture of room i",
    "V = 30",
    "never measured"
   ],
   [
    "State (hidden)",
    "W_i",
    "outside wall of room i (only rooms that have one)",
    "25 rooms",
    "never measured"
   ],
   [
    "State (hidden)",
    "P_ij",
    "wall shared by rooms i and j",
    "58 walls",
    "never measured"
   ],
   [
    "Control",
    "a",
    "the action: K + 1 on/off switches, 3K + 2 + 2V setpoints",
    "4 + 71",
    "we choose it"
   ],
   [
    "Control, as heat",
    "g_i, T_s, q_i",
    "supply airflow (W/K), supply temperature, reheat (W)",
    "per room",
    "from the action, via the kit's equipment laws"
   ],
   [
    "Disturbance",
    "T_out, Q_gain,i, n_i",
    "outside air; lights and equipment; people",
    "1 + V + V",
    "measured; schedule; building total"
   ],
   [
    "Output",
    "y = T",
    "the room-air temperatures",
    "V = 30",
    "the measurement"
   ]
  ],
  "colw": [
   0.15,
   0.13,
   0.36,
   0.16,
   0.2
  ],
  "size": 20,
  "font": 10,
  "callout": "State x = (T, M, W, P): 143 numbers for building_train_8. We measure the 30 room-air temperatures, and our commands act directly only on those.",
  "src": "[audit] STATE_SPACE.md §5, §13.2, §14.3; [our code] seeds_cf/seed_cf_network.py; [kit] building_train_8 action spec (4 discrete + 71 continuous = 75)"
 },
 {
  "id": "ssobs",
  "kind": "cards",
  "eyebrow": "Step 4 · observable and controllable",
  "title": "We see and push only the room air; the hidden stores are reached through it",
  "cards": [
   [
    "Observed",
    "Only the V room-air temperatures, one step late. Slab, wall and shared-wall temperatures are never measured.",
    "y = H x",
    "accent2"
   ],
   [
    "Controlled",
    "Only the room-air rows: airflow g and reheat q act on the air nodes. The slab and walls change only by exchanging heat with the air.",
    "B₀ acts on air rows",
    "accent1"
   ],
   [
    "Hidden but recoverable",
    "Each hidden node is tied to a room's air by a positive conductance, so it shows in how that air changes. So it can be estimated from the measured history (part 6).",
    "observability",
    "accent3"
   ]
  ],
  "callout": "The plant's other state, the boiler water temperature, is observed and enters through q (reheat), computed from our own water-setpoint commands.",
  "src": "[audit] STATE_SPACE.md §13.1-13.2, §14.3; [our code] runtime/inputs.py (T_water from commands)"
 },
 {
  "id": "matrix",
  "kind": "image",
  "eyebrow": "Step 4 · a real building's matrix",
  "title": "building_train_8's conductance matrix: sparse, and built from the floor plan",
  "img": "phys_matrix.png",
  "side": "right",
  "points": [
   "525 non-zero entries of the 143 × 143 matrix (the diagonal included); each of the 382 off-diagonal dots is one conductance between two nodes",
   "Air-to-air dots are doors; air-to-wall dots are shared walls; every room has a slab node, and the 25 rooms with an outside wall have an outside-wall node",
   "A bigger building gives a bigger matrix from the same formulas: nothing about the size is hard-coded"
  ],
  "caption": "[our code] the node network of this part (the formulas of seeds_cf/seed_cf_network.py) on building_train_8's metadata",
  "src": "[kit] building_train_8 floor plan; [our code] dlib.LinearNetwork"
 },
 {
  "id": "structok",
  "kind": "table",
  "eyebrow": "Step 4 · is the structure right?",
  "title": "With the right materials the structure reproduces the simulator; with wrong ones it does not",
  "rows": [
   [
    "Model (3-day open-loop runs, no measurement correction)",
    "Error, K (min / median / max over 18 runs)"
   ],
   [
    "Cell-level rebuild from the metadata, HVAC off",
    "0.0004 K"
   ],
   [
    "Room network with spreading, true materials",
    "0.17 / 0.52 / 0.95"
   ],
   [
    "Room network without spreading, true materials",
    "0.34 / 1.08 / 2.22"
   ],
   [
    "Air node only (1 per room), true materials",
    "0.63 / 1.59 / 4.32"
   ],
   [
    "Room network, materials at the middle of their ranges",
    "up to 9.6 K (building_train_1, Fairbanks)"
   ]
  ],
  "colw": [
   0.62,
   0.38
  ],
  "size": 25,
  "callout": "So the equations are right, the slow stores matter, and the remaining difficulty is the material numbers.",
  "src": "[audit] STATE_SPACE.md §14.5 ([sim data] 6 building-weather pairs × HVAC off / random / pid_water, 864 steps)",
  "font": 11
 },
 {
  "id": "unknowns",
  "kind": "cards",
  "eyebrow": "Step 5 · what is unknown",
  "title": "Everything in A, B, E is known except six material numbers per building",
  "cards": [
   [
    "Drawn per building",
    "k_ext, ρc_ext: outside wall conductivity and heat capacity\nk_iw, ρc_iw: shared walls\nρc_mass: slab and furniture",
    "hidden",
    "accent1"
   ],
   [
    "Set per weather station",
    "h: the outside film, how fast wind takes heat off the wall (8.8–26.7 W/m²K)",
    "hidden",
    "accent1"
   ],
   [
    "Effectively 4–5 numbers",
    "k_ext and h act only together (the wall's U-value); ρc_iw barely shows in the data",
    "identifiability",
    "accent3"
   ]
  ],
  "callout": "This is the whole modelling problem: infer a handful of material numbers for a building we have never seen, from its first hours of measurements, with nothing fitted.",
  "src": "[audit] STATE_SPACE.md §14.2, §14.4; [kit] materials.py, weather/*.yaml"
 },
 {
  "kind": "section",
  "id": "s5",
  "num": "5",
  "title": "The usual approaches",
  "question": "How is building control normally done, and why does it not fit this competition?",
  "notes": ""
 },
 {
  "kind": "image",
  "id": "rules",
  "eyebrow": "Option 1 · rules",
  "title": "Option 1, rules: react to the error now, never plan ahead",
  "img": "baselines.png",
  "side": "right",
  "points": [
   "Fixed setpoints, bang-bang, PID (the kit's baselines) look only at how far rooms are from the band right now",
   "No forecast: heating starts after a room is already cold, so every working day starts outside the band (part 3)",
   "Tuned once for all buildings: pid_water is best on average, yet on train_1 in Fairbanks a fixed setpoint beats it (0.147 vs 0.212)",
   "Safe and simple: the base our own controller builds on"
  ],
  "caption": "[kit] baseline_returns.json, 30 public scenarios",
  "src": "[kit] baseline_returns.json",
  "notes": "Rules do not adapt to the building."
 },
 {
  "kind": "cards",
  "id": "smarter",
  "eyebrow": "Options 2–4",
  "title": "Options 2–4 learn or fit a model, and each needs something we do not have",
  "cards": [
   [
    "Reinforcement learning",
    "Learns a policy by trial and error over many episodes.\n\nHere every test building is new, its size changes, and nothing can be trained during the evaluation.",
    "needs: training on the building",
    "accent1"
   ],
   [
    "A neural-network model",
    "Learns the building's behaviour, then plans with it.\n\nHere there is no data on the new building at the start, and layouts differ.",
    "needs: the building's data",
    "accent2"
   ],
   [
    "Physics, fitted per building",
    "A resistor-capacitor (RC) heat model whose constants are fitted to the building.\n\nHere there is no day of data to fit on, and only 0.125 s per step.",
    "needs: a fitting period",
    "accent3"
   ]
  ],
  "src": "[website] variable-dimensional hybrid action space; [measured] time budget; reasoning, not experiments",
  "notes": "These are arguments, not measured comparisons; leaderboard teams may use any of them."
 },
 {
  "kind": "table",
  "id": "comparison",
  "eyebrow": "Side by side",
  "title": "Side by side: only a universal formula needs no data from the new building",
  "rows": [
   [
    "Approach",
    "Data from the new building?",
    "Transfers to new layouts?",
    "Plans ahead?",
    "Readable?"
   ],
   [
    "Rules (PID)",
    "no",
    "yes, but untuned",
    "no",
    "yes"
   ],
   [
    "Reinforcement learning",
    "much",
    "poorly",
    "implicitly",
    "no"
   ],
   [
    "Black-box model + MPC",
    "much",
    "poorly",
    "yes",
    "no"
   ],
   [
    "Physics fitted per building + MPC",
    "a day or more",
    "yes",
    "yes",
    "yes"
   ],
   [
    "Ours: universal formulas + MPC",
    "only what it measures as it runs",
    "yes: the formulas take any floor plan",
    "yes",
    "yes"
   ]
  ],
  "colw": [
   0.3,
   0.2,
   0.18,
   0.14,
   0.18
  ],
  "src": "[decision] our assessment (not measured)",
  "notes": "The last row is the target.",
  "callout": "The bottom row is what this deck builds: physics structure so it transfers, nothing fitted so it works from the first step, and MPC so it plans ahead."
 },
 {
  "kind": "image",
  "id": "tried",
  "eyebrow": "What we tried first",
  "title": "What we tried first: fitting the constants works, but needs a day of each building",
  "img": "horizon.png",
  "side": "right",
  "points": [
   "We wrote a heat network by hand and fitted its material numbers on day 1 of each building",
   "It then forecasts far better than persistence ('the temperature stays where it is'): the structure is right",
   "But the hidden test gives no day 1 to fit on, and any fitting would have to run inside 0.125 s per step",
   "The way forward: keep the structure, and compute the material numbers from the measurements with a formula (part 6)"
  ],
  "caption": "[sim data] audit, 3-day episodes: a hand-built heat network with materials fitted on day 1, against persistence (the forecast 'temperature stays where it is')",
  "src": "[sim data] outputs/room_model_forecast.json (day-1 fitting: an alternative evaluated and not adopted)",
  "notes": "The question became: can formulas read the materials from history, so nothing is fitted?"
 },
 {
  "kind": "section",
  "id": "s6",
  "num": "6",
  "title": "Our approach: formulas for the unknown numbers",
  "question": "The structure is known and only a few material numbers are missing. How do we fill them in for an unseen building without fitting?",
  "notes": ""
 },
 {
  "kind": "statement",
  "id": "goal",
  "eyebrow": "What we are after",
  "title": "One set of formulas that predicts every room’s temperature in any building",
  "sub": "Written out as readable code, with no trained weights. Inputs: the floor plan, the measured history and universal constants. Nothing is fitted: not on the new building, not beforehand.",
  "src": "[decision] planning notes",
  "notes": "The only estimation allowed is a closed-form expression of the history buffer inside the formulas."
 },
 {
  "kind": "compare",
  "id": "observation",
  "eyebrow": "The observation behind our approach",
  "title": "What changes between buildings, and what does not",
  "left": [
   "Changes",
   [
    "Geometry: visible in the floor plan",
    "Six materials: hidden, but they show in how rooms warm and cool",
    "Weather and people: measured as they happen"
   ]
  ],
  "right": [
   "Does not change",
   [
    "The simulator’s physics and equipment laws",
    "Air and water properties, the 0.2 m grid, the 3 m height, 5-minute step",
    "Start at 294 K; 70 W per occupant; 10 W/m² internal gains"
   ]
  ],
  "callout": "Write the physics once (part 4), fill in the geometry from the floor plan, and compute the materials from the measured history.",
  "src": "[kit] config.gin (initial_temp 294, schedule), materials.py; [audit] STATE_SPACE.md §12.2 constants (70 W, 10 W/m²)",
  "notes": "Universal values in the simulator are generator choices; on real buildings we first try the same constants."
 },
 {
  "id": "threeways",
  "kind": "table",
  "eyebrow": "Filling the gap",
  "title": "Three ways to get the material numbers for a new building",
  "rows": [
   [
    "Way",
    "How",
    "Problem / strength"
   ],
   [
    "Fit them on the building",
    "adjust θ until the model matches the first day",
    "needs a day of data and fitting time inside 0.125 s per step (part 5)"
   ],
   [
    "One universal value, state corrected by measurements",
    "mid-range materials; each step the room-air nodes are reset to the measured temperatures",
    "the hidden slab and wall nodes follow the measurements; wrong materials still bias the forecast"
   ],
   [
    "A formula of the measured history",
    "compute the material number each step from the last hours of data, by a fixed closed-form expression",
    "works from the first hours; nothing is stored or learned"
   ]
  ],
  "colw": [
   0.24,
   0.4,
   0.36
  ],
  "size": 24,
  "callout": "Ours is the second and third, and their mixtures. Which formulas, which hidden nodes and which history summaries work best is what the LLM searches for.",
  "src": "[our code] seeds_cf/seed_cf_network.py (second), seed_cf_history.py (third); [decision] no fitting",
  "font": 11
 },
 {
  "id": "histformula",
  "kind": "eq",
  "eyebrow": "Example · a material number from history",
  "title": "The outside-wall U-value, computed from the last 24 hours of measurements",
  "eq": "Û  =  ( Σ x·r  +  s·U₀ )  /  ( Σ x²  +  s )\nclipped to 0.3 – 8 W/m²K",
  "eqsize": 40,
  "eqpt": 20,
  "points": [
   "r: how much each exterior room's measured 5-minute change differs from what every other term predicts",
   "x: how much that change would be with U = 1 W/m²K: 300 s × wall area × (T_out − T) ÷ C, divided by the same end-of-step denominator as the other terms",
   "So Û is the least-squares ratio r/x over every exterior room and step in the buffer, pulled toward U₀ = 2 W/m²K (s = 5) while the buffer is short",
   "A fixed formula, recomputed every 3 hours (REFRESH = 36 steps) on the latest 288 steps; no parameter is fitted, stored or tuned"
  ],
  "src": "[our code] seeds_cf/seed_cf_history.py facade_u (U_NOMINAL 2.0, SHRINK 5.0, U_LO 0.3, U_HI 8.0, HISTORY 288, REFRESH 36)"
 },
 {
  "id": "histnotfit",
  "kind": "points",
  "eyebrow": "A common question",
  "title": "Isn't the history U-value formula a fit in disguise?",
  "points": [
   "It is a fixed function of the last 288 measurements, recomputed every 3 hours (REFRESH = 36), like a moving average: Û = (Σ x·r + 5·2.0) / (Σ x² + 5)",
   "The formula and its constants (2.0, 5, 0.3, 8) are the same in every building; nothing is chosen to reduce an error",
   "Nothing is stored or carried between steps, buildings or episodes: it has no memory of the buildings it was chosen on",
   "Fitting = search for parameters that minimise an error, then keep them. This = compute a statistic of the data in front of you",
   "A formula like this is allowed precisely because it uses only what the policy measures at test time"
  ],
  "src": "[our code] seeds_cf/seed_cf_history.py facade_u; prompts.CF_SYSTEM ('the only estimation allowed is a closed-form expression of the history buffer')"
 },
 {
  "id": "observer",
  "kind": "eq",
  "eyebrow": "Example · hidden stores from measurements",
  "title": "The slab and wall temperatures, carried by the model and corrected by the measurements",
  "eq": "predict:  x⁺ = A(g) x + B(g) u + E(g) d\ncorrect:  room-air part of x ← measured T;   slab and wall parts unchanged",
  "eqsize": 30,
  "eqpt": 15,
  "points": [
   "Every 5 minutes the framework predicts all nodes with the LLM's coefficients, then overwrites the room-air nodes with what was measured",
   "Hidden nodes keep their predicted values, so they are pulled toward the truth by the measured air over the following steps",
   "Like a Kalman filter with gain 1 on the air and 0 on the hidden nodes: a fixed rule, nothing tuned",
   "Real data, network model: its 3-hour forecasts are off by 0.87 K (root-mean-square) over the 12 thirty-day runs, 0.902 K when the future weather is unknown; it still loses to persistence in 3 of the 12 runs"
  ],
  "src": "[our code] discovery/coefform.py (TermModel.correct), seeds_cf/seed_cf_network.py; [measured] closedform.evaluate"
 },
 {
  "id": "bigpicture",
  "kind": "flow",
  "eyebrow": "The whole method in one line",
  "title": "Floor plan + universal constants + measured history → formulas → A, B, E → forecast → control",
  "steps": [
   [
    "Floor plan",
    "areas, wall lengths, doors (metadata)"
   ],
   [
    "History",
    "the last minutes to 24 h of measurements"
   ],
   [
    "Formulas",
    "written by the LLM: coefficients and hidden stores"
   ],
   [
    "State space",
    "A(g), B(g), E(g) for this building"
   ],
   [
    "Forecast + MPC",
    "predict 2 h ahead, choose commands"
   ]
  ],
  "points": [
   "The formulas are the only part that is searched for. They take geometry, universal constants and the measured history, and return every coefficient of the state-space model",
   "The LLM writes candidate formulas; simulator data on buildings it was not shown decides which are better",
   "Learning happens once, offline, over the structure of the formulas; at test time nothing is fitted"
  ],
  "src": "[our code] discovery/ (contract, closedform, evolve), runtime/closedform_tracker.py, control/",
  "notes": "Control (part 10) is a later step; this deck's focus is the forecasting model."
 },
 {
  "kind": "cards",
  "id": "whyllm",
  "eyebrow": "Why an LLM",
  "title": "An LLM writes the formulas; data decides which are right",
  "cards": [
   [
    "Too many structures to write by hand",
    "Which heat stores, which couplings, how coefficients follow from geometry, how to read materials from history.",
    "",
    "accent1"
   ],
   [
    "LLMs write physics code",
    "It proposes readable programs, each a hypothesis, and reacts to diagnostics.",
    "",
    "accent2"
   ],
   [
    "The scorer decides",
    "Every candidate is checked and scored on validation runs, and the winner tested on held-out tiers. The LLM never supplies a verdict or a fitted number.",
    "",
    "accent3"
   ]
  ],
  "callout": "The result is readable physics that can be checked and tried on real buildings.",
  "src": "[our code] discovery/evolve.py, closedform.py; [plan] no real LLM generation has run yet",
  "notes": "The machinery is tested with a mechanical stand-in for the LLM."
 },
 {
  "id": "nofitdef",
  "kind": "table",
  "eyebrow": "What 'nothing fitted' means",
  "title": "No backpropagation and no fitting, anywhere the model is made or used",
  "rows": [
   [
    "Where fitting usually happens",
    "What we do instead"
   ],
   [
    "Before submission: tune weights or constants on the training data (backpropagation, least squares)",
    "There is no parameter vector to tune. A candidate with any fitted-constant table is rejected"
   ],
   [
    "On the test building: estimate constants from its first day (online fitting)",
    "No optimiser runs inside the policy. Each step evaluates a fixed formula"
   ],
   [
    "Gradients: differentiate an error and step the parameters",
    "No error is ever differentiated. The code imports only numpy, math and our small dlib"
   ],
   [
    "Where the model's numbers come from",
    "Only three kinds: universal constants cited from the kit, counts from the floor plan, and expressions of the current measured history"
   ],
   [
    "What changes from one candidate to the next",
    "Only the program text, written by the LLM"
   ]
  ],
  "colw": [
   0.45,
   0.55
  ],
  "size": 23,
  "font": 11,
  "callout": "So the model is the same code with the same numbers in every building. A new building changes only its floor plan and its measurements.",
  "src": "[our code] contract.load_closedform, contract._exec (sandbox), closedform.evaluate; prompts.CF_CONTRACT"
 },
 {
  "id": "nobpcompare",
  "kind": "compare",
  "eyebrow": "Backpropagation vs our search",
  "title": "Backprop adjusts numbers inside a fixed model; we choose between whole programs",
  "left": [
   "Backpropagation",
   [
    "Fix an architecture with free weights",
    "Define a loss on training data",
    "Differentiate the loss with respect to every weight",
    "Step every weight downhill, thousands of times",
    "The trained weights are the model"
   ]
  ],
  "right": [
   "Our search",
   [
    "No free weights: the LLM writes a complete program",
    "Run it on the validation runs; the score is one number used only to rank",
    "No derivative of anything is taken",
    "Keep the better programs, discard the rest, ask for new ones",
    "The program text is the model; its numbers never change after it is written"
   ]
  ],
  "callout": "The search learns which formulas work. It never adjusts the numbers inside a formula.",
  "src": "[our code] evolve.py (tournament, elites), closedform.evaluate (score only)"
 },
 {
  "kind": "table",
  "id": "nobackprop",
  "eyebrow": "The alternatives, side by side",
  "title": "Four ways to get the model, and what each one updates",
  "rows": [
   [
    "Option",
    "What is updated, and when",
    "Strength",
    "Why it is not our main path"
   ],
   [
    "Neural network trained offline on many buildings",
    "thousands of weights, once, before submission",
    "uses all generated data; nothing to fit on the new building",
    "room count and layout change per building (needs a graph network); opaque; cannot pass physics checks. A real alternative we have not tested"
   ],
   [
    "Physics model, constants fitted online per building",
    "a few constants, by gradient steps during the episode",
    "adapts to the actual building",
    "no model in the first hours or day; fitting inside 0.125 s per step; can run away (part 8). Tried first (part 5)"
   ],
   [
    "Physics model, one constant set fitted offline",
    "a few constants, once",
    "simple and cheap",
    "materials differ by building (part 4), so one set is wrong for most buildings"
   ],
   [
    "Ours: LLM-written formula, nothing fitted",
    "the formula's structure, offline, by evolutionary search scored on validation runs",
    "works from the first hours; one code for any size; readable; checkable",
    "only as good as the formula: the gap to the fitted ceiling (part 8) measures what is left"
   ]
  ],
  "colw": [
   0.2,
   0.24,
   0.24,
   0.32
  ],
  "size": 21,
  "font": 9.5,
  "callout": "We use the last row. Whether to add online updates on top of a discovered formula is an open question (part 12).",
  "src": "[decision] our reasoning, not experiments, except: online fitting tried (part 5), ceiling measured (part 8)"
 },
 {
  "kind": "compare",
  "id": "nofit",
  "eyebrow": "What no fitting buys, and costs",
  "title": "Why we insist on no fitting at all",
  "left": [
   "Buys",
   [
    "Works from step 1: no identification period",
    "A bad score means the formula is wrong, not the optimizer",
    "Cheap: about 0.1 ms per step for 30 rooms (measured inside the policy)"
   ]
  ],
  "right": [
   "Costs",
   [
    "Some error in the first hours while the measured history builds up (accepted)",
    "Materials must be inferable from history"
   ]
  ],
  "src": "[decision] planning notes; [measured] step time",
  "notes": "Decisions: no fitting, universal constants allowed, early error accepted, discrete-time maps allowed."
 },
 {
  "id": "honestfit",
  "kind": "cards",
  "eyebrow": "Choices that remain",
  "title": "Where human or LLM choices still enter, and how they are checked",
  "cards": [
   [
    "Hand-chosen constants",
    "The hand-written models' constants (U₀ = 2.0 W/m²K, shrink 5, slab coupling 4 W/m²K, 6 h store time constant; mid-range materials) were set once by reasoning, never optimised. They are references, not parents.",
    "hand-written models",
    "accent3"
   ],
   [
    "Constants the LLM writes",
    "The LLM may add physically sensible constants. Selection could favour lucky values indirectly; guarded by the size price, Pareto selection on validation runs and three test tiers never seen by the search.",
    "the search",
    "accent1"
   ],
   [
    "The one fitted thing",
    "The 'ceiling': a per-room model fitted by non-negative least squares on half of each episode. A diagnostic only; never deployed or used in selection.",
    "diagnostic",
    "accent2"
   ]
  ],
  "callout": "The fitted variants we tried first (part 5) are not part of the method; their measured results are in outputs/. Closed-form discovery, with nothing fitted, is the only mode.",
  "src": "[our code] seeds_cf/*.py, closedform.ceiling (scipy nnls, diagnostic)"
 },
 {
  "id": "enforce",
  "kind": "table",
  "eyebrow": "Checked by the code",
  "title": "How 'no fitting' is checked automatically",
  "rows": [
   [
    "Check",
    "Where",
    "Effect"
   ],
   [
    "No fitted-constant tables",
    "contract.load_closedform",
    "a candidate with non-empty PARAMS or THETA is rejected before it runs"
   ],
   [
    "Sandbox imports",
    "contract._exec",
    "only numpy, math and dlib can be imported: no optimisers, no autodiff libraries, no files"
   ],
   [
    "Scorer fits nothing",
    "closedform.evaluate",
    "only rollouts and error statistics; the single optimiser call in the module is the diagnostic ceiling"
   ],
   [
    "No special-casing of buildings",
    "contract._reject_building_specific",
    "code comparing a building's size or geometry to a fixed number is rejected"
   ],
   [
    "Test-time data only",
    "tests/test_coefform.py",
    "the coefficient formulas only receive geometry, universal constants and measured history up to now; changing later data changes nothing they saw"
   ],
   [
    "Same path live",
    "runtime/closedform_tracker.py, discovery/coefform.py",
    "the policy evaluates the same coefficient formulas and builds the same model as the scorer: no estimation step exists"
   ]
  ],
  "colw": [
   0.24,
   0.3,
   0.46
  ],
  "size": 22,
  "font": 11,
  "src": "[our code] discovery/contract.py, closedform.py, runtime/closedform_tracker.py; tests/test_deployment.py"
 },
 {
  "kind": "table",
  "id": "decmodel",
  "eyebrow": "Decisions · model form",
  "title": "What a candidate model is allowed to be, and why",
  "rows": [
   [
    "Decision",
    "What we chose",
    "Why",
    "Rejected alternative / evidence"
   ],
   [
    "Fitting",
    "none: not on the new building, not beforehand",
    "works from step 1; a bad score means a bad formula, not a bad optimiser",
    "fitting per building (tried first): costs a day, can fail"
   ],
   [
    "Numbers",
    "universal constants (cited from the kit) + geometry + history",
    "a universal constant does not make a model building-specific",
    "per-building constants"
   ],
   [
    "Form",
    "one fixed linear state-space function; the LLM writes a formula for each of its coefficients",
    "the search is over the physics itself; every candidate is a valid model, stable for any coefficients ≥ 0",
    "free-form one-step maps (still loadable)"
   ],
   [
    "Memory",
    "three hidden stores per room (slab, outside wall, inner walls), coefficients may read up to one day of history",
    "stands in for the unmeasured slab and wall temperatures without fitting",
    "one temperature per room: misses the slow part"
   ],
   [
    "Correction",
    "room-air temperatures set to the measurement every step; the stores keep their predictions",
    "the stores follow the measured history; nothing tuned",
    "tuned correction weights (a Kalman gain): fitted numbers"
   ],
   [
    "Supply air",
    "implicit (end-of-step) in the assembled form",
    "supply air can replace a room's air in under 5 minutes; a plain forward step then overshoots",
    "plain forward (Euler) step: fails the strong-airflow check"
   ],
   [
    "Safety",
    "sandbox: numpy, math and dlib (our small helper library) only; no files; deterministic",
    "candidates are untrusted code",
    "-"
   ],
   [
    "Shortcuts",
    "comparing a building's size or geometry to fixed numbers is rejected",
    "would special-case the buildings seen in discovery",
    "-"
   ]
  ],
  "colw": [
   0.17,
   0.25,
   0.32,
   0.26
  ],
  "callout": None,
  "src": "[decision] planning notes; [our code] discovery/contract.py, dlib.py, prompts.CF_CONTRACT",
  "notes": "",
  "size": 21,
  "font": 9.5
 },
 {
  "kind": "section",
  "id": "s7",
  "num": "7",
  "title": "LLM discovery",
  "question": "How does an LLM write the coefficient formulas of the state-space function, what does it see, how is each candidate judged, and how does the search keep the best?",
  "notes": ""
 },
 {
  "id": "discflow",
  "kind": "flow",
  "eyebrow": "Discovery in one picture",
  "title": "An LLM proposes formulas; validation runs judge them; held-out tiers test the winner",
  "steps": [
   [
    "Data",
    "finished episodes from the kit; validation only"
   ],
   [
    "LLM proposes",
    "a Python module: the formulas"
   ],
   [
    "Checks",
    "physics sanity on a real floor plan"
   ],
   [
    "Score",
    "five forecast errors on validation runs"
   ],
   [
    "Analysis",
    "where, when and through which term it is wrong, fed back"
   ]
  ],
  "points": [
   "The loop repeats for many generations; the programs no other program beats on all five errors survive (evolutionary search)",
   "The LLM never sees the test tiers, never supplies a number and never gives a verdict",
   "Status: the loop, prompts, scorer and logs are built and tested (a stand-in for the API client in the tests); no real API generation has run yet"
  ],
  "src": "[our code] discovery/evolve.py, prompts.py, closedform.py, pareto.py, splits.py; docs/DISCOVERY_WALKTHROUGH.md"
 },
 {
  "kind": "eq",
  "id": "discproblem",
  "eyebrow": "The problem the LLM solves",
  "title": "Find a formula for every coefficient of one state-space function, that works in every building",
  "eq": "coefficient_k = f_k( floor plan, universal constants, measured history to now )\nx(t + 5 min)  = fixed function( x(t), commands, weather; coefficients )",
  "points": [
   "The function is fixed: a few terms per room (supply air, heat, outside air, neighbours, three heat stores). The LLM writes f_k for each term's coefficient",
   "Each f_k may read only what exists at test time: the floor plan, universal constants, the measured history. Never the building's configuration file",
   "The same formulas must work for a 30-room or an 85-room building, in Fairbanks or Phoenix",
   "Fitting is ruled out, so we search over formulas: the LLM proposes, a scorer on validation runs decides, evolution keeps the best"
  ],
  "src": "[decision] planning notes; [our code] contract.load_closedform, prompts.CF_CONTRACT",
  "notes": "This is symbolic discovery: the output is code a person can read, not a black box.",
  "eqsize": 28,
  "eqpt": 14
 },
 {
  "id": "available",
  "kind": "compare",
  "eyebrow": "What the formulas may read",
  "title": "Available on an unseen building at deployment, and never available",
  "left": [
   "Available: the formulas may use it",
   [
    "Metadata: floor plan, zone map, air-handler map, schedule → each room's area, walls, doors",
    "Every 5 min: room temperatures (one step late), outside air, people in the building, comfort flag, clock",
    "Our own commands → supply air, reheat heat, boiler water",
    "Universal constants: air and water properties, cell size, height, step, 70 W per person"
   ]
  ],
  "right": [
   "Never available: the formulas must not use it",
   [
    "The building's configuration file (YAML): materials, seed, generator settings",
    "Slab and wall temperatures (never measured)",
    "People per room; future weather and occupancy",
    "Which building or weather window it is; anything after 'now'"
   ]
  ],
  "callout": "The materials are hidden in the configuration file, so any formula that needs them must read them from the measured history.",
  "src": "[website] submission.html (building metadata); [Kaggle] data tab; [our code] closedform.EXOGENOUS, runtime/inputs.py, tests/test_coefform.py, tests/test_deployment.py"
 },
 {
  "id": "episode",
  "kind": "table",
  "eyebrow": "What the LLM's formulas get to read",
  "title": "One validation episode: geometry, inputs every 5 minutes, measured temperatures",
  "rows": [
   [
    "Part",
    "Contents (exactly as in the prompt)",
    "Source at test time"
   ],
   [
    "Geometry, per room",
    "area, facade faces (total and by side), number of sides with outside wall, boundary faces, neighbours, bounding box, aspect, floor, air handler",
    "floor_plan, zone_map, AHU map"
   ],
   [
    "Geometry, per pair",
    "shared wall cells, door cells, number of doors",
    "same"
   ],
   [
    "Inputs, per room",
    "g (supply air, W/K, 0 to ~2,400), T_supply (K), Q_reheat (W), Q_gain (W)",
    "own commands + the kit's equipment laws"
   ],
   [
    "Inputs, building",
    "T_out, occupants (total), comfort flag, hour and weekday (sin, cos), T_water",
    "observation"
   ],
   [
    "Target",
    "every room's measured air temperature after the step; every episode starts at 294 K",
    "observation"
   ]
  ],
  "colw": [
   0.18,
   0.56,
   0.26
  ],
  "size": 22,
  "callout": "Until generation runs, the episodes are the six shipped buildings' 30-day runs, split into validation and the three test tiers by the rule in part 2.",
  "src": "[our code] prompts.CF_DATA, episodes.geometry, starterkit_data.episode_inputs",
  "font": 11
 },
 {
  "kind": "image",
  "id": "geomscript",
  "eyebrow": "The model · geometry",
  "title": "A script reads every room’s numbers from the floor plan",
  "img": "rooms_classes.png",
  "side": "right",
  "points": [
   "Input: floor_plan and zone_map from the metadata, nothing else",
   "Per room: area, outside-wall length, bounding box, floor, AHU",
   "Per pair: shared wall length, door width, number of doors",
   "Corner, edge, interior fall out of these numbers"
  ],
  "caption": "[audit] building_train_2 room classes (outputs/building_train_2_rooms.png)",
  "src": "[our code] discovery/episodes.py geometry()",
  "notes": "The same function runs inside the policy."
 },
 {
  "kind": "cards",
  "id": "templates",
  "eyebrow": "What the LLM writes",
  "title": "One function returning a formula for every coefficient; the rest is fixed",
  "cards": [
   [
    "coefficients(geo, hist, U)",
    "A closed-form formula for each coefficient of the fixed function: air, heat, outside, neighbours, slab, wall and inner stores",
    "the A, B, E entries",
    "accent1"
   ],
   [
    "heat(geo, u, U)",
    "Optional: the direct heat into each room (reheat, lights and equipment, people). Default: people split evenly",
    "inputs",
    "accent3"
   ],
   [
    "HISTORY, REFRESH",
    "How much measured history the formulas may read (up to 24 h), and how often the coefficients are recomputed",
    "settings",
    "accent3"
   ],
   [
    "The fixed function",
    "Ours, not the LLM's: the state-space function the coefficients go into, stepped and corrected by the framework",
    "fixed",
    "accent2"
   ]
  ],
  "callout": "Each formula may read only test-time data: the floor plan, universal constants and the measured history up to now. Nothing is fitted.",
  "src": "[our code] discovery/coefform.py, prompts.CF_CONTRACT",
  "notes": "One physical question per formula, so crossover can recombine them."
 },
 {
  "kind": "eq",
  "id": "assemble",
  "eyebrow": "The fixed state-space function",
  "title": "Per room, every 5 minutes: a weighted average of its heat sources, plus three heat stores",
  "eq": "T_i⁺ = ( T_i + air·g_i·T_supply + outside·T_out + Σ_j nb_ij·T_j + slab·S_i + wall·W_i + inner·P_i + heat·Q_i )\n       / ( 1 + air·g_i + outside + Σ_j nb_ij + slab + wall + inner )\nS_i⁺ = ( S_i + slab_back·T_i⁺ ) / ( 1 + slab_back )                      slab store\nW_i⁺ = ( W_i + wall_back·T_i⁺ + wall_out·T_out ) / ( 1 + wall_back + wall_out )   outside-wall store\nP_i⁺ = ( P_i + inner_back·T_i⁺ ) / ( 1 + inner_back )                     inner-wall store",
  "points": [
   "Each line is the backward-Euler (end-of-step) form of x⁺ = x + Σ_k a_k (source_k − x): stable for any coefficients ≥ 0",
   "Together it is the linear state-space model x(t+1) = A(g) x(t) + B(g) u(t) + E(g) d(t) with x = (T, S, W, P); the coefficients are the entries of A, B and E",
   "After each step the room air T is set to the measurement; the three stores keep their predictions and so follow the measured history"
  ],
  "src": "[our code] discovery/coefform.py (TermModel), prompts.CF_CONTRACT",
  "notes": "The supply air is implicit because it can turn a room’s air over faster than one step.",
  "eqsize": 20,
  "eqpt": 10
 },
 {
  "id": "terms",
  "kind": "table",
  "eyebrow": "The coefficients the LLM writes formulas for",
  "title": "Eleven coefficients per room, each with a physical meaning",
  "rows": [
   [
    "Coefficient",
    "Term it scales",
    "Physical meaning (per 5-minute step)",
    "In the hand-written network model"
   ],
   [
    "air, heat",
    "supply air g·T_supply; direct heat Q",
    "300 s ÷ the room air's heat capacity",
    "300 s ÷ (1,206 J/m³K × 3 m × floor area)"
   ],
   [
    "outside",
    "outside air T_out, directly",
    "300 s × conductance to outside ÷ air capacity",
    "0 (it goes through the outside-wall store)"
   ],
   [
    "neighbours",
    "room j's air T_j",
    "300 s × conductance through shared walls and doors ÷ air capacity",
    "wall halves in series + doors, from the floor plan"
   ],
   [
    "slab, slab_back",
    "the slab-and-furniture store S",
    "exchange with the slab, seen from the air and from the slab",
    "5 W/m²K × area, ÷ air or ÷ slab capacity"
   ],
   [
    "wall, wall_back, wall_out",
    "the outside-wall store W",
    "air ↔ outside wall ↔ outside air",
    "facade faces × material, ÷ capacities"
   ],
   [
    "inner, inner_back",
    "the inner-wall store P",
    "exchange with the room's half of its shared walls",
    "half-wall conductance and capacity"
   ]
  ],
  "colw": [
   0.17,
   0.22,
   0.31,
   0.3
  ],
  "size": 21,
  "font": 10,
  "callout": "Every coefficient is 300 s × a conductance ÷ a heat capacity: a floor-plan count times a material property. The materials are what the formulas must read from the measured history.",
  "src": "[our code] discovery/coefform.py TERMS_ROOM; seeds_cf/seed_cf_network.py"
 },
 {
  "id": "sysprompt",
  "kind": "table",
  "eyebrow": "What the LLM reads first",
  "title": "The system prompt, part by part (identical in every request)",
  "rows": [
   [
    "Part",
    "What it says"
   ],
   [
    "Task",
    "discover ONE set of closed-form formulas for the coefficients of a fixed state-space function that predicts every room's temperature in ANY building, for MPC; nothing fitted"
   ],
   [
    "Physics hypotheses",
    "heat flow is linear in temperatures; commands → heat is already done; one temperature per room is probably not enough (fast, then slow)"
   ],
   [
    "Model form",
    "the fixed function, written out; one formula per coefficient, one physical question per formula"
   ],
   [
    "Data dictionary",
    "the geometry, inputs and target of one episode (above), with units and shapes"
   ],
   [
    "Universal constants",
    "10 values with their kit sources: Δt, cell size, height, air and water properties, 294 K start, 70 W per person"
   ],
   [
    "Contract and rules",
    "the module to write (next slide); no fitted numbers; conductance = conductivity × area ÷ distance; layers add"
   ],
   [
    "Scoring",
    "validation only, test tiers kept aside; the checks, the screen, the test-time 3 h score with the size price, and the five objectives used for selection"
   ]
  ],
  "colw": [
   0.2,
   0.8
  ],
  "size": 23,
  "src": "[our code] prompts.CF_SYSTEM, CF_DATA, CF_CONTRACT, CF_SCORING; discovery/universal_constants.json"
 },
 {
  "kind": "code",
  "id": "llmreads",
  "eyebrow": "What the LLM reads (excerpt of the real prompt)",
  "title": "The contract: the fixed function, and the coefficients to write",
  "code": "The state-space function is FIXED; you write the FORMULA for every\ncoefficient in it.  Per room i, every 5-minute step:\n  T_i+ = ( T_i + air_i g_i T_supply_i + outside_i T_out + ... ) / ( 1 + ... )\n  S_i+ = ( S_i + slab_back_i T_i+ ) / ( 1 + slab_back_i )      ...\n\nHISTORY = 288     # steps of measured history your formulas may read\nREFRESH = 36      # coefficients recomputed every REFRESH steps\n\ndef coefficients(geo, hist, U):\n    return {\"air\": ..., \"heat\": ..., \"outside\": ...,\n            \"neighbours\": ...,                       # (V, V)\n            \"slab\": ..., \"slab_back\": ...,\n            \"wall\": ..., \"wall_back\": ..., \"wall_out\": ...,\n            \"inner\": ..., \"inner_back\": ...}        # each >= 0\n\n  hist  ONLY data a policy has at that moment on an unseen building\n        ... the building's configuration file is never available",
  "points": [
   "Around it: the task, physics hypotheses, data dictionary, universal constants, scoring rules",
   "Then the operator-specific part: parents, their analysis, recent rejected children and children that scored worse"
  ],
  "src": "[our code] prompts.CF_CONTRACT (the text sent to the LLM, generated by prompts.cf_system_prompt)",
  "notes": "The system prompt is the same every generation, so it is cached."
 },
 {
  "kind": "code",
  "id": "llmwrites",
  "eyebrow": "What the LLM writes back, shown with the hand-written history model",
  "title": "A candidate: a formula for every coefficient",
  "code": "HISTORY = 288                              # one day of 5-minute steps\nREFRESH = 36                               # U recomputed every 3 hours\n\ndef coefficients(geo, hist, U):\n    dt = U[\"DT\"]\n    C_air = U[\"RHO_C_AIR\"] * U[\"HEIGHT\"] * geo[\"area\"]\n    U_wall = facade_u(geo, hist, U)          # U-value FROM HISTORY\n    return {\"air\": dt / C_air, \"heat\": dt / C_air,\n            \"neighbours\": dt * G_nb / C_air[:, None],\n            \"slab\": dt * H_SLOW * geo[\"area\"] / C_air,\n            \"slab_back\": dt / TAU_SLOW,           # 6 h store\n            \"outside\": dt * facade_area * U_wall / C_air}",
  "points": [
   "Every number is universal, geometric, or computed from the measured history",
   "Terms it leaves out (the wall and inner stores here) are zero",
   "The answer comes as JSON: name, one-paragraph hypothesis, code"
  ],
  "src": "[our code] code/discovery/seeds_cf/seed_cf_history.py (abridged; a hand-written model in the format the LLM must use)",
  "notes": "No LLM generation has run yet; this is a hand-written model written in the same contract."
 },
 {
  "id": "netseed",
  "kind": "code",
  "eyebrow": "A second hand-written model: the network model (excerpt)",
  "title": "Part 4's physics, written as coefficient formulas",
  "code": "K_EXT = (0.15 * 0.55) ** 0.5            # W/mK   middle of the generator's range\nH_FILM = 17.75                          # W/m2K  middle of the stations' range\n\ndef coefficients(geo, hist, U):\n    C_air  = RHO_C_AIR * H * A                       # air capacity\n    G_slab = H_SLAB * A;  C_slab = RC_MASS * H * A   # slab\n    G_wall = 1 / (1/g_fac + spread(A, nf * c))       # air -> outside wall\n    C_wall = RC_EXT * nf * face * c / 2\n    G_wall_out = nf * face / (c/(4*K_EXT) + 1/H_FILM) # wall -> outside\n    ...                                              # shared walls, doors\n    return {\"air\": dt/C_air, \"heat\": dt/C_air,\n            \"neighbours\": dt * (G_wall_ij + G_door) / C_air,\n            \"slab\": dt*G_slab/C_air, \"slab_back\": dt*G_slab/C_slab,\n            \"wall\": dt*G_wall/C_air, \"wall_back\": dt*G_wall/C_wall,\n            \"wall_out\": dt*G_wall_out/C_wall,\n            \"inner\": dt*G_inner/C_air, \"inner_back\": dt*G_inner/C_inner}",
  "points": [
   "Each coefficient is 300 s × a row of part 4's coefficient table ÷ a heat capacity",
   "The materials are fixed mid-range values; the LLM's job is to replace them with formulas that read the building's materials from its measured history",
   "Hand-written network model: 0.87 K at 3 h on the 12 runs"
  ],
  "src": "[our code] code/discovery/seeds_cf/seed_cf_network.py (abridged)"
 },
 {
  "id": "checks",
  "kind": "table",
  "eyebrow": "How a candidate is judged · stage 0",
  "title": "Before any scoring: five checks over two simulated days on a real floor plan, plus a one-day shift warning",
  "rows": [
   [
    "Check",
    "Set-up",
    "Must hold"
   ],
   [
    "Equilibrium",
    "outside 294 K, rooms 294 K, no heat, no air",
    "no room drifts more than 0.05 K"
   ],
   [
    "Bounded cooling",
    "outside 274 K",
    "rooms stay within 274–294 K and every room ends cooler than it started"
   ],
   [
    "Monotone",
    "outside 284 K instead of 274 K",
    "no room ends colder"
   ],
   [
    "Heating",
    "2 kW of reheat into one room",
    "that room warms; no other room cools"
   ],
   [
    "Stiff air",
    "2,400 W/K of 290 K supply air in every room, outside 294 K",
    "rooms stay within 290–294 K and approach 290 K"
   ],
   [
    "Shift (warning only)",
    "one day with every temperature (start, outside, supply, water) raised by 5 K",
    "the forecast rises by 5 K too, within 0.05 K"
   ]
  ],
  "colw": [
   0.17,
   0.43,
   0.4
  ],
  "size": 24,
  "callout": "A failed check rejects the candidate before any data is used; the reason goes back to the LLM in the island's next requests.",
  "src": "[our code] closedform.physics_checks; DISCOVERY_WALKTHROUGH.md (shift warning)",
  "font": 11
 },
 {
  "id": "scoreeq",
  "kind": "eq",
  "eyebrow": "How a candidate is judged · stages 1 and 2",
  "title": "The main score is the 3-hour forecast error on the validation runs, plus a small price for size",
  "eq": "main  =  mean over validation episodes of RMSE_3h\n        +  0.0001 K × (number of nodes in the code's syntax tree)",
  "eqsize": 32,
  "eqpt": 16,
  "points": [
   "Stage 1 screen: the 5-minute error from measured history; more than 5× the error of 'temperature stays' (persistence) and the candidate is dropped",
   "Stage 2: from every 30 minutes after a 3 h warm-up, the model runs on its own predictions for 1, 3 and 6 h; RMSE_h = √(mean over starts and rooms of (forecast − measured)²)",
   "As at test time: the controller's own future commands are known; future outside temperature and occupancy are held at their last values",
   "Any forecast leaving 230–350 K marks the candidate unstable: its score becomes infinite and it cannot survive"
  ],
  "src": "[our code] closedform.evaluate, episode_scores (every 6, warmup 36, screen 5.0), evolve.py defaults (main horizon 36 steps)"
 },
 {
  "id": "pareto",
  "kind": "table",
  "eyebrow": "How a candidate is judged · selection",
  "title": "Five errors at once: a program survives unless another is at least as good on all five",
  "rows": [
   [
    "Objective",
    "What it measures (validation runs)",
    "Mostly decided by these coefficients"
   ],
   [
    "main",
    "3 h forecast error as at test time, + the size price",
    "all"
   ],
   [
    "fast",
    "1 h forecast error",
    "air, heat, neighbours"
   ],
   [
    "slow",
    "6 h forecast error",
    "slab, wall, inner and their _back terms"
   ],
   [
    "exterior",
    "3 h error in rooms with an outside wall",
    "outside, wall, wall_out, wall_back"
   ],
   [
    "interior",
    "3 h error in rooms without one",
    "neighbours, inner, inner_back"
   ]
  ],
  "colw": [
   0.14,
   0.46,
   0.4
  ],
  "size": 22,
  "font": 11,
  "callout": "A dominates B if A is no worse on every objective and better on one. Each island keeps its 6 best by front, then by how sparse their part of the trade-off is (crowding), so a program that gets one group of coefficients right survives while another group is still wrong. The deployed model is the one with the lowest main score.",
  "src": "[our code] discovery/pareto.py (fronts, crowding, select), closedform.OBJECTIVES, DEFAULT_OBJECTIVES, evolve.DB.elites, evolve.tournament",
  "notes": "Size is not its own objective: on its own axis the smallest program is never dominated and would stay a parent forever (seen in a test run: the one-node model kept being picked). HVAC-on and HVAC-off errors are available as objectives too; --selection scalar uses the main score alone."
 },
 {
  "kind": "table",
  "id": "decscoring",
  "eyebrow": "Decisions · scoring",
  "title": "How a candidate is judged, and why each setting",
  "rows": [
   [
    "Decision",
    "What we chose",
    "Why",
    "Rejected alternative / evidence"
   ],
   [
    "Physics checks",
    "equilibrium · bounded cooling · warmer outside never cools · heat warms · stiff air",
    "cheap, catches impossible physics before scoring",
    "score only: slow, and accepts lucky nonsense"
   ],
   [
    "Screen",
    "5-minute error above 5× persistence's → dropped",
    "saves the expensive 3 h forecasts for plausible candidates",
    "2×: dropped every hand-written model (tried)"
   ],
   [
    "Horizons",
    "1, 3 and 6 h; main horizon 3 h",
    "MPC horizon not chosen yet; 3 h covers a pre-heat",
    "one horizon only"
   ],
   [
    "Spacing, warm-up",
    "a 3 h forecast started every 30 min, after a 3 h warm-up (every hour in the part 8 evidence)",
    "dense enough to average days; the buffer needs time to fill",
    "every step: 6x slower"
   ],
   [
    "Stability",
    "any forecast leaving 230–350 K: status 'unstable', score infinite",
    "a good 5-minute fit does not guarantee a stable 3 h forecast",
    "seen: one-node model unstable on one episode"
   ],
   [
    "Test-time inputs",
    "future outside temperature and occupancy held at their last values",
    "a policy cannot know them at test time",
    "true future inputs: kept as a diagnostic"
   ],
   [
    "Complexity",
    "+0.0001 K per node of the code's syntax tree (≈ per operation)",
    "prefers readable formulas at equal accuracy",
    "no price: bloated candidates"
   ],
   [
    "Selection",
    "Pareto front over 5 validation errors; the three test tiers scored once at the end",
    "one number hides which coefficient is wrong; test stays unseen",
    "one score only: available as --selection scalar"
   ]
  ],
  "colw": [
   0.17,
   0.25,
   0.32,
   0.26
  ],
  "callout": None,
  "src": "[our code] discovery/closedform.py (evaluate, physics_checks, episode_scores), evolve.py defaults",
  "notes": "",
  "size": 21,
  "font": 9.5
 },
 {
  "kind": "cards",
  "id": "analysis",
  "eyebrow": "The feedback the LLM gets",
  "title": "The LLM is told where, when and through which term the parents are wrong",
  "cards": [
   [
    "When (temporal)",
    "error vs horizon\nby hour, occupancy\nHVAC on, off, switching\ncold or warm outside\nhow long errors persist",
    "pre-mutation",
    "accent1"
   ],
   [
    "Where (spatial)",
    "by room class, facade side, floor\nwhich room features the bias follows\nwhether adjacent rooms err together\nworst rooms",
    "pre-mutation",
    "accent1"
   ],
   [
    "Which term",
    "the 5-minute error regressed on each term's contribution, overall, per room type and per condition: slope s → the term should be (1 + s) times as large",
    "pre-mutation",
    "accent1"
   ],
   [
    "Before a crossover",
    "both parents on every group, rooms each wins, which parent is closer to right per term, per room type and condition",
    "pre-crossover",
    "accent2"
   ]
  ],
  "callout": "Computed by code on all validation episodes; sent with a real hourly trajectory excerpt. Saved as analysis.md next to each prompt.",
  "src": "[our code] closedform.analysis, temporal_stats, spatial_stats, term_regression, term_contributions, crossover_analysis",
  "notes": "The term contributions add up exactly to the model's one-step change (tested), so the regression attributes the residual to the formulas the LLM wrote."
 },
 {
  "id": "llmanalysis",
  "src": "[measured] closedform.analysis on seeds_cf/seed_cf_network.py, the 6 validation runs of the local split (outputs/discovery/real12/generations/g001/island_1/mutation/analysis.md)",
  "notes": "The local split from the 12 cached runs: validation = buildings 4, 5, 7 (denver_2, phoenix_1, singapore_6) × pid_water / HVAC off.",
  "kind": "code",
  "eyebrow": "Pre-mutation analysis (real output, validation runs)",
  "title": "What the LLM is told about the network model: when and where it is wrong",
  "code": "error vs horizon (rmse / bias / persistence, K):\n   5 min 0.247/-0.097/0.527   1 h 0.582/-0.271/0.764\n   3 h   0.757/-0.368/0.931   6 h 0.980/-0.498/1.358\n  -> half of the long-horizon error is already there after 1 h:\n     the fast terms (air, heat, neighbours) are off\none-step error autocorrelation at 1 h lag: +0.82 (persistent)\n3 h error (rmse, bias): target hour 04-08  0.870 -0.578\n                        HVAC air off       0.773 -0.396\n                        colder half        0.852 -0.514\ncorrelation of a room's bias with its geometry:\n   floor area -0.47  <- the formula's dependence on this is wrong\nerror correlation, adjacent rooms +0.87 vs non-adjacent +0.84\n   (similar: a building-wide cause)",
  "points": [
   "Too cold almost everywhere, worst before and during the morning start and with the HVAC off",
   "Larger rooms are biased colder: a coefficient scales wrongly with floor area",
   "Neighbouring rooms err together no more than distant ones: the cause is building-wide, not the coupling between rooms"
  ]
 },
 {
  "id": "llmterms",
  "kind": "code",
  "eyebrow": "Pre-mutation analysis (real output, validation runs)",
  "title": "…and through which term: the error regressed on each term's contribution",
  "code": "residual rms 0.2411 K per step; the terms explain 58% of it\n  air         size 0.3168 K/step  slope +0.015 +- 0.002  -> consistent\n  neighbours  size 0.2230 K/step  slope -0.742 +- 0.004  -> too strong by ~74%\n  slab        size 0.1361 K/step  slope +0.418 +- 0.011  -> too weak by ~42%\n  wall        size 0.4035 K/step  slope -0.450 +- 0.002  -> too strong by ~45%\n  inner       size 0.0826 K/step  slope -0.722 +- 0.019  -> too strong by ~72%\n  heat        size 0.4642 K/step  slope -0.001 +- 0.002  -> consistent\nexterior rooms only: wall -0.447 (too strong by ~45%)",
  "points": [
   "Each line: how big the term's 5-minute push is, and how the leftover error follows it",
   "Air and heat are right; the stores and the room-to-room exchange are not: the next mutation should change those formulas",
   "Terms that move together (neighbours, wall, inner) share the error, so the LLM is told to read their slopes together"
  ],
  "src": "[measured] closedform.term_regression on seeds_cf/seed_cf_network.py, 6 validation runs (same analysis.md)"
 },
 {
  "id": "termsbytype",
  "kind": "code",
  "eyebrow": "Pre-mutation analysis (real output, validation runs)",
  "title": "Which term is wrong, for which rooms, under which conditions",
  "code": "slope per term (s: the term should be about (1 + s) times as large); explained = share of the error\n                                 air  neighbours   slab    wall   inner   heat  explained\ncorner rooms                    -0.01   -0.68     +0.35   -0.44   -0.73  +0.02    46%\none-side exterior rooms         +0.03   -0.85     +0.68   -0.42   -1.17  -0.03    61%\ninterior rooms                  +0.16   -0.76     +1.06           -1.05  +0.00    88%\nlarger half of rooms            +0.01   -0.75     +0.33   -0.49   -0.74  -0.03    69%\nsmaller half of rooms           +0.02   -0.72     +0.50   -0.43   -0.77  +0.01    49%\nHVAC air on                     +0.03   -0.51     +0.04   -0.22   +0.20  +0.07    28%\nHVAC air off                    -0.50   -0.80     +0.07   -0.61   -1.14  -0.31    74%\noccupied hours                  +0.02   -0.66     +0.26   -0.38   -0.21  +0.03    40%\ncolder half of outside temps    +0.05   -0.83     +0.48   -0.47   -0.60  +0.01    68%",
  "points": [
   "Interior rooms: the slab term should be about twice as strong (+1.06), and the room-to-room exchange weaker: a formula for interior rooms, not a global factor",
   "With the HVAC off the stores (wall, inner) are far too strong; with it on they are nearly right: they need to depend on what the formula ignores",
   "This is the spatial and temporal detail the LLM writes formulas from"
  ],
  "src": "[measured] closedform.term_regression with ROOM_TYPES and CONDITIONS, network model, 6 validation runs"
 },
 {
  "id": "profiletab",
  "kind": "code",
  "eyebrow": "In every mutation request (real output)",
  "title": "A real day, hourly, one room of each type: measured vs the forecast made 3 h earlier",
  "code": "hour | corner (63 m2)  air  reheat | one-side (110 m2) | interior (200 m2) | T_out\n   6 |  9.45 /  8.48  2414 18525 | 18.36 / 17.48     | 20.55 / 19.49     | -1.7\n   8 | 21.96 / 20.64     0     0 | 20.50 / 19.94     | 21.66 / 21.21     |  1.1\n  10 | 17.37 / 15.72   645 11058 | 21.21 / 20.65     | 22.35 / 21.86     | 12.2\n  12 | 21.90 / 19.75     0     0 | 21.42 / 20.91     | 22.94 / 22.46     | 15.6\n  16 | 19.62 / 18.47     0     0 | 22.53 / 22.35     | 23.95 / 23.70     | 16.1\n  20 | 17.50 / 17.12     0     0 | 22.19 / 22.17     | 23.79 / 23.61     |  4.5\n   0 | 14.42 / 13.94     0     0 | 21.64 / 21.19     | 23.62 / 22.98     |  2.2\n   4 | 12.34 / 11.56     0     0 | 21.16 / 20.44     | 23.38 / 22.46     |  1.8",
  "points": [
   "The prompt has all 24 hours, plus each room's supply air and reheat and its geometry (area, outside wall, doors, neighbours)",
   "The corner room is forecast too cold all day, by up to 2.2 K while it warms up",
   "The interior room drifts too cold overnight with the HVAC off, matching the store slopes on the previous slide"
  ],
  "src": "[measured] evolve.profile_text, network model, validation episode 1 of 6 (54 rooms), day 2 (every 2 h shown here)"
 },
 {
  "id": "operators",
  "kind": "table",
  "eyebrow": "The search · four kinds of request",
  "title": "Four kinds of request, with the exact instruction each one ends with",
  "rows": [
   [
    "Operator",
    "What is sent",
    "The instruction (from prompts.py)"
   ],
   [
    "Init (generation 0)",
    "the system prompt alone (plus the island's recent rejected programs, if any)",
    "“Write n first candidates from the physics and the contract above. Make them differ from each other in structure (…).”"
   ],
   [
    "Mutation",
    "one parent's code, its five scores, the pre-mutation analysis, a measured-vs-predicted table; the island's recent rejected children and children that scored worse",
    "“Each must change ONE thing that the analysis points to (…). Use the TERMS slopes, the SPATIAL correlations and the TEMPORAL breakdown to decide which coefficient to change and in which direction, and say … which objective it should improve.”"
   ],
   [
    "Crossover",
    "two parents' code and the pre-crossover analysis (where and when each wins, which term each gets right); recent rejected and worse children",
    "“Write n children that COMBINE the parents: take each coefficient formula from the parent that is better where that coefficient matters (the TERMS lines say which parent's formula is closer to right for each term; …), make the formulas consistent, and state in the hypothesis which formula came from which parent.”"
   ],
   [
    "Explore",
    "the island's programs (names, scores, five objectives, hypotheses); recent rejected and worse children",
    "“Write n candidates with a STRUCTURALLY different hypothesis from all of the above (a different use of the stores, a different way to infer per-building materials from history, different coupling laws), not a re-tuning.”"
   ]
  ],
  "colw": [
   0.13,
   0.4,
   0.47
  ],
  "size": 20,
  "callout": "Every answer is JSON: for each candidate a name, a one-paragraph hypothesis, and the complete module.",
  "src": "[our code] prompts.cf_user_mutation, cf_user_crossover, cf_user_explore, RESPONSE_SCHEMA",
  "font": 11
 },
 {
  "id": "proposers",
  "kind": "compare",
  "eyebrow": "What goes to the API, and what does not",
  "title": "Every candidate comes from one stateless Claude API request; the episodes never leave the machine",
  "left": [
   "In one mutation request (real sizes)",
   [
    "System prompt (same every time, cached): ≈ 11 KB",
    "Parent's code ≈ 4 KB, its five scores; recent rejected children and children that scored worse",
    "Analysis (when, where, which term): ≈ 6 KB",
    "A real day, hourly, one room per type: ≈ 3 KB",
    "≈ 3,400 tokens (≈ 500 more with the worse-children list) plus the system prompt; no history"
   ]
  ],
  "right": [
   "Never sent",
   [
    "The episodes: 6 validation runs = 15 million numbers (≈ 26 million tokens)",
    "Test-tier numbers; building or station names",
    "Earlier requests or answers: nothing is kept",
    "Instead: the scorer runs each candidate on all of them here"
   ]
  ],
  "callout": "To write a formula the LLM needs which term is wrong, for which rooms, when, and a real day that shows it. More episodes cost more local computation, not a longer prompt.",
  "src": "[measured] prompt parts on the 6 validation runs (seeds_cf/seed_cf_network.py as parent); episode sizes n × V × (1 + 4) + building inputs; [our code] proposers.ClaudeProposer, prompts.py; tests/test_api_loop.py"
 },
 {
  "id": "llmout",
  "kind": "table",
  "eyebrow": "What comes back from the API, and what is done with it",
  "title": "The LLM returns code; the numbers come from running that code",
  "rows": [
   [
    "Step",
    "What",
    "Where"
   ],
   [
    "Answer",
    "JSON: for each candidate a name, a one-paragraph hypothesis, and the complete module",
    "generations/…/response.json"
   ],
   [
    "The module",
    "HISTORY, REFRESH, coefficients(geo, hist, U) returning the 11 coefficients as formulas; optionally heat()",
    "candidates/<id>.py"
   ],
   [
    "Checked",
    "allowed imports only, no fitted tables, no branching on building size, coefficients ≥ 0 and finite, physics checks",
    "database.jsonl (reason if rejected)"
   ],
   [
    "Scored",
    "on every validation episode: five objectives, the analysis for the next request",
    "database.jsonl, analysis.md"
   ],
   [
    "Kept",
    "Pareto front per island; best.py = lowest main score",
    "best.py, summary.json"
   ],
   [
    "At the end",
    "best.py and the front scored once on each test tier",
    "log.md, summary.json"
   ],
   [
    "Deployed",
    "best.py inside the policy: its formulas evaluated on the new building's floor plan and measured history",
    "the Kaggle package"
   ]
  ],
  "colw": [
   0.13,
   0.6,
   0.27
  ],
  "size": 21,
  "font": 11,
  "src": "[our code] prompts.RESPONSE_SCHEMA, contract.load_closedform, closedform.evaluate, evolve.run, score_tests"
 },
 {
  "id": "islands",
  "src": "[our code] evolve.py defaults (--islands 3, --elites 6, --n-mut 2, --n-cx 2, --n-exp 1, --migrate-every 3, --selection pareto), evolve.tournament, pareto.py",
  "notes": "Each operator is a separate LLM request per island per generation.",
  "kind": "flow",
  "eyebrow": "The search · evolution",
  "title": "Three islands; generation 0 is written by the LLM, then every island asks for five children per generation",
  "steps": [
   [
    "Pick",
    "2 random programs of the island's 6; keep the better on the Pareto front"
   ],
   [
    "Mutate",
    "1 request, 2 children"
   ],
   [
    "Cross",
    "1 request, 2 children"
   ],
   [
    "Explore",
    "1 request, 1 child"
   ],
   [
    "Keep",
    "the island's 6 best by front and crowding"
   ]
  ],
  "points": [
   "Generation 0: per island, one request for 3 first candidates from the system prompt alone. Then 3 islands × 5 children = 15 new programs per generation",
   "Every 3 generations each island's best is copied to the next island (1 → 2 → 3 → 1), so good ideas spread but islands stay different",
   "Every request, analysis, answer and score is written to disk (part 8: where to look)"
  ]
 },
 {
  "kind": "flow",
  "id": "genloop",
  "eyebrow": "One generation, step by step",
  "title": "One generation on one island, step by step",
  "steps": [
   [
    "1 Pick parents",
    "better of 2 random, on the island"
   ],
   [
    "2 Analyse",
    "where each parent is wrong"
   ],
   [
    "3 Ask the LLM",
    "mutate · cross · explore"
   ],
   [
    "4 Check",
    "physics; 5-min error"
   ],
   [
    "5 Forecast",
    "1/3/6 h on validation runs"
   ],
   [
    "6 Keep",
    "record; migrate best"
   ]
  ],
  "points": [
   "Each LLM request is one operator: mutate (fix one parent), cross (combine two) or explore (a new idea); all of it is saved to disk",
   "Main score = 3 h forecast error on the validation runs: the model runs on its own predictions for 3 h (a 'rollout')",
   "Every 3 generations each island sends its best program to the next island"
  ],
  "src": "[our code] discovery/evolve.py run(), closedform.evaluate",
  "notes": "Generation 0: each island asks the API for 3 first candidates written from the system prompt alone."
 },
 {
  "kind": "table",
  "id": "decsearch",
  "eyebrow": "Decisions · search settings",
  "title": "How the evolutionary search is configured, and why",
  "rows": [
   [
    "Decision",
    "What we chose",
    "Why",
    "Rejected alternative / evidence"
   ],
   [
    "Islands",
    "3 islands; each island's best copied to the next every 3 generations",
    "independent populations keep different ideas alive; migration shares winners",
    "one population: collapses onto one idea"
   ],
   [
    "Requests",
    "generation 0: 1 init request per island (3 candidates); then per island per generation: 1 mutation (2 children), 1 crossover (2), 1 explore (1)",
    "balance refining, combining and new structure",
    "explore only: no refinement"
   ],
   [
    "Parents",
    "pick 2 at random from the island's 6; keep the one on the better Pareto front",
    "favours good parents without always picking the same one",
    "always the best: no diversity"
   ],
   [
    "Analysis",
    "where, when and through which term, in every request",
    "tells the LLM which coefficient is wrong and in which direction",
    "score only: the LLM guesses"
   ],
   [
    "LLM",
    "Claude API only: one stateless request per operator; a --max-usd budget",
    "the context is the prompt alone, never a chat history",
    "one long chat: earlier answers leak into later ones"
   ],
   [
    "Robustness",
    "tolerant parsing + one retry; answers saved before evaluation; resume",
    "never pay twice for a request; a crash loses nothing",
    "-"
   ],
   [
    "Speed",
    "--workers N forked evaluation (same results as serial)",
    "evaluation dominates wall time",
    "-"
   ],
   [
    "Data per candidate",
    "up to 30 validation episodes (deterministic sample); no training episodes",
    "bounded cost per candidate",
    "all episodes: slow on a large dataset"
   ]
  ],
  "colw": [
   0.17,
   0.25,
   0.32,
   0.26
  ],
  "callout": None,
  "src": "[our code] discovery/evolve.py, proposers.py, prompts.py",
  "notes": "",
  "size": 21,
  "font": 9.5
 },
 {
  "kind": "flow",
  "id": "pipeline",
  "eyebrow": "The whole pipeline",
  "title": "From generated data to a controller",
  "steps": [
   [
    "Episodes",
    "kit runs, read"
   ],
   [
    "Split",
    "whole episodes"
   ],
   [
    "Init",
    "API writes generation 0"
   ],
   [
    "Score",
    "validation, 5 objectives"
   ],
   [
    "Evolve",
    "API: mutate, cross, explore"
   ],
   [
    "Test",
    "held-out tiers, once"
   ]
  ],
  "points": [
   "Run on the machine that holds the data; nothing is generated or simulated in this pipeline",
   "One command: ./run_pipeline.sh (episodes → split → discovery)",
   "Control (part 10) comes after"
  ],
  "src": "[our code] run_pipeline.sh, control/ (part 10)",
  "notes": "Nothing touches the kit."
 },
 {
  "kind": "cards",
  "id": "discoutput",
  "eyebrow": "What comes out",
  "title": "The output of discovery, and where it goes",
  "cards": [
   [
    "best.py",
    "The winning coefficient formulas: coefficients(), and heat() if the LLM wrote one. Readable code (the hand-written models are 20–65 lines); about 0.1 ms per step.",
    "the model",
    "accent1"
   ],
   [
    "REPORT.md, log.md",
    "How we got there: leaderboard, family tree, operator success, every prompt and answer, measured-vs-predicted plots.",
    "the evidence",
    "accent2"
   ],
   [
    "Into the policy",
    "Built (part 9): the policy computes geometry from the metadata and tracks the model live; a controller can then plan with it (part 10, a later step).",
    "parts 9–10",
    "accent3"
   ]
  ],
  "src": "[our code] evolve.py, report.py; runtime/, control/",
  "notes": "summary.json holds the scores and the test-tier results."
 },
 {
  "kind": "table",
  "id": "bigmachine",
  "eyebrow": "Running the pipeline",
  "title": "One command runs the whole pipeline",
  "rows": [
   [
    "Step",
    "What it does",
    "Command (inside ./run_pipeline.sh)"
   ],
   [
    "1",
    "turn the kit's finished runs into episodes (one file per run)",
    "code/build_dataset.py --results R --scenarios S --out D"
   ],
   [
    "2",
    "split whole episodes: validation + test tiers",
    "code/make_splits.py --dataset D"
   ],
   [
    "3",
    "discovery through the Claude API; at the end the chosen model and the Pareto front on the test tiers",
    "code/run_discovery.py --dataset D --generations 20 --test --max-usd 50"
   ]
  ],
  "colw": [
   0.08,
   0.5,
   0.42
  ],
  "mono_first": False,
  "callout": "Data generation happens beforehand with the starter kit; this pipeline only reads it.",
  "src": "[our code] run_pipeline.sh, docs/RUNBOOK.md",
  "notes": "Settings (paths, generations, budget, model) are environment variables at the top of run_pipeline.sh."
 },
 {
  "kind": "section",
  "id": "s8",
  "num": "8",
  "title": "Evidence",
  "question": "How do we measure how wrong a model is, and what do the six shipped buildings' 30-day runs say about the hand-written models?",
  "notes": ""
 },
 {
  "kind": "table",
  "id": "devmodel",
  "eyebrow": "Model deviations",
  "title": "How we measure how wrong a model is",
  "rows": [
   [
    "Measure",
    "Definition",
    "Why"
   ],
   [
    "One-step error",
    "RMSE (root-mean-square error, K) of the 5-minute prediction from measured history",
    "quick screen (stage 1)"
   ],
   [
    "Forecast error at h",
    "RMSE when the model runs on its own predictions for 1, 3, 6 h; started every 30 min (every hour in this part)",
    "what MPC uses; decides (stage 2)"
   ],
   [
    "Skill",
    "forecast error ÷ persistence error at the same horizon",
    "below 1 = better than ‘temperature stays’"
   ],
   [
    "Bias",
    "mean signed error",
    "a wrong steady-state gain shows here"
   ],
   [
    "By group",
    "exterior / interior / most exposed; HVAC on/off; comfort hours; reheat; cold/warm",
    "says where to fix (pre-mutation)"
   ],
   [
    "Stability",
    "any forecast leaving 230–350 K",
    "rejects the candidate"
   ],
   [
    "Gap to ceiling",
    "candidate vs the 'ceiling': a per-room fitted model, never deployed",
    "is the formula or the data the limit?"
   ]
  ],
  "colw": [
   0.17,
   0.53,
   0.3
  ],
  "src": "[our code] closedform.episode_scores, analysis, ceiling",
  "notes": "In a real run all of these are computed on validation buildings the search was not shown."
 },
 {
  "kind": "cards",
  "id": "canfit",
  "eyebrow": "Three reference points",
  "title": "Is a bad score the formula's fault or the data's?",
  "cards": [
   [
    "Floor: persistence",
    "The forecast 'temperature stays where it is'. Every candidate must beat it.",
    "floor",
    "accent3"
   ],
   [
    "Reference: the hand-written models",
    "The network and history models we wrote by hand, scored on the same real runs (next slide). They are references only; the search starts from what the LLM writes.",
    "reference",
    "accent1"
   ],
   [
    "Ceiling: fitted per room",
    "Non-negative per-room weights fitted on half of each run; the non-negative weights keep its forecasts stable. Never deployed.",
    "diagnostic only",
    "accent2"
   ]
  ],
  "callout": "Why non-negative weights: an unconstrained per-room fit ran away when forecasting on its own predictions (101 K error at 3 h on real data). A good 5-minute fit is not a good forecast.",
  "src": "[our code] closedform.ceiling; [measured]",
  "notes": "A good one-step fit does not guarantee a good forecast."
 },
 {
  "kind": "image",
  "id": "seedscompare",
  "eyebrow": "Real data · the hand-written models, nothing fitted",
  "title": "The network model is the best hand-written model; all are far from the ceiling",
  "img": "seeds_compare.png",
  "side": "bottom",
  "points": [
   "Network model: 0.68 / 0.87 / 1.11 K at 1 / 3 / 6 h; persistence 0.66 / 0.97 / 1.47",
   "History model: 1.17 / 1.40 / 1.64 K: one store, materials partly read from history",
   "Score with weather unknown, 3 h: network 0.902, history 1.262 K",
   "Ceiling 0.31 / 0.46 / 0.62 K; the one-node model goes unstable on one run"
  ],
  "caption": "[sim data] 12 thirty-day runs, forecasts started every hour; nothing fitted except the diagnostic ceiling",
  "src": "[measured] closedform.evaluate and closedform.ceiling on the kit runs replayed by our starterkit_data (make_figs9.py)",
  "notes": ""
 },
 {
  "kind": "image",
  "id": "byepisode",
  "eyebrow": "Real data, per episode",
  "title": "Where the history model loses: 9 of 12 runs at 3 h, including all six with the HVAC off",
  "img": "ep30_by_episode.png",
  "side": "bottom",
  "points": [
   "The network model loses to persistence in 3 of 12 runs; the history model in 9, every HVAC-off run among them",
   "The ceiling (per-room fitted, diagnostic) beats persistence in all 12 runs, on the second half of each run",
   "The gap between the hand-written models and the ceiling is what the LLM search has to close"
  ],
  "caption": "[sim data] 3 h forecast RMSE per 30-day run, history model; b1 = building_train_1, pid = pid_water, off = HVAC off",
  "src": "[measured] closedform.evaluate (every 12 steps) and closedform.ceiling on the 12 thirty-day runs",
  "notes": ""
 },
 {
  "id": "tiers",
  "kind": "table",
  "eyebrow": "Held-out tiers (real runs)",
  "title": "On unseen buildings the network model beats 'temperature stays'; in the unseen climate it does not",
  "rows": [
   [
    "Model",
    "Validation (6 runs)",
    "Test · medium: buildings 2, 8 (4 runs)",
    "Extra: building 1 at Fairbanks (2 runs)"
   ],
   [
    "Persistence ('temperature stays')",
    "0.889",
    "0.948",
    "1.274"
   ],
   [
    "Network model (hand-written)",
    "0.740",
    "0.538",
    "2.114"
   ],
   [
    "History model (hand-written)",
    "1.099",
    "1.270",
    "1.732"
   ],
   [
    "One-node model (hand-written)",
    "5.707",
    "5.043 (unstable)",
    "10.636"
   ]
  ],
  "colw": [
   0.34,
   0.18,
   0.24,
   0.24
  ],
  "size": 21,
  "font": 11,
  "callout": "3 h forecast error (K) as at test time, hand-written models. The network model is better than persistence on unseen buildings, but every model loses to it at Fairbanks: the climate the hard tier is for.",
  "src": "[measured] closedform.evaluate on 12 thirty-day runs (6 shipped buildings × pid_water / HVAC off), split by splits.split_from_episodes",
  "notes": "Each building has one scenario among these 12 runs, so the easy and hard tiers are empty in this check."
 },
 {
  "kind": "image",
  "id": "realprofile",
  "eyebrow": "Real data · measured vs predicted",
  "title": "Week 2 of a 30-day run: what the network model gets right, and where it is wrong",
  "img": "ep30_profile_pred.png",
  "side": "right",
  "points": [
   "Daily shape: right in both rooms",
   "Under the baseline's rapid switching, the forecast swings lower than the room does",
   "Exposed room: up to about 0.8 K too warm at night; interior room a little too cold",
   "These are the errors the pre-mutation analysis points the LLM to"
  ],
  "caption": "[sim data] building_train_2, Atlanta window 3, pid_water, days 7–14; network model (seed_cf_network), nothing fitted",
  "src": "[measured] closedform.profile on the replayed 30-day episode",
  "notes": ""
 },
 {
  "kind": "table",
  "id": "wheretolook",
  "eyebrow": "Where to look",
  "title": "Every proposal, mutation and crossover leaves a file",
  "rows": [
   [
    "File (outputs/discovery/<run>/)",
    "What it shows"
   ],
   [
    "REPORT.md",
    "best model, leaderboard, operator success, family tree, plots"
   ],
   [
    "log.md",
    "each request and each child, with its change against the parent"
   ],
   [
    "generations/gNNN/island_K/<operator>/prompt.md",
    "exactly what the LLM was asked"
   ],
   [
    "…/analysis.md · …/response.json",
    "what it was told · what it answered"
   ],
   [
    "candidates/<id>_<name>.py",
    "every candidate’s code"
   ],
   [
    "plots/<id>_profile.png · _horizon.png",
    "measured vs predicted · error vs horizon"
   ],
   [
    "database.jsonl",
    "island, operator, parents, scores per candidate"
   ]
  ],
  "colw": [
   0.45,
   0.55
  ],
  "mono_first": True,
  "src": "[our code] evolve.py, report.py; docs/DISCOVERY_WALKTHROUGH.md",
  "notes": ""
 },
 {
  "kind": "section",
  "id": "s9",
  "num": "9",
  "title": "Inside the policy",
  "question": "Does the model behave the same inside a submission-style policy as it did offline, using only test-time information?",
  "notes": ""
 },
 {
  "kind": "table",
  "id": "parity",
  "eyebrow": "Running the model inside the policy",
  "title": "Inside the policy, the forecasts match the offline scores",
  "rows": [
   [
    "Check (building_train_8, atlanta_1, 576 steps)",
    "Live (inside the policy)",
    "Offline (episode)"
   ],
   [
    "Geometry: all 17 fields",
    "from the Kaggle-style metadata",
    "identical"
   ],
   [
    "Inputs: outside air, comfort, occupants, gains",
    "from observations",
    "identical"
   ],
   [
    "Inputs: air streams, reheat",
    "from own commands",
    "within 0.04 W/K and 0.3 W (CSV rounding)"
   ],
   [
    "History model: 5-min / 3 h error",
    "0.5467 / 2.3963 K",
    "0.5467 / 2.3963 K"
   ],
   [
    "Network model: 5-min / 3 h error",
    "0.1950 / 0.5775 K",
    "0.1950 / 0.5775 K"
   ],
   [
    "Model time per step",
    "0.1 ms (30 rooms)",
    "-"
   ]
  ],
  "colw": [
   0.42,
   0.3,
   0.28
  ],
  "callout": "The live check showed that the occupancy observation arrives two steps late, so the offline episodes use that observable value too.",
  "src": "[measured] run_policy.py --check-replay on building_train_8 / atlanta_1 (seeds_cf/seed_cf_network.py, seed_cf_history.py); runtime/closedform_tracker.py"
 },
 {
  "kind": "table",
  "id": "deploy",
  "eyebrow": "Deployment parity",
  "title": "Discovery uses only what a policy has at Kaggle test time",
  "rows": [
   [
    "What discovery uses",
    "At test time?",
    "Source at run time"
   ],
   [
    "geometry",
    "yes",
    "floor plan, zone map, AHU assignment (metadata)"
   ],
   [
    "room temperatures, outside air, comfort, clock",
    "yes",
    "observations"
   ],
   [
    "occupants (building total)",
    "yes, 2 steps late",
    "observation"
   ],
   [
    "air streams, supply temperature, reheat, boiler water",
    "yes",
    "own commands + the kit's equipment laws"
   ],
   [
    "internal gains, universal constants",
    "assumed",
    "kit-wide values with cited sources"
   ],
   [
    "future outside temperature and occupancy",
    "no",
    "held at last observed values in the score"
   ],
   [
    "building identity, materials, weather window",
    "no",
    "never given; LLM analysis text is anonymous"
   ]
  ],
  "colw": [
   0.4,
   0.18,
   0.42
  ],
  "size": 25,
  "src": "docs/DISCOVERY_WALKTHROUGH.md (Deployment parity); tests/test_deployment.py"
 },
 {
  "kind": "table",
  "id": "decruntime",
  "eyebrow": "Decisions · runtime",
  "title": "How the model runs inside the policy, and why",
  "rows": [
   [
    "Decision",
    "What we chose",
    "Why",
    "Rejected alternative / evidence"
   ],
   [
    "Timing",
    "the same conventions as the offline episodes (observation lag, AHU start delay, occupancy lag)",
    "a model trained on one timing must run on the same",
    "checked by the parity test"
   ],
   [
    "Shared code",
    "one command-to-heat module for the tracker and the fitted runtime",
    "two copies would drift apart",
    "-"
   ],
   [
    "Safety layer",
    "clip into bounds, replace NaN/inf, fall back on any exception",
    "an invalid action is rejected or corrupts the episode",
    "fuzz test: 500 malformed actions all repaired"
   ],
   [
    "Fallback",
    "the kit's pid_water if a model-based controller fails to start or errors",
    "never worse than the baseline because of a crash",
    "no fallback: one bug loses the scenario"
   ],
   [
    "Tracker errors",
    "caught; after 10 errors the tracker is switched off",
    "tracking must never break control",
    "-"
   ]
  ],
  "colw": [
   0.17,
   0.25,
   0.32,
   0.26
  ],
  "callout": None,
  "src": "[our code] runtime/, submission/safety.py, submission/transfer_policy.py",
  "notes": "",
  "size": 21,
  "font": 9.5
 },
 {
  "kind": "section",
  "id": "s10",
  "num": "10",
  "title": "After discovery: control",
  "question": "Once the forecasting model exists, how would a controller use it? (a later step; discovery comes first)",
  "notes": ""
 },
 {
  "id": "whympc",
  "src": "[our code] control/policies.py (MPCPolicy: horizon 24, replan 3), control/planner.py (MODES, cost), control/costs.py, control/schedule.py",
  "notes": "Known gaps of an energy proxy: fan power ∝ damper × pressure^1.5; cooling and gas depend on temperatures; the cost reference scales with the number of AHUs; comfort is weighted by unobserved occupants per room.",
  "kind": "eq",
  "eyebrow": "A later step, after discovery",
  "title": "What 'model predictive control' means here, in plain words",
  "eq": "every 15 minutes:  for each mode m in {schedule, comfort, eco, off}:\n    cost(m) = Σ over the next 2 h of  0.6·comfort + 0.2·energy + 0.2·carbon,  rooms forecast by the discovered model\nrun the mode with the lowest cost for the next 15 minutes, then ask again",
  "eqsize": 21,
  "eqpt": 11,
  "points": [
   "It does not search over raw commands. It compares 4 ready-made ways of running the building: schedule = the kit's pid_water as usual; comfort = hold the comfort band now (start pre-heating early); eco = hold the wider eco band now (coast on stored heat); off = air handlers and boiler off",
   "To price a mode, a copy of pid_water is run on the model's forecast rooms for 24 five-minute steps; its commands give energy and carbon, the forecast temperatures give comfort",
   "So the controller is only as good as the forecast at 1 to 3 hours. That is why discovery scores 1, 3 and 6 h forecasts, and why discovery comes first",
   "Feed-forward is a simpler use: pid_water, with the model only deciding when to start pre-heating and when to coast"
  ]
 },
 {
  "kind": "image",
  "id": "costval",
  "eyebrow": "The reward as a cost",
  "title": "Our energy estimate matches the kit's logged costs",
  "img": "cost_validation.png",
  "side": "bottom",
  "points": [
   "Electric power correlation 0.999 (within 3–7%); gas 0.977; normalised energy term within ~3%",
   "Comfort term exact: the kit's shortfall (linear to 5 K, then quadratic), occupancy-weighted"
  ],
  "caption": "[sim data] two 30-day pid_water runs; estimate from commands vs power implied by reward_components.csv",
  "src": "[our code] control/costs.py; [measured] docs/SOURCES_AND_FACTS.md §3b; price and carbon tables from the kit",
  "notes": "Mild weather: boiler losses (not modelled) explain the small gas gap."
 },
 {
  "kind": "image",
  "id": "occfc",
  "eyebrow": "Occupancy forecast",
  "title": "The comfort term needs occupants ahead of time: learn each building's day",
  "img": "occupancy_forecast.png",
  "side": "bottom",
  "points": [
   "Schedule prior (arrivals 06–12, departures 13–19), then the building's own weekday profile from the observed head count",
   "3 h ahead error 4–16 people vs persistence 15–43 on four 30-day runs; split between rooms by floor area"
  ],
  "caption": "[sim data] building_train_2, atlanta_3, 30 days",
  "src": "[our code] control/occupancy.py"
 },
 {
  "id": "gaincalc",
  "kind": "eq",
  "eyebrow": "Early control checks · how the gains are calculated",
  "title": "Same building, same weather, same two days: only the controller changes",
  "eq": "score = mean over the 576 steps of the competition's loss per step (0.6·comfort + 0.2·energy + 0.2·carbon)\nchange = ( score_controller − score_pid_water ) / score_pid_water\nFairbanks, MPC:  ( 0.02230 − 0.02309 ) / 0.02309 = −3.4%      (negative = better)",
  "eqsize": 22,
  "eqpt": 11,
  "points": [
   "Each run is the first 2 days (576 five-minute steps) of the shipped scenario, through the kit's own harness; the simulator is deterministic, so a difference is due to the controller alone",
   "pid_water: the kit's best rule-based controller. Feed-forward: pid_water, but the model chooses when to start pre-heating and when to coast. MPC: every 15 minutes, for each of 4 modes (schedule, comfort, eco, off), the model predicts 2 h ahead with pid_water acting on the predicted rooms, prices it with the competition's loss, and the cheapest mode runs",
   "'Mean' is the average of the two buildings' scores. These are 2-day scores, so they are not comparable with the 30-day baseline score of 0.054"
  ],
  "src": "[our code] rehearse_submission.py, control/policies.py (MPCPolicy, FeedForwardPolicy), control/planner.py, control/costs.py; [measured] outputs/rehearsal/check_*/report.json"
 },
 {
  "kind": "image",
  "id": "ctrlres",
  "eyebrow": "Early control checks (not the focus yet)",
  "title": "Early check: model-based control gains a little over pid_water in two short runs, mostly in the cold",
  "img": "controllers.png",
  "side": "right",
  "points": [
   "Score change against pid_water (negative = better). MPC: −3.4% in fairbanks_1, +0.5% in atlanta_1; −1.4% on the mean",
   "Feed-forward (pid_water, but the model decides when to start pre-heating and when to coast in the evening): −1.8% in fairbanks_1, equal in atlanta_1",
   "Two buildings, two days each: a hint, not the benchmark",
   "Planning takes 35–45 ms every 3 steps on 30–39 rooms"
  ],
  "caption": "[measured] code/rehearse_submission.py through the kit's harness, network model (seeds_cf/seed_cf_network.py)",
  "src": "[measured] rehearse_submission.py --controller pid_water|ff|mpc --steps 576 (outputs/rehearsal/check_*/report.json); make_figs10.py",
  "notes": "The 30-scenario benchmark at 8,640 steps (task I5) decides."
 },
 {
  "kind": "table",
  "id": "decctrl",
  "eyebrow": "Decisions · controllers",
  "title": "How the controllers are built, and why",
  "rows": [
   [
    "Decision",
    "What we chose",
    "Why",
    "Rejected alternative / evidence"
   ],
   [
    "Base",
    "the kit's pid_water (vendored, Apache-2.0)",
    "a proven, safe inner loop; the model decides when and how hard",
    "raw-command MPC: 200+ dimensions per step"
   ],
   [
    "Reduced action",
    "modes: schedule / comfort / eco / off",
    "4 candidates are cheap to simulate and cover pre-heat, coast and switch-off",
    "continuous setpoint search"
   ],
   [
    "Planning",
    "copy pid_water, step it on predicted observations, price with the reward",
    "plans are closed-loop, like the real controller",
    "open-loop held actions"
   ],
   [
    "Horizon",
    "MPC 2 h, replanned every 15 min; feed-forward looks up to 3 h ahead",
    "covers the morning pre-heat within the time budget",
    "longer horizon: more time per step"
   ],
   [
    "Cost",
    "the reward itself: comfort with forecast occupants, energy and carbon from commands",
    "optimises the scored objective",
    "hand-tuned weights"
   ],
   [
    "Unknowns",
    "outside air held at its last value; gas price = annual mean; holidays ignored",
    "the policy cannot read the date or future weather",
    "open item"
   ],
   [
    "On/off",
    "the 'off' mode: every AHU and the boiler off when predicted cheapest",
    "switching off is the biggest saving at night",
    "its effect alone not yet measured"
   ]
  ],
  "colw": [
   0.17,
   0.25,
   0.32,
   0.26
  ],
  "callout": None,
  "src": "[our code] control/planner.py, control/policies.py, control/costs.py, control/schedule.py",
  "notes": "",
  "size": 21,
  "font": 9.5
 },
 {
  "kind": "section",
  "id": "s11",
  "num": "11",
  "title": "The Kaggle submission",
  "question": "What gets uploaded, how is it checked, and when is it ready?",
  "notes": ""
 },
 {
  "kind": "table",
  "id": "upload",
  "eyebrow": "What gets uploaded",
  "title": "One notebook plus one attached dataset",
  "rows": [
   [
    "Piece",
    "What it is",
    "Check"
   ],
   [
    "kaggle/submission.ipynb",
    "subclass of smart_buildings_inference_server.Policy wrapping TransferPolicyCore",
    "runs the public example scenario"
   ],
   [
    "dist/kaggle_dataset/code",
    "our runtime, controllers, safety layer, vendored pid_water (26 files)",
    "imports in a clean interpreter without the kit"
   ],
   [
    "dist/kaggle_dataset/best.py",
    "the model file: today the hand-written network model (seeds_cf/seed_cf_network.py); the discovered formulas after LLM runs",
    "MPC used when present; pid_water otherwise"
   ],
   [
    "code/rehearse_submission.py",
    "the same policy through the kit's harness on the 30 shipped scenarios",
    "score, seconds per step, safety repairs"
   ]
  ],
  "colw": [
   0.3,
   0.45,
   0.25
  ],
  "src": "[our code] package_submission.py, rehearse_submission.py, kaggle/submission.ipynb"
 },
 {
  "kind": "table",
  "id": "gate",
  "eyebrow": "Readiness gate",
  "title": "What must be true before the model-based notebook is submitted",
  "rows": [
   [
    "Item",
    "Status (task)"
   ],
   [
    "pid_water-only notebook scored on Kaggle (proves the interface)",
    "to do (Planning 11)"
   ],
   [
    "Per-step time guard for large buildings",
    "to do (Planning 08)"
   ],
   [
    "Unsupported HVAC variants detected, fall back to pid_water",
    "to do (Planning 09)"
   ],
   [
    "Benchmark: 30 shipped scenarios at 8,640 steps + hold-out buildings and station",
    "to do (Planning 10)"
   ],
   [
    "Packaged dataset rehearsed locally; fallback verified by fault injection",
    "to do (Planning 12)"
   ],
   [
    "Deployment parity: test-time information only",
    "done"
   ]
  ],
  "colw": [
   0.72,
   0.28
  ],
  "callout": "Ready to submit today: the pid_water-only notebook. The MPC notebook waits for this gate.",
  "src": "docs/RUNBOOK.md §11; Planning/index.md"
 },
 {
  "kind": "section",
  "id": "s12",
  "num": "12",
  "title": "Where we are",
  "question": "What is built, what is open, and what comes next?",
  "notes": ""
 },
 {
  "kind": "table",
  "id": "figsources",
  "eyebrow": "Which figures are data",
  "title": "Every figure in this deck, and where it comes from",
  "rows": [
   [
    "Figure",
    "Kind",
    "Source"
   ],
   [
    "30-day profiles, week zoom, weather, HVAC off vs baseline, loss by day, comfort deviation",
    "real data",
    "kit evaluate.py, 6 buildings × 30 days"
   ],
   [
    "Linear in temperature, fast-then-slow, commands → heat",
    "real data",
    "same runs, replayed by our starterkit_data"
   ],
   [
    "Cooling vs outside wall, hand-written models vs persistence vs ceiling, per episode, measured vs predicted",
    "real data",
    "same runs; closedform.py"
   ],
   [
    "Hand-written models compared, cost check, occupancy forecast, controller results",
    "real data / measured",
    "same 30-day runs; closedform.py, control/*.py, rehearse_submission.py"
   ],
   [
    "Every observation group",
    "real data (3 days)",
    "kept audit file outputs/roomdata_building_train_2_atlanta_3_pidwater.npz"
   ],
   [
    "Fitting constants (persistence vs network)",
    "real data",
    "kept audit outputs/room_model_forecast.json"
   ],
   [
    "Room classes floor plan",
    "real (geometry)",
    "outputs/building_train_2_rooms.png"
   ],
   [
    "Plant, layouts, overview",
    "kit images",
    "starter kit images/"
   ],
   [
    "Baselines",
    "kit numbers",
    "baseline_returns.json"
   ],
   [
    "Floor plans of the six buildings; weather year with the 30 simulated episodes",
    "real (geometry, weather)",
    "kit zone_map.npy, weather/csv; make_figs11.py"
   ],
   [
    "Cell grid, one room, the network matrix",
    "real (geometry)",
    "building_train_8 floor plan"
   ]
  ],
  "colw": [
   0.5,
   0.17,
   0.33
  ],
  "src": "this deck",
  "notes": ""
 },
 {
  "kind": "table",
  "id": "status",
  "eyebrow": "Status",
  "title": "What is built, what is partial, what is not",
  "rows": [
   [
    "Piece",
    "Status"
   ],
   [
    "Data generation with the kit (outside this pipeline)",
    "not run yet: results use the 6 shipped buildings"
   ],
   [
    "Dataset building and audits (kit tools only)",
    "built"
   ],
   [
    "Geometry from floor plan and zone map",
    "built"
   ],
   [
    "Coefficient contract: the LLM writes a formula for each coefficient of a fixed state-space function; no fitting",
    "built and tested"
   ],
   [
    "Splits from the data: validation + easy / medium / hard test tiers",
    "built and tested"
   ],
   [
    "Analysis: temporal, spatial, per-term",
    "built and tested on real runs"
   ],
   [
    "Scoring with test-time information; floor and ceiling",
    "built and tested"
   ],
   [
    "Islands, operators, Pareto selection, logs, report, resume, parallel",
    "built; exercised without an LLM"
   ],
   [
    "Model inside the policy; live = offline",
    "built and verified"
   ],
   [
    "Cost model, occupancy forecast, feed-forward, MPC",
    "built; short checks only"
   ],
   [
    "Kaggle notebook, package, rehearsal, safety layer",
    "built; not yet run on Kaggle"
   ],
   [
    "Real API generations; 30-scenario benchmark; time guard",
    "not yet"
   ]
  ],
  "colw": [
   0.6,
   0.4
  ],
  "src": "[our code]; Planning/index.md (pending tasks)",
  "notes": "",
  "size": 22
 },
 {
  "kind": "table",
  "id": "decisions",
  "eyebrow": "Decisions",
  "title": "Settled, and still open",
  "rows": [
   [
    "Open question",
    "Options"
   ],
   [
    "Goal",
    "best Kaggle score, or a generic discovered model"
   ],
   [
    "Footprint range for generated buildings",
    "the generator's 400–700 × 300–500 ft, the shipped six's smaller sizes, or both"
   ],
   [
    "Simulator structure in the prompt",
    "graph only, or the full physics chain"
   ],
   [
    "MPC horizon (a later step)",
    "2 h in use; 1, 3, 6 h all reported"
   ],
   [
    "History summary",
    "raw lags, differences, window averages, LLM-proposed features"
   ],
   [
    "Occupants per room",
    "split equally (the hand-written models) or by floor area (the controller's cost); a time-of-day rule is untested"
   ],
   [
    "Command-to-heat conversion",
    "keep hand-written, or let the LLM discover it"
   ],
   [
    "Online updates",
    "keep nothing fitted, or let the policy adjust a few constants as data arrives"
   ]
  ],
  "colw": [
   0.38,
   0.62
  ],
  "callout": "Settled: no fitting · no training split · universal constants · a fixed state-space function whose coefficients the LLM writes · hidden stores corrected by measurements · test-time scoring · Pareto selection · discovery before control.",
  "src": "[decision] planning notes",
  "notes": ""
 },
 {
  "kind": "flow",
  "id": "roadmap",
  "eyebrow": "Roadmap",
  "title": "From here to a submission, with a fallback at every stage",
  "steps": [
   [
    "Kaggle check",
    "pid_water notebook"
   ],
   [
    "Time guard",
    "large buildings"
   ],
   [
    "Real LLM runs",
    "Claude API, test tiers"
   ],
   [
    "Benchmark",
    "30 scenarios"
   ],
   [
    "Control",
    "MPC notebook, after the gate"
   ]
  ],
  "points": [
   "Every stage is a complete submission with a working fallback",
   "5 submissions a day; 2 final picks; deadline January 15, 2027"
  ],
  "src": "[plan] Planning/index.md (14 steps); [Kaggle] timeline, limits",
  "notes": "Each stage is a complete submission."
 },
 {
  "kind": "statement",
  "id": "close",
  "eyebrow": "In one line",
  "title": "Formulas we can read, written once, run on any building",
  "sub": "Geometry from the floor plan · materials from the measured history · everything else universal · nothing fitted",
  "src": "",
  "notes": "Questions?"
 }
]

SECTIONS = [
 [
  "s1",
  "1 · The problem"
 ],
 [
  "s2",
  "2 · The data"
 ],
 [
  "s3",
  "3 · The data over 30 days"
 ],
 [
  "s4",
  "4 · The physics, from cell to state space"
 ],
 [
  "s5",
  "5 · The usual approaches"
 ],
 [
  "s6",
  "6 · Our approach: formulas for the unknowns"
 ],
 [
  "s7",
  "7 · LLM discovery"
 ],
 [
  "s8",
  "8 · Evidence"
 ],
 [
  "s9",
  "9 · Runtime and parity"
 ],
 [
  "s10",
  "10 · After discovery: control"
 ],
 [
  "s11",
  "11 · Submission"
 ],
 [
  "s12",
  "12 · Next"
 ]
]
