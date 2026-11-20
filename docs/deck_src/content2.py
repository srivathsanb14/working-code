"""Restructured storyline (nine parts, for an uninitiated audience).  Reuses slides from content.py by id, replaces
the drawn physics figures with real 30-day data, and adds the data-generation facts.  DRAWN figures say so."""
import copy
from content import S as OLD

B = {s["id"]: copy.deepcopy(s) for s in OLD}
S, SECTIONS = [], []


def use(*ids):
    for i in ids:
        S.append(B[i])


def new(kind, id, **kw):
    d = dict(kind=kind, id=id, **kw); B[id] = d; S.append(d)


def section(id, num, title, question, desc, notes=""):
    SECTIONS.append((id, desc)); S.append(dict(kind="section", id=id, num=num, title=title, question=question, notes=notes))


# ---------------------------------------------------------------- opening
B["questions"]["points"] = ["1 · The problem: what we must submit and how it is scored", "2 · The data: what the kit ships, what a seed and a weather window are, what we may change",
                            "3 · What the data look like over a 30-day episode", "4 · What the physics says, checked on the data",
                            "5 · The usual approaches, and why not here", "6 · Our approach, and why", "7 · LLM discovery, end to end",
                            "8 · Measuring deviations, and what the data say so far", "9 · Where we are and what comes next"]
B["questions"]["title"] = "Nine parts, each answering one question"; B["questions"]["eyebrow"] = "The story"
use("cover", "questions")
new("table", "legend", eyebrow="How to read this deck", title="Every slide names its sources; every drawing says it is a drawing",
    rows=[["Tag in the footer", "Means"], ["[website]  [Kaggle]", "the official competition pages (read Nov 19, 2026)"], ["[kit]", "the starter kit’s code, YAML and images"],
          ["[audit]", "our earlier reading of the simulator code (STATE_SPACE.md, annotated CSVs)"], ["[sim data]", "real simulator output: kit evaluate.py runs, mostly six full 30-day episodes"],
          ["[measured]", "numbers we computed from that data"], ["[our code]", "working-code/code"], ["[synthetic]", "the planted-truth self-test (made-up data by design)"],
          ["[DRAWN]", "a schematic or illustration, not data"], ["[decision] · [plan]", "agreed in planning · not built yet"]],
    colw=[0.3, 0.7], src="this deck", notes="If a figure is not data, its caption starts with DRAWN.")

# ---------------------------------------------------------------- 1 problem
section("s1", "1", "The problem", "What are we asked to do, what exactly must we submit, and how is it scored?", "1 · The problem")
use("why", "plant", "evaluation", "submission", "budget", "scoring", "leaderboard", "objective")

# ---------------------------------------------------------------- 2 data
section("s2", "2", "The data", "What does the kit ship, what is a seed, what is a weather window, and what may we change to make more?", "2 · The data")
new("table", "shipped", eyebrow="What the kit ships: six buildings", title="Six training buildings, each made by the generator from one seed",
    rows=[["Building", "Seed", "Floors", "Footprint", "Rooms", "AHUs", "People / room", "Facade k, density"],
          ["building_train_1", "0", "3", "373 × 188", "39", "4", "2", "0.21, 1,967"], ["building_train_2", "1", "2", "420 × 301", "85", "9", "2", "0.34, 578"],
          ["building_train_4", "3001", "3", "430 × 211", "54", "6", "1", "0.18, 475"], ["building_train_5", "4007", "4", "340 × 199", "79", "8", "2", "0.30, 1,187"],
          ["building_train_7", "8", "2", "325 × 218", "36", "4", "1", "0.55, 415"], ["building_train_8", "9008", "2", "318 × 198", "30", "3", "2", "0.27, 1,139"]],
    colw=[0.17, 0.08, 0.08, 0.13, 0.08, 0.08, 0.13, 0.25],
    callout="There are six shipped buildings (numbered 1, 2, 4, 5, 7, 8), not three. Each YAML holds its seed, geometry parameters and six materials; the floor plan and zone map are built from it.",
    src="[kit] smart_buildings/buildings/building_train_*.yaml; rooms and AHUs from training_scenarios/*/config.gin", notes="Footprint units as written in the YAML. Facade k in W/mK, density in kg/m³.")
new("table", "scenarios30", eyebrow="What the kit ships: 30 scenarios", title="Each building is paired with one window from each of five stations",
    rows=[["Building", "Atlanta", "Denver", "Fairbanks", "Phoenix", "Singapore"],
          ["building_train_1", "atlanta_2", "denver_7", "fairbanks_1", "phoenix_5", "singapore_8"], ["building_train_2", "atlanta_3", "denver_8", "fairbanks_2", "phoenix_6", "singapore_1"],
          ["building_train_4", "atlanta_5", "denver_2", "fairbanks_4", "phoenix_8", "singapore_3"], ["building_train_5", "atlanta_6", "denver_3", "fairbanks_5", "phoenix_1", "singapore_4"],
          ["building_train_7", "atlanta_8", "denver_5", "fairbanks_7", "phoenix_3", "singapore_6"], ["building_train_8", "atlanta_1", "denver_6", "fairbanks_8", "phoenix_4", "singapore_7"]],
    colw=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16],
    callout="6 buildings × 5 stations = 30 scenarios; no window is used twice. Minneapolis ships with the kit but is in no scenario. The hidden test uses 6 other buildings and 5 other stations.",
    src="[kit] smart_buildings/training_set.yaml", notes="Running run_scenario_generator with explicit --buildings/--weathers gives the cross product instead.")
new("table", "windows", eyebrow="What a weather window is", title="atlanta_1 and atlanta_3: the same city, different 30-day stretches of 2023",
    rows=[["Station (outdoor film W/m²K)", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8"],
          ["Atlanta (19.1)", "Jan 22", "Mar 5", "Apr 16", "May 28", "Jul 9", "Aug 20", "Oct 1", "Nov 12"],
          ["Denver (26.7)", "Jan 15", "Feb 26", "Apr 9", "May 21", "Jul 2", "Aug 13", "Sep 24", "Nov 5"],
          ["Fairbanks (8.8)", "Jan 1", "Feb 12", "Mar 26", "May 7", "Jun 18", "Jul 30", "Sep 10", "Oct 22"],
          ["Minneapolis (21.5)", "Jan 8", "Feb 19", "Apr 2", "May 14", "Jun 25", "Aug 6", "Sep 17", "Oct 29"],
          ["Phoenix (19.5)", "Jan 29", "Mar 12", "Apr 23", "Jun 4", "Jul 16", "Aug 27", "Oct 8", "Nov 19"],
          ["Singapore (17.6)", "Feb 5", "Mar 19", "Apr 30", "Jun 11", "Jul 23", "Sep 3", "Oct 15", "Nov 26"]],
    colw=[0.24] + [0.095] * 8,
    callout="A window = recorded hourly weather from one station for 30 days starting 06:00 on that date. Same city, different season: atlanta_1 is late January, atlanta_3 is mid April. 6 × 8 = 48 windows in total.",
    src="[kit] smart_buildings/weather/<station>_<n>.yaml (start_timestamp, convection_coefficient) and weather/csv", notes="The window's start date overrides the building YAML's start_timestamp (scenario_generator: episode_start = weather.start_timestamp or sim.start_timestamp).")
new("cards", "seed", eyebrow="What a seed is", title="A seed is one random building; the generator turns it into everything else",
    cards=[("From the seed alone", "Footprint size and shape, number of floors, room sizes and layout, number of AHUs, people per room.\n\nSame seed → same geometry, always.", "geometry", "accent2"),
           ("From the batch", "The six materials are a Latin-hypercube draw over the whole batch of seeds generated together.\n\nSame seed in a different batch → different materials.", "materials", "accent1"),
           ("Not from the seed", "The weather (chosen separately: one of 48 windows), the policy that drives it, the length of the run.", "chosen per run", "accent3")],
    callout="Shipped buildings used seeds 0, 1, 3001, 4007, 8, 9008. Our splits use seeds 500–599 (train), 700–719 (val), 800–819 (test): different seeds = different buildings. Seeds whose plan has a room under 20 m² are rejected by the kit.",
    src="[kit] run_floor_generator.py (build_config, draw_materials), materials.py, scenario_generator._check_generated_zone_areas", notes="")
new("table", "change", eyebrow="Making more data", title="What we change to generate data, and what we must not touch",
    rows=[["", "Change it?", "How / why"],
          ["Building seed", "YES: the only building knob", "run_floor_generator --start-seed N --count M (or --seeds)"],
          ["Weather window", "YES: choose among the 48", "run_scenario_generator --weathers atlanta_1 …"],
          ["Policy that drives the building", "YES", "evaluate.py --policies all_off bang_bang fixed_setpoint pid pid_water"],
          ["Episode length", "yes, with care", "8,640 is scored; we use 2,016 (7 days) for training data"],
          ["Generator ranges (DEFAULT_RANGES), footprint, floors, room size, AHUs", "NO", "they define the distribution the hidden buildings come from"],
          ["The building YAML, by hand", "NO", "changing any field changes the building away from what the generator makes"],
          ["config.gin interface constants (action ranges, normalisers, reward weights)", "NO", "they fix what the Kaggle gateway hands the policy and how it is scored"],
          ["The kit’s code", "NO", "our code only calls the kit’s own commands; nothing is written into the kit"]],
    colw=[0.36, 0.22, 0.42],
    callout="A new seed regenerates the whole YAML. That is how new buildings are made: never by editing a YAML.",
    src="[kit] run_floor_generator.py arguments; [audit] STATE_SPACE.md §12 (interface constants); [our code] generate_data.py passes only seeds, weather names, policies, steps",
    notes="Observation: the shipped YAMLs have footprints 318–430 × 188–301, below the generator's current DEFAULT_RANGES (400–700 × 300–500), so generated buildings can be larger than the shipped six.")
use("generator", "layouts", "climates")
B["climates"]["title"] = "Weather over the six episodes used on the next slides"
B["climates"]["points"] = ["Each line is one 30-day window: fairbanks_1 (Jan), atlanta_3 (Apr), denver_2 (Feb), phoenix_1 (Jan), singapore_6 (Oct), atlanta_1 (Jan)",
                           "Fairbanks drops below −40 °C; Singapore stays near 25–30 °C"]

# ---------------------------------------------------------------- 3 what data look like
section("s3", "3", "What the data look like", "What happens to every room over a full 30-day episode?", "3 · The data over 30 days")
use("profiles", "week", "control", "geometry", "devcontrol", "comfortdev")
B["devcontrol"]["eyebrow"] = "Where the loss comes from"; B["comfortdev"]["eyebrow"] = "Comfort deviation"

# ---------------------------------------------------------------- 4 physics on data
section("s4", "4", "What the physics says", "Which physical facts can we read off the simulator’s data, and which come only from its code?", "4 · Physics, checked on data")
B["roomsees"]["caption"] = "DRAWN: a schematic of the structure found in the simulator code, not data"
B["roomsees"]["src"] = "[DRAWN] schematic; [audit] STATE_SPACE.md §14 (structure read from the simulator code)"
use("roomsees")
new("image", "linear", eyebrow="Checked on data · linear in temperature", title="With HVAC off, temperatures change in proportion to the indoor–outdoor gap",
    img="real_linear.png", side="bottom",
    points=["One straight line per building, r = 0.71–1.00", "Slopes differ by building: materials the formulas must infer",
            "Crosses (19:00 hour): lights drop to 20%, an input change"],
    caption="[sim data] six 30-day HVAC-off episodes; mean room temperature, hourly changes at night",
    src="[sim data] kit evaluate.py all_off (scratch); [measured] straight-line fit per building", notes="The code says the solver is linear in temperatures; the data agree.")
new("image", "fastslowreal", eyebrow="Checked on data · stored heat", title="When the heating stops, cold-climate rooms drop fast, then slowly",
    img="real_fastslow.png", side="bottom",
    points=["fairbanks_1: 1.9 K in 30 min, then 0.14 K/h; denver_2, phoenix_1, atlanta_1 alike", "atlanta_3 (April), singapore_6: rooms warm up instead",
            "Two speeds = air + heavy materials: history must stand in for the slow part"],
    caption="[sim data] six 30-day pid_water episodes; mean room temperature after every weekday 19:00 (21 evenings each)",
    src="[sim data] kit evaluate.py (scratch); [measured] averages over evenings", notes="")
new("image", "commands", eyebrow="Checked on data · nonlinear in commands", title="Commands turn into heat nonlinearly: products, floors, caps and on/off",
    img="real_commands.png", side="bottom",
    points=["Air stream = damper × √pressure × constant, with a 10% damper floor, and 0 whenever the AHU is off", "Reheat heat is capped by air flow, water temperature and a discharge limit: not proportional to the valve"],
    caption="[sim data] commands replayed from 30-day pid_water episodes (flow matches the logged AHU flow to 0.02%)",
    src="[sim data] kit evaluate.py (scratch); [our code] starterkit_data.episode_inputs (the kit’s equipment laws)", notes="This conversion is hand-written from the simulator audit; the formulas receive its outputs.")
B["obsgroups"]["caption"] = "[sim data] first 3 days of an episode: building_train_2, Atlanta, pid_water, as delivered to the policy (normalised)"
use("obsgroups", "obsroles", "actions", "constants", "coefficients")
B["coefficients"]["src"] = "[audit] STATE_SPACE.md §14 (read from the finite-volume solver code, not measured)"

# ---------------------------------------------------------------- 5 usual
section("s5", "5", "The usual approaches", "How is this normally done, and why does it fall short here?", "5 · The usual approaches")
use("rules", "smarter", "comparison", "tried")

# ---------------------------------------------------------------- 6 ours
section("s6", "6", "Our approach", "What do we do instead, and why should it work?", "6 · Our approach")
B["whympc"]["caption"] = "DRAWN: an illustration of two heating plans, not data"
use("observation", "goal", "whympc", "whyllm", "nofit")

# ---------------------------------------------------------------- 7 discovery
section("s7", "7", "LLM discovery, end to end", "What problem does the LLM solve, what does it read and write, and what comes out?", "7 · LLM discovery, end to end")
B["islands"]["caption"] = "DRAWN: an illustrative family tree, not a real run"
use("discproblem", "genloop", "geomscript", "templates", "assemble", "pipeline", "llmio", "islands", "analysis", "llmreads", "llmanalysis", "llmwrites", "discoutput", "bigmachine")

# ---------------------------------------------------------------- 8 deviations
section("s8", "8", "Measuring deviations", "How do we measure how wrong a model is, and what do the data say so far?", "8 · Deviations and evidence")
use("devmodel", "realhorizon", "byepisode", "realprofile", "canfit", "selftest", "wheretolook")
new("table", "figsources", eyebrow="Which figures are data", title="Every figure in this deck, and where it comes from",
    rows=[["Figure", "Kind", "Source"],
          ["30-day profiles, week zoom, weather, HVAC off vs baseline, loss by day, comfort deviation", "real data", "kit evaluate.py, 6 buildings × 30 days (scratch)"],
          ["Linear in temperature, fast-then-slow, commands → heat", "real data", "same runs, replayed by our starterkit_data"],
          ["Cooling vs outside wall, seed vs persistence vs ceiling, per episode, measured vs predicted", "real data", "same runs; closedform.py"],
          ["Every observation group", "real data (3 days)", "kept audit file outputs/roomdata_building_train_2_atlanta_3_pidwater.npz"],
          ["Fitting constants (persistence vs network)", "real data", "kept audit outputs/room_model_forecast.json"],
          ["Room classes floor plan", "real (geometry)", "outputs/building_train_2_rooms.png"],
          ["Plant, layouts, overview", "kit images", "starter kit images/"], ["Baselines", "kit numbers", "baseline_returns.json"],
          ["Self-test profile", "synthetic by design", "outputs/discovery/selftest"],
          ["What one room sees; MPC plans; island family tree", "DRAWN", "illustrations"]],
    colw=[0.5, 0.17, 0.33], src="this deck", notes="")

# ---------------------------------------------------------------- 9 next
section("s9", "9", "Where we are", "What is built, what is not, and what comes next?", "9 · Next")
use("status", "decisions", "roadmap", "close")
