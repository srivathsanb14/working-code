"""Nov 24, 2026 (5): what data exists, said plainly.  Six buildings ship pre-built but the generator makes any number;
the weather files hold all of 2023, but only the 30 shipped 30-day windows have ever been simulated.  Adds the real
floor-plan and weather figures used in the meeting deck."""
import content17 as C17

B, S, SECTIONS = C17.B, C17.S, C17.SECTIONS
new, insert_after = C17.new, C17.insert_after

# ---- buildings: six ship pre-built; the generator makes as many as we like
B["shipped"]["callout"] = ("Six buildings ship pre-built, as examples; the kit's generator makes as many more as we like (next slides). "
                           "Final scores come only from 6 other buildings at 5 other weather stations, never released.")
new("floorplans", "figure", eyebrow="What the kit ships · buildings, drawn",
    title="The six pre-built buildings at one scale: 30 to 85 rooms",
    img="buildings.png", h=520,
    callout="Floor 1 of each, from the kit's own floor-plan files. Any new seed gives the generator another building from the same family.",
    src="[kit] training_scenarios/*/zone_map.npy; room types: code/explore_zone_map.py (outputs/building_*_rooms.csv); make_figs11.py")
insert_after("shipped", "floorplans")

# ---- weather: full-year files, 48 fixed windows, only 30 ever simulated
B["windows"]["callout"] = ("A window = 30 days of recorded hourly weather from 06:00 on that date. The weather files hold all of 2023; the kit fixes "
                           "48 windows (6 × 8), and only the 30 in shipped scenarios have ever been simulated. Wind film = how easily an "
                           "outside wall loses heat at that station.")
new("weather30", "figure", eyebrow="What the kit ships · weather, drawn",
    title="Only 30 episodes ever simulated: −42 °C to +46 °C",
    img="weather.png", h=520,
    callout="Orange: the 30 shipped episodes, all we have. Grey: the rest of 2023, in the weather file but in no run. The test uses 5 other stations, never released.",
    src="[kit] weather/csv/*.csv (2023, hourly), weather/<scenario>.yaml start dates, training_set.yaml; [website] scoring.html; make_figs11.py")
insert_after("windows", "weather30")
B["generator"]["cards"][2] = ["Weather: chosen per run", "48 fixed 30-day windows (6 stations × 8); 30 of them in shipped scenarios\n"
                              "The files hold all of 2023, hourly\nWind film 8.8–26.7 W/m²K, one value per station", "chosen", "accent3"]
B["climates"]["title"] = "The 12 runs we simulated: six buildings, one of their 30-day windows each, −40 °C to 30 °C"

# ---- figure sources
B["figsources"]["rows"].insert(-1, ["Floor plans of the six buildings; weather year with the 30 simulated episodes",
                                    "real (geometry, weather)", "kit zone_map.npy, weather/csv; make_figs11.py"])

# ---- no data generator in this repository: the data is made with the kit's own commands
B["change"]["src"] = ("[kit] README and run_floor_generator.py arguments (the kit's commands take seeds, weather names, controllers, steps); "
                      "[audit] STATE_SPACE.md §12 (interface constants)")
B["change"]["eyebrow"] = "Making more data, with the kit's own commands"
B["datarecipe"]["title"] = "The plan, run with the kit's commands: new buildings × weather windows × controllers"
B["datarecipe"]["steps"][0] = ["100 new buildings", "kit generator, seeds 500–599; ask for ~30% more (it rejects about 1 in 5)"]
B["datarecipe"]["notes"] = "Run with the kit's own commands on the data machine; this repository generates and simulates nothing."
B["decdatagen"]["title"] = "How more data will be made with the kit, and why"
B["decdatagen"]["src"] = "[kit] run_floor_generator.py, run_scenario_generator.py, evaluate.py, README; [our code] run_pipeline.sh (reads the runs)"
B["decdatagen"]["rows"] = [r if r[0] != "Rejected seeds?" else ["Rejected seeds?", "ask the kit's generator for ~30% extra seeds",
                           "the kit rejects about 1 seed in 5 (a room under 20 m²)"] for r in B["decdatagen"]["rows"]]
B["episode"]["src"] = B["episode"]["src"].replace("; generate_data.py defaults", "")

# ---- each request now also lists the island's recent children that ran but scored worse than their parent
B["proposers"]["left"][1][1] = "Parent's code ≈ 4 KB, its five scores; recent rejected children and children that scored worse"

# ---- audit Nov 25: what each request carries, as prompts.cf_user_* build it
_ops = {r[0]: r for r in B["operators"]["rows"]}
_ops["Init (generation 0)"][1] = "the system prompt alone (plus the island's recent rejected programs, if any)"
_ops["Mutation"][1] = ("one parent's code, its five scores, the pre-mutation analysis, a measured-vs-predicted table; the island's "
                       "recent rejected children and children that scored worse")
_ops["Crossover"][1] = ("two parents' code and the pre-crossover analysis (where and when each wins, which term each gets right); "
                        "recent rejected and worse children")
_ops["Explore"][1] = "the island's programs (names, scores, five objectives, hypotheses); recent rejected and worse children"
B["proposers"]["left"][1][4] = "≈ 3,400 tokens (≈ 500 more with the worse-children list) plus the system prompt; no history"
B["histnotfit"]["points"][0] = ("It is a fixed function of the last 288 measurements, recomputed every 3 hours (REFRESH = 36), like a moving "
                                "average: Û = (Σ x·r + 5·2.0) / (Σ x² + 5)")
B["histnotfit"]["src"] = B["histnotfit"]["src"].replace("prompts.CF_CONTRACT (", "prompts.CF_SYSTEM (")
B["llmreads"]["points"][1] = "Then the operator-specific part: parents, their analysis, recent rejected children and children that scored worse"
B["datarecipe"]["points"][1] = "Every run becomes one episode: one building, one weather window (its first 7 days in this plan), one controller"

# ---- the data plan (Nov 25): every shipped building in every window first, then new buildings
B["datarecipe"]["title"] = "The plan: every shipped building in every window, then new buildings"
B["datarecipe"]["steps"] = [["Stage 1: 6 shipped buildings", "× all 48 windows (6 stations × 8) = 288 scenarios"],
                            ["× 5 controllers", "all_off, bang_bang, fixed_setpoint, pid, pid_water"],
                            ["= 1,440 runs", "30 days each (8,640 steps); about 100–200 CPU-hours"],
                            ["Stage 2: new buildings", "kit generator from seed 500; 20 to start, one window per station each: 600 runs"]]
B["datarecipe"]["points"] = [
    "Made beforehand with the starter kit's own commands; this pipeline only reads the finished runs",
    "Every run becomes one episode: one building, one 30-day weather window, one controller",
    "Stage 1 gives every building every season at every station; stage 2 adds new floor plans, which is what the hidden test varies",
    "The split (next slides) is computed from whatever episodes exist. Not generated yet: every result in this deck uses the 30 shipped scenarios"]
B["datarecipe"]["src"] = "[kit] run_floor_generator.py, run_scenario_generator.py (naming --buildings takes the cross product), evaluate.py; GUIDE.md A.4; [measured] 0.025–0.057 s per step"
_r = {r[0]: r for r in B["decdatagen"]["rows"]}
_r["Which weather?"][1:] = ["shipped buildings: all 48 windows; new buildings: one window per station, rotated",
                            "every season at every station; the split then holds out the farthest station (Wasserstein)"]
_r["How long?"][1:] = ["30 days per run (8,640 steps)", "the scored length; every phase of the month"]
B["change"]["left"][1][1] = "Which weather: all 48 windows for the shipped buildings, one per station for new ones"
B["change"]["left"][1][3] = "How long: full 30-day runs, as scored"

# ---- the old backlog (docs/next_tasks.*) is gone: Planning/ is the task list
_task = {"K1": "Planning 14", "K7": "Planning 12", "K8": "Planning 13", "I5": "Planning 11", "L6": "Planning 15"}
B["gate"]["rows"][0] = ["Item", "Status (task)"]
for _r in B["gate"]["rows"][1:]:
    for _k, _v in _task.items():
        _r[1] = _r[1].replace(f"({_k})", f"({_v})")
    _r[1] = _r[1].replace("done (I7)", "done")
B["gate"]["src"] = "docs/RUNBOOK.md §11; Planning/index.md"
B["status"]["src"] = "[our code]; Planning/index.md (pending tasks)"
B["roadmap"]["src"] = "[plan] Planning/index.md (18 pending tasks); [Kaggle] timeline, limits"
