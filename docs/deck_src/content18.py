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
