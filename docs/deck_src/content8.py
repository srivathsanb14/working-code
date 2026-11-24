"""Seeds told plainly: the kit ships six ready-made buildings AND the generator that made them; a seed is the number
you give the generator.  New buildings are planned, not yet generated: every result so far uses the six shipped ones."""
import content7 as C7

B, S, SECTIONS = C7.B, C7.S, C7.SECTIONS
NOTYET = "Not generated yet: every result in this deck uses only the six shipped buildings."

B["s2"]["question"] = "What does the kit ship (six buildings, and the generator that made them), what is a weather window, and how would we make more data?"
B["decmap"]["rows"][1][1] = "use only the kit's tools · new buildings from the kit's generator (planned, not yet run) · hold out whole weather stations · all 5 controllers"

B["shipped"].update(eyebrow="What the kit ships (1 of 2): six ready-made buildings",
    title="Six ready-made buildings; each is what the kit's generator built from one number",
    callout=("These six buildings are fixed and we use them exactly as given. The kit also ships the program that made them, "
             "the floor generator, which can build other buildings (next slides)."))
B["shipped"]["rows"][0][1] = "Seed"

C7.C6.put("seed", "cards", eyebrow="What the kit ships (2 of 2): the generator", title="Give the generator a number (the 'seed') and it builds a whole building",
    cards=[["You give it", "One integer, for example 500.\n\nCommand: run_floor_generator --start-seed 500 --count 100 builds buildings 500, 501, … 599.", "input", "accent3"],
           ["It builds", "A floor plan, rooms and doors, air handlers, people per room, and the wall, slab and partition materials.", "output", "accent2"],
           ["Same number, same layout", "The six shipped buildings are what it built for 0, 1, 3001, 4007, 8 and 9008. Any other number gives a building nobody has seen.", "why it matters", "accent1"]],
    callout="The hidden test buildings come from this same generator. So new numbers give new buildings from the same family as the test, which is what makes them good training data.",
    src="[kit] run_floor_generator.py (--start-seed, --count, --seeds; build_config, draw_materials); [website] scoring.html ('the generator supports unlimited scenario creation')")

B["change"]["left"] = ("We vary", ["Which building: run the generator with a new number",
                                   "Which weather: one of the kit's 48 windows",
                                   "Which controller drives it: the 5 shipped ones",
                                   "How long: 7-day runs (the score uses 30 days)"])
B["change"]["callout"] = "So every training building would be one the hidden test could have drawn. " + NOTYET

B["datarecipe"].update(eyebrow="How more data would be made [plan]", title="The plan: new buildings × weather windows × controllers",
    points=["Status: the code that does this (generate_data.py) is built; it has not been run. " + NOTYET,
            "Validation: 20 other buildings (700–719) × Phoenix × 5 controllers = 100 runs; test: 20 more (800–819) × 3 Minneapolis windows × 5 = 300 runs",
            "Whole stations are held out, because the hidden test uses stations we never see",
            "The window number rotates with the building number, so every season appears"])
B["datarecipe"]["steps"][0] = ("100 new buildings", "generator numbers 500–599")

B["decdatagen"]["title"] = "How the training data will be generated, and why"
B["decdatagen"]["rows"][2] = ["How to get new buildings?", "run the kit's generator with new numbers", "it builds whole buildings from the same ranges as the hidden test"]
B["decdatagen"]["rows"][3] = ["How to split?", "different numbers (so different buildings) for train, validation, test", "test buildings must be unseen in training"]
B["decdatagen"]["callout"] = NOTYET

B["episode"]["callout"] = ("Planned training set: 100 generated buildings × 4 stations × 5 controllers, 7 days each; scored on 20 other buildings at an unseen station. "
                           "Until then, the episodes are the six shipped buildings' 30-day runs.")
B["discflow"]["steps"][0] = ("Data", "now: the 6 shipped buildings; planned: generated ones (part 2)")
B["bigpicture"]["points"][1] = "The LLM writes candidate formulas; simulator data on buildings it was not shown decides which are better"
B["pipeline"]["points"] = ["Every step is built; data generation and real LLM runs have not been run yet",
                           "One command: ./run_pipeline.sh"]
B["status"]["rows"].insert(1, ["Generating new buildings (generate_data.py)", "built; not run yet (results use the 6 shipped buildings)"])
for r in B["status"]["rows"]:
    if r[0].startswith("Data generation, dataset, audits"):
        r[0] = "Dataset building and audits (kit tools only)"
