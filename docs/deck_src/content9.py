"""What the organisers say about the six buildings (checked on the website and the Kaggle data page, Nov 23, 2026):
they are the public development set, not the test set; the test buildings are never released; generating more is
allowed.  We hold the six out as our own local test set (generate_data.py puts them in the test split)."""
import content8 as C8

B, S, SECTIONS = C8.B, C8.S, C8.SECTIONS

B["shipped"].update(
    title="The six buildings we are given are for development; the test uses six other buildings",
    callout=("Final scores come only from 6 different, unseen buildings at 5 unseen weather stations, which are never released. "
             "We go one step further: we never train on these six either, and keep them as our own local test (next slides)."),
    src="[kit] smart_buildings/buildings/building_train_*.yaml, training_scenarios/*/config.gin; [website] scoring.html; [Kaggle] data page (read Nov 23, 2026)")

C8.C7.C6.new("official", "table", eyebrow="What the organisers say, checked", title="Which buildings are for training, which are for the test, and what we may generate",
    rows=[["Question", "Answer", "Source"],
          ["Are the six shipped buildings the test set?", "No. They are the public development set: 6 buildings × 5 stations = 30 scenarios", "[website] scoring; [Kaggle] data"],
          ["What is the test set?", "6 other buildings × 5 other weather stations, from the same generator and parameter ranges", "[website] scoring; [Kaggle] data"],
          ["Do we ever see the test buildings or their seeds?", "No. Never released; the policy gets each one's metadata only at run time, inside Kaggle's sandbox", "[Kaggle] data"],
          ["May we generate more buildings?", "Yes, as many buildings and scenarios as we like, for training", "[Kaggle] data; [website] scoring"],
          ["What do we do with the six shipped buildings?", "Hold them out as our local test: never used to choose formulas, scored once at the end", "[decision]; generate_data.py"]],
    colw=[0.3, 0.5, 0.2], size=23, font=11,
    callout="So the generator is how we get training buildings; the six shipped buildings are how we check, before submitting, that the result works on buildings it never saw.",
    src="[website] scoring.html, competition.html; [Kaggle] data tab, read Nov 23, 2026; [our code] generate_data.split_json")
k = [s["id"] for s in S].index("shipped")
S.insert(k + 1, B["official"])

B["datarecipe"]["points"][1] = ("Validation: 20 other generated buildings (700–719) at Phoenix. Test: 20 more (800–819) at Minneapolis, "
                                "plus the six shipped buildings in all 30 of their scenarios, never trained on")
B["decdatagen"]["rows"].insert(4, ["The six shipped buildings?", "held out: local test only", "they are the organisers' own buildings, the closest stand-in for the hidden test"])
B["episode"]["callout"] = ("Planned: train on 100 generated buildings × 4 stations × 5 controllers (7 days each); validate on 20 others at an unseen station; "
                           "test on 20 more plus the six shipped buildings. Until generation runs, the episodes are the six shipped buildings' 30-day runs.")
