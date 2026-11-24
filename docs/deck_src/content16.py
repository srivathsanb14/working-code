"""Nov 24, 2026 (3): the pipeline reads data generated beforehand with the kit; every candidate, generation 0 included,
comes from the Claude API; what one request carries vs the size of the data; analysis by room type and condition;
a real trajectory excerpt in every mutation request."""
import content15 as C15

B, S, SECTIONS = C15.B, C15.S, C15.SECTIONS
new, insert_after = C15.new, C15.insert_after


def ids():
    return [s["id"] for s in S]


# ============================================================ part 2: the data comes in finished
B["datarecipe"]["eyebrow"] = "How the data is made (on the data machine, before this pipeline)"
B["datarecipe"]["points"] = [
    "The data is generated beforehand with the starter kit's own tools; this pipeline only reads the finished runs",
    "Every run becomes one episode: one building, one 30-day weather window, one controller",
    "The split (next slides) is then computed from whatever episodes exist: whole episodes go to validation or to one test tier",
    "Not generated yet: every result in this deck uses the six shipped buildings"]
B["datarecipe"]["src"] = "[kit] run_floor_generator.py, run_scenario_generator.py, evaluate.py; [our code] build_dataset.py (runs → episodes)"
B["splits"]["callout"] = ("Whole episodes only: both controllers' runs of a scenario always go to the same split, and a run stops if any episode "
                          "or scenario would be in two splits. No training split: nothing is fitted.")
B["splits"]["src"] += "; splits.check_whole_episodes"

# ============================================================ part 7: what goes to the API and what comes back
B["discflow"]["steps"][0] = ["Data", "finished episodes from the kit; validation only"]
B["discflow"]["points"][2] = ("Status: the loop, prompts, scorer and logs are built and tested (a stand-in for the API client in the tests); "
                              "no real API generation has run yet")
B["operators"]["rows"].insert(1, ["Init (generation 0)", "the system prompt alone: physics, the fixed function, the contract, the scoring",
                                  "“Write n first candidates from the physics and the contract above. Make them differ from each other in structure (…).”"])
B["operators"]["title"] = "Four kinds of request, with the exact instruction each one ends with"
B["operators"]["size"] = 20
B["operators"]["eyebrow"] = "The search · four kinds of request"
B["islands"]["title"] = "Three islands; generation 0 is written by the LLM, then every island asks for five children per generation"
B["islands"]["points"][0] = "Generation 0: per island, one request for 3 first candidates from the system prompt alone. Then 3 islands × 5 children = 15 new programs per generation"

_P = dict(id="proposers", kind="compare", eyebrow="What goes to the API, and what does not",
    title="Every candidate comes from one stateless Claude API request; the episodes never leave the machine",
    left=["In one mutation request (real sizes)", ["System prompt (same every time, cached): ≈ 11 KB",
                                                   "Parent's code ≈ 4 KB, and its five scores",
                                                   "Analysis (when, where, which term): ≈ 6 KB",
                                                   "A real day, hourly, one room per type: ≈ 3 KB",
                                                   "≈ 3,400 tokens plus the system prompt; no history"]],
    right=["Never sent", ["The episodes: 6 validation runs = 15 million numbers (≈ 26 million tokens)",
                          "Test-tier numbers; building or station names",
                          "Earlier requests or answers: nothing is kept",
                          "Instead: the scorer runs each candidate on all of them here"]],
    callout=("To write a formula the LLM needs which term is wrong, for which rooms, when, and a real day that shows it. "
             "More episodes cost more local computation, not a longer prompt."),
    src="[measured] prompt parts on the 6 validation runs (seeds_cf/seed_cf_network.py as parent); episode sizes n × V × (1 + 4) + building inputs; [our code] proposers.ClaudeProposer, prompts.py; tests/test_api_loop.py")
B["proposers"].clear(); B["proposers"].update(_P)

B["analysis"]["cards"][2] = ["Which term", "the 5-minute error regressed on each term's contribution, overall, per room type and per condition: slope s → the term should be (1 + s) times as large", "pre-mutation", "accent1"]
B["analysis"]["cards"][3] = ["Before a crossover", "both parents on every group, rooms each wins, which parent is closer to right per term, per room type and condition", "pre-crossover", "accent2"]
B["analysis"]["callout"] = "Computed by code on all validation episodes; sent with a real hourly trajectory excerpt. Saved as analysis.md next to each prompt."

new("termsbytype", "code", eyebrow="Pre-mutation analysis (real output, validation runs)",
    title="Which term is wrong, for which rooms, under which conditions",
    code=("slope per term (s: the term should be about (1 + s) times as large); explained = share of the error\n"
          "                                 air  neighbours   slab    wall   inner   heat  explained\n"
          "corner rooms                    -0.01   -0.68     +0.35   -0.44   -0.73  +0.02    46%\n"
          "one-side exterior rooms         +0.03   -0.85     +0.68   -0.42   -1.17  -0.03    61%\n"
          "interior rooms                  +0.16   -0.76     +1.06           -1.05  +0.00    88%\n"
          "larger half of rooms            +0.01   -0.75     +0.33   -0.49   -0.74  -0.03    69%\n"
          "smaller half of rooms           +0.02   -0.72     +0.50   -0.43   -0.77  +0.01    49%\n"
          "HVAC air on                     +0.03   -0.51     +0.04   -0.22   +0.20  +0.07    28%\n"
          "HVAC air off                    -0.50   -0.80     +0.07   -0.61   -1.14  -0.31    74%\n"
          "occupied hours                  +0.02   -0.66     +0.26   -0.38   -0.21  +0.03    40%\n"
          "colder half of outside temps    +0.05   -0.83     +0.48   -0.47   -0.60  +0.01    68%"),
    points=["Interior rooms: the slab term should be about twice as strong (+1.06), and the room-to-room exchange weaker: a formula for interior rooms, not a global factor",
            "With the HVAC off the stores (wall, inner) are far too strong; with it on they are nearly right: they need to depend on what the formula ignores",
            "This is the spatial and temporal detail the LLM writes formulas from"],
    src="[measured] closedform.term_regression with ROOM_TYPES and CONDITIONS, network model, 6 validation runs")
insert_after("llmterms", "termsbytype")

B["profiletab"].update(eyebrow="In every mutation request (real output)",
    title="A real day, hourly, one room of each type: measured vs the forecast made 3 h earlier",
    code=("hour | corner (63 m2)  air  reheat | one-side (110 m2) | interior (200 m2) | T_out\n"
          "   6 |  9.45 /  8.48  2414 18525 | 18.36 / 17.48     | 20.55 / 19.49     | -1.7\n"
          "   8 | 21.96 / 20.64     0     0 | 20.50 / 19.94     | 21.66 / 21.21     |  1.1\n"
          "  10 | 17.37 / 15.72   645 11058 | 21.21 / 20.65     | 22.35 / 21.86     | 12.2\n"
          "  12 | 21.90 / 19.75     0     0 | 21.42 / 20.91     | 22.94 / 22.46     | 15.6\n"
          "  16 | 19.62 / 18.47     0     0 | 22.53 / 22.35     | 23.95 / 23.70     | 16.1\n"
          "  20 | 17.50 / 17.12     0     0 | 22.19 / 22.17     | 23.79 / 23.61     |  4.5\n"
          "   0 | 14.42 / 13.94     0     0 | 21.64 / 21.19     | 23.62 / 22.98     |  2.2\n"
          "   4 | 12.34 / 11.56     0     0 | 21.16 / 20.44     | 23.38 / 22.46     |  1.8"),
    points=["The prompt has all 24 hours, plus each room's supply air and reheat and its geometry (area, outside wall, doors, neighbours)",
            "The corner room is forecast too cold all day, by up to 2.2 K while it warms up",
            "The interior room drifts too cold overnight with the HVAC off, matching the store slopes on the previous slide"],
    src="[measured] evolve.profile_text, network model, validation episode 1 of 6 (54 rooms), day 2 (every 2 h shown here)")

new("llmout", "table", eyebrow="What comes back from the API, and what is done with it", title="The LLM returns code; the numbers come from running that code",
    rows=[["Step", "What", "Where"],
          ["Answer", "JSON: for each candidate a name, a one-paragraph hypothesis, and the complete module", "generations/…/response.json"],
          ["The module", "HISTORY, REFRESH, coefficients(geo, hist, U) returning the 11 coefficients as formulas; optionally heat()", "candidates/<id>.py"],
          ["Checked", "allowed imports only, no fitted tables, no branching on building size, coefficients ≥ 0 and finite, physics checks", "database.jsonl (reason if rejected)"],
          ["Scored", "on every validation episode: five objectives, the analysis for the next request", "database.jsonl, analysis.md"],
          ["Kept", "Pareto front per island; best.py = lowest main score", "best.py, summary.json"],
          ["At the end", "best.py and the front scored once on each test tier", "log.md, summary.json"],
          ["Deployed", "best.py inside the policy: its formulas evaluated on the new building's floor plan and measured history", "the Kaggle package"]],
    colw=[0.13, 0.6, 0.27], size=21, font=11,
    src="[our code] prompts.RESPONSE_SCHEMA, contract.load_closedform, closedform.evaluate, evolve.run, score_tests")
insert_after("proposers", "llmout")

B["pipeline"]["steps"] = [["Episodes", "kit runs, read"], ["Split", "whole episodes"], ["Init", "API writes generation 0"],
                          ["Score", "validation, 5 objectives"], ["Evolve", "API: mutate, cross, explore"], ["Test", "held-out tiers, once"]]
B["pipeline"]["points"] = ["Run on the machine that holds the data; nothing is generated or simulated in this pipeline",
                           "One command: ./run_pipeline.sh (episodes → split → discovery)", "Control (part 10) comes after"]
B["bigmachine"]["rows"] = [["Step", "What it does", "Command (inside ./run_pipeline.sh)"],
                           ["1", "turn the kit's finished runs into episodes (one file per run)", "code/build_dataset.py --results R --scenarios S --out D"],
                           ["2", "split whole episodes: validation + test tiers", "code/make_splits.py --dataset D"],
                           ["3", "discovery through the Claude API; at the end the chosen model and the Pareto front on the test tiers", "code/run_discovery.py --dataset D --generations 20 --test --max-usd 50"]]
B["bigmachine"]["callout"] = "Data generation happens beforehand with the starter kit; this pipeline only reads it."
B["bigmachine"]["notes"] = "Settings (paths, generations, budget, model) are environment variables at the top of run_pipeline.sh."

# ============================================================ part 8: only real models in the evidence
B["tiers"]["rows"] = [r for r in B["tiers"]["rows"] if "mechanical" not in r[0] and "Pareto front" not in r[0]]
B["tiers"]["callout"] = ("3 h forecast error (K) as at test time, hand-written models. The network model is better than persistence on unseen "
                         "buildings, but every model loses to it at Fairbanks: the climate the hard tier is for.")
B["tiers"]["src"] = "[measured] closedform.evaluate on 12 thirty-day runs (6 shipped buildings × pid_water / HVAC off), split by splits.split_from_episodes"
B["tiers"]["notes"] = "Each building has one scenario among these 12 runs, so the easy and hard tiers are empty in this check."

B["status"]["rows"][-1] = ["Real API generations; 30-scenario benchmark; time guard", "not yet"]
