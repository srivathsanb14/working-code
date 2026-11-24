"""Nov 24, 2026 (2): no training split (validation + three held-out test tiers computed from the data), multi-objective
(Pareto) selection, spatial / temporal / per-term analysis, API vs coding-agent proposers, and control placed after
discovery with a plain explanation of the MPC."""
import content14 as C14

B, S, SECTIONS = C14.B, C14.S, C14.SECTIONS
new = C14.C13.C12.C11.C10.C9.C8.C7.C6.new


def card(sid, i, j, text):
    c = list(B[sid]["cards"][i]); c[j] = text; B[sid]["cards"][i] = c


def ids():
    return [s["id"] for s in S]


def insert_after(after, *sids):
    k = ids().index(after) + 1
    for j, sid in enumerate(sids):
        S.insert(k + j, B[sid])


def move_after(sid, after):
    S.pop(ids().index(sid))
    insert_after(after, sid)


# ============================================================ part 2: the data, and how it is split
B["official"]["rows"][5] = ["What do we do with the six shipped buildings?",
                            "They are data like any other: a rule (next slides) decides which runs the search sees and which are held out",
                            "[decision]; make_splits.py"]
B["official"]["callout"] = ("Whatever data exists (the six shipped buildings now, generated ones later) is split by one rule into a validation "
                            "set the search sees and three test tiers it never sees.")
B["official"]["src"] = "[website] scoring.html, competition.html; [Kaggle] data tab, read Nov 23, 2026; [our code] discovery/splits.py, make_splits.py"
B["official"]["notes"] = ("A limit: our hand-written starting models and the physics audit were developed while looking at the six shipped "
                          "buildings, so for those models the shipped test tiers are not a perfectly clean test.")
B["datarecipe"]["points"] = [
    "Status: the code that does this (generate_data.py) is built; it has not been run. Not generated yet: every result in this deck uses only the six shipped buildings.",
    "After generation, the split is computed from whatever runs exist (make_splits.py): validation for the search, three test tiers held out",
    "Whole stations and whole buildings are held out, because the hidden test uses unseen stations and unseen buildings",
    "The window number rotates with the building number, so every season appears"]
B["datarecipe"]["notes"] = "About 20% of seeds are rejected by the kit, so 30% extra seeds are generated and the planned number kept."
B["decdatagen"]["rows"][3] = ["How to split?", "no training split; validation + easy / medium / hard test tiers, computed from the data",
                              "nothing is fitted, so nothing needs training data; the tiers measure how far the formulas generalize"]
B["decdatagen"]["rows"][4] = ["The six shipped buildings?", "split by the same rule as any other data", "no special case: the rule decides from the data"]
B["decdatagen"]["rows"][5] = ["Which weather?", "the station farthest from the others (Wasserstein distance) is held out", "the hidden test uses unseen stations"]
B["decdatagen"]["title"] = "How more data will be generated, and why"

new("splits", "table", eyebrow="How the data is split · no training set", title="The search sees only validation runs; three test tiers stay hidden until the end",
    rows=[["Split", "What is held out", "On the six shipped buildings (computed)", "What it tests"],
          ["Validation", "nothing: every candidate is scored here and every analysis comes from here",
           "buildings 1, 4, 5, 7 at Atlanta, Denver, Phoenix, Singapore: 12 scenarios", "what the search optimizes"],
          ["Test · easy", "the weather window: seen building, seen station, different 30 days",
           "1/atlanta_2, 4/denver_2, 5/phoenix_1, 7/singapore_6", "a new month in a known building"],
          ["Test · medium", "the building", "buildings 2 and 8 at the four seen stations: 8 scenarios", "a building never scored on"],
          ["Test · hard", "the building and the climate", "buildings 2 and 8 at Fairbanks: 2 scenarios", "a new building in a new climate"],
          ["Extra", "the climate only", "buildings 1, 4, 5, 7 at Fairbanks: 4 scenarios (kept out so Fairbanks stays unseen)", "a diagnostic"]],
    colw=[0.13, 0.3, 0.35, 0.22], size=21, font=10.5,
    callout="There is no training split because nothing is fitted. The test tiers are scored once, at the end of a run, for the chosen model and the best few others.",
    src="[our code] discovery/splits.py (assign, split_from_episodes, make_split), make_splits.py; [measured] make_splits.py --from-scenarios --only-shipped (outputs/split_shipped.json)",
    notes="Easy windows are chosen so that their building and their station both stay in validation.")
new("whichheld", "eq", eyebrow="How the held-out station and buildings are chosen", title="Nothing is hard-coded: the farthest climate and the most unusual floor plans are held out",
    eq="W₁(station, rest) = ∫ | F_station(x) − F_rest(x) | dx\n(F = share of hours colder than x: the average distance each hourly temperature must move to match the others)",
    eqsize=22, eqpt=11,
    points=["Wasserstein-1 distance of each station's hourly outside temperatures to the other stations pooled (K): Fairbanks 21.9, Singapore 14.6, Phoenix 11.9, Denver 7.5, Atlanta 5.1. The farthest, Fairbanks, is the hard station",
            "Buildings: four numbers from the floor plan (floor area, rooms, mean room area, outside wall per floor area), standardized; the farthest from the average and then from each other are held out: building 2 (2.9 standard deviations), then building 8 (2.0)",
            "One building in four is held out (rounded up); one validation window in four becomes an easy test",
            "The same rule runs on whatever data exists. On the 12 runs we have locally it again picks Fairbanks and buildings 2 and 8"],
    src="[our code] splits.station_distances, atypical_buildings, split_from_episodes; [kit] weather/csv/*.csv, scenario config.yaml windows; [measured] make_splits.py output (_why)")
insert_after("decdataset", "splits", "whichheld")

# ============================================================ part 6/10: control is a later step
S.pop(ids().index("whympc"))
B["s10"].update(title="After discovery: control", question="Once the forecasting model exists, how would a controller use it? (a later step; discovery comes first)")
SECTIONS[:] = [(a, b if a != "s10" else "10 · After discovery: control") for a, b in SECTIONS]
B["questions"]["rows"][4] = ["Using it", "10 After discovery: control · 11 The Kaggle submission · 12 Where we are",
                             "Once the model exists, how could a controller use it, how is it submitted, and what is left to do?"]
B["questions"]["rows"][3] = ["The machinery", "7 LLM discovery · 8 Evidence · 9 Inside the policy",
                             "How does an LLM find the formulas, how are they scored on held-out data, and do they behave the same in a live policy?"]
B["whympc"].update(eyebrow="A later step, after discovery", title="What 'model predictive control' means here, in plain words",
    eq="every 15 minutes:  for each mode m in {schedule, comfort, eco, off}:\n    cost(m) = Σ over the next 2 h of  0.6·comfort + 0.2·energy + 0.2·carbon,  rooms forecast by the discovered model\nrun the mode with the lowest cost for the next 15 minutes, then ask again",
    eqsize=21, eqpt=11,
    points=["It does not search over raw commands. It compares 4 ready-made ways of running the building: schedule = the kit's pid_water as usual; comfort = hold the comfort band now (start pre-heating early); eco = hold the wider eco band now (coast on stored heat); off = air handlers and boiler off",
            "To price a mode, a copy of pid_water is run on the model's forecast rooms for 24 five-minute steps; its commands give energy and carbon, the forecast temperatures give comfort",
            "So the controller is only as good as the forecast at 1 to 3 hours. That is why discovery scores 1, 3 and 6 h forecasts, and why discovery comes first",
            "Feed-forward is a simpler use: pid_water, with the model only deciding when to start pre-heating and when to coast"],
    src="[our code] control/policies.py (MPCPolicy: horizon 24, replan 3), control/planner.py (MODES, cost), control/costs.py, control/schedule.py")
insert_after("s10", "whympc")
B["gaincalc"]["eyebrow"] = "Early control checks · how the gains are calculated"
B["ctrlres"]["eyebrow"] = "Early control checks (not the focus yet)"
B["ctrlres"]["title"] = "Early check: model-based control gains a little over pid_water in two short runs, mostly in the cold"
B["bigpicture"]["notes"] = (B["bigpicture"].get("notes", "") + " Control (part 10) is a later step; this deck's focus is the forecasting model.").strip()

# ============================================================ part 7: discovery
B["discflow"]["title"] = "An LLM proposes formulas; validation runs judge them; held-out tiers test the winner"
B["discflow"]["steps"] = [["Data", "validation runs only; test tiers held out"], ["LLM proposes", "a Python module: the formulas"],
                          ["Checks", "physics sanity on a real floor plan"], ["Score", "five forecast errors on validation runs"],
                          ["Analysis", "where, when and through which term it is wrong, fed back"]]
B["discflow"]["points"] = ["The loop repeats for many generations; the programs no other program beats on all five errors survive (evolutionary search)",
                           "The LLM never sees the test tiers, never supplies a number and never gives a verdict",
                           "Status: the loop, prompts, scorer and logs are built and tested on real data with hand-written models and mechanical stand-ins for the LLM; no LLM generation has run yet"]
B["discflow"]["src"] = "[our code] discovery/evolve.py, prompts.py, closedform.py, pareto.py, splits.py; docs/DISCOVERY_WALKTHROUGH.md"
B["episode"]["title"] = "One validation episode: geometry, inputs every 5 minutes, measured temperatures"
B["episode"]["callout"] = ("Until generation runs, the episodes are the six shipped buildings' 30-day runs, split into validation and the three "
                           "test tiers by the rule in part 2.")
B["sysprompt"]["rows"][4] = ["Data dictionary", "the geometry, inputs and target of one episode (above), with units and shapes"]
B["sysprompt"]["rows"][7] = ["Scoring", "validation only, test tiers kept aside; the checks, the screen, the test-time 3 h score with the size price, and the five objectives used for selection"]

B["scoreeq"].update(title="The main score is the 3-hour forecast error on the validation runs, plus a small price for size",
                    eq="main  =  mean over validation episodes of RMSE_3h\n        +  0.0001 K × (number of nodes in the code's syntax tree)")
B["scoreeq"]["points"][0] = "Stage 1 screen: the 5-minute error from measured history; more than 5× the error of 'temperature stays' (persistence) and the candidate is dropped"
new("pareto", "table", eyebrow="How a candidate is judged · selection", title="Five errors at once: a program survives unless another is at least as good on all five",
    rows=[["Objective", "What it measures (validation runs)", "Mostly decided by these coefficients"],
          ["main", "3 h forecast error as at test time, + the size price", "all"],
          ["fast", "1 h forecast error", "air, heat, neighbours"],
          ["slow", "6 h forecast error", "slab, wall, inner and their _back terms"],
          ["exterior", "3 h error in rooms with an outside wall", "outside, wall, wall_out, wall_back"],
          ["interior", "3 h error in rooms without one", "neighbours, inner, inner_back"]],
    colw=[0.14, 0.46, 0.4], size=22, font=11,
    callout=("A dominates B if A is no worse on every objective and better on one. Each island keeps its 6 best by front, then by how sparse "
             "their part of the trade-off is (crowding), so a program that gets one group of coefficients right survives while another group is "
             "still wrong. The deployed model is the one with the lowest main score."),
    src="[our code] discovery/pareto.py (fronts, crowding, select), closedform.OBJECTIVES, DEFAULT_OBJECTIVES, evolve.DB.elites, evolve.tournament",
    notes=("Size is not its own objective: on its own axis the smallest program is never dominated and would stay a parent forever "
           "(seen in a test run: the one-node model kept being picked). HVAC-on and HVAC-off errors are available as objectives too; "
           "--selection scalar uses the main score alone."))
insert_after("scoreeq", "pareto")
B["decscoring"]["rows"][8] = ["Selection", "Pareto front over 5 validation errors; the three test tiers scored once at the end",
                              "one number hides which coefficient is wrong; test stays unseen", "one score only: available as --selection scalar"]

B["analysis"].update(title="The LLM is told where, when and through which term the parents are wrong",
    cards=[["When (temporal)", "error vs horizon\nby hour, occupancy\nHVAC on, off, switching\ncold or warm outside\nhow long errors persist", "pre-mutation", "accent1"],
           ["Where (spatial)", "by room class, facade side, floor\nwhich room features the bias follows\nwhether adjacent rooms err together\nworst rooms", "pre-mutation", "accent1"],
           ["Which term", "the 5-minute error regressed on each term's contribution: slope s → the term should be (1 + s) times as large", "pre-mutation", "accent1"],
           ["Before a crossover", "both parents on every group above, rooms each wins, which parent is closer to right per term, coefficient values", "pre-crossover", "accent2"]],
    callout="Each analysis is computed by code on the validation runs only and saved as analysis.md next to its prompt.",
    src="[our code] closedform.analysis, temporal_stats, spatial_stats, term_regression, term_contributions, crossover_analysis",
    notes="The term contributions add up exactly to the model's one-step change (tested), so the regression attributes the residual to the formulas the LLM wrote.")
B["llmanalysis"].update(title="What the LLM is told about the network model: when and where it is wrong",
    eyebrow="Pre-mutation analysis (real output, validation runs)",
    code=("error vs horizon (rmse / bias / persistence, K):\n   5 min 0.247/-0.097/0.527   1 h 0.582/-0.271/0.764\n   3 h   0.757/-0.368/0.931   6 h 0.980/-0.498/1.358\n"
          "  -> half of the long-horizon error is already there after 1 h:\n     the fast terms (air, heat, neighbours) are off\n"
          "one-step error autocorrelation at 1 h lag: +0.82 (persistent)\n"
          "3 h error (rmse, bias): target hour 04-08  0.870 -0.578\n"
          "                        HVAC air off       0.773 -0.396\n"
          "                        colder half        0.852 -0.514\n"
          "correlation of a room's bias with its geometry:\n   floor area -0.47  <- the formula's dependence on this is wrong\n"
          "error correlation, adjacent rooms +0.87 vs non-adjacent +0.84\n   (similar: a building-wide cause)"),
    points=["Too cold almost everywhere, worst before and during the morning start and with the HVAC off",
            "Larger rooms are biased colder: a coefficient scales wrongly with floor area",
            "Neighbouring rooms err together no more than distant ones: the cause is building-wide, not the coupling between rooms"],
    src="[measured] closedform.analysis on seeds_cf/seed_cf_network.py, the 6 validation runs of the local split (outputs/discovery/real12/generations/g001/island_1/mutation/analysis.md)",
    notes="The local split from the 12 cached runs: validation = buildings 4, 5, 7 (denver_2, phoenix_1, singapore_6) × pid_water / HVAC off.")
new("llmterms", "code", eyebrow="Pre-mutation analysis (real output, validation runs)", title="…and through which term: the error regressed on each term's contribution",
    code=("residual rms 0.2411 K per step; the terms explain 58% of it\n"
          "  air         size 0.3168 K/step  slope +0.015 +- 0.002  -> consistent\n"
          "  neighbours  size 0.2230 K/step  slope -0.742 +- 0.004  -> too strong by ~74%\n"
          "  slab        size 0.1361 K/step  slope +0.418 +- 0.011  -> too weak by ~42%\n"
          "  wall        size 0.4035 K/step  slope -0.450 +- 0.002  -> too strong by ~45%\n"
          "  inner       size 0.0826 K/step  slope -0.722 +- 0.019  -> too strong by ~72%\n"
          "  heat        size 0.4642 K/step  slope -0.001 +- 0.002  -> consistent\n"
          "exterior rooms only: wall -0.447 (too strong by ~45%)"),
    points=["Each line: how big the term's 5-minute push is, and how the leftover error follows it",
            "Air and heat are right; the stores and the room-to-room exchange are not: the next mutation should change those formulas",
            "Terms that move together (neighbours, wall, inner) share the error, so the LLM is told to read their slopes together"],
    src="[measured] closedform.term_regression on seeds_cf/seed_cf_network.py, 6 validation runs (same analysis.md)")
insert_after("llmanalysis", "llmterms")
B["profiletab"].update(title="Measured vs predicted, 3 h ahead, for two rooms of a validation run",
    code=("time | room #15 (most exposed) | room #37 (largest interior) | T_out | HVAC\n"
          "  0h |    9.45 /  8.48 C       |   20.55 / 19.49 C           | -1.7  | on\n"
          "  2h |   21.96 / 20.64 C       |   21.66 / 21.21 C           |  1.1  | on\n"
          "  4h |   17.37 / 15.72 C       |   22.35 / 21.86 C           | 12.2  | off\n"
          "  6h |   21.90 / 19.75 C       |   22.94 / 22.46 C           | 15.6  | off\n"
          " 10h |   19.62 / 18.47 C       |   23.95 / 23.70 C           | 16.1  | off\n"
          " 14h |   17.50 / 17.12 C       |   23.79 / 23.61 C           |  4.5  | off\n"
          " 18h |   14.42 / 13.94 C       |   23.62 / 22.98 C           |  2.2  | off\n"
          " 22h |   12.34 / 11.56 C       |   23.38 / 22.46 C           |  1.8  | off"),
    points=["Hours after 06:00. The exposed room is forecast 0.4–2.2 K too cold all day, worst while it warms up in the morning",
            "The interior room is within about 1 K, also too cold: the same building-wide bias the analysis shows",
            "The LLM reads this table next to the analysis"],
    src="[measured] evolve.profile_text, network model, first validation run (building_train_4 denver_2), day 2")
B["operators"]["rows"][1][2] = ("“Each must change ONE thing that the analysis points to (…). Use the TERMS slopes, the SPATIAL correlations and the TEMPORAL "
                                "breakdown to decide which coefficient to change and in which direction, and say … which objective it should improve.”")
B["operators"]["rows"][2][1] = "two parents' code and the pre-crossover analysis: where and when each wins, which term each gets right, which functions differ"
B["operators"]["rows"][2][2] = ("“Write n children that COMBINE the parents: take each coefficient formula from the parent that is better where that coefficient "
                                "matters (the TERMS lines say which parent's formula is closer to right for each term; …), make the formulas consistent, "
                                "and state in the hypothesis which formula came from which parent.”")
B["operators"]["rows"][3][1] = "the island's programs (names, scores, five objectives, hypotheses)"

new("proposers", "compare", eyebrow="Who answers the requests", title="Two ways to call the LLM; both see only the prompt of that one request",
    left=["Claude API (--proposer api), preferred", ["One request per operator: the cached system prompt + this request",
                                                     "No chat history: parents and analysis are inside the request",
                                                     "JSON answer, one retry; cost logged; --max-usd caps the run",
                                                     "Answers saved before evaluation: a crash never pays twice"]],
    right=["Coding-agent harness (--proposer agent)", ["A coding-agent CLI (default claude -p) run headless, a new session per request",
                                                       "Same prompt in, JSON answer out",
                                                       "No history from other requests or chats",
                                                       "Also: dir (answers dropped in a folder), dry (prompts only), mock (no LLM)"]],
    callout="Neither has run a real generation yet. Every result in this deck comes from hand-written models and mechanical stand-ins for the LLM.",
    src="[our code] discovery/proposers.py (ClaudeProposer, AgentCLIProposer, DirectoryProposer, DryRunProposer, MockProposer), evolve.make_proposer")
insert_after("operators", "proposers")
B["islands"]["steps"] = [["Pick", "2 random programs of the island's 6; keep the better on the Pareto front"], ["Mutate", "1 request, 2 children"],
                         ["Cross", "1 request, 2 children"], ["Explore", "1 request, 1 child"], ["Keep", "the island's 6 best by front and crowding"]]
B["islands"]["src"] = "[our code] evolve.py defaults (--islands 3, --elites 6, --n-mut 2, --n-cx 2, --n-exp 1, --migrate-every 3, --selection pareto), evolve.tournament, pareto.py"
B["decsearch"]["rows"][3] = ["Parents", "pick 2 at random from the island's 6; keep the one on the better Pareto front", "favours good parents without always picking the same one", "always the best: no diversity"]
B["decsearch"]["rows"][4] = ["Analysis", "where, when and through which term, in every request", "tells the LLM which coefficient is wrong and in which direction", "score only: the LLM guesses"]
B["decsearch"]["rows"][5] = ["LLM", "Claude API, one stateless request per operator (or a coding-agent CLI, a fresh session per request)", "the context is the prompt alone, never a chat history", "one long chat: earlier answers leak into later ones"]
B["decsearch"]["rows"][8] = ["Data per candidate", "up to 30 validation episodes (deterministic sample); no training episodes", "bounded cost per candidate", "all episodes: slow on a large dataset"]
B["pipeline"]["steps"] = [["Generate", "kit tools only"], ["Split", "validation + 3 test tiers"], ["Propose", "LLM writes formulas"],
                          ["Score", "validation, 5 objectives"], ["Keep best", "islands, Pareto"], ["Test", "held-out tiers, once"]]
B["pipeline"]["points"] = ["Every step is built; data generation and real LLM runs have not been run yet", "One command: ./run_pipeline.sh", "Control (part 10) comes after"]
B["bigmachine"]["rows"][4] = ["4", "build the dataset (replay checked), then the split: validation + test tiers from the runs that exist", "code/build_dataset.py; code/make_splits.py"]
B["bigmachine"]["rows"][2] = ["2", "generate buildings, scenarios and runs with the kit's tools (and the 30 shipped scenarios)", "code/generate_data.py --oversample 1.3 --include-shipped"]
B["bigmachine"]["rows"][5] = ["6", "discovery (PROPOSER=dry by default: writes the prompts; api or agent for a real run); the chosen model and the Pareto front scored once on the test tiers", "code/run_discovery.py --proposer $PROPOSER --test"]
B["bigmachine"]["notes"] = "Settings are environment variables at the top of run_pipeline.sh."

# ============================================================ part 8: evidence on the held-out tiers
new("tiers", "table", eyebrow="Held-out tiers (real runs)", title="On unseen buildings the network model beats 'temperature stays'; in the unseen climate it does not",
    rows=[["Model", "Validation (6 runs)", "Test · medium: buildings 2, 8 (4 runs)", "Extra: building 1 at Fairbanks (2 runs)"],
          ["Persistence ('temperature stays')", "0.889", "0.948", "1.274"],
          ["Network model (hand-written)", "0.740", "0.538", "2.114"],
          ["History model (hand-written)", "1.099", "1.270", "1.732"],
          ["One-node model (hand-written)", "5.707", "5.043 (unstable)", "10.636"],
          ["Best on validation after 1 mechanical generation", "0.708", "0.589", "2.016"],
          ["Another program on the Pareto front (better on interior rooms)", "0.716", "0.509", "2.057"]],
    colw=[0.34, 0.18, 0.24, 0.24], size=21, font=11,
    callout=("3 h forecast error (K) as at test time. The best on validation is not the best on unseen buildings: that gap is why test tiers "
             "exist. Every model loses to persistence at Fairbanks: the climate the hard tier is for."),
    src="[measured] closedform.evaluate on the 12 cached 30-day runs, split by splits.split_from_episodes; run_discovery.py --proposer mock --generations 1 --islands 2 --test (outputs/discovery/real12/summary.json)",
    notes=("Locally each building has one scenario per run, so the easy and hard tiers are empty in this check; with the 30 shipped scenarios "
           "or generated data all tiers fill. 'Mechanical' = the mock proposer (rescaled numbers), not an LLM."))
insert_after("byepisode", "tiers")

# ============================================================ part 12: status and decisions
B["status"]["rows"][6] = ["Islands, operators, Pareto selection, logs, report, resume, parallel", "built; exercised without an LLM"]
B["status"]["rows"].insert(5, ["Splits from the data: validation + easy / medium / hard test tiers", "built and tested"])
B["status"]["rows"].insert(6, ["Analysis: temporal, spatial, per-term", "built and tested on real runs"])
B["status"]["rows"][-1] = ["Real LLM generations (API or coding-agent); 30-scenario benchmark; time guard", "not yet"]
B["status"]["size"] = 22
B["decisions"]["callout"] = ("Settled: no fitting · no training split · universal constants · a fixed state-space function whose coefficients the LLM writes · "
                             "hidden stores corrected by measurements · test-time scoring · Pareto selection · discovery before control.")
B["decisions"]["rows"][4] = ["MPC horizon (a later step)", "2 h in use; 1, 3, 6 h all reported"]
B["roadmap"]["steps"] = [["Kaggle check", "pid_water notebook"], ["Time guard", "large buildings"], ["Real LLM runs", "Claude API, test tiers"],
                         ["Benchmark", "30 scenarios"], ["Control", "MPC notebook, after the gate"]]


# ============================================================ wording: validation and test tiers, not "held-out buildings"
def _sub(obj, a, b):
    if isinstance(obj, str):
        return obj.replace(a, b)
    if isinstance(obj, (list, tuple)):
        return [_sub(x, a, b) for x in obj]
    if isinstance(obj, dict):
        return {k: _sub(v, a, b) for k, v in obj.items()}
    return obj


for sid, a, b in [
        ("whyllm", "checked and scored on unseen buildings", "checked and scored on validation runs, and the winner tested on held-out tiers"),
        ("nobpcompare", "Run it on held-out buildings", "Run it on the validation runs"),
        ("nobackprop", "by evolutionary search on held-out buildings", "by evolutionary search scored on validation runs"),
        ("honestfit", "held-out buildings and a separate test station", "Pareto selection on validation runs and three test tiers never seen by the search"),
        ("discproblem", "a scorer on unseen buildings decides", "a scorer on validation runs decides"),
        ("genloop", "1/3/6 h on unseen buildings", "1/3/6 h on validation runs"),
        ("genloop", "Score = 3 h forecast error on held-out buildings", "Main score = 3 h forecast error on the validation runs"),
        ("parity", "Offline (training episode)", "Offline (episode)"),
        ("histnotfit", "it has no memory of the training buildings", "it has no memory of the buildings it was chosen on"),
        ("questions", "how are they scored on held-out data", "how are they scored and tested on held-out data")]:
    new_s = _sub(B[sid], a, b)
    assert new_s != B[sid], (sid, a)
    B[sid].clear(); B[sid].update(new_s)
