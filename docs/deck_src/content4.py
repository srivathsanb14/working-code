"""Plain-language pass over content3: define every term on the slide where it first appears, one name per thing
(history seed / network seed), numbers that agree across slides, no stale 'not built' claims, better order in parts 7-8."""
import re

import content3 as C3

B, S, SECTIONS = C3.B, C3.S, C3.SECTIONS


def sub(sid, old, new):
    """Replace text inside every string of one slide; fail loudly if the old text is not there."""
    hit = [False]

    def walk(x):
        if isinstance(x, str):
            if old in x:
                hit[0] = True
                return x.replace(old, new)
            return x
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, tuple):
            return tuple(walk(v) for v in x)
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x

    s = B[sid]
    for k in list(s):
        if k not in ("id", "kind", "img"):
            s[k] = walk(s[k])
    assert hit[0], (sid, old)


# ------------------------------------------------------------------ everywhere: drop working-folder remarks from sources
for s in S:
    if s.get("src"):
        s["src"] = re.sub(r" \((scratch folder|scratch)\)", "", s["src"])
for r in B["figsources"]["rows"]:
    r[2] = r[2].replace(" (scratch)", "")

# ------------------------------------------------------------------ opening
B["questions"]["points"][3] = "8 Evidence: how error is measured, the hand-written starting models ('seeds') on real data, the self-test"
B["questions"]["src"] = "the next slide explains the tag at the start of every source line"
sub("legend", "real simulator output: kit evaluate.py runs, mostly six full 30-day episodes",
    "simulator output from the kit's evaluate.py: the 6 shipped buildings, 30 days each, under pid_water and with HVAC off (12 runs)")
sub("legend", "agreed in planning · not built yet", "a choice agreed in planning · planned, not built yet")
sub("decmap", "Decisions (slide)", "Decisions")
sub("decmap", "no fitting · universal constants", "state-space form · formulas, not backprop · no fitting · universal constants")
sub("decmap", "one-step screen · rollout horizons and spacing", "5-minute screen · forecast horizons and spacing")
sub("legend", "under pid_water and with HVAC off", "under the kit's best baseline controller (pid_water) and with HVAC off")
sub("budget", "a small closed-form formula", "a small formula rather than a large network")

# ------------------------------------------------------------------ 1 problem
B["plant"]["eyebrow"] = "The equipment"
sub("plant", "Air handlers (AHUs) cool and mix air", "Air handlers (AHUs) mix outside and return air, cool it,")
sub("plant", "Each zone has a VAV box: a damper (its share of air) and a reheat valve",
    "Each zone (a room) has a variable-air-volume (VAV) box: a damper sets its share of air, a reheat valve warms it")
sub("submission", "{continuous_action: float32 [3K+2+2V]", "K = number of AHUs, V = number of zones: {continuous_action: float32 [3K+2+2V]")
sub("scoring", "r = 0.6 · comfort regret − 0.2 · energy cost − 0.2 · carbon; never positive",
    "Reward per step = −(0.6 × comfort penalty + 0.2 × energy cost + 0.2 × carbon): never positive")
sub("leaderboard", "best baseline per scenario (oracle)", "best baseline picked per scenario, in hindsight")
B["leaderboard"]["callout"] = ("The sets differ, so the numbers are not strictly comparable. Still, leading teams score about "
                              "ten times lower than pid_water: there is a lot of room above PID-style control.")

# ------------------------------------------------------------------ 2 data
sub("windows", "Station (outdoor film W/m²K)", "Station (wind film, W/m²K)")
B["windows"]["callout"] = ("A window = recorded hourly weather from one station for 30 days starting 06:00 on that date; atlanta_1 is late "
                          "January, atlanta_3 mid April. 6 × 8 = 48 windows. Wind film = how easily an outside wall loses heat to the "
                          "outside air at that station.")
sub("seed", "The six materials are a Latin-hypercube draw over the whole batch of seeds generated together.",
    "The six materials are drawn jointly for all seeds generated in one batch (spread evenly over their ranges).")
sub("generator", "Weather: station, 30-day window, outdoor film", "Weather: station, 30-day window, wind film")
sub("decdatagen", "window index rotated by seed (strip_for)", "the window number rotates with the seed")
sub("decdatagen", "passive cooling, on/off swings and closed-loop behaviour all excite the dynamics",
    "the data must show rooms drifting freely, swinging on/off and held steady")
sub("decdatagen", "pid_water only: too little excitation", "pid_water only: rooms mostly held in the band")
sub("decdatagen", "oversample 1.3x from seeds +10,000; keep the planned number of accepted",
    "generate 30% extra seeds (numbered from +10,000) and keep the planned number that pass")
B["climates"]["eyebrow"] = "The weather in our 30-day runs"
B["climates"]["title"] = "From −40 °C in Fairbanks to a steady 25–30 °C in Singapore"
sub("climates", "outside air in the six 30-day episodes used on the next slides", "outside air in the six 30-day windows used on the following slides")
sub("decdataset", "recovered by REPLAY: the deterministic shipped policy re-run on observations rebuilt from the CSVs",
    "recovered by replay: we re-run the (deterministic) shipped policy on the observations rebuilt from the CSVs")
sub("decdataset", "discovering it too (open, D5)", "letting the LLM discover it too (still open)")
sub("decdataset", "it is the one plant state the policy observes and can predict", "it is the one equipment state the policy observes and can predict")

# ------------------------------------------------------------------ 3 data over 30 days
sub("control", "mean of all rooms, band = 10–90 %", "line = mean of all rooms; shading = middle 80 % of rooms")
sub("comfortdev", "band from the schedule (294–297 K comfort)", "comfort band 294–297 K (21–24 °C) in occupied hours")

# ------------------------------------------------------------------ 4 physics
sub("roomsees", "Outside walls, then the outside air through a film", "Outside walls, then the outside air (through the wind film)")
B["obsgroups"]["title"] = "Every observation field, by group, over three days"
sub("obsgroups", "Per zone 6, per AHU 15, building 58 (boiler 7, clock 48, comfort 2, occupants 1)",
    "6 per zone, 15 per AHU, 58 for the building (boiler 7, clock 48, comfort 2, occupants 1): 6V + 15K + 58 in all")
sub("obsroles", "clock (12 phases ×4)", "clock (48 fields)")
sub("obsroles", "yes (occupants, comfort, 4 clock)", "yes: occupants, comfort, time of day and weekday")

sub("obsroles", "the one observed plant state", "the one observed equipment state")
sub("obsroles", "(each role confirmed in rollouts)", "(each role confirmed in simulator runs)")

# ------------------------------------------------------------------ 5 usual approaches
sub("smarter", "An RC model whose constants are fitted", "A resistor-capacitor (RC) thermal model whose constants are fitted")
B["tried"]["caption"] = ("[sim data] earlier audit, 3-day episodes: a hand-built heat network with materials fitted on day 1, "
                        "against persistence (the forecast 'temperature stays where it is')")

# ------------------------------------------------------------------ 6 our approach
B["goal"]["title"] = "One set of formulas that predicts every room’s temperature in any building"
B["goal"]["sub"] = ("Written out as readable code, with no trained weights. Inputs: the floor plan, the measured history and "
                    "universal constants. Nothing is fitted: not on the new building, not beforehand.")
sub("whympc", "Apply the first step; plan again 5 minutes later", "Apply the start of the best plan; plan again a few steps later (every 15 min in ours)")
sub("whyllm", "The space is huge", "Too many structures to write by hand")
sub("nofit", "Cheap: 0.23 ms per step on 85 rooms", "Cheap: under a millisecond per step (0.15–0.7 ms measured)")
sub("nofit", "while the history buffer fills (accepted)", "while the measured history builds up (accepted)")
sub("decmodel", "fixed rule: room-air nodes take the measurement (stateful form)",
    "optional 'stateful' form: the model keeps its own estimates, and each room's air temperature is reset to the measurement")
sub("decmodel", "Kalman gain tuning: a fitted number", "tuned correction weights (a Kalman gain): fitted numbers")
sub("decmodel", "an air stream can replace a room's air faster than one step; explicit steps blow up",
    "supply air can replace a room's air in under 5 minutes; a plain forward step then overshoots")
sub("decmodel", "explicit Euler: rejected by the stiff-air check", "plain forward (Euler) step: fails the strong-airflow check")

# ------------------------------------------------------------------ 7 discovery
sub("discproblem", "No number may be fitted: per-building materials must be read from the history inside the formula",
    "No number may be fitted: each building's materials must be inferred from its history inside the formula")
sub("genloop", "The score is the 3 h rollout error on held-out buildings; nothing is fitted at any stage",
    "Score = 3 h forecast error on held-out buildings: the model runs on its own predictions for 3 h (a 'rollout')")
B["genloop"]["steps"] = [("1 Pick parents", "better of 2 random, on the island"), ("2 Analyse", "where each parent is wrong"),
                         ("3 Ask the LLM", "mutate · cross · explore"), ("4 Check", "physics; 5-min error"),
                         ("5 Forecast", "1/3/6 h on unseen buildings"), ("6 Keep", "record; migrate best")]
B["templates"]["cards"][3] = ["History [per building]", "summaries of the measured history that stand in for the hidden materials", "read, not fitted", "accent3"]
sub("assemble", "Inputs: geometry, history buffer (≤ 1 day), this step’s inputs, universal constants",
    "Inputs: geometry, the measured history (up to 1 day) or the model's own corrected estimates, this step’s inputs, universal constants")
sub("pipeline", "Steps 1–5 are built and tested; step 6 (the controller) is next",
    "All six steps are built and tested (discovery with a scripted stand-in for the LLM); real LLM runs have not started")
B["pipeline"]["src"] = "[our code] run_pipeline.sh, control/ (part 10)"
sub("llmio", "Data dictionary: geometry, inputs, history buffer", "Data dictionary: geometry, inputs, measured history")
sub("llmio", "prepare(geometry); step(history, inputs) → next temperatures",
    "prepare(geometry); step(history, inputs) → next temperatures\nor the stateful trio: init_state / predict / correct")
sub("islands", "Each island evolves on its own", "An island = a separate population of candidates that evolves on its own")
sub("decscoring", "one-step error > 5x persistence -> dropped", "5-minute error above 5× persistence's → dropped")
sub("decscoring", "a rollout every 30 min after the first 3 h", "a 3 h forecast started every 30 min, after a 3 h warm-up (every hour in the part 8 evidence)")
sub("decscoring", "one-step accuracy does not guarantee stable free runs", "a good 5-minute fit does not guarantee a stable 3 h forecast")
sub("decscoring", "+0.0001 K per AST node", "+0.0001 K per node of the code's syntax tree (≈ per operation)")
sub("decscoring", "saves the expensive rollouts", "saves the expensive 3 h forecasts")
sub("decscoring", "any rollout outside 230–350 K rejects", "any forecast leaving 230–350 K rejects the candidate")
sub("decmodel", "sandbox: numpy, math, dlib only", "sandbox: numpy, math and dlib (our small helper library) only")
B["llmwrites"]["eyebrow"] = "What the LLM writes back: a hand-written starting model (the 'history seed'), abridged"
sub("bigmachine", "fitted ceiling (diagnostic)", "per-room fitted reference, the 'ceiling' (diagnostic only)")
sub("decsearch", "2-way tournament among the island's best 6", "pick 2 at random from the island's best 6; keep the better")
B["discoutput"]["cards"][2] = ["Into the policy", "Built (parts 9–10): the policy computes geometry from the metadata, tracks the model live, "
                               "and the controller predicts hours ahead to choose commands.", "parts 9–10", "accent3"]
B["discoutput"]["src"] = "[our code] evolve.py, report.py; runtime/, control/"
B["bigmachine"]["eyebrow"] = "Running the pipeline"
sub("bigmachine", "audit the raw data and the dataset", "audit the raw data (3) and the dataset (5)")

# ------------------------------------------------------------------ 8 evidence
sub("devmodel", "RMSE of the 5-minute prediction from measured history",
    "RMSE (root-mean-square error, K) of the 5-minute prediction from measured history")
sub("devmodel", "RMSE of free-run forecasts 1, 3, 6 h ahead, started every 30 min",
    "RMSE when the model runs on its own predictions for 1, 3, 6 h; started every 30 min (every hour in this part)")
sub("devmodel", "rollout error ÷ persistence error at the same horizon", "forecast error ÷ persistence error at the same horizon")
sub("devmodel", "any rollout leaving 230–350 K", "any forecast leaving 230–350 K")
sub("devmodel", "Rollout error at h", "Forecast error at h")
sub("devmodel", "candidate vs per-room fitted reference", "candidate vs the 'ceiling': a per-room fitted model, never deployed")
sub("canfit", "‘The temperature stays.’ Every candidate must beat it.", "The forecast 'temperature stays where it is'. Every candidate must beat it.")
sub("canfit", "Data from a known formula: the scorer must give it zero and rank it first.",
    "Data made by a known formula: the scorer must give that formula zero error and rank it first.")
B["canfit"]["callout"] = ("Our first ceiling (unconstrained fit) ran away when forecasting on its own predictions: 101 K error at 3 h on real "
                         "data. A good 5-minute fit is not a good forecast. Non-negative weights fixed it.")
B["seedscompare"]["eyebrow"] = "Real data · the hand-written starting models ('seeds'), nothing fitted"
B["seedscompare"]["title"] = "The network seed starts best; every seed is far from the ceiling"
B["seedscompare"]["points"] = [
    "Network seed: 0.74 / 0.93 / 1.16 K at 1 / 3 / 6 h",
    "History seed: 0.99 / 1.16 / 1.34 K; persistence 0.66 / 0.97 / 1.47",
    "Score with weather unknown, 3 h: network 0.959, history 1.011 K",
    "Ceiling 0.31 / 0.46 / 0.62 K; one-node seed unstable"]
B["seedscompare"]["caption"] = "[sim data] 12 thirty-day runs. Network seed: heat network, air reset to measurements; history seed: past temperatures"
B["byepisode"]["title"] = "Where the history seed loses: almost every run at 3 h, worst with HVAC off"
sub("byepisode", "3 h rollout RMSE per 30-day episode", "3 h forecast RMSE per 30-day run, history seed")
sub("byepisode", "The gap between seed and ceiling is what the LLM search has to close",
    "The gap between the seeds and the ceiling is what the LLM search has to close")
B["realprofile"]["title"] = "Week 2 of a 30-day run: what the history seed gets right, and where it is wrong"
sub("realprofile", "seed_cf_history_slab, nothing fitted", "history seed (seed_cf_history_slab), nothing fitted")
sub("selftest", "mechanical stand-in for the LLM", "a scripted stand-in for the LLM")

# ------------------------------------------------------------------ 9-12
sub("parity", "Stateful seed: one-step / 3 h error", "Network seed: 5-min / 3 h error")
sub("parity", "History seed: one-step / 3 h error", "History seed: 5-min / 3 h error")
sub("decruntime", "(obs lag, AHU delay, occupancy lag)", "(observation lag, AHU start delay, occupancy lag)")
sub("ctrlres", "stateful seed as the model", "network seed as the model")
B["ctrlres"]["points"] = [
    "Score change against pid_water (negative = better). MPC: −9.5% in fairbanks_1, +0.2% in atlanta_1; −4.5% on the mean",
    "Feed-forward (pid_water, but the model decides when to start pre-heating and when to coast in the evening): −1.8% in fairbanks_1, equal in atlanta_1",
    "Two buildings, two days each: a hint, not the benchmark",
    "Planning takes ~50 ms every 3 steps on 30–39 rooms"]
sub("decctrl", "tracked as K9", "open item")
sub("decctrl", "ablation pending (K6)", "its effect alone not yet measured")
sub("decctrl", "MPC 2 h, replanned every 3 steps; feed-forward lead 3 h", "MPC 2 h, replanned every 15 min; feed-forward looks up to 3 h ahead")
sub("gate", "Status", "Status (backlog ID)")
sub("decisions", "early error accepted", "error in the first hours accepted")
sub("roadmap", "(68 tasks)", "(77 tasks)")

# ------------------------------------------------------------------ order: part 7 = what a candidate is -> what the LLM sees -> one generation -> settings -> output
def reorder(first, last, ids):
    i, j = [s["id"] for s in S].index(first), [s["id"] for s in S].index(last)
    block = {s["id"]: s for s in S[i + 1:j]}
    assert set(block) == set(ids), set(block) ^ set(ids)
    S[i + 1:j] = [block[k] for k in ids]


reorder("s7", "s8", ["discproblem", "geomscript", "templates", "assemble", "llmio", "llmreads", "llmwrites",
                     "genloop", "analysis", "llmanalysis", "islands", "decscoring", "decsearch", "pipeline", "discoutput", "bigmachine"])
reorder("s8", "s9", ["devmodel", "canfit", "seedscompare", "byepisode", "realprofile", "selftest", "wheretolook"])

# genloop: name the operators where they first appear
B["genloop"]["points"] = ["Each LLM request is one operator: mutate (fix one parent), cross (combine two) or explore (a new idea); all of it is saved to disk",
                          "Score = 3 h forecast error on held-out buildings: the model runs on its own predictions for 3 h (a 'rollout')",
                          "Every 3 generations each island sends its best program to the next island"]


# ================================================================== round 2: stand-alone wording, data recipe, why not backprop
def insert_after(after, slide):
    B[slide["id"]] = slide
    S.insert([s["id"] for s in S].index(after) + 1, slide)


# no wording that assumes an earlier version of this deck or of the code
B["shipped"]["callout"] = ("Six shipped buildings, numbered 1, 2, 4, 5, 7 and 8. Each was made from exactly one seed; its YAML records "
                           "that seed, the generator's geometry settings and the six materials, and the floor plan and zone map are built from it.")
sub("legend", "our earlier reading of the simulator code", "our reading of the simulator code")
sub("tried", "earlier audit, 3-day episodes", "audit, 3-day episodes")
sub("decdataset", "found by the live parity check and fixed", "confirmed by the live parity check")
sub("decscoring", "2x (first version) dropped every seed", "2×: dropped every seed (tried)")
B["canfit"]["callout"] = ("Why non-negative weights: an unconstrained per-room fit ran away when forecasting on its own predictions "
                         "(101 K error at 3 h on real data). A good 5-minute fit is not a good forecast.")
B["parity"]["callout"] = ("The live check showed that the occupancy observation arrives two steps late, so the offline episodes use "
                          "that observable value too.")
B["decisions"]["rows"][3][1] = "2 h in use; 1, 3, 6 h all reported"
B["decisions"]["rows"][5][1] = "by area (in use), equally, or a time-of-day rule"
B["decisions"]["rows"].append(["Online updates", "keep nothing fitted, or let the policy adjust a few constants as data arrives"])

# how more data is made: building (seed) x weather window x policy
insert_after("change", dict(
    kind="table", id="datarecipe", eyebrow="How more data is made",
    title="Training data = new buildings × weather windows × policies",
    rows=[["Axis", "What we vary", "Train", "Validation", "Test"],
          ["Building", "a new seed = a new building. One building has one seed; we never re-seed a shipped building",
           "100 (seeds 500–599)", "20 (700–719)", "20 (800–819)"],
          ["Weather", "windows from the kit's 48; the window number rotates with the seed, so seasons spread",
           "4 per building: Fairbanks, Denver, Atlanta, Singapore", "1: Phoenix", "3: Minneapolis"],
          ["Policy", "who drives the building in the run", "all 5 shipped", "all 5", "all 5"],
          ["Length", "steps per run", "2,016 (7 days)", "2,016", "2,016"],
          ["Runs", "buildings × windows × policies", "2,000", "100", "300"]],
    colw=[0.12, 0.40, 0.2, 0.13, 0.15], size=22,
    callout=("So the same building never appears with two seeds. Variety comes from many buildings, each seen in several climates "
             "and under several policies. The 30 shipped scenarios can join training, except the Phoenix ones (held out for validation)."),
    src="[our code] generate_data.py defaults (make_plan, strip_for, split_json), run_pipeline.sh; [kit] run_floor_generator.py, run_scenario_generator.py",
    notes="Train and validation buildings get one window from each of the five stations; training keeps the four non-Phoenix ones, validation only Phoenix. "
          "About 20% of seeds are rejected by the kit, so 30% extra seeds are generated and the planned number kept."))

# the formula is a state-space model whose matrices are written, not learned
insert_after("observation", dict(
    kind="eq", id="statespace", eyebrow="Why a formula is enough",
    title="Every building is the same state-space model; only the numbers inside differ",
    eq="T(t+5 min)  =  A · T(t)  +  B · u(t)  +  d(t)",
    points=["T: every room's temperature (plus hidden slab and wall stores); u: heat from the commands; d: outside air, people, lights",
            "For given commands the simulator is linear in temperature (part 4); supply airflow enters A, so A changes with the commands",
            "Each entry of A is 300 s × conductance ÷ heat capacity; conductance = conductivity × contact area ÷ distance",
            "Areas and distances come from the floor plan; conductivity and capacity are universal or read from history",
            "So the unknowns are a few material numbers per building, not thousands of weights; A grows with the building by itself"],
    src="[audit] STATE_SPACE.md §14 (finite-volume solver); [sim data] linear-in-temperature check (part 4); [our code] dlib.assemble",
    notes="The LLM's job is therefore narrow: which stores and couplings to keep, and how to compute the hidden material numbers from the measured history."))

# why not learn the model by gradient descent
insert_after("whympc", dict(
    kind="table", id="nobackprop", eyebrow="Why not train a model by backprop?",
    title="Four ways to get the model, and what each one updates",
    rows=[["Option", "What is updated, and when", "Strength", "Why it is not our main path"],
          ["Neural network trained offline on many buildings", "thousands of weights, once, before submission",
           "uses all generated data; nothing to fit on the new building", "room count and layout change per building (needs a graph network); opaque; cannot pass physics checks. A real alternative we have not tested"],
          ["Physics model, constants fitted online per building", "a few constants, by gradient steps during the episode",
           "adapts to the actual building", "no model in the first hours or day; fitting inside 0.125 s per step; can run away (part 8). Tried first (part 5)"],
          ["Physics model, one constant set fitted offline", "a few constants, once", "simple and cheap",
           "materials differ by building (part 4), so one set is wrong for most buildings"],
          ["Ours: LLM-written formula, nothing fitted", "the formula's structure, offline, by evolutionary search on held-out buildings",
           "works from the first hours; one code for any size; readable; checkable", "only as good as the formula: the gap to the fitted ceiling (part 8) measures what is left"]],
    colw=[0.2, 0.24, 0.24, 0.32], size=21, font=9.5,
    callout=("Learning still happens, but offline and over program structure: the held-out score is the signal, not a gradient. "
             "Whether to add online updates on top of a discovered formula is an open question (part 12)."),
    src="[decision] our reasoning, not experiments, except: online fitting tried (part 5), ceiling measured (part 8)"))
