"""Comprehensive storyline: every decision gets a row (what we chose, why, what we rejected or the evidence).
Reuses still-correct slides from content2 by id, updates stale ones, adds the runtime, controller and submission parts."""
import copy

import content2 as C2

B = {s["id"]: copy.deepcopy(s) for s in C2.S}
S, SECTIONS = [], []
DEC = ["Decision", "What we chose", "Why", "Rejected alternative / evidence"]
W4 = [0.17, 0.25, 0.32, 0.26]


def use(*ids):
    for i in ids:
        S.append(B[i])


def new(kind, id, **kw):
    d = dict(kind=kind, id=id, **kw); B[id] = d; S.append(d)


def section(id, num, title, question, desc, notes=""):
    SECTIONS.append((id, desc)); S.append(dict(kind="section", id=id, num=num, title=title, question=question, notes=notes))


def dec(id, eyebrow, title, rows, src, callout=None, notes=""):
    new("table", id, eyebrow=eyebrow, title=title, rows=[DEC] + rows, colw=W4, callout=callout, src=src, notes=notes, size=21, font=9.5)


# ================================================================ opening
B["questions"]["title"] = "Twelve parts, each answering one question"
B["questions"]["points"] = ["1 Problem · 2 Data · 3 The data over 30 days · 4 Physics checked on data",
                            "5 Usual approaches · 6 Our approach and the model form", "7 LLM discovery: loop, prompt, scoring, search settings",
                            "8 Evidence: deviations, seeds on real data, self-test", "9 Running the model inside the policy, and deployment parity",
                            "10 Controllers: cost, occupancy, feed-forward, MPC", "11 The Kaggle submission", "12 Status, open questions, roadmap"]
use("cover", "questions", "legend")
new("table", "decmap", eyebrow="Every decision, and where it is explained", title="The decisions this deck walks through",
    rows=[["Area", "Decisions (slide)"],
          ["Data generation", "kit tools only · seed as the only building knob · splits and stations · season rotation · policies · run length · seed top-up"],
          ["Dataset", "replay instead of logging · replay acceptance thresholds · observation timing · occupancy lag · inputs from commands"],
          ["Model form", "no fitting · universal constants · discrete-time map · history buffer or corrected state · implicit supply air · sandbox"],
          ["Scoring", "physics checks · one-step screen · rollout horizons and spacing · warm-up · stability · test-time inputs · complexity price · held-out selection"],
          ["Search", "islands and migration · operators and their counts · parent choice · LLM settings · robustness (retry, resume, parallel)"],
          ["Runtime", "same timing as offline · parity check · safety layer and fallback"],
          ["Controllers", "build on pid_water · modes not raw commands · cost = reward · occupancy forecast · horizons and replanning"],
          ["Submission", "one notebook · vendored code · packaging · rehearsal · readiness gate"]],
    colw=[0.18, 0.82], size=24, src="this deck; docs/DISCOVERY_WALKTHROUGH.md, docs/SOURCES_AND_FACTS.md", notes="Each area has a decision table with the reason and the alternative or evidence.")

# ================================================================ 1 problem
section("s1", "1", "The problem", "What are we asked to do, what exactly must we submit, and how is it scored?", "1 · The problem")
use("why", "plant", "evaluation", "submission", "budget", "scoring", "leaderboard", "objective")

# ================================================================ 2 data
section("s2", "2", "The data", "What does the kit ship, what is a seed and a weather window, and what did we decide about generating data?", "2 · The data")
use("shipped", "scenarios30", "windows", "seed", "change", "generator", "layouts")
dec("decdatagen", "Decisions · data generation", "How the training data is generated, and why",
    [["Tools", "only the kit's own commands (floor generator, scenario generator, evaluate.py)", "data must come from the same simulator the hidden test uses", "a modified kit: would make data unrepresentative"],
     ["Building knob", "the seed (start-seed / count)", "a seed regenerates the whole building from the generator's distribution", "editing YAMLs: moves buildings off the distribution"],
     ["Splits", "train 500–599 · val 700–719 · test 800–819", "disjoint seeds = buildings never seen in training", "random split of one pool: leaks look-alike buildings"],
     ["Stations", "val = Phoenix only, test = Minneapolis only, train = the other four", "the hidden test uses unseen stations: hold out stations, not only buildings", "same stations everywhere: overstates transfer"],
     ["Seasons", "window index rotated by seed (strip_for)", "spreads the 8 windows of each station across the buildings", "one window per station: one season only"],
     ["Policies", "all five shipped: all_off, bang_bang, fixed_setpoint, pid, pid_water", "passive cooling, on/off swings and closed-loop behaviour all excite the dynamics", "pid_water only: too little excitation"],
     ["Run length", "2,016 steps (7 days) for training data", "covers weekdays and a weekend at a quarter of the cost of 8,640", "full 8,640: kept for scoring and rehearsal"],
     ["Rejected seeds", "oversample 1.3x from seeds +10,000; keep the planned number of accepted", "the kit rejects ~20% of seeds (a room under 20 m²)", "accept smaller splits: sizes would vary"]],
    src="[our code] generate_data.py, run_pipeline.sh; [kit] run_floor_generator.py, scenario_generator.py")
use("climates")
dec("decdataset", "Decisions · dataset", "How runs become training episodes, and why",
    [["Commands", "recovered by REPLAY: the deterministic shipped policy re-run on observations rebuilt from the CSVs", "the kit's CSVs do not hold per-room commands; replay needs no kit changes", "patching the kit to log commands"],
     ["Acceptance", "AHU flow error <= 2% and run-command agreement >= 99.5%", "a replay that diverges would teach wrong inputs", "measured: <= 0.02% error on all 12 thirty-day runs"],
     ["Timing", "observation t shows temperatures of step t-1; first AHU start one step late", "the same convention live and offline", "found in the audit; checked live"],
     ["Occupancy", "the head count as observed: two steps behind the kit's log", "only what the policy can see may be used", "found by the live parity check and fixed"],
     ["Inputs", "air streams, supply temperature, reheat heat from commands via the kit's equipment laws", "the formulas get heat flows; the nonlinear command-to-heat part is known physics", "discovering it too (open, D5)"],
     ["Boiler water", "integrated from the water-setpoint commands with the kit's ramp limits from 340 K", "it is the one plant state the policy observes and can predict", "-"],
     ["Geometry", "computed from floor_plan + zone_map + AHU assignment only", "exactly what the metadata gives a new building", "materials or YAML values: not available at test time"]],
    src="[our code] discovery/starterkit_data.py, build_dataset.py, runtime/inputs.py; [audit] STATE_SPACE.md")

# ================================================================ 3 data over 30 days
section("s3", "3", "What the data look like", "What happens to every room over a full 30-day episode?", "3 · The data over 30 days")
use("profiles", "week", "control", "geometry", "devcontrol", "comfortdev")

# ================================================================ 4 physics
section("s4", "4", "What the physics says", "Which physical facts can we read off the simulator's data, and which come only from its code?", "4 · Physics, checked on data")
use("roomsees", "linear", "fastslowreal", "commands", "obsgroups", "obsroles", "actions", "constants", "coefficients")

# ================================================================ 5 usual
section("s5", "5", "The usual approaches", "How is this normally done, and why does it fall short here?", "5 · The usual approaches")
use("rules", "smarter", "comparison", "tried")

# ================================================================ 6 ours
section("s6", "6", "Our approach", "What do we do instead, why, and what exactly is a model?", "6 · Our approach and the model form")
B["whympc"]["src"] = "[DRAWN] illustration; [our code] control/policies.py (MPC built, part 10)"
use("observation", "goal", "whympc", "whyllm", "nofit")
dec("decmodel", "Decisions · model form", "What a candidate model is allowed to be, and why",
    [["Fitting", "none: not on the new building, not beforehand", "works from step 1; a bad score means a bad formula, not a bad optimiser", "fitting per building (tried first): costs a day, can fail"],
     ["Numbers", "universal constants (cited from the kit) + geometry + history", "a universal constant does not make a model building-specific", "per-building constants"],
     ["Form", "a discrete-time map: next 5-minute temperatures", "matches the controller's step; no ODE solver needed", "continuous-time equations"],
     ["Memory", "a history buffer (up to one day) or a state corrected by measurements", "stands in for hidden slab and wall temperatures without fitting", "one temperature per room: misses the slow part"],
     ["Correction", "fixed rule: room-air nodes take the measurement (stateful form)", "estimates hidden stores from history; nothing tuned", "Kalman gain tuning: a fitted number"],
     ["Supply air", "implicit (end-of-step) in the assembled form", "an air stream can replace a room's air faster than one step; explicit steps blow up", "explicit Euler: rejected by the stiff-air check"],
     ["Safety", "sandbox: numpy, math, dlib only; no files; deterministic", "candidates are untrusted code", "-"],
     ["Shortcuts", "comparing a building's size or geometry to fixed numbers is rejected", "would special-case the buildings seen in discovery", "-"]],
    src="[decision] planning notes; [our code] discovery/contract.py, dlib.py, prompts.CF_CONTRACT")

# ================================================================ 7 discovery
section("s7", "7", "LLM discovery, end to end", "What problem does the LLM solve, what does it read and write, how is a candidate scored, and how does the search run?", "7 · LLM discovery")
B["bigmachine"]["rows"] = [["Step", "What it does", "Command (inside ./run_pipeline.sh)"],
                           ["1", "check the setup and the kit", "code/setup_check.py"], ["2", "generate buildings, scenarios and runs with the kit's tools", "code/generate_data.py --oversample 1.3"],
                           ["3, 5", "audit the raw data and the dataset", "code/audit_data.py"], ["4", "build the dataset (replay checked)", "code/build_dataset.py"],
                           ["6", "LLM discovery; best scored once on test", "code/run_discovery.py --proposer claude --test"],
                           ["7", "fitted ceiling (diagnostic)", "code/run_discovery.py --ceiling"],
                           ["8", "best model inside a policy, live vs offline", "code/run_policy.py --check-replay"]]
B["bigmachine"]["title"] = "One command runs the whole pipeline"
use("discproblem", "genloop", "geomscript", "templates", "assemble", "pipeline", "llmio", "llmreads", "llmwrites", "islands", "analysis", "llmanalysis")
dec("decscoring", "Decisions · scoring", "How a candidate is judged, and why each setting",
    [["Physics checks", "equilibrium · bounded cooling · warmer outside never cools · heat warms · stiff air", "cheap, catches impossible physics before scoring", "score only: slow, and accepts lucky nonsense"],
     ["Screen", "one-step error > 5x persistence -> dropped", "saves the expensive rollouts for plausible candidates", "2x (first version) dropped every seed"],
     ["Horizons", "1, 3 and 6 h; main horizon 3 h", "MPC horizon not chosen yet; 3 h covers a pre-heat", "one horizon only"],
     ["Spacing, warm-up", "a rollout every 30 min after the first 3 h", "dense enough to average days; the buffer needs time to fill", "every step: 6x slower"],
     ["Stability", "any rollout outside 230–350 K rejects", "one-step accuracy does not guarantee stable free runs", "seen: one-node seed unstable on one episode"],
     ["Test-time inputs", "future outside temperature and occupancy held at their last values", "a policy cannot know them at test time", "true future inputs: kept as a diagnostic"],
     ["Complexity", "+0.0001 K per AST node", "prefers readable formulas at equal accuracy", "no price: bloated candidates"],
     ["Selection", "validation buildings and station; test scored once at the end", "the test split stays unseen", "selecting on test: overfits"]],
    src="[our code] discovery/closedform.py (evaluate, physics_checks, episode_scores), evolve.py defaults")
dec("decsearch", "Decisions · search settings", "How the evolutionary search is configured, and why",
    [["Islands", "3 islands; each island's best copied to the next every 3 generations", "independent populations keep different ideas alive; migration shares winners", "one population: collapses onto one idea"],
     ["Requests", "per island per generation: 1 mutation (2 children), 1 crossover (2), 1 explore (1)", "balance refining, combining and new structure", "explore only: no refinement"],
     ["Parents", "2-way tournament among the island's best 6", "favours good parents without always picking the same one", "always the best: no diversity"],
     ["Analysis", "pre-mutation and pre-crossover tables in every request", "tells the LLM where a parent is wrong, by room type and condition", "score only: the LLM guesses"],
     ["LLM", "Claude Opus 5.5, adaptive thinking, effort high, JSON schema, cached system prompt", "structured answers; the system prompt is identical each generation", "free text: fragile parsing"],
     ["Robustness", "tolerant parsing + one retry; answers saved before evaluation; resume", "never pay twice for a request; a crash loses nothing", "-"],
     ["Speed", "--workers N forked evaluation (same results as serial)", "evaluation dominates wall time", "-"],
     ["Data per candidate", "up to 30 validation episodes and 8 training episodes (deterministic sample)", "bounded cost per candidate", "all episodes: slow on a large dataset"]],
    src="[our code] discovery/evolve.py, proposers.py, prompts.py")
use("discoutput", "bigmachine")

# ================================================================ 8 evidence
section("s8", "8", "Evidence", "How do we measure how wrong a model is, and what do real data say about the seeds?", "8 · Evidence")
new("image", "seedscompare", eyebrow="Real data · three seeds, nothing fitted", title="The measurement-corrected network is the best seed; all are far from the ceiling",
    img="seeds_compare.png", side="bottom",
    points=["Stateful network 0.74 / 0.93 / 1.16 K at 1 / 3 / 6 h; history buffer 0.99 / 1.16 / 1.34; persistence 0.66 / 0.97 / 1.47",
            "Test-time score (future weather unknown): network 0.959 K, history 1.011 K at 3 h", "Fitted ceiling 0.31 / 0.46 / 0.62 K: the gap is the search's job"],
    caption="[sim data] 12 thirty-day episodes, 6 shipped buildings × (pid_water, HVAC off), inputs rebuilt from commands",
    src="[measured] closedform.evaluate on kit evaluate.py runs replayed by our starterkit_data", notes="The one-node seed with nominal materials is unstable on one episode.")
use("devmodel", "byepisode", "realprofile", "canfit", "selftest", "wheretolook")

# ================================================================ 9 runtime
section("s9", "9", "Inside the policy", "Does the model behave the same inside a submission-style policy as it did offline, using only test-time information?", "9 · Runtime and parity")
new("table", "parity", eyebrow="Running the model inside the policy", title="Live forecasts equal the offline scores, exactly",
    rows=[["Check (building_train_8, atlanta_1, 576 steps)", "Live (inside the policy)", "Offline (training episode)"],
          ["Geometry: all 17 fields", "from the Kaggle-style metadata", "identical"], ["Inputs: outside air, comfort, occupants, gains", "from observations", "identical"],
          ["Inputs: air streams, reheat", "from own commands", "within 0.04 W/K and 0.3 W (CSV rounding)"],
          ["History seed: one-step / 3 h error", "0.4584 / 2.0203 K", "0.4584 / 2.0203 K"],
          ["Stateful seed: one-step / 3 h error", "0.2142 / 0.6014 K", "0.2142 / 0.6014 K"],
          ["Model time per step", "0.15–0.7 ms", "-"]],
    colw=[0.42, 0.3, 0.28], callout="The live check also found that the occupancy observation is two steps late; the offline episodes now use the observable value.",
    src="[measured] run_policy.py --check-replay; runtime/closedform_tracker.py, runtime/inputs.py")
new("table", "deploy", eyebrow="Deployment parity", title="Discovery uses only what a policy has at Kaggle test time",
    rows=[["What discovery uses", "At test time?", "Source at run time"],
          ["geometry", "yes", "floor plan, zone map, AHU assignment (metadata)"], ["room temperatures, outside air, comfort, clock", "yes", "observations"],
          ["occupants (building total)", "yes, 2 steps late", "observation"], ["air streams, supply temperature, reheat, boiler water", "yes", "own commands + the kit's equipment laws"],
          ["internal gains, universal constants", "assumed", "kit-wide values with cited sources"],
          ["future outside temperature and occupancy", "no", "held at last observed values in the score"],
          ["building identity, materials, weather window", "no", "never given; LLM analysis text is anonymous"]],
    colw=[0.4, 0.18, 0.42], size=25, src="docs/DISCOVERY_WALKTHROUGH.md (Deployment parity); tests/test_deployment.py")
dec("decruntime", "Decisions · runtime", "How the model runs inside the policy, and why",
    [["Timing", "the same conventions as the offline episodes (obs lag, AHU delay, occupancy lag)", "a model trained on one timing must run on the same", "checked by the parity test"],
     ["Shared code", "one command-to-heat module for the tracker and the fitted runtime", "two copies would drift apart", "-"],
     ["Safety layer", "clip into bounds, replace NaN/inf, fall back on any exception", "an invalid action is rejected or corrupts the episode", "fuzz test: 500 malformed actions all repaired"],
     ["Fallback", "the kit's pid_water if a model-based controller fails to start or errors", "never worse than the baseline because of a crash", "no fallback: one bug loses the scenario"],
     ["Tracker errors", "caught; after 10 errors the tracker is switched off", "tracking must never break control", "-"]],
    src="[our code] runtime/, submission/safety.py, submission/transfer_policy.py")

# ================================================================ 10 controllers
section("s10", "10", "Controllers", "How does the model become control, and does it help?", "10 · Controllers")
new("image", "costval", eyebrow="The reward as a cost", title="Our energy estimate matches the kit's logged costs",
    img="cost_validation.png", side="bottom",
    points=["Electric power correlation 0.999 (within 3–7%); gas 0.977; normalised energy term within ~3%", "Comfort term exact: the kit's shortfall (linear to 5 K, then quadratic), occupancy-weighted"],
    caption="[sim data] two 30-day pid_water runs; estimate from commands vs power implied by reward_components.csv",
    src="[our code] control/costs.py, control/validate_costs.py; price and carbon tables from the kit", notes="Mild weather: boiler losses (not modelled) explain the small gas gap.")
new("image", "occfc", eyebrow="Occupancy forecast", title="The comfort term needs occupants ahead of time: learn each building's day",
    img="occupancy_forecast.png", side="bottom",
    points=["Schedule prior (arrivals 06–12, departures 13–19), then the building's own weekday profile from the observed head count",
            "3 h ahead error 4–16 people vs persistence 15–43 on four 30-day runs; split between rooms by floor area"],
    caption="[sim data] building_train_2, atlanta_3, 30 days", src="[our code] control/occupancy.py")
new("image", "ctrlres", eyebrow="Does it help?", title="Model-based control beats pid_water in short checks",
    img="controllers.png", side="right",
    points=["MPC: −9.5% in fairbanks_1, +0.2% in atlanta_1; −4.5% on average", "Feed-forward: −1.8% in fairbanks_1, equal in atlanta_1",
            "Two buildings, two days each: a hint, not the benchmark", "Planning ~50 ms every 3 steps on 30–39 rooms"],
    caption="[measured] code/rehearse_submission.py through the kit's harness, stateful seed as the model",
    src="[measured] rehearse_submission.py --controller pid_water|ff|mpc", notes="The 30-scenario benchmark at 8,640 steps (task I5) decides.")
dec("decctrl", "Decisions · controllers", "How the controllers are built, and why",
    [["Base", "the kit's pid_water (vendored, Apache-2.0)", "a proven, safe inner loop; the model decides when and how hard", "raw-command MPC: 200+ dimensions per step"],
     ["Reduced action", "modes: schedule / comfort / eco / off", "4 candidates are cheap to simulate and cover pre-heat, coast and switch-off", "continuous setpoint search"],
     ["Planning", "copy pid_water, step it on predicted observations, price with the reward", "plans are closed-loop, like the real controller", "open-loop held actions"],
     ["Horizon", "MPC 2 h, replanned every 3 steps; feed-forward lead 3 h", "covers the morning pre-heat within the time budget", "longer horizon: more time per step"],
     ["Cost", "the reward itself: comfort with forecast occupants, energy and carbon from commands", "optimises the scored objective", "hand-tuned weights"],
     ["Unknowns", "outside air held at its last value; gas price = annual mean; holidays ignored", "the policy cannot read the date or future weather", "tracked as K9"],
     ["On/off", "the 'off' mode: every AHU and the boiler off when predicted cheapest", "switching off is the biggest saving at night", "ablation pending (K6)"]],
    src="[our code] control/planner.py, control/policies.py, control/costs.py, control/schedule.py")

# ================================================================ 11 submission
section("s11", "11", "The Kaggle submission", "What exactly gets uploaded, how is it checked, and when is it ready?", "11 · Submission")
new("table", "upload", eyebrow="What gets uploaded", title="One notebook plus one attached dataset",
    rows=[["Piece", "What it is", "Check"],
          ["kaggle/submission.ipynb", "subclass of smart_buildings_inference_server.Policy wrapping TransferPolicyCore", "runs the public example scenario"],
          ["dist/kaggle_dataset/code", "our runtime, controllers, safety layer, vendored pid_water (25 files)", "imports in a clean interpreter without the kit"],
          ["dist/kaggle_dataset/best.py", "the discovered formulas (optional)", "MPC used when present; pid_water otherwise"],
          ["code/rehearse_submission.py", "the same policy through the kit's harness on the 30 shipped scenarios", "score, seconds per step, safety repairs"]],
    colw=[0.3, 0.45, 0.25], src="[our code] package_submission.py, rehearse_submission.py, kaggle/submission.ipynb")
new("table", "gate", eyebrow="Readiness gate", title="What must be true before the model-based notebook is submitted",
    rows=[["Item", "Status"],
          ["pid_water-only notebook scored on Kaggle (proves the interface)", "to do (K1)"],
          ["Per-step time guard for large buildings", "to do (K7)"], ["Unsupported HVAC variants detected, fall back to pid_water", "to do (K8)"],
          ["Benchmark: 30 shipped scenarios at 8,640 steps + hold-out buildings and station", "to do (I5)"],
          ["Packaged dataset rehearsed locally; fallback verified by fault injection", "to do (L6)"],
          ["Deployment parity: test-time information only", "done (I7)"]],
    colw=[0.72, 0.28], callout="Deployable today: the pid_water-only notebook. The MPC version waits for the gate.",
    src="docs/RUNBOOK.md §11; docs/next_tasks.csv")

# ================================================================ 12 next
section("s12", "12", "Where we are", "What is built, what is open, and what comes next?", "12 · Next")
B["status"]["rows"] = [["Piece", "Status"],
                       ["Data generation, dataset, audits (kit tools only)", "built"], ["Geometry from floor plan and zone map", "built"],
                       ["Closed-form contract incl. stateful form; no fitting", "built"], ["Scoring with test-time information; floor, ceiling, self-test", "built and tested"],
                       ["Islands, operators, logs, report, resume, parallel", "built and tested (mock LLM)"],
                       ["Model inside the policy; live = offline", "built and verified"], ["Cost model, occupancy forecast, feed-forward, MPC", "built; short checks only"],
                       ["Kaggle notebook, package, rehearsal, safety layer", "built; not yet run on Kaggle"],
                       ["Real LLM generations; 30-scenario benchmark; time guard", "not yet"]]
B["status"]["src"] = "[our code]; docs/next_tasks.csv (77 tasks with status)"
B["status"]["size"] = 24
B["decisions"]["rows"] = [["Open question", "Options"], ["Goal", "best Kaggle score, or a generic discovered model"],
                          ["Simulator structure in the prompt", "graph only, or the full physics chain"], ["MPC horizon", "2 h used now; 1, 3, 6 h all reported"],
                          ["History summary", "raw lags, differences, window averages, LLM-proposed features"],
                          ["Occupants per room", "by area (used now), equally, or a time-of-day rule"],
                          ["Command-to-heat conversion", "keep hand-written, or let the LLM discover it"]]
B["decisions"]["callout"] = "Settled: no fitting · universal constants · stateful candidates allowed (fixed correction) · test-time scoring · early error accepted · discrete-time maps."
B["roadmap"]["steps"] = [("Kaggle check", "pid_water notebook"), ("Time guard", "large buildings"), ("Real LLM runs", "better formulas"),
                         ("Benchmark", "30 scenarios + hold-out"), ("MPC notebook", "after the gate")]
B["roadmap"]["points"] = ["Every stage is a complete submission with a working fallback", "5 submissions a day; 2 final picks; deadline January 15, 2027"]
B["figsources"]["rows"] = B["figsources"]["rows"][:4] + [["Seeds compared, cost check, occupancy forecast, controller results", "real data / measured",
                                                          "same 30-day runs; closedform.py, control/*.py, rehearse_submission.py"]] + B["figsources"]["rows"][4:]
use("figsources", "status", "decisions", "roadmap", "close")
