"""The meeting version: 9 main slides, then the full working deck as the appendix.  Every number below is taken from a
slide of the full deck (named in each source line), which carries its own sources."""
import copy

import content17 as FULL

M = []


def s(**kw):
    M.append(kw)


s(id="m_cover", kind="statement", eyebrow="NeurIPS 2026 · Smart Buildings Challenge", title="One controller for every building",
  sub="How we let an LLM discover the formulas of a physics model that forecasts every room of any building, and what we know so far",
  foot="9 slides · the full working deck follows as the appendix", notes="")

s(id="m_problem", kind="table", eyebrow="The problem", title="One policy, dropped into buildings and weather it has never seen",
  rows=[["", ""],
        ["Task", "one policy, run unchanged on 6 unseen buildings × 5 unseen weather stations: 30 scenarios of 8,640 five-minute steps"],
        ["Score", "mean loss per step = 0.6 × comfort + 0.2 × energy + 0.2 × carbon (lower is better)"],
        ["Time", "one 9-hour Kaggle run plays all 30 scenarios: about 0.125 s per step, simulator included"],
        ["What it sees", "the building's floor plan and metadata at the start; every 5 minutes room temperatures, outside air, occupancy, clock"],
        ["The bar", "the kit's best rule-based controller (pid_water) scores 0.054; the leaderboard leader 0.0045"]],
  colw=[0.18, 0.82], size=25, font=12,
  callout="To plan heating ahead, a controller needs a forecast of every room's temperature, in a building it has never seen, from its first hours.",
  src="appendix: evaluation, budget, scoring, leaderboard slides ([website] scoring.html; [kit] baseline_returns.json; [Kaggle] leaderboard)")

s(id="m_idea", kind="eq", eyebrow="Our approach", title="A physics model of every room, whose coefficients are formulas instead of fitted numbers",
  eq=("T_i(t+1) = ( T_i + air_i·g_i·T_supply + outside_i·T_out + Σ_j nb_ij·T_j + slab_i·S_i + wall_i·W_i + inner_i·P_i + heat_i·Q_i )\n"
      "           / ( 1 + air_i·g_i + outside_i + Σ_j nb_ij + slab_i + wall_i + inner_i )          + three hidden heat stores S, W, P\n"
      "each coefficient = formula( floor-plan geometry, measured history up to now, universal constants )"),
  eqsize=19, eqpt=10,
  points=["The function is fixed: the simulator's own physics (backward Euler, the same scheme), the same in every building",
          "Geometry comes from the floor plan; the six hidden materials per building must be read from the measured history by closed-form expressions",
          "No backpropagation and no fitting, anywhere (checked by the code); the formulas run in about 0.1 ms per step",
          "An LLM writes the formulas; data on buildings it never saw decides which are right"],
  src="appendix: assemble, terms, nofitdef, enforce, parity slides; [our code] discovery/coefform.py, contract.py")

s(id="m_data", kind="table", eyebrow="Data and split", title="Whole episodes, split by a rule; there is no training set because nothing is trained",
  rows=[["Split", "What is new in it", "On the six shipped buildings"],
        ["Validation", "nothing: every score and every analysis the LLM sees comes from here", "12 scenarios (buildings 1, 4, 5, 7)"],
        ["Test · easy", "a different 30-day window of a seen building and station", "4 scenarios"],
        ["Test · medium", "an unseen building", "buildings 2 and 8: 8 scenarios"],
        ["Test · hard", "an unseen building in an unseen climate", "buildings 2 and 8 at Fairbanks: 2 scenarios"]],
  colw=[0.18, 0.5, 0.32], size=23, font=11.5,
  callout=("Hard climate = the station farthest from the rest in Wasserstein distance (Fairbanks, 21.9 K; next Singapore 14.6 K); held-out buildings "
           "= the most unusual floor plans. An episode (one building, one window, one controller) is never cut between splits."),
  src="appendix: splits, whichheld slides; [our code] discovery/splits.py; [measured] make_splits.py on the 30 shipped scenarios")

s(id="m_loop", kind="flow", eyebrow="LLM discovery", title="The Claude API writes formulas; the scorer judges them; evolution keeps the best",
  steps=[["Init", "API writes 3 programs per island"], ["Check", "physics sanity on a real floor plan"], ["Score", "5 forecast errors on validation"],
         ["Analyse", "where, when, which term is wrong"], ["Ask", "mutate · cross · explore: one API request each"], ["Keep", "Pareto front; test tiers at the end"]],
  points=["3 islands, 5 new programs per island per generation; every 3 generations each island's best moves to the next island",
          "Five objectives at once (3 h, 1 h, 6 h, exterior rooms, interior rooms): a program survives unless another beats it on all five",
          "Every prompt, answer, analysis and score is saved to disk; a budget cap stops the run"],
  src="appendix: discflow, islands, pareto, operators slides; [our code] discovery/evolve.py, pareto.py")

s(id="m_io", kind="compare", eyebrow="The LLM's input and output", title="It gets a diagnosis, not the data; it returns code, not numbers",
  left=["In one request (≈ 3,400 tokens + system prompt)",
        ["Physics, the fixed function, the rules (same every time)",
         "The parent's code and its five scores",
         "Where, when, through which term it is wrong (all validation steps)",
         "One real day, hourly, one room of each type",
         "No chat history"]],
  right=["Back from the API",
         ["JSON: a name, a hypothesis, and a Python module per candidate",
          "A formula for each of the 11 coefficients (heat input optional)",
          "Rejected if it fits constants or special-cases buildings",
          "Numbers appear only when the formulas run on a building's own data"]],
  callout="The 6 validation runs hold 15 million numbers (≈ 26 million tokens): too many to send, so the scorer runs every candidate on all of them here and sends what it found.",
  src="appendix: proposers, llmout, sysprompt, llmreads slides; [our code] prompts.py, proposers.py, contract.py; tests/test_api_loop.py")

s(id="m_feedback", kind="code", eyebrow="The feedback that drives each mutation (real output)", title="Which term is wrong, for which rooms, under which conditions",
  code=("slope s: the term should be about (1 + s) times as large\n"
        "                               neighbours   slab    wall   inner\n"
        "corner rooms                     -0.68     +0.35   -0.44   -0.73\n"
        "one-side exterior rooms          -0.85     +0.68   -0.42   -1.17\n"
        "interior rooms                   -0.76     +1.06           -1.05\n"
        "HVAC air on                      -0.51     +0.04   -0.22   +0.20\n"
        "HVAC air off                     -0.80     +0.07   -0.61   -1.14"),
  points=["The hand-written network model, on the 6 validation runs",
          "Interior rooms want a slab term about twice as strong: a formula for that room type, not a global factor",
          "The wall and inner-wall stores are far too strong with the HVAC off and nearly right with it on",
          "The LLM writes its next formulas from exactly this"],
  src="appendix: termsbytype, llmterms, profiletab slides; [measured] closedform.term_regression")

s(id="m_evidence", kind="table", eyebrow="Evidence so far (hand-written models; no API generation has run yet)",
  title="On unseen buildings the network model beats 'temperature stays'; in the cold climate nothing does yet",
  rows=[["3 h forecast error (K), as at test time", "Validation", "Unseen buildings 2, 8", "Building 1 at Fairbanks"],
        ["Persistence ('temperature stays')", "0.889", "0.948", "1.274"],
        ["Network model (hand-written)", "0.740", "0.538", "2.114"],
        ["History model (hand-written)", "1.099", "1.270", "1.732"]],
  colw=[0.4, 0.18, 0.21, 0.21], size=23, font=12,
  callout=("The starting point the LLM has to beat. The cold climate is the weak spot, which is why the hard test tier uses it. "
           "Locally we have 12 thirty-day runs (6 buildings × 2 controllers), one window per building, so the easy and hard tiers are empty here."),
  src="appendix: tiers, seedscompare slides; [measured] closedform.evaluate on 12 thirty-day runs, split by splits.split_from_episodes")

s(id="m_next", kind="table", eyebrow="Where we are", title="Built and tested; next come data at scale and the first API runs",
  rows=[["", "Status"],
        ["Pipeline: episodes → split → API discovery → test tiers", "built and tested (a stand-in API client in the tests)"],
        ["Model inside a Kaggle policy; live forecasts = offline scores", "built and verified"],
        ["1. Generate data with the kit's tools on the big machine", "next"],
        ["2. Real API generations, scored on the test tiers", "next, with a budget cap"],
        ["3. Control (later): every 15 min, compare 4 modes over the next 2 h with the model, run the cheapest", "built; early 2-day check −1.4% vs pid_water"],
        ["4. Submit: a pid_water notebook is ready as the fallback", "deadline Jan 15, 2027"]],
  colw=[0.62, 0.38], size=22, font=11.5,
  src="appendix: status, whympc, gaincalc, ctrlres, roadmap slides")

APPX = dict(id="appx", kind="section", num="A", title="Appendix: the full working deck",
            question="Every detail behind the nine slides: the problem, the data, the physics, the search, the evidence, the runtime and control.", notes="")

S = M + [APPX] + [copy.deepcopy(x) for x in FULL.S]
SECTIONS = [("m_cover", "Main (9 slides)"), ("appx", "Appendix · Opening")] + [(a, f"Appendix · {b}") for a, b in FULL.SECTIONS]
