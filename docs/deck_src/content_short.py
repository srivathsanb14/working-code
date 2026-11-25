"""The meeting version: 9 main slides, then the full working deck as the appendix.  Every number below is taken from a
slide of the full deck (named in each source line), which carries its own sources."""
import copy

import content18 as FULL

M = []


def s(**kw):
    M.append(kw)


s(id="m_cover", kind="statement", eyebrow="NeurIPS 2026 · Smart Buildings Challenge", title="One controller for every building",
  sub="How we let an LLM discover the formulas of a physics model that forecasts every room of any building, and what we know so far",
  foot="12 slides · the full working deck follows as the appendix", notes="")

s(id="m_problem", kind="figure", eyebrow="The problem: every building is different",
  title="One policy, dropped into buildings it has never seen",
  img="buildings.png", h=530,
  callout="Six buildings ship pre-built (floor 1 shown, one scale). The kit's generator makes as many more as we like, one per seed; the 6 test buildings come from the same generator.",
  src="[kit] training_scenarios/*/zone_map.npy; run_floor_generator.py and README (generate any number; test from the same generator and ranges); room types: code/explore_zone_map.py; make_figs11.py")

s(id="m_weather", kind="figure", eyebrow="The problem: every climate is different",
  title="…and weather from −42 °C to +46 °C: only 30 episodes",
  img="weather.png", h=530,
  callout="Orange: the only 30-day episodes ever simulated. Grey: the rest of 2023, in the weather file but in no run. The test uses 5 other stations, never released.",
  src="[kit] weather/csv/*.csv (2023, hourly) and weather/<scenario>.yaml start dates of the 30 shipped scenarios; training_set.yaml; [website] scoring.html (5 unseen stations); make_figs11.py")

s(id="m_episode", kind="figure", eyebrow="The data: one episode = one kit run",
  title="1 building × 1 thirty-day window × 1 controller",
  img="episode.png", h=530,
  callout="Made with the starter kit's own commands; this repo only reads the runs. Per room: temperature (the target) and four heat inputs. Building-wide: five more. Plus the floor plan.",
  src="[sim data] building_train_4 / denver_2 / pid_water, kit evaluate.py (8,640 steps), read by build_dataset.py; [kit] zone_map.npy; make_figs11.py")

s(id="m_idea", kind="image", side="right", eyebrow="Our approach", title="Forecast every room with physics whose numbers are formulas, not fits",
  img="ep30_profile_pred.png",
  caption="[sim data] building 2 (never used to choose formulas), Atlanta, days 7–14: simulator (dark) vs the hand-written network model 3 h ahead (orange)",
  points=["Same heat physics in every building", "Geometry: read from the floor plan", "Materials: read from the room's own history",
          "Each coefficient = a formula the LLM writes", "Nothing fitted, anywhere"],
  src="appendix: assemble, terms, nofitdef, realprofile slides; [our code] discovery/coefform.py")

s(id="m_data", kind="figure", eyebrow="Data and split", title="Whole scenarios are split by a rule: no training set, three held-out tiers",
  img="split_grid.png", h=480,
  callout="The search only sees blue. Hard: unseen building, most different climate. Medium: unseen building. Easy: a new month.",
  src="appendix: splits, whichheld slides; [measured] make_splits.py on the 30 shipped scenarios (outputs/split_shipped.json); make_figs11.py")

s(id="m_split", kind="table", eyebrow="How episodes are split (computed from the data, nothing hard-coded)",
  title="Whole scenarios go to validation or to one test tier, by rule",
  rows=[["", "Rule", "On the 30 shipped scenarios"],
        ["Unit", "a whole scenario (building + window): every controller's run on it goes together", "30 scenarios"],
        ["Hard climate", "the station whose outside temperatures are farthest from the rest (Wasserstein-1)", "Fairbanks: 21.9 K (next: Singapore 14.6 K)"],
        ["Unseen buildings", "the most unusual quarter by floor-plan numbers", "buildings 2 and 8"],
        ["test · hard", "unseen building in the hard climate", "2"],
        ["test · medium", "unseen building, other climates", "8"],
        ["test · easy", "seen building and station, a new month (1 window in 4)", "4"],
        ["validation", "everything else: the only data the search ever sees", "12 (+4 extra: seen building, hard climate)"]],
  colw=[0.18, 0.5, 0.32], size=22, font=12,
  callout="No training split: nothing is fitted. The test tiers are scored once, at the end.",
  src="[our code] discovery/splits.py, make_splits.py; [measured] outputs/split_shipped.json (_why)")

s(id="m_loop", kind="flow", eyebrow="LLM discovery", title="The Claude API writes formulas; the scorer judges them; evolution keeps the best",
  steps=[["Write", "API: 3 first programs per island"], ["Check", "physics sanity"], ["Score", "5 forecast errors on validation"],
         ["Diagnose", "where, when, which term"], ["Improve", "mutate · cross · explore"], ["Test", "held-out tiers, once"]],
  callout="3 islands, 5 new programs each per generation. A program survives unless another beats it on all five errors (Pareto).",
  src="appendix: discflow, islands, pareto, operators slides; [our code] discovery/evolve.py, pareto.py")

s(id="m_io", kind="compare", eyebrow="The LLM's input and output",
  title="In: rules, the parent and a diagnosis. Out: code for 11 formulas",
  left=["Into the LLM (one request: ≈ 3,400 tokens + fixed 11 KB rules)",
        ["Fixed: the room equation, the column names and units, the rules, how code is scored",
         "The parent's code and five errors; recent edits that made a child worse, and rejected ones",
         "A diagnosis from all validation runs: when, where and through which term it is wrong",
         "One real day: 24 hourly rows × 3 rooms (measured vs forecast, supply air, reheat, outside)"]],
  right=["Out of the LLM (JSON)",
         ["name, hypothesis and a Python module",
          "coefficients(geo, hist, U): a formula for each of the 11 coefficients",
          "HISTORY (≤ 24 h it reads) and REFRESH (how often it is recomputed)",
          "We run it on every validation run: numbers per room → five errors → the next request"]],
  callout="Never sent: the episodes (15 M numbers), test results, building or station names, earlier requests.",
  src="[our code] prompts.py (CF_CONTRACT, CF_DATA, CF_SCORING, RESPONSE_SCHEMA), evolve.profile_text; [measured] prompt sizes; tests/test_api_loop.py")

s(id="m_feedback", kind="figure", eyebrow="The diagnosis that drives each mutation (real output)", title="Which term is wrong, for which rooms, under which conditions",
  img="term_heatmap.png", h=470,
  callout="Network model, 6 validation runs: interior rooms need a stronger slab; with the HVAC off the stores are far too strong.",
  src="appendix: termsbytype, llmterms slides; [measured] closedform.term_regression (outputs/term_slopes_network.json); make_figs11.py")

s(id="m_evidence", kind="figure", eyebrow="Evidence so far (hand-written models; no API generation has run yet)",
  title="The network model beats 'temperature stays' on unseen buildings, not yet in the cold",
  img="tiers_bars.png", h=470,
  callout="The starting point the LLM must beat. The cold climate is the weak spot, which is why the hard tier uses it.",
  src="appendix: tiers, seedscompare slides; [measured] closedform.evaluate on 12 thirty-day runs (6 buildings × 2 controllers); make_figs11.py")

s(id="m_next", kind="flow", eyebrow="Where we are", title="Built and tested; next come data at scale and the first API runs",
  steps=[["Built", "pipeline, tests, Kaggle policy"], ["Data", "generate with the kit"], ["API runs", "budget-capped"],
         ["Benchmark", "all 30 scenarios"], ["Control", "plan with the model (later)"], ["Submit", "by Jan 15, 2027"]],
  callout="Fallback ready today: a pid_water notebook. Control later: every 15 min, try 4 modes for 2 h on the model, run the cheapest (early check −1.4%).",
  src="appendix: status, whympc, gaincalc, ctrlres, roadmap slides")

APPX = dict(id="appx", kind="section", num="A", title="Appendix: the full working deck",
            question="Every detail behind the twelve slides: the problem, the data, the physics, the search, the evidence, the runtime and control.", notes="")

S = M + [APPX] + [copy.deepcopy(x) for x in FULL.S]
SECTIONS = [("m_cover", "Main (12 slides)"), ("appx", "Appendix · Opening")] + [(a, f"Appendix · {b}") for a, b in FULL.SECTIONS]
