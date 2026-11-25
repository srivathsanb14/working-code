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

s(id="m_problem", kind="figure", eyebrow="The problem: every building is different",
  title="One policy, dropped into buildings it has never seen",
  img="buildings.png", h=530,
  callout="Six buildings ship pre-built (floor 1 shown, one scale). The kit's generator makes as many more as we like, one per seed; the 6 test buildings come from the same generator.",
  src="[kit] training_scenarios/*/zone_map.npy; run_floor_generator.py and README (generate any number; test from the same generator and ranges); room types: code/explore_zone_map.py; make_figs11.py")

s(id="m_weather", kind="figure", eyebrow="The problem: every climate is different",
  title="…and weather from −42 °C to +46 °C: only 30 episodes",
  img="weather.png", h=530,
  callout="Orange: the only 30-day episodes ever simulated. Grey: the rest of 2023, in the weather file but in no run. The test uses other 30-day windows, never released.",
  src="[kit] weather/csv/*.csv (2023, hourly) and weather/<scenario>.yaml start dates of the 30 shipped scenarios; training_set.yaml; README (test strips are not shipped strips); make_figs11.py")

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

s(id="m_loop", kind="flow", eyebrow="LLM discovery", title="The Claude API writes formulas; the scorer judges them; evolution keeps the best",
  steps=[["Write", "API: 3 first programs per island"], ["Check", "physics sanity"], ["Score", "5 forecast errors on validation"],
         ["Diagnose", "where, when, which term"], ["Improve", "mutate · cross · explore"], ["Test", "held-out tiers, once"]],
  callout="Each request is ≈3,400 tokens of code and diagnosis; the 15 M data values never leave our machine. A program survives unless another beats it on all five errors.",
  src="appendix: discflow, islands, pareto, operators slides; [our code] discovery/evolve.py, pareto.py")

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
            question="Every detail behind the nine slides: the problem, the data, the physics, the search, the evidence, the runtime and control.", notes="")

S = M + [APPX] + [copy.deepcopy(x) for x in FULL.S]
SECTIONS = [("m_cover", "Main (9 slides)"), ("appx", "Appendix · Opening")] + [(a, f"Appendix · {b}") for a, b in FULL.SECTIONS]
