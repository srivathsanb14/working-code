"""The meeting version: 10 main slides, then the full working deck as the appendix.  Every number below is taken from a
slide of the full deck (named in each source line), which carries its own sources."""
import copy

import content18 as FULL

M = []


def s(**kw):
    M.append(kw)


s(id="m_cover", kind="statement", eyebrow="NeurIPS 2026 · Smart Buildings Challenge", title="One controller for every building",
  sub="How we let an LLM discover the formulas of a physics model that forecasts every room of any building, and what we know so far",
  foot="10 slides · the full working deck follows as the appendix", notes="")

s(id="m_problem", kind="figure", eyebrow="The problem: every building is different",
  title="One policy, dropped into buildings it has never seen",
  img="buildings.png", h=700,
  src="[kit] training_scenarios/*/zone_map.npy; run_floor_generator.py and README (generate any number; test from the same generator and ranges); room types: code/explore_zone_map.py; make_figs11.py")

s(id="m_weather", kind="figure", eyebrow="The problem: every climate is different",
  title="…and weather from −42 °C to +46 °C: only 30 episodes",
  img="weather.png", h=700,
  src="[kit] weather/csv/*.csv (2023, hourly) and weather/<scenario>.yaml start dates of the 30 shipped scenarios; training_set.yaml; [website] scoring.html (5 unseen stations); make_figs11.py")

s(id="m_episode", kind="figure", eyebrow="The data: one episode = one kit run",
  title="1 building × 1 thirty-day window × 1 controller",
  img="episode.png", h=700,
  src="[sim data] building_train_4 / denver_2 / pid_water, kit evaluate.py (8,640 steps), read by build_dataset.py; [kit] zone_map.npy; make_figs11.py")

s(id="m_idea", kind="image", side="right", eyebrow="Our approach", title="Forecast every room with physics whose numbers are formulas, not fits",
  img="ep30_profile_pred.png",
  caption="[sim data] building 2 (never used to choose formulas), Atlanta, days 7–14: simulator (dark) vs the hand-written network model 3 h ahead (orange)",
  points=["Same heat physics in every building", "Geometry: read from the floor plan", "Materials: read from the room's own history",
          "Each coefficient = a formula the LLM writes", "Nothing fitted, anywhere"],
  src="appendix: assemble, terms, nofitdef, realprofile slides; [our code] discovery/coefform.py")

s(id="m_data", kind="figure", eyebrow="Data and split", title="Whole scenarios are split by a rule: no training set, three held-out tiers",
  img="split_grid.png", h=640,
  src="appendix: splits, whichheld slides; [measured] make_splits.py on the 30 shipped scenarios (outputs/split_shipped.json); make_figs11.py")

s(id="m_model", kind="image", side="below", eyebrow="The model: a closed-form equation, not a neural network",
  title="Fixed physics; the LLM writes only the coefficients",
  img="eq_model.png",
  points=["Fixed: every room mixes its own temperature with supply air, outside air, its neighbours and three hidden heat stores "
          "(slab S, outside wall W, inner walls P). For the whole building this is a linear state-space model x(t+1) = A x(t) + B u(t) + E d(t)",
          "Searched: the 11 coefficients, each a closed-form formula of the floor plan, the measured history and physical constants",
          "Why not a neural network or fitted constants: a test building is new, with no data to train or fit on, and 0.125 s per step. "
          "A formula works from the first step, and we can read it"],
  src="[our code] prompts.CF_CONTRACT (the fixed function), discovery/coefform.py (TermModel); appendix: discproblem, terms slides")

s(id="m_loop", kind="flow", eyebrow="LLM discovery", title="The Claude API writes formulas; the scorer judges them; evolution keeps the best",
  steps=[["Write", "API: 3 first programs per island"], ["Check", "physics sanity"], ["Score", "5 forecast errors on validation"],
         ["Diagnose", "where, when, which term"], ["Improve", "mutate · cross · explore"], ["Test", "held-out tiers, once"]],
  points=["3 islands; each asks for 5 new programs per generation; every 3 generations an island's best moves to the next",
          "A program stays unless another is at least as good on all five errors (Pareto)"],
  src="appendix: discflow, islands, pareto, operators slides; [our code] discovery/evolve.py, pareto.py")

s(id="m_io", kind="flow", eyebrow="The LLM's input and output",
  title="One request: what the LLM reads, and what it writes back",
  steps=[["Fixed rules", "equation, data columns, constraints"], ["The parent", "its code and its 5 errors"],
         ["Diagnosis", "which term is wrong, where, when"], ["One real day", "24 h × 3 rooms"],
         ["Claude", "one request, no memory"], ["Code back", "11 formulas + the reason"]],
  points=["Never sent: the episodes (15 million numbers), test results, building or station names"],
  src="[our code] prompts.py (CF_CONTRACT, CF_DATA, RESPONSE_SCHEMA), evolve.profile_text; [measured] prompt sizes; tests/test_api_loop.py")

s(id="m_feedback", kind="figure", eyebrow="The diagnosis that drives each mutation (real output)", title="Which term is wrong, for which rooms, under which conditions",
  img="term_heatmap.png", h=640,
  src="appendix: termsbytype, llmterms slides; [measured] closedform.term_regression (outputs/term_slopes_network.json); make_figs11.py")

APPX = dict(id="appx", kind="section", num="A", title="Appendix: the full working deck",
            question="Every detail behind the ten slides: the problem, the data, the physics, the search, the evidence, the runtime and control.", notes="")

S = M + [APPX] + [copy.deepcopy(x) for x in FULL.S]
SECTIONS = [("m_cover", "Main (10 slides)"), ("appx", "Appendix · Opening")] + [(a, f"Appendix · {b}") for a, b in FULL.SECTIONS]
