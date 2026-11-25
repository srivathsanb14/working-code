# 12 · Choose the two final submissions, and write the paper

**Needs:** 05, 09, 10 · **Deadlines:** entry and team merger 3 Jan 2027; final submission and paper 15 Jan 2027
(OpenReview). Kaggle allows 5 submissions a day and 2 final picks.

- [ ] Decide what to optimise when they conflict: best Kaggle score, or the most generic discovered model.
- [ ] Two picks among the submitted notebooks: the best-scoring one and the most robust one (the one that beat
  pid_water in the worst scenario in 09); record the rationale with the benchmark numbers.
- [ ] Bring the decks in line with the results: edit `docs/deck_src/deck.py` (one file holds every slide), regenerate the
  figures with `docs/deck_src/make_figs11.py`, rebuild with `story.py` / `render.js`; replace the hand-written-model
  evidence with discovered-model results.
- [ ] Paper: method (fixed state-space structure, LLM-written closed-form coefficients, computed split), results on every
  tier, ablations (worse-children feedback, analysis sections), the controller benchmark.
