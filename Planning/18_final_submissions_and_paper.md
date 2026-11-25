# 18 · Choose the two final submissions, and write the paper

**Phase:** Final · **Priority:** P0 (selection), P1 (paper) · **Size:** L · **Depends on:** 08, 11, 15

## Deadlines

Entry and team-merger deadline 3 Jan 2027; final submission and paper round 15 Jan 2027 (OpenReview). Kaggle allows 5
submissions a day and 2 final picks.

## Steps

1. **Decide the goal when they conflict:** best Kaggle score, or the most generic discovered model. This decides
   which of the two picks is "ambitious".
2. **Two picks:** one safe (pid_water, or the model-based notebook if it beat pid_water everywhere in task 11) and
   one ambitious (the best model-based controller); rationale recorded with the benchmark numbers.
3. **Deck in sync:** regenerate the deck figures from the main run (`docs/deck_src/make_figs11.py` and the
   content layers); replace hand-written-model evidence with discovered-model results; every source line correct.
4. **Paper:** method (fixed state-space structure, LLM-written closed-form coefficients, computed split),
   results on every tier, ablations (worse-children feedback, analysis sections), controller benchmark.

## Done when

- Both final submissions are selected on Kaggle with the rationale recorded; the paper is submitted.
