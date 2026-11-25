# 10 · Make candidate scoring fast enough for the full dataset

**Phase:** Discovery · **Priority:** P0 · **Size:** M · **Depends on:** 03

## Goal

Every candidate is scored on the validation sample (30 episodes by default) at 1, 3 and 6 h horizons, with a rollout
started every 30 minutes. With 15 candidates per generation this must take minutes, not hours, including on large
generated buildings.

## Steps

1. Time `closedform.evaluate` for one candidate on the stage-1 validation sample; `prepare()` is already cached per
   building and the per-candidate time is logged.
2. If too slow: parallelise across episodes (today candidates run in parallel with `--workers`, episodes in series),
   and profile the rollout loop on the largest building.
3. Record the time per candidate and per generation in `docs/RUNBOOK.md`.

## Done when

- One candidate scores on the full validation sample in under about 2 minutes on the data machine, and a generation
  of 15 candidates with 16 workers in under about 10 minutes (or the chosen budget is recorded with the measured time).
