# 08 · Full discovery run, then the test tiers once

**Phase:** Discovery · **Priority:** P0 · **Size:** L · **Depends on:** 07 (and 09 if the first run shows problems)

## Goal

Run the search until it stops improving, then score the chosen model and the Pareto front once on the held-out tiers.

## Run

```bash
SCENARIOS=$D/scenarios RESULTS=$D/results DATASET=$D/dataset GENS=<from 07> MAX_USD=<from 07> FROM=3 ./run_pipeline.sh
```
(step 3 of the pipeline = `run_discovery.py ... --test`.) For the per-building fitted ceiling, which is a diagnostic
and never deployed, run `run_discovery.py` directly once with `--ceiling` added.

**Stopping rule:** stop when the best main score has not improved by more than 1% for 5 generations, or the budget is
spent. If interrupted, run the same command with the same `--run` name: it finishes the unfinished generation and reuses the answers already saved, so no request is paid twice.

## Acceptance

| Tier | best.py must |
|---|---|
| validation | beat persistence and the hand-written network model at 1, 3 and 6 h |
| test_easy, test_medium | beat persistence at 1, 3 and 6 h |
| test_hard (unseen building, hard climate) | beat persistence at 3 h, or the gap is explained (every hand-written model loses to persistence at Fairbanks today: 2.11 K network vs 1.27 K persistence) |

The test numbers are read once and never used to choose a model.

## Done when

- `outputs/discovery/main/REPORT.md`, `best.py` and `summary.json` (with test-tier rows) are committed.
- The ceiling is in REPORT.md, so the remaining gap is known.
