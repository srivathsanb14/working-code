# 06 · Full discovery run, then the test tiers once

**Needs:** 04 (go), 05 · **Produces:** `outputs/discovery/main/` with `best.py`, `REPORT.md`, `summary.json` with
test-tier rows · **Time:** many hours

```bash
cd $WC
caffeinate -i $PY code/run_discovery.py --dataset $D/dataset --run main --generations <from 04> --islands 3 --max-usd <from 03> --workers $J --baselines --test
```
The test tiers are scored at the end in chunks of 16 episodes across the `$J` workers, so memory stays small however
large a tier is; `--max-test N` gives a quicker stratified read. For the per-building fitted ceiling (a diagnostic,
never deployed) add `--ceiling` once.

**Stopping rule:** stop when the best main score has not improved by more than 1% in 5 generations, or the budget is
spent. Interrupted: run the same command again.

## Acceptance

| Tier | best.py must |
|---|---|
| validation | beat persistence and the hand-written network model at 1, 3 and 6 h |
| test_easy, test_medium | beat persistence at 1, 3 and 6 h |
| test_hard | beat persistence at 3 h, or explain the gap (today every hand-written model loses to persistence at Fairbanks: network 2.11 K vs persistence 1.27 K) |

The test numbers are read once and never used to choose a model.

## Handoff to 08 and 10

`best.py` committed, and checked inside a policy built from the metadata alone:
```bash
$PY code/run_policy.py --model outputs/discovery/main/best.py --scenarios-dir $D/scenarios --split $D/dataset/split.json --on test_medium --check-replay
```
Live and offline forecast errors and geometry must match.
