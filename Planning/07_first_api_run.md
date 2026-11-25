# 07 · First real API run (small)

**Phase:** Discovery · **Priority:** P0 · **Size:** M · **Depends on:** 04, 06

## Goal

Answer, for little money: does the loop run end to end with real answers, how much does a generation cost, how often do
candidates fail and why, and do LLM-written formulas beat the hand-written network model on validation? No real API
generation has run yet; every result so far is from hand-written models.

## Run

```bash
python code/run_discovery.py --dataset $D/dataset --run g1 --generations 3 --islands 3 --max-usd 10 --max-val 30 --workers 16 --baselines
```

No `--test`: the test tiers stay unseen until task 08. `--baselines` scores the hand-written models as reference rows
(never parents).

## What to check (write the answers in `outputs/discovery/g1/NOTES.md`)

- **Cost:** dollars and tokens per request and per generation (`log.md`, `generations/*/*/candidates.json` usage).
- **Failures:** how many candidates were rejected, by reason: contract (imports, fitted tables, building-size branches),
  physics checks, unstable rollouts, timeouts, unparseable answers. Each reason should be fixable by prompt wording.
- **Duplicates:** how many answers repeat an existing program (skipped by hash).
- **Worse children:** confirm the "scored WORSE than their parent" section appears in later requests and is used.
- **Scores:** best main score per island per generation against persistence and the hand-written network model
  (validation 3 h error 0.740 K vs persistence 0.889 K on the 12-run set).
- **Prompt sizes** as measured, against the estimate in task 06.

## Done when

- `REPORT.md` exists with real candidates; failure reasons reviewed and the prompt fixes made.
- Cost per generation is known and the full-run budget (task 08) is set from it.
