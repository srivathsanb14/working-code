# 04 · First real discovery run (small)

**Needs:** 03 · **Produces:** `outputs/discovery/g1/` and written notes; a go / no-go · **Time:** a few hours

No real API generation has run yet: every result so far is from hand-written models. This run answers, for little
money, whether LLM-written formulas beat them and what goes wrong.

```bash
cd $WC
caffeinate -i $PY code/run_discovery.py --dataset $D/dataset --run g1 --generations 3 --islands 3 --max-usd 10 --workers $J --baselines
```
No `--test`: the test tiers stay unseen until 05. Interrupted: run the same command again (it finishes the open
generation and reuses saved answers, so no request is paid twice).

## Write down (in `outputs/discovery/g1/NOTES.md`)

- [ ] Cost per request and per generation; minutes per generation.
- [ ] Rejected candidates by reason: contract (imports, fitted tables, building-size branches), physics checks,
  unstable rollouts, timeouts, unparseable answers. Each frequent reason → a prompt fix.
- [ ] Duplicates (same code as an earlier program) per generation.
- [ ] That later requests contain the "scored WORSE than their parent" section and the answers react to it.
- [ ] Best main score per island per generation, against persistence and the hand-written network model.
- [ ] Which items of 06 the run calls for.

## Handoff to 05

Go: LLM candidates beat the hand-written network model on validation → set generations and budget for 05.
No-go: do the improvements in 06 that the notes call for, then repeat this step.
