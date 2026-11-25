# 10 · Package the discovered model and pass the readiness gate

**Needs:** 05, 07, 08, 09 · **Time:** half a day

Submit the model-based notebook only when every item holds (record them in `docs/RUNBOOK.md` §11):

- [ ] Time guard in place and tested (07).
- [ ] Unsupported HVAC variants fall back (08).
- [ ] Controller benchmark passed (09).
- [ ] `$PY code/package_submission.py --model outputs/discovery/main/best.py --out dist/kaggle_dataset` builds (it checks
  the package imports without the kit); it is uploaded as a Kaggle Dataset and attached to `kaggle/submission.ipynb`;
  the notebook loads it in `__init__` well within the 15-minute first-call limit and runs the public example scenario.
- [ ] Local rehearsal with the packaged files matches the benchmark.
- [ ] Fault injection: a crashing or NaN-producing model makes the policy fall back to pid_water, never break the
  episode.

## Handoff to 12

Every box ticked; the model-based notebook submitted and its public score recorded.
