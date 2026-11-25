# 12 · Package the discovered model and pass the readiness gate

**Needs:** 06, 08, 09, 10, 11 · **Time:** half a day

Submit the model-based notebook only when every item holds (record them in `docs/RUNBOOK.md` §11):

- [ ] pid_water notebook scored on Kaggle (11).
- [ ] Time guard in place and tested on the largest building (08).
- [ ] Unsupported HVAC variants fall back (09).
- [ ] Controller benchmark passed (10).
- [ ] `$PY code/package_submission.py --model outputs/discovery/main/best.py --out dist/kaggle_dataset` builds; it is
  uploaded and attached; the notebook loads it in `__init__` well within the 15-minute first-call limit.
- [ ] Local rehearsal with the packaged files matches the benchmark.
- [ ] Fault injection: a crashing or NaN-producing model makes the policy fall back to pid_water, never break the
  episode.

## Handoff to 14

Every box ticked; the model-based notebook submitted and its public score recorded.
