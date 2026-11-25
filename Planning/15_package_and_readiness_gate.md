# 15 · Package the discovered model and pass the readiness gate

**Phase:** Submission · **Priority:** P0 · **Size:** S · **Depends on:** 08, 11, 12, 13, 14

## Goal

Submit the model-based notebook only when every safety item is checked.

## Checklist (record each in `docs/RUNBOOK.md` §11)

- [ ] pid_water notebook scored on Kaggle (task 14).
- [ ] Time guard in place and tested on the largest building (task 12).
- [ ] Unsupported HVAC variants fall back (task 13).
- [ ] Controller benchmark passed (task 11).
- [ ] `python code/package_submission.py --model outputs/discovery/main/best.py` builds the dataset; it is uploaded and
      attached; the notebook loads it in `__init__` well within the 15-minute first-call limit.
- [ ] Local rehearsal with the packaged files (`rehearse_submission.py --model ...`) matches the benchmark.
- [ ] Fault injection: a crashing or NaN-producing model makes the policy fall back to pid_water, never break the episode.

## Done when

- Every item is ticked; then the model-based notebook is submitted and its public score recorded.
