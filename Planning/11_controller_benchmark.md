# 11 · Benchmark the controllers on all 30 scenarios and on held-out buildings

**Phase:** Control · **Priority:** P0 · **Size:** M · **Depends on:** 08 (a discovered best.py), 12

## Goal

Decide whether the model-based controllers beat the kit's best rule-based controller over full 30-day episodes. So far
there are only 2-day checks on two scenarios: MPC −1.4% on the mean against pid_water (fairbanks_1 −9.5%, atlanta_1
+0.5%), feed-forward −0.9%. That is a hint, not a result.

## Controllers (all in `code/control/policies.py`)

- `pid_water`: the kit's best baseline (mean loss ≈ 0.054 on the 30 scenarios).
- `FeedForwardPolicy`: pid_water, with the model deciding when to start pre-heating and when to set back early.
- `MPCPolicy`: every 15 minutes, compare 4 modes (schedule, comfort, eco, off) over 2 h by running pid_water on the
  model's forecast rooms and pricing each with the competition loss (0.6 comfort + 0.2 energy + 0.2 carbon); run the cheapest.

## Steps

```bash
python code/rehearse_submission.py --controller pid_water
python code/rehearse_submission.py --controller ff  --model outputs/discovery/main/best.py
python code/rehearse_submission.py --controller mpc --model outputs/discovery/main/best.py
```
(default: the 30 shipped scenarios at 8,640 steps.) Then the same on held-out buildings from task 02 at the hard
station. Report per scenario: score, comfort / energy / carbon terms, seconds per step, safety repairs; and the worst
scenario per controller. Include an ablation of the MPC's 'off' mode.

## Done when

- A table in `docs/` shows that the MPC beats pid_water on average and in the worst scenario within the time budget,
  or explains why not; the MPC horizon (2 h) is confirmed or changed from these numbers.
