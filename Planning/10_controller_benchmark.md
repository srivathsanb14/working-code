# 10 · Benchmark the controllers

**Needs:** 06 (`best.py`), 08 · **Produces:** a results table in `docs/` · **Time:** about a day of compute

Do the model-based controllers beat the kit's best rule-based controller over full 30-day episodes? So far there are
only 2-day checks on two scenarios: MPC −1.4% on the mean against pid_water (fairbanks_1 −9.5%, atlanta_1 +0.5%),
feed-forward −0.9%.

Controllers (`code/control/policies.py`): `pid_water` (the kit's best baseline, mean loss ≈ 0.054 on the 30
scenarios); `FeedForwardPolicy` (pid_water, with the model choosing when to pre-heat and set back); `MPCPolicy` (every
15 minutes, compare 4 modes over 2 h on the model's forecast and run the cheapest).

```bash
cd $WC
caffeinate -i $PY code/rehearse_submission.py --controller pid_water
caffeinate -i $PY code/rehearse_submission.py --controller ff  --model outputs/discovery/main/best.py
caffeinate -i $PY code/rehearse_submission.py --controller mpc --model outputs/discovery/main/best.py
```
(the 30 shipped scenarios at 8,640 steps, compared with the kit's `baseline_returns.json`). Then the same on held-out
stage-2 buildings at the hard station, compared with the kit's own runs of them:
```bash
caffeinate -i $PY code/rehearse_submission.py --controller mpc --model outputs/discovery/main/best.py --scenarios-dir $D/scenarios --scenarios <building>/<window> ... --baseline-results $D/results --tag mpc_holdout
```

- [ ] Per scenario: score, comfort / energy / carbon terms, seconds per step, safety repairs; the worst scenario per
  controller; an ablation of the MPC's "off" mode.
- [ ] Confirm or change the MPC horizon (2 h) from the numbers.

## Handoff to 12

The MPC beats pid_water on average and in the worst scenario within the time budget, or the table explains why not
(then the model-based notebook is not submitted).
