# 08 · Per-step time guard and large buildings

**Needs:** 05 (large generated buildings to test on) · **Produces:** a guarded submission policy · **Time:** 1–2 days

The whole hidden evaluation (30 scenarios × 8,640 steps) must finish in one 9-hour Kaggle session: about 0.125 s per
step including the simulator. The simulator alone took up to 0.109 s per step on a 203-room building.

- [ ] Measure on the largest stage-2 buildings: tracker + forecast time per step (0.7 ms on a 30-room building today)
  and MPC plan time (35–50 ms every 3 steps on 30–39 rooms), with `rehearse_submission.py --scenarios-dir $D/scenarios
  --scenarios <building>/<window> --controller mpc --model outputs/discovery/main/best.py --steps 576`.
- [ ] In `code/submission/transfer_policy.py`: track elapsed time per step and per episode; when the budget would be
  exceeded, skip planning (keep the last mode, or fall back to pid_water); lengthen the replanning interval on large
  buildings automatically.
- [ ] Test: force a slow planner; the fallback engages and the episode completes.

## Handoff to 10 and 12

On the largest generated building the policy stays within its share of the budget; the forced-slow test passes.
