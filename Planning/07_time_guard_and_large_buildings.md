# 07 · Per-step time guard and large buildings

**Needs:** 05 (`best.py`) · **Produces:** a guarded submission policy · **Time:** 1–2 days

The whole hidden evaluation (30 scenarios × 8,640 steps) must finish in one 9-hour Kaggle session: about 0.125 s per
step including the simulator. The hidden buildings come from the kit's generator and can be much larger than the
shipped ones: on a 203-room generated building the simulator alone took up to 0.109 s per step. The guard must
therefore hold whatever the building size.

- [ ] Measure on the largest shipped building (building 2, 85 rooms): tracker + forecast time per step (0.7 ms on a
  30-room building) and MPC plan time (35–50 ms every 3 steps on 30–39 rooms):
  ```bash
  $PY code/rehearse_submission.py --scenarios-dir $D/scenarios --scenarios building_train_2/atlanta_3 --controller mpc --model outputs/discovery/main/best.py --steps 576 --tag timing
  ```
  Scale the per-room costs to a 200-room building to set the budget shares.
- [ ] In `code/submission/transfer_policy.py`: track elapsed time per step and per episode; when the budget would be
  exceeded, skip planning (keep the last mode, or fall back to pid_water); lengthen the replanning interval
  automatically as the room count grows.
- [ ] Test: force a slow planner; the fallback engages and the episode completes.

## Handoff to 09 and 10

The measured timings and the budget shares are recorded; the forced-slow test passes.
