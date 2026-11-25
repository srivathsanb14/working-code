# 12 · Per-step time guard and large buildings

**Phase:** Control · **Priority:** P0 · **Size:** M · **Depends on:** 02 (large buildings to test on)

## Goal

The whole hidden evaluation (30 scenarios × 8,640 steps) must finish in one 9-hour Kaggle session: about 0.125 s per
step including the simulator. The simulator alone took up to 0.109 s per step on a 203-room building, which leaves
almost nothing for the policy there.

## Steps

1. Measure tracker + forecast time (0.7 ms per step on a 30-room building today) and MPC plan time (about 35–50 ms every
   3 steps on 30–39 rooms) on the largest generated buildings.
2. Implement the guard in the submission policy (`code/submission/transfer_policy.py`): track elapsed time per step and
   per episode; when the budget would be exceeded, skip planning and keep the last mode or fall back to pid_water;
   lengthen the replanning interval automatically on large buildings.
3. Test: force a slow planner and check that the fallback engages and the episode still completes.

## Done when

- On the largest generated building the policy never exceeds its share of the budget; the forced-slow test passes.
