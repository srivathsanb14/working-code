# 17 · Bring the long docs in line with the current design

**Phase:** Docs · **Priority:** P2 · **Size:** M · **Depends on:** nothing

## Goal

A few documents still describe earlier designs. Anyone reading them should find the current one.

## Items

- `STATE_SPACE.md`: still describes fitted constants (offline/online deployment), a training split and the
  Kalman-filter scoring of the legacy loop. Bring every section in line with closed-form discovery (nothing fitted),
  the computed split and the kit-only data plan, or mark the old sections as history.
- `docs/RUNBOOK.md` §7–§8: still describe the legacy fitted test (`test_model.py`) and say "`run_pipeline.sh` step 8
  runs `run_policy.py --check-replay`"; the pipeline now has three steps.
- The legacy fitted mode (`code/discovery/loop.py`, `run_discovery.py --deployment`, `code/test_model.py`) is kept only
  for comparison and is cited by the deck's "what we tried first" slide: decide whether to keep or remove it.

Already done (25 Nov 2026): the old backlog (`docs/next_tasks.*`), `split_example.json` and `docs/LLM_DISCOVERY.md` were
removed; this folder is the plan.

## Done when

- No document contradicts `docs/DISCOVERY_WALKTHROUGH.md`, `docs/RUNBOOK.md` or this folder.
