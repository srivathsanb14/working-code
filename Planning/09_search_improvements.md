# 09 · Search improvements (only where the first run shows a need)

**Phase:** Discovery · **Priority:** P1–P3 · **Size:** M each · **Depends on:** 07

Do not build these speculatively: pick the ones the first run's notes (task 07) call for.

| Item | When it is needed | What to build | Done when |
|---|---|---|---|
| Screen calibration | good structures are screened out at stage 1 | histogram of one-step/persistence ratios for the hand-written models and early candidates; choose `--screen` (default 5.0) from it | value chosen and recorded |
| Term ablation in the analysis | mutations do not move the term the diagnosis names | zero one term at a time and report the change in each error, next to the slope table in `closedform.analysis` | ablation table in `analysis.md`, tested |
| Diversity control | islands converge to near-copies | AST-similarity filter and error-profile diversity when choosing parents in `evolve.tournament` | duplicate rate per generation in REPORT.md |
| Adaptive operator mix | one operator rarely improves parents | track improvement rate per operator; shift requests toward the better ones, behind a flag | operator table shows the shift |
| History-summary options | formulas use only crude history statistics | name the allowed patterns in the contract (lags, differences, window and exponential averages) | contract text updated; tests pass |
| Physics detail in the prompt | answers ignore or misuse the store structure | compare "graph only" against the current physics notes for a few generations | comparison recorded; decision made |
