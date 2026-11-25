# 06 · Search improvements, only where 04 shows a need

**Needs:** 04 notes · **Time:** M each

| Item | When it is needed | What to build | Done when |
|---|---|---|---|
| Screen calibration | good structures screened out early; the hand-written network model is screened on HVAC-off episodes, where persistence is nearly perfect | histogram of one-step / persistence ratios by controller; screen per controller group or on active-HVAC episodes only; choose `--screen` (default 5.0) | value chosen and recorded |
| Term ablation in the analysis | mutations do not move the term the diagnosis names | zero one term at a time, report the change in each error next to the slope table (`closedform.analysis`) | ablation table in `analysis.md`, tested |
| Diversity control | islands converge to near-copies | AST-similarity filter; error-profile diversity when choosing parents (`evolve.tournament`) | duplicate rate per generation in REPORT.md |
| Adaptive operator mix | one operator rarely improves parents | improvement rate per operator; shift requests toward the better ones, behind a flag | operator table shows the shift |
| History-summary options | formulas use only crude history statistics | name the allowed patterns in the contract (lags, differences, window and exponential averages) | contract updated; tests pass |
| Physics detail in the prompt | answers ignore or misuse the store structure | compare "graph only" with the current physics notes for a few generations | comparison recorded |
