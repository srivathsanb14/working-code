# 16 · Robustness checks on assumptions

**Phase:** Validation · **Priority:** P2 · **Size:** M · **Depends on:** 08, 11

| Check | Why | What to do | Done when |
|---|---|---|---|
| Early episode | the history buffer (up to 24 h) is short in the first hours of each episode | plot forecast error against hours since episode start for best.py | plot in REPORT.md; decision whether the early hours need a different rule |
| Assumed constants | 10 W/m² gains, 0.2 standby, 70 W per person and the equipment laws are assumed equal in the hidden set | in labelled copies of scenarios (never in the data), perturb each and measure forecast and controller degradation | degradation table; any fragile assumption gets a task |
| Occupants per room | only the building total is observed | compare splitting it by floor area (default) vs equally on model error | decision recorded |
| Holidays and season | the clock has no date, so holidays are ignored and gas price uses the annual mean | detect holidays from zero occupancy on a workday; estimate season from outside temperature | pre-heating not triggered on detected holidays; cost error reported |
