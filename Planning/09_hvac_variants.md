# 09 · Detect HVAC variants we do not model, and fall back

**Needs:** nothing (can start any time) · **Time:** 1–2 days

The website says hidden buildings vary in "HVAC system"; the kit's code has an air-source heat-pump variant no shipped
scenario uses. Our command-to-heat laws (`code/runtime/inputs.py`) and cost estimate (`code/control/costs.py`) assume a
boiler with hot-water reheat; a wrong heat estimate would drive wrong decisions.

- [ ] Find how the variant is configured in the kit, and what changes in the metadata and observation fields.
- [ ] Generate one heat-pump scenario with the kit's commands into `$D/probe/` (outside the kit).
- [ ] In the submission policy, detect the variant from the metadata / observation layout and fall back to pid_water (or
  add the heat-pump laws to `runtime/inputs.py` and `control/costs.py` if time allows).

## Handoff to 12

`rehearse_submission.py --scenarios-dir $D/probe/scenarios` runs the heat-pump scenario end to end with the fallback
engaged; documented in the RUNBOOK.
