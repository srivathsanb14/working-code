# 13 · Detect HVAC variants we do not model, and fall back

**Phase:** Control · **Priority:** P0 · **Size:** M · **Depends on:** nothing

## Goal

The website says hidden buildings vary in "HVAC system", and the kit's code contains an air-source heat-pump variant
that no shipped scenario uses. Our command-to-heat laws (`code/runtime/inputs.py`) and cost estimate
(`code/control/costs.py`) assume a boiler and hot-water reheat. A wrong heat estimate would drive wrong decisions.

## Steps

1. Find how the variant is configured in the kit and what changes in the metadata and observation fields.
2. Generate a heat-pump scenario with the kit's own commands (outside the kit folder).
3. In the submission policy, detect the variant from the metadata / observation layout and fall back to pid_water; or,
   if time allows, add the heat-pump laws to `runtime/inputs.py` and `control/costs.py`.

## Done when

- The heat-pump scenario runs end to end through `rehearse_submission.py` with the fallback engaged (or correct laws),
  and the behaviour is documented in `docs/RUNBOOK.md`.
