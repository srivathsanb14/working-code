# 02 · Stage 2 data: new buildings from the kit's generator

**Phase:** Data · **Priority:** P1 · **Size:** L · **Depends on:** 01 (speed table)

## Goal

Add floor plans nobody has seen. The hidden test is 6 new buildings at 5 new stations, drawn from the same generator
and ranges as the shipped six. With only six buildings, holding out the most unusual quarter (two) leaves four to
discover formulas on, which is thin for learning what generalises across floor plans. New seeds fix that.

## Plan

| | Count |
|---|---|
| New buildings | 20 to start (seeds from 500); ask the generator for about 26, because it rejects about 1 seed in 5 (a room under 20 m²) |
| Windows per building | 6: one per station, window index rotated across buildings (building k gets `<station>_k`, k = 1..8 cycling) |
| Controllers | the same 5 |
| Runs | 20 × 6 × 5 = **600**, 30 days each |

Scale to more buildings when the first 20 are through tasks 03–08 and compute allows.

## Steps

1. Generate the building configs (geometry depends only on the seed; materials are stratified per batch, so keep one
   batch per call and record its `--count`):
   ```bash
   python -m smart_buildings.run_floor_generator --start-seed 500 --count 26 --output-dir $D/buildings --no-pdf
   ```
2. Scenarios, one call per building with its rotated windows; skip rejected seeds:
   ```bash
   k=1; for b in $D/buildings/*.yaml; do python -m smart_buildings.run_scenario_generator --buildings $b --weathers atlanta_$k denver_$k fairbanks_$k minneapolis_$k phoenix_$k singapore_$k --output-dir $D/scenarios || true; k=$((k % 8 + 1)); done
   ```
   Delete partial scenario folders of rejected seeds; keep the first 20 accepted buildings.
3. Runs: the same `evaluate.py` command as stage 1, into the same results folder.
4. Record per building: seed, rooms, floors, AHUs, footprint, accepted/rejected, seconds per step.

## Decisions to record

- **Footprint range (open question).** The generator draws 400–700 × 300–500 ft; the six shipped buildings are only
  318–430 × 188–301 ft, and the organisers say the test uses "the same generator and ranges as the shipped buildings".
  Decide whether to sample the generator's full range, the shipped range, or both, and say which in GUIDE.md A.2.
- **Large buildings.** Generated buildings reach 120+ rooms and 12 AHUs; the simulator took up to 0.109 s per step on a
  203-room building. Check runtime before scaling up.
- **Extra excitation controllers (optional).** Only if the first discovery runs show slow dynamics are
  poorly identified: deterministic policies that read only zone temperatures, the clock, comfort flags and outside air
  (otherwise their per-room actions cannot be recovered by replay).

## Done when

- At least 20 accepted new buildings, each with 6 windows × 5 controllers of complete 30-day runs.
- The footprint-range decision is written in GUIDE.md A.2.
