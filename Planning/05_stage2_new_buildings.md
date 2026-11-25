# 05 · Stage 2 data: new buildings (start right after 02, in the background)

**Needs:** 01 (speed known), 02 done, a footprint decision · **Produces:** about 600 more runs in the same `$D` folders,
then the combined dataset re-built, re-split and re-audited · **Time:** generated buildings are larger and slower

The hidden test is 6 new buildings at 5 new stations. With only six buildings the split holds out two and leaves four to
discover formulas on: thin for learning what generalises across floor plans. Start this as soon as 02 has finished, so
it runs while 03 and 04 happen (they use little CPU except 03 step 1 and 04).

## Decide first

- [ ] **Footprint range.** The generator draws 400–700 × 300–500 ft (two test seeds gave 544 × 460 and 636 × 434 ft);
  the six shipped buildings are 318–430 × 188–301 ft, and the organisers say the test uses "the same generator and ranges
  as the shipped buildings". Choose the generator's default ranges, a range matching the shipped six, or both; write the
  choice in GUIDE.md A.2.

## Steps

- [ ] **1. Buildings** (ask for about 26 to keep 20: the kit rejects about 1 seed in 5; one batch, so materials are
  stratified across it):
  ```bash
  cd $KIT
  $PY -m smart_buildings.run_floor_generator --start-seed 500 --count 26 --output-dir $D/buildings --no-pdf
  ```
- [ ] **2. Scenarios**, one window per station, the window index rotated across buildings:
  ```bash
  bash <<'GEN'
  cd $KIT; k=1
  for b in $D/buildings/*.yaml; do
    $PY -m smart_buildings.run_scenario_generator --buildings $b --weathers atlanta_$k denver_$k fairbanks_$k minneapolis_$k phoenix_$k singapore_$k --output-dir $D/scenarios || true
    k=$((k % 8 + 1)); done
  GEN
  ```
  Rejected seeds fail and are skipped; delete their partial folders. Keep the first 20 accepted buildings.
- [ ] **3. Runs:** the driver block of 01 step 4 again; it only runs what is missing (target: 1,440 + 600 summaries).
- [ ] **4. Repeat 02** on the combined data. With 26 buildings the held-out quarter becomes 7 buildings; review the
  tiers. Add `--buildings $D/buildings` to the audit so it checks the generated parameters against the generator's
  ranges.

## Handoff to 06

The combined dataset is built, split and audited (exit 0); counts recorded.
