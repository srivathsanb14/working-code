# 04 · Compute the split on the full data and review it

**Phase:** Dataset · **Priority:** P0 · **Size:** S · **Depends on:** 03

## Goal

Run the split rule on the stage-1 (later stage-2) episodes and check that every tier is meaningful before discovery.

## Steps

1. `python code/make_splits.py --dataset $D/dataset --out $D/dataset/split.json`. The rule: whole scenarios only; the
   hard station is the one whose hourly outside temperatures are farthest from the rest (Wasserstein-1); the most
   unusual quarter of buildings by four floor-plan numbers is held out (`--hard-frac 0.25`, rounded up); about one
   window in four of a seen building at a seen station is test_easy (`--easy-frac`).
2. Read `_why` in `split.json` and check, for stage 1 (six buildings, 48 windows), roughly:

   | Tier | Expected |
   |---|---|
   | test_hard | 2 held-out buildings × 8 hard-station windows = 16 scenarios |
   | test_medium | 2 held-out buildings × the other stations' windows |
   | test_station (extra) | 4 seen buildings at the hard station |
   | test_easy | about a quarter of the seen-building, seen-station windows |
   | validation | the rest: the only data the search sees |

   Check that the hard station is still Fairbanks with Minneapolis included, and that `check_whole_episodes` passed.
3. **Validation sample (code change).** Discovery scores each candidate on `--max-val 30` validation episodes, drawn
   at random with seed 0 (`evolve.load_split`). With ~480 validation episodes that sample can miss whole buildings,
   stations or controllers. Make the sample stratified (every building × station × controller group represented, then
   fill at random) and add a test that checks coverage.
4. After stage 2, re-run the split: with 26 buildings the held-out quarter becomes 7 buildings; check the tiers again.

## Done when

- `split.json` for the stage-1 dataset is reviewed and its tier counts recorded in `docs/RUNBOOK.md`.
- The validation sample is stratified, tested, and its coverage is printed at the start of every discovery run.
