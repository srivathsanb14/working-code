# 02 · Build the episodes, compute the split, audit everything

**Needs:** 01 · **Produces:** `$D/dataset/` (1,440 `.npz` + `index.json` + `split.json`), `$D/audit/audit.md`,
`docs/DATASET_REPORT.md` · **Time:** about an hour (estimate)

## Steps

- [ ] **1. Episodes** (skips runs already built, so it resumes):
  ```bash
  cd $WC
  caffeinate -i $PY code/build_dataset.py --scenarios $D/scenarios --results $D/results --out $D/dataset --workers $J
  ```
  Per run: re-runs only the controller's decision function on the logged observations to recover the per-room commands
  (no simulation), checks them against the logged per-AHU airflow (typically within 1e-5), computes the
  per-room inputs. A run that does not match is `replay mismatch` in `index.json` and never used.
- [ ] **2. Split** (reads `index.json` and one light episode per scenario; refuses a missing dataset):
  ```bash
  $PY code/make_splits.py --dataset $D/dataset --out $D/dataset/split.json
  ```
  Expected for stage 1 (six buildings, 48 windows):

  | Tier | Expected (approximately) |
  |---|---|
  | test_hard | 2 held-out buildings × 8 windows at the hard station = 16 scenarios |
  | test_medium | 2 held-out buildings × 40 other windows = 80 |
  | test_station | 4 seen buildings × 8 hard-station windows = 32 |
  | test_easy | about a quarter of the remaining 160 = 40 |
  | validation | about 120 scenarios = about 600 episodes |

  Check the hard station (Fairbanks on the shipped set) and the held-out buildings (2 and 8 on the shipped set).
- [ ] **3. Audit:**
  ```bash
  $PY code/audit_data.py --scenarios $D/scenarios --results $D/results --dataset $D/dataset
  ```
  Exit 0 and no `[FAIL]` in `$D/audit/audit.md`: kit untouched; every run present with 8,640 rows, no NaN, 294 K reset
  row, Sunday 06:00 start, same controllers per scenario; no scenario in two tiers, held-out buildings only in test_medium
  / test_hard, hard-station scenarios only in test_hard / test_station, validation not empty; replay all `ok`; arrays
  finite; temperatures in 250–330 K.
- [ ] **4. Report:** `$PY code/dataset_report.py --dataset $D/dataset --out docs/DATASET_REPORT.md`; record the tier counts
  and the split's `_why` block in `docs/RUNBOOK.md`.

## Handoff to 03

`index.json` lists 1,440 runs, all `ok` (or each exclusion explained); `split.json` has every tier non-empty; the audit
exits 0.
