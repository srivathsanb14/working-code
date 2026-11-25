# 03 · Build the episodes and audit them

**Phase:** Dataset · **Priority:** P0 · **Size:** M · **Depends on:** 01 (then again after 02)

## Goal

Turn the kit's finished runs into episode files (one `.npz` per run) and prove they are correct before any discovery
run uses them.

## Steps

1. **Audit the raw runs:** `python code/audit_data.py` (writes `data/audit/audit.md`, exits 1 on a hard failure).
   Checks: the kit is unmodified; every run has its four files; row counts reach 8,640; no NaN; CSV row 0 is the reset
   state (294 K); every episode starts on a Sunday 06:00; building parameters inside the generator's ranges.
2. **Build the episodes:**
   ```bash
   python code/build_dataset.py --scenarios $D/scenarios --results $D/results --out $D/dataset --workers 16
   ```
   For each run it re-runs only the shipped controller's decision function on the logged observations to recover the
   per-room damper and reheat commands, checks them against the logged per-AHU airflow and run commands (a run that does
   not match is marked `replay mismatch` and not used), then computes the per-room inputs `g`, `T_supply`, `Q_reheat`,
   `Q_gain` and the building-wide `T_out`, occupants, comfort flag, clock and `T_water`.
3. **Re-audit:** `python code/audit_data.py` again (adds replay status and dataset sanity), and
   `python code/dataset_report.py` for the summary by station, season, controller and building size
   (`docs/DATASET_REPORT.md`).
4. **Speed (only if needed):** if replay of a large generated building is slow, profile
   `starterkit_data.replay` and `episode_inputs` and vectorise; the output must be hash-identical.

## Done when

- `data/dataset/index.json` lists every run with status; accepted runs have replay airflow error ≤ 2%.
- Both audits pass, or every hard failure is explained and the run excluded.
- `docs/DATASET_REPORT.md` is written for the stage-1 data (and again after stage 2).
