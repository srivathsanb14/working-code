# 01 · The data: 288 scenarios, 1,440 runs, checked against the kit's baseline scores

**Needs:** a laptop with this repository and the kit side by side, `.venv` installed (docs/RUNBOOK.md §0–1), ≥ 100 GB
free on the data disk · **Produces:** `$D/scenarios/` (288), `$D/results/` (1,440 runs) · **Time:** about 10–20 hours of
compute on 8 parallel jobs (estimate; step 2 measures it)

Six shipped buildings × all 48 weather windows (6 stations × 8) × 5 controllers, full 30-day runs (8,640 steps). All
with the kit's own commands, run **from the kit's root folder** (each scenario's weather path is relative to it), with
every output under `$D` (left out, the kit writes into itself).

## Steps

- [ ] **1. Set the paths every command below uses** (put these lines in a file such as `~/sb_env.sh` and `source` it in
  every new terminal; set `D` to the data disk and `J` to the core count minus 2):
  ```bash
  export SB=~/smart-buildings KIT=~/smart-buildings/smart-buildings-starter-kit WC=~/smart-buildings/working-code
  export D=/Volumes/data/sb SB_DATA=/Volumes/data/sb J=8
  export PY=$WC/.venv/bin/python NUMBA_CACHE_DIR=$WC/.numba_cache PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
  export POLICIES="all_off bang_bang fixed_setpoint pid pid_water"
  ```
- [ ] **2. Measure speed and size on one scenario** (5 controllers, one day), then delete the probe:
  ```bash
  cd $KIT
  $PY -m smart_buildings.run_scenario_generator --buildings building_train_8 --weathers atlanta_1 --output-dir $D/probe/scenarios
  time $PY -m smart_buildings.evaluate --scenarios-dir $D/probe/scenarios --results-dir $D/probe/results --policies $POLICIES --steps 288
  du -sh $D/probe/results && rm -rf $D/probe
  ```
  For reference, on a 10-core laptop a 30-room building runs at about 0.015 s per step and one day of 4 runs takes 2.8 MB, so a 30-day run is about
  4 minutes and 21 MB; larger buildings are slower and bigger. Total hours ≈ 1,440 × 8,640 × s/step ÷ J.
- [ ] **3. Generate the 288 scenarios** (naming only `--buildings` takes all 48 windows; about 2 s each):
  ```bash
  cd $KIT
  $PY -m smart_buildings.run_scenario_generator --buildings building_train_1 building_train_2 building_train_4 building_train_5 building_train_7 building_train_8 --output-dir $D/scenarios
  ls -d $D/scenarios/*/* | wc -l        # 288
  ```
  **Never move `$D/scenarios` afterwards:** each `config.gin` stores the absolute path of its floor plan. If the data
  must move, regenerate the scenarios at the new place (seconds; results stay valid).
- [ ] **4. Run the controllers.** `evaluate.py` cannot resume and runs one scenario at a time, so this driver runs one
  process per scenario, `J` at a time, and skips scenarios already done. Re-run the same block to resume:
  ```bash
  bash <<'RUN'
  cd $KIT && mkdir -p $D/logs
  for d in $D/scenarios/*/*/; do b=$(basename $(dirname $d)); w=$(basename $d)
    [ "$(find $D/results/$b/$w -name summary.json 2>/dev/null | wc -l)" -ge 5 ] || echo "$b $w"; done > $D/todo.txt
  echo "$(wc -l < $D/todo.txt) scenarios to run"
  caffeinate -i xargs -P $J -n 2 sh -c '"$PY" -m smart_buildings.evaluate --scenarios-dir "$D/scenarios" --results-dir "$D/results" --buildings "$0" --weathers "$1" --policies $POLICIES > "$D/logs/$0__$1.log" 2>&1 && echo "done $0 $1"' < $D/todo.txt
  RUN
  ```
  `caffeinate` (macOS) keeps the laptop awake; drop it on Linux. Run it inside `tmux` so a closed terminal does not stop
  it. Progress: `find $D/results -name summary.json | wc -l` (target 1,440); a failed scenario leaves its log in
  `$D/logs/`.
- [ ] **5. Check the 30 shipped scenarios reproduce the kit's baselines** (the same simulator as the leaderboard):
  ```bash
  $PY - <<'CHK'
  import json, os
  D = os.environ["D"]; KIT = os.environ["KIT"]
  ref = json.load(open(f"{KIT}/baseline_returns.json")); bad = 0
  for scen, pols in ref.items():
      for pol, r in pols.items():
          s = json.load(open(f"{D}/results/{scen}/{pol}/summary.json"))["total_reward"]
          if abs(s - r) > 1e-6 * abs(r): bad += 1; print("MISMATCH", scen, pol, s, r)
  print("checked", sum(len(p) for p in ref.values()), "runs;", bad, "mismatches")
  CHK
  git -C $KIT status --porcelain      # must print nothing: the kit is untouched
  ```

## Handoff to 02

- 288 folders `$D/scenarios/<building>/<window>/` with `config.gin`, `floor_plan.npy`, `zone_map.npy`.
- 1,440 folders `$D/results/<building>/<window>/<controller>/` with `timeseries.csv`, `ahu_log.csv`,
  `reward_components.csv`, `summary.json`.
- Step 5 printed 0 mismatches and the kit's `git status` is empty.
