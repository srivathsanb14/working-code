#!/usr/bin/env bash
# The whole pipeline, in order, for a big machine.
#
# Layout (the only assumption about paths):
#   <root>/smart-buildings-starter-kit/   the kit, untouched (override with STARTER_KIT=...)
#   <root>/working-code/                  this repository; everything is written under it:
#       data/buildings, data/scenarios, data/results, data/plan.json, data/split.json   (step 2, the kit's own tools)
#       data/audit/                       (steps 3 and 5)
#       data/dataset/                     (step 4: one .npz per building x weather x policy, plus split.json)
#       outputs/discovery/<run>/      (step 6: REPORT.md, log.md, generations/, candidates/, plots/)
#       outputs/test/, outputs/policy/    (steps 7 and 8)
#   Generated data can live elsewhere with SB_DATA=/big/disk/data (every script reads it).
#
# Usage, from anywhere:
#   ./run_pipeline.sh                      # everything, LLM step as a dry run (writes every prompt, calls nothing)
#   PROPOSER=mock GENS=3 ./run_pipeline.sh               # no LLM: mechanical mutation/crossover, to inspect the logs
#   PROPOSER=claude GENS=20 ./run_pipeline.sh            # real LLM discovery (needs requirements-llm.txt + key)
#   DEPLOY=offline ./run_pipeline.sh                     # the earlier FITTED modes instead (offline|online|both)
#   FROM=6 PROPOSER=claude GENS=20 ./run_pipeline.sh     # data already generated: start at discovery
#   WORKERS=32 STEPS=8640 ./run_pipeline.sh              # more cores, full 30-day runs
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
WORKERS=${WORKERS:-16}
STEPS=${STEPS:-2016}                 # 864 = 3 days (minimum), 2016 = 7 days, 8640 = 30 days like the hidden test
TRAIN=${TRAIN:-500:600}              # building seed ranges (start:stop): disjoint batches = unseen buildings
VAL=${VAL:-700:720}
TEST=${TEST:-800:820}
VAL_STATION=${VAL_STATION:-phoenix}
TEST_STATION=${TEST_STATION:-minneapolis}
POLICIES=${POLICIES:-"all_off bang_bang fixed_setpoint pid pid_water"}
RUN=${RUN:-main}
DEPLOY=${DEPLOY:-closedform}         # closedform (default: no fitting anywhere) | offline | online | both (fitted modes)
PROPOSER=${PROPOSER:-dry}            # dry (no LLM call) | dir (inbox) | mock (no LLM) | claude
ISLANDS=${ISLANDS:-3}
GENS=${GENS:-1}
FROM=${FROM:-1}                      # first step to run
TO=${TO:-8}                          # last step to run
REHEARSE_STEPS=${REHEARSE_STEPS:-8640}

step() { local n=$1; shift; if (( n >= FROM && n <= TO )); then echo; echo "=== step $n: $*"; "$@"; fi; }

step 1 "$PY" code/setup_check.py
# shellcheck disable=SC2086
step 2 "$PY" code/generate_data.py --train-seeds "$TRAIN" --val-seeds "$VAL" --test-seeds "$TEST" \
        --val-station "$VAL_STATION" --test-station "$TEST_STATION" --policies $POLICIES --steps "$STEPS" --workers "$WORKERS"
step 3 "$PY" code/audit_data.py
step 4 "$PY" code/build_dataset.py --workers "$WORKERS"
step 5 "$PY" code/audit_data.py
if [[ "$DEPLOY" == closedform ]]; then
  # step 6 also scores the final best on the test split (--test); step 7 = the fitted ceiling, a diagnostic
  step 6 "$PY" code/run_discovery.py --run "$RUN" --proposer "$PROPOSER" --generations "$GENS" --islands "$ISLANDS" --test
  step 7 "$PY" code/run_discovery.py --run "$RUN" --ceiling --generations 0
  echo; echo "done: data in ${SB_DATA:-data}/, results in outputs/discovery/$RUN/REPORT.md"
  echo "(step 8, the submission rehearsal, runs the fitted-mode model; the closed-form runtime tracker is not built yet)"
  exit 0
fi
step 6 "$PY" code/run_discovery.py --run "$RUN" --deployment "$DEPLOY" --proposer "$PROPOSER" --generations "$GENS"

# the discovery run(s) step 6 wrote
if [[ "$DEPLOY" == both ]]; then OFF="outputs/discovery/${RUN}_offline"; ON="outputs/discovery/${RUN}_online"
elif [[ "$DEPLOY" == offline ]]; then OFF="outputs/discovery/${RUN}"; ON=""
else OFF=""; ON="outputs/discovery/${RUN}"; fi

if [[ -n "$OFF" ]]; then
  step 7 "$PY" code/test_model.py "$OFF/best.py" --params "$OFF/summary.json" --on test --theta all
  step 8 "$PY" code/run_policy.py --model "$OFF/best.py" --params "$OFF/summary.json" --on test --steps "$REHEARSE_STEPS"
fi
if [[ -n "$ON" ]]; then
  step 7 "$PY" code/test_model.py "$ON/best.py" --params "$ON/summary.json" --on test --theta day1
fi
echo; echo "done: data in ${SB_DATA:-data}/, results in outputs/"
