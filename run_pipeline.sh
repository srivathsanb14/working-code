#!/usr/bin/env bash
# Discovery on data that already exists.  Run it on the machine where the data was generated.
#
# The data is made beforehand with the starter kit (its scenario generator and evaluate.py, any number of buildings,
# weather windows and controllers).  This script does not generate or simulate anything; it reads.
#
#   step 1  episodes   turn the kit's result folders into episodes (one .npz per run) -> $DATASET
#                      (skipped for runs already converted; skip the step entirely if $DATASET is already built)
#   step 2  split      whole episodes -> validation + test_easy / test_medium / test_hard (+ test_station)
#   step 3  discovery  every candidate written by the Claude API, scored on validation only; at the end the chosen
#                      model and the Pareto front are scored once on the test tiers
#
# Usage, from anywhere:
#   ANTHROPIC_API_KEY=... RESULTS=/data/results SCENARIOS=/data/scenarios GENS=20 ./run_pipeline.sh
#   FROM=2 DATASET=/data/dataset GENS=20 ./run_pipeline.sh       # episodes already built
#   FROM=2 TO=2 DATASET=/data/dataset ./run_pipeline.sh          # only (re)compute the split
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
DATA_ROOT=${SB_DATA:-data}
RESULTS=${RESULTS:-$DATA_ROOT/results}          # the kit's evaluate.py output: <building>/<weather>/<controller>/
SCENARIOS=${SCENARIOS:-$DATA_ROOT/scenarios}    # the kit's scenario folders: <building>/<weather>/config.yaml ...
DATASET=${DATASET:-$DATA_ROOT/dataset}          # episodes: index.json + <building>__<weather>__<controller>.npz
WORKERS=${WORKERS:-16}
RUN=${RUN:-main}
ISLANDS=${ISLANDS:-3}
GENS=${GENS:-20}
MAX_USD=${MAX_USD:-50}
MODEL=${MODEL:-claude-opus-5-5}
FROM=${FROM:-1}
TO=${TO:-3}

step() { local n=$1; shift; if (( n >= FROM && n <= TO )); then echo; echo "=== step $n: $*"; "$@"; fi; }

step 1 "$PY" code/build_dataset.py --results "$RESULTS" --scenarios "$SCENARIOS" --out "$DATASET" --workers "$WORKERS"
step 2 "$PY" code/make_splits.py --dataset "$DATASET" --out "$DATASET/split.json"
step 3 "$PY" code/run_discovery.py --dataset "$DATASET" --run "$RUN" --generations "$GENS" --islands "$ISLANDS" \
        --model "$MODEL" --max-usd "$MAX_USD" --workers "$WORKERS" --test
echo; echo "done: outputs/discovery/$RUN/REPORT.md (split: $DATASET/split.json)"
