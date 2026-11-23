#!/bin/bash
# Worker for the main controls; the controller submits it with partition, resources and array size.
# $1: cpu (proposals), gpu (live measurements) or bridge (bridging re-measurements).
set -euo pipefail
source "${TAILOR_PROJECT:?}/experiments/runtime.sh"
wm_storage_preflight
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${TAILOR_MAIN_CONTROLS_RUN:?set by the controller}"
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} kind=$1"
PY="${TAILOR_PYTHON:-python}"
echo "python=$PY $($PY --version 2>&1)"
exec "$PY" -B -u "$TAILOR_MAIN_CONTROLS_RUN/code/worker.py" "$1"
