#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export BASELINE_RUN_ROOT="$1"
BASELINE_ROLE="$2"
BASELINE_PLAN="$3"
BASELINE_INDEX="${SLURM_ARRAY_TASK_ID:-${4:-0}}"
exec "${TAILOR_PYTHON:-python}" -B -u \
  "${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/baselines/code/final_measurement.py \
  task "$BASELINE_PLAN" "$BASELINE_INDEX" --cache-root "$BASELINE_RUN_ROOT/$BASELINE_ROLE/cache"
