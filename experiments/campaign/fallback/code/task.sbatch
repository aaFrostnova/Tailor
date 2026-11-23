#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY="${TAILOR_PYTHON:-python}"
SCRIPT="${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/fallback/code/campaign.py
if [ "$1" = enum_batch ]; then
    exec "$PY" -u "$SCRIPT" enum_batch "$SLURM_ARRAY_TASK_ID"
else
    exec "$PY" -u "$SCRIPT" "$1" "$2" "$SLURM_ARRAY_TASK_ID"
fi
