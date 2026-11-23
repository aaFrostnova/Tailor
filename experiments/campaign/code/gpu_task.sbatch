#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
PY="${TAILOR_PYTHON:-python}"
R="${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914
exec "$PY" -u "$R/code/$1" task "$2" "${SLURM_ARRAY_TASK_ID:-${3:-0}}"
