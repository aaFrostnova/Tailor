#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "${TAILOR_PYTHON:-python}" -B -u \
  "${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/baselines/code/baseline_controller.py "$@"
