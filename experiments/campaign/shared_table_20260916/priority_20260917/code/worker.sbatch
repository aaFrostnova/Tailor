#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
RUN="${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/shared_table_20260916/priority_20260917
exec "${TAILOR_PYTHON:-python}" -B -u "$RUN/code/worker.py" "$1"
