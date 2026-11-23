#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
U="${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914
exec "${TAILOR_PYTHON:-python}" -B -u "$U/code/rigor_controller.py" main
