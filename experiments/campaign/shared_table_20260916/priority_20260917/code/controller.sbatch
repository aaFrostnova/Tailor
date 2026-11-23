#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
RUN="${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/shared_table_20260916/priority_20260917
set +e
timeout 22h "${TAILOR_PYTHON:-python}" -B -u "$RUN/code/controller.py"
rc=$?
set -e
if [ ! -f "$RUN/pipeline_audit.json" ]; then
  # Renew before the scheduler's 24-hour limit; all work is checkpointed.
  sbatch --parsable -p cpu -c 2 --mem=24G -t 24:00:00 -J shared_controller \
    -o "$RUN/logs/controller_%j.log" --export=ALL "$RUN/code/controller.sbatch" >> "$RUN/controller_renewals.log"
fi
exit "$rc"
