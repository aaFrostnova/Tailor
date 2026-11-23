#!/bin/bash
set -euo pipefail
source "${TAILOR_PROJECT:?}/experiments/runtime.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
"${TAILOR_PYTHON:-python}" -B -u "${TAILOR_OUTPUTS:?}"/qualitative_timing_20260925/worker.py prepare --index "$SLURM_ARRAY_TASK_ID"
exec "${TAILOR_PYTHON:-python}" -B -u "${TAILOR_OUTPUTS:?}"/qualitative_timing_20260925/worker.py measure --index "$SLURM_ARRAY_TASK_ID"
