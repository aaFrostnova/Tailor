#!/bin/bash
set -euo pipefail
source "${TAILOR_PROJECT:?}/experiments/runtime.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
W="${TAILOR_OUTPUTS:?}"/request_solution_examples_20260925/timing/worker.py
"${TAILOR_PYTHON:-python}" -B -u "$W" prepare --index "$SLURM_ARRAY_TASK_ID"
exec "${TAILOR_PYTHON:-python}" -B -u "$W" measure --index "$SLURM_ARRAY_TASK_ID"
