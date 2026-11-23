#!/bin/bash
set -euo pipefail
source "${TAILOR_PROJECT:?}/experiments/runtime.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
exec "${TAILOR_PYTHON:-python}" -B -u \
  "${TAILOR_OUTPUTS:?}"/request_solution_examples_20260925/ring_probe/structure.py
