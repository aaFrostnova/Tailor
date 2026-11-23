#!/bin/bash
set -euo pipefail
source "${TAILOR_PROJECT:?}/experiments/runtime.sh"
wm_storage_preflight
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${TAILOR_PYTHON:-python}" -B -u "$WM_RUN_ROOT/component_ablation_20260923/code/worker.py" "$1"
