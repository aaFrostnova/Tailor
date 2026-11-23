#!/bin/bash
#SBATCH --job-name=exp_mc_controller
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
# Controller for the main controls (A1 SingleOptimized, A3 AlwaysAllThree): sweeps proposals and cells,
# submits CPU and GPU worker arrays. Submit with:
#   sbatch --export=ALL,TAILOR_MAIN_CONTROLS_RUN=$RUN -o $RUN/logs/controller_%j.log $RUN/code/controller.sbatch
set -euo pipefail
source "${TAILOR_PROJECT:?}/experiments/runtime.sh"
wm_storage_preflight
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${TAILOR_MAIN_CONTROLS_RUN:?export TAILOR_MAIN_CONTROLS_RUN}"
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-local}"
PY="${TAILOR_PYTHON:-python}"
echo "python=$PY $($PY --version 2>&1)"
exec "$PY" -B -u "$TAILOR_MAIN_CONTROLS_RUN/code/mc_controller.py"
