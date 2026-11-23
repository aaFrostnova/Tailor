#!/bin/bash
#SBATCH --job-name=exp_mc_rejudge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
# CPU re-judge of the frozen TAILOR-F records from the frozen cells (entry condition for the new arms).
# Submit from anywhere:
#   sbatch --export=ALL,TAILOR_MAIN_CONTROLS_RUN=$RUN -o $RUN/logs/rejudge_%j.log experiments/main_controls/rejudge.sbatch
set -euo pipefail
source "${TAILOR_PROJECT:?}/experiments/runtime.sh"
: "${TAILOR_MAIN_CONTROLS_RUN:?export TAILOR_MAIN_CONTROLS_RUN}"
REPO="${TAILOR_PROJECT:?}"
PY="${TAILOR_PYTHON:-python}"
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-local} commit=$(git -C "$REPO" rev-parse HEAD)"
echo "python=$PY $($PY --version 2>&1)"
exec "$PY" -B -u "$REPO/experiments/main_controls/rejudge.py" "$TAILOR_MAIN_CONTROLS_RUN/rejudge/smt06_rejudge.json"
