#!/bin/bash
#SBATCH --job-name=exp_ssim_pr1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# SSIM of newly accepted main-control configurations (measure_ssim.py measure), the same measurement the OnDemand
# 1080 Ti loop runs. Workers are interchangeable: each configuration is claimed with a per-config lock and an
# existing cell is skipped, so any number of workers (and the OnDemand loop) run side by side without overlap.
# Submit: sbatch --array=0-3%4 -o $OUT/logs/ssim_%A_%a.log experiments/main_controls/ssim.sbatch
set -euo pipefail
REPO="${TAILOR_PROJECT:?}"
B="${TAILOR_RUNS:?}"
PY="${TAILOR_PYTHON:-python}"
source "${TAILOR_PROJECT:?}/experiments/runtime.sh"
wm_storage_preflight >/dev/null 2>&1 || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TAILOR_MAIN_CONTROLS_RUN=$B/main_controls_20261109
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} commit=$(git -C $REPO rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "python=$PY $($PY --version 2>&1)"
"$PY" -B -u "$REPO/experiments/main_controls/measure_ssim.py" measure "$B/ssim_pr1_20261110" --variants AlwaysAllThree,SingleOptimized
echo "done $(date -Is)"
