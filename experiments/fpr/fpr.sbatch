#!/bin/bash
# PR4 negatives: array task $SLURM_ARRAY_TASK_ID of $NSHARD decodes its shard of RUNF/manifest.json tasks.
# Submit (L40S example):
#   sbatch -p gpu --gres=gpu:l40s:1 -c 6 --mem=64G -t 08:00:00 --array=0-7%8 -J exp_fpr \
#     -o $RUNF/logs/fpr_%A_%a.log --export=ALL,TAILOR_MAIN_CONTROLS_RUN=$RUNF,NSHARD=8 experiments/fpr/fpr.sbatch
# Local helper on an interactive GPU: NSHARD=1 FPR_REVERSE=1 bash experiments/fpr/fpr.sbatch (walks from the end).
set -euo pipefail
source "${TAILOR_PROJECT:?}/experiments/runtime.sh"
wm_storage_preflight
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${TAILOR_MAIN_CONTROLS_RUN:?}" "${NSHARD:?}"
REPO="${TAILOR_PROJECT:?}"
PY="${TAILOR_PYTHON:-python}"
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-local} shard=${SLURM_ARRAY_TASK_ID:-0}/$NSHARD commit=$(git -C $REPO rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$PY" -B -u "$REPO/experiments/fpr/decode_negatives.py" "${SLURM_ARRAY_TASK_ID:-0}" "$NSHARD" ${FPR_REVERSE:+--reverse}
