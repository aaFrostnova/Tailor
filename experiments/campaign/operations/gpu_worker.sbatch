#!/bin/bash
#SBATCH --job-name=unified_gpu_participant
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gpu_participant_%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"/hub
export UNIFIED_ATTACK_REUSE_INDEX="${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/plans/pixel_reuse_index.json
export UNIFIED_EMBED_REUSE_CERTIFICATE="${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/plans/embedding_migration.json
hostname
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
exec "${TAILOR_PYTHON:-python}" -B -u "${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/operations/local_worker.py loop --execute
