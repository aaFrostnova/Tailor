#!/bin/bash
#SBATCH --job-name=unified_gpu_recovery
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/gpu_recovery_%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${TAILOR_PYTHON:-python}" -B -u \
  "${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/operations/gpu_recovery_watchdog.py "$@"
