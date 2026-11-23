#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --job-name=margin_fallback
#SBATCH --output=logs/controller_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "${TAILOR_PYTHON:-python}" -u "${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914/fallback/code/controller.py
