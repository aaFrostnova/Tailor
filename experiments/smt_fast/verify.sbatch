#!/bin/bash
#SBATCH --job-name=exp_smtfast_verify
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --requeue
# Verify the accelerated top-k against the original, one shard of the case list per array task.
# Submit: sbatch --array=0-39%40 --export=ALL,TAILOR_MAIN_CONTROLS_RUN=$RUN,CASES=...,OUT=...,NSHARD=40 \
#         -o $OUT/logs/verify_%A_%a.log experiments/smt_fast/verify.sbatch
set -euo pipefail
: "${TAILOR_MAIN_CONTROLS_RUN:?}" "${CASES:?}" "${OUT:?}" "${NSHARD:?}"
REPO="${TAILOR_PROJECT:?}"
PY="${TAILOR_PYTHON:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-local} shard=${SLURM_ARRAY_TASK_ID:-0}/$NSHARD commit=$(git -C $REPO rev-parse HEAD)"
echo "python=$PY $($PY --version 2>&1)"
exec "$PY" -B -W ignore -u "$REPO/experiments/smt_fast/verify_fast.py" "$CASES" "$OUT" "${SLURM_ARRAY_TASK_ID:-0}" "$NSHARD"
