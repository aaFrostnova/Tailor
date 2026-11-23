#!/bin/bash
#SBATCH --job-name=exp_freshid_report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
# Fresh-identity transfer report: after every worker of the fresh-identity run has ended (submit with
# --dependency=afterany:<all worker jobs>), judge the accepted requests on the fresh-identity cells
# (report_heldout.py, unchanged) and pair them with the calibration-identity held-out run (compare_identity.py).
# Submit: sbatch --dependency=afterany:J1:J2:J3 -o $RUNF/logs/finalize_%j.log experiments/heldout/finalize_freshid.sbatch
set -euo pipefail
REPO="${TAILOR_PROJECT:?}"
B="${TAILOR_RUNS:?}"
RUNF=$B/heldout_freshid_20261113
RUNH=$B/heldout_20261109
PY="${TAILOR_PYTHON:-python}"
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-local} commit=$(git -C $REPO rev-parse HEAD)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
TAILOR_MAIN_CONTROLS_RUN=$RUNF "$PY" -B -W ignore "$REPO/experiments/heldout/report_heldout.py" "$RUNF/report_final"
"$PY" -B "$REPO/experiments/heldout/compare_identity.py" "$RUNH/report_final" "$RUNF/report_final" "$RUNF/report_final"
echo "done $(date -Is)"
