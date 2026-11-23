#!/bin/bash
#SBATCH --job-name=exp_finalize_pr1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
# Close the main-control rows (PR1) without depending on an interactive session: wait until the controller marks
# the run complete, back-fill the SSIM of the last accepted configurations on a 1080 Ti (ssim.sbatch, blocking),
# then write RUN/report_final (export_pr1.py) and restage the paper tables (export_staging.py). Polls every 5 min.
# Submit: sbatch -o $RUN/logs/finalize_pr1_%j.log experiments/main_controls/finalize_pr1.sbatch
set -euo pipefail
REPO="${TAILOR_PROJECT:?}"
B="${TAILOR_RUNS:?}"
RUN=$B/main_controls_20261109
SSIM=$B/ssim_pr1_20261110
PY="${TAILOR_PYTHON:-python}"
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-local} commit=$(git -C $REPO rev-parse HEAD)"
until "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$RUN/progress.json')).get('complete') else 1)"; do
  sleep 300
done
echo "main controls complete $(date -Is)"
"$PY" -c "import json; d=json.load(open('$RUN/progress.json')); print(json.dumps(d.get('counts')))"
sbatch --wait --array=0-1%2 -o "$SSIM/logs/ssim_final_%A_%a.log" "$REPO/experiments/main_controls/ssim.sbatch" || true
echo "ssim back-fill finished $(date -Is)"
export TAILOR_MAIN_CONTROLS_RUN=$RUN OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
"$PY" -B -W ignore "$REPO/experiments/main_controls/export_pr1.py" "$SSIM" "$RUN/report_final"
"$PY" -B "$REPO/experiments/tools/export_staging.py"
echo "done $(date -Is)"
