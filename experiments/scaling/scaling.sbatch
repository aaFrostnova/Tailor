#!/bin/bash
#SBATCH --job-name=exp_scaling
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --requeue
# One task of the solver-scaling experiment per array index: TASKS is a JSON list of {"n", "case", "method"};
# the result goes to $OUT/<method>/n<n>_<case id>.json (an existing result is kept, so requeues resume).
# Submit (memory and walltime per task group on the command line):
#   sbatch --array=0-19%20 --mem=8G -t 12:00:00 --export=ALL,TAILOR_MAIN_CONTROLS_RUN=$RUN,TASKS=...,OUT=... \
#          -o $OUT/logs/%x_%A_%a.log experiments/scaling/scaling.sbatch
set -euo pipefail
: "${TAILOR_MAIN_CONTROLS_RUN:?}" "${TASKS:?}" "${OUT:?}"
REPO="${TAILOR_PROJECT:?}"
PY="${TAILOR_PYTHON:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} commit=$(git -C $REPO rev-parse HEAD)"
echo "python=$PY $($PY --version 2>&1)"
read -r N CASE METHOD DEST < <("$PY" - "$TASKS" "${SLURM_ARRAY_TASK_ID:-0}" "$OUT" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))[int(sys.argv[2])]
print(t['n'], json.dumps(t['case'], separators=(',', ':')), t['method'],
      f"{sys.argv[3]}/{t['method']}/n{t['n']}_{t['case']['id']}.json")
EOF
)
if [ -s "$DEST" ]; then echo "exists: $DEST"; exit 0; fi
echo "task n=$N method=$METHOD case=$CASE -> $DEST"
"$PY" -B -W ignore "$REPO/experiments/scaling/run_scaling.py" "$N" "$CASE" "$METHOD" "$DEST"
echo "done $(date -Is)"
