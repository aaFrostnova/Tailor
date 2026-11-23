#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
RUN="${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914
PY="${TAILOR_PYTHON:-python}"

finished() {
  "$PY" -B -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); a=p/"pipeline_audit.json"; s=p/"progress.json"; sys.exit(0 if a.exists() and s.exists() and json.loads(a.read_text()).get("passed") is True and json.loads(s.read_text()).get("phase")=="complete" else 1)' "$RUN"
}

if finished; then
  exit 0
fi

# Keep the frozen controller and its checkpoint validation unchanged. Stop with
# time to arrange a successor; recorded Slurm measurement arrays survive this.
set +e
timeout --signal=TERM --kill-after=90s 22h "$PY" -B -u "$RUN/code/rigor_controller.py" main
rc=$?
set -e
if ! finished; then
  sbatch --parsable -p cpu -c 2 --mem=24G -t 24:00:00 \
    --dependency="afterany:${SLURM_JOB_ID}" -J unified_main_resume \
    -o "$RUN/logs/main_resume_%j.log" --export=ALL \
    "$RUN/operations/resume_main_renew.sbatch" >> "$RUN/operations/main_controller_renewals.log"
fi
exit "$rc"
