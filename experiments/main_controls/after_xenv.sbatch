#!/bin/bash
#SBATCH --job-name=exp_after_xenv
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
# After the L40S cross-environment check (crossgpu_xenv.py) completes: decide by the registered rule
# (protocol.md section 13), add the passing attacks to the L40S approval of the result runs, restore the L40S
# worker caps lowered for the check, and give the held-out run L40S workers. Polls every 10 minutes; stops
# without acting if decide fails for any reason other than an incomplete check.
# Submit: sbatch -o $X/logs/after_xenv_%j.log experiments/main_controls/after_xenv.sbatch
set -euo pipefail
REPO="${TAILOR_PROJECT:?}"
B="${TAILOR_RUNS:?}"
PY="${TAILOR_PYTHON:-python}"
X=$B/crossgpu_l40s_xenv_20261110
RUN=$B/main_controls_20261109
RUNH=$B/heldout_20261109
RUNL=$B/grid_c2_live_20261110
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-local} commit=$(git -C $REPO rev-parse HEAD)"
echo "python=$PY $($PY --version 2>&1)"

while true; do
  set +e
  out=$(TAILOR_MAIN_CONTROLS_RUN=$X "$PY" -B -W ignore "$REPO/experiments/main_controls/crossgpu_xenv.py" decide 2>&1)
  rc=$?
  set -e
  echo "$(date -Is) decide rc=$rc"
  echo "$out" | tail -n 8
  if [ "$rc" -eq 0 ]; then break; fi
  if ! grep -q 'check incomplete' <<< "$out"; then
    echo "decide failed for a reason other than an incomplete check: no approval, caps unchanged"
    exit 1
  fi
  sleep 600
done

approved=$(echo "$out" | awk '$2 == "APPROVED" {print $1}' | paste -sd, -)
echo "approved by the rule: ${approved:-none}"
if [ -n "$approved" ]; then
  TAILOR_MAIN_CONTROLS_RUN=$X "$PY" -B -W ignore "$REPO/experiments/main_controls/crossgpu_xenv.py" approve "$approved" \
    "$RUN" "$RUNH" "$RUNL"
fi

# restore the L40S caps lowered on 2026-11-10 15:54 for the check (capacity_changes.jsonl records both moves)
"$PY" - "$RUN" "$RUNL" <<'EOF'
import datetime, json, os, sys
from pathlib import Path
caps = {sys.argv[1]: {"gpu": 16, "gpu_light": 8}, sys.argv[2]: {"cpu": 0, "gpu": 4, "gpu_light": 4}}
for run, cap in caps.items():
    run = Path(run)
    tmp = run / 'capacity.json.tmp'
    tmp.write_text(json.dumps(cap))
    os.replace(tmp, run / 'capacity.json')
    with (run / 'capacity_changes.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), capacity=cap,
                                reason='cross-environment check decided; L40S caps restored')) + '\n')
    print('CAPACITY', run.name, cap, flush=True)
EOF

if [ -n "$approved" ]; then
  TAILOR_MAIN_CONTROLS_RUN=$RUNH "$PY" -B -W ignore "$REPO/experiments/heldout/prepare_heldout.py" submit 4 --light
fi
echo "done $(date -Is)"
