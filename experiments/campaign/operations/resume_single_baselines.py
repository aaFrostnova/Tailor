"""Recover only the frozen single-baseline controller, after shared smoke is ready."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from smoke_dependencies import pending_dependencies

U = Path(__file__).resolve().parents[1]
O = U / 'operations'
PY = os.environ.get('TAILOR_PYTHON', 'python')


def write_status(phase, **fields):
    path = O / 'single_baselines_recovery_state.json'
    value = dict(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 job=os.environ.get('SLURM_JOB_ID'), pid=os.getpid(), **fields)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)
    print(json.dumps(value), flush=True)


def main():
    with (O / 'single_baselines_recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = json.loads((U / 'source_hashes.json').read_text())
        for name, expected in frozen.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError('frozen source changed: ' + name)
        while True:
            pending = pending_dependencies(U, 'single_baselines')
            if not pending:
                break
            write_status('waiting_dependencies', pending=pending, retry_budget_consumed=False)
            time.sleep(20)
        args = [PY, '-B', '-u', str(U / 'single_baselines/code/baseline_supervisor.py'),
                '--poll', '15', '--', '--slots', '2', '--max-attempts', '3']
        env = dict(os.environ, BASELINE_RUN_ROOT=str(U / 'single_baselines/S_VINE'),
            UNIFIED_ATTACK_REUSE_INDEX=str(U / 'plans/pixel_reuse_index.json'),
            UNIFIED_EMBED_REUSE_CERTIFICATE=str(U / 'plans/embedding_migration.json'),
            HF_HOME=os.environ.get('HF_HOME', '~/.cache/huggingface'),
            HF_HUB_CACHE=os.environ.get('HF_HOME', '~/.cache/huggingface') + '/hub',
            HUGGINGFACE_HUB_CACHE=os.environ.get('HF_HOME', '~/.cache/huggingface') + '/hub',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        write_status('starting_frozen_supervisor', controller_argv=args, frozen_files=len(frozen))
        proc = subprocess.Popen(args, env=env)
        write_status('running', child_pid=proc.pid, controller_argv=args, frozen_files=len(frozen))
        try:
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
        write_status('complete' if code == 0 else 'failed', exit_code=code)
        return code


if __name__ == '__main__':
    raise SystemExit(main())
