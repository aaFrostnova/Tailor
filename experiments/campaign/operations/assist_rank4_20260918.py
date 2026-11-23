"""Use the idle interactive GPU for one already submitted native rank-4 task."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

U = Path(__file__).resolve().parents[1]
R = U / 'shared_table_20260916/priority_20260917'
PY = os.environ.get('TAILOR_PYTHON', 'python')
job = os.environ['SLURM_JOB_ID']
end = int(os.environ['SLURM_JOB_END_TIME'])
record = json.loads((R / 'operations/local_gpu_current.json').read_text())
assert record['job_id'] == job
state = json.loads((R / 'status' / f'local_{job}.json').read_text())
assert state['phase'] == 'waiting_for_available_task', 'do not interrupt a live measurement'
memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                 '--format=csv,noheader,nounits'], text=True).strip().splitlines()
assert memory == ['0'], 'interactive GPU is in use'
assert end - time.time() > 600
pid = record['pid']
cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'local_gpu_worker.py' in cmdline and os.getpgid(pid) == pid
os.killpg(pid, signal.SIGTERM)
time.sleep(1)

manifest = U / 'fallback/stages/00/plans/measure_rank4.json'
index = 3
item = json.loads(manifest.read_text())[index]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec = dict(kind='measure_task', manifest=str(manifest), manifest_sha256=sha(manifest),
            index=index, stage='00', linked_plan_sha256=sha(item['plan']))
task = U / 'operations' / f'local_rank4_{job}.json'
task.write_text(json.dumps(spec, indent=2) + '\n')
log = U / 'operations' / f'local_rank4_{job}.log'
env = os.environ.copy()
env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1')
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
with log.open('ab', buffering=0) as f:
    child = subprocess.Popen(['timeout', '--kill-after=15s', str(int(end-time.time()-90)),
        PY, '-B', '-u', str(U/'operations/local_worker.py'), '_execute', '--task', str(task)],
        env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
    status = U / 'operations' / f'local_rank4_{job}_status.json'
    status.write_text(json.dumps(dict(started_at=started, pid=child.pid, log=str(log),
        phase='running', spec=spec), indent=2) + '\n')
    print('NATIVE_RANK4_STARTED', child.pid, str(log), flush=True)
    rc = child.wait()
    status.write_text(json.dumps(dict(started_at=started, ended_at=datetime.datetime.now(
        datetime.timezone.utc).isoformat(), log=str(log), returncode=rc,
        phase='native_validated' if rc == 0 else 'unresolved', spec=spec), indent=2) + '\n')
    print('NATIVE_RANK4_EXIT', rc, flush=True)
if end - time.time() > 120:
    subprocess.run([PY, '-B', str(R/'operations/start_local_gpu.py')], env=env, check=True)
