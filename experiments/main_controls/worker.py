"""Bounded workers: CPU proposals never wait for a GPU; a GPU worker never holds a busy configuration."""
import os
import signal
import subprocess
import sys
import time
import traceback
import fcntl
from common import (PY, RUN, atomic, light_attacks, light_eligible, load_cell, output, read, task_id, transient,
                    verify_sources)


QUEUES = dict(cpu=RUN / 'queue_cpu.json', gpu=RUN / 'queue_gpu.json', gpu_light=RUN / 'queue_gpu.json',
              bridge=RUN / 'bridging/queue.json', bridge_light=RUN / 'bridging/queue.json')
LIGHT = ('gpu_light', 'bridge_light')     # L40S workers: approved attacks only (cross-GPU checks)


HEAVY = {'unmarker', 'ctrlregen_s03', 'ctrlregen_s05', 'ctrlregen_s07'}


def priority(kind, task):
    """Visiting order of a worker. A100 workers start the work only they can do, the long cross-environment
    attacks first (UnMarker takes about two hours), then other A100-only work, then work an L40S may also do,
    long first. L40S workers start the long work they may do. Ordering is scheduling only; it never changes what
    is measured. Ranked from the task's attack list (no cache lookups, so ranking a long queue stays cheap)."""
    if kind not in ('gpu', 'gpu_light'):  # CPU proposals carry no attack list; bridge queues keep their order
        return 0
    attacks = set(task['attacks'])
    if kind == 'gpu_light':
        return 0 if attacks & HEAVY else 1
    light = light_eligible(attacks)
    if not light:
        return 0 if attacks & HEAVY else 1
    return 2 if attacks & HEAVY else 3


def eligible(kind, task):
    """An L40S worker takes a task only when a cross-GPU check approved every attack the task would still
    measure; attacks whose cells exist already (measured on an A100) do not bind it."""
    if kind not in LIGHT:
        return True
    allowed = light_attacks()
    if not allowed:
        return False
    blocked = [a for a in task['attacks'] if a not in allowed]
    # same as "every still-missing attack is allowed", but looks up only the blocked attacks and stops at the
    # first one without a cell, so scanning a long queue stays light on the shared file system
    frozen = not task.get('run_cells_only')
    return all(load_cell(task['candidate'], a, frozen=frozen) is not None for a in blocked)


def complete(kind, task):
    if kind == 'cpu':
        return output(task).exists()
    # Bridging tasks are complete only with the run's own cells, never with frozen ones.
    frozen = not task.get('run_cells_only')
    return all(load_cell(task['candidate'], a, frozen=frozen) is not None for a in task['attacks'])


def stop(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def active_gpu():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = [r.split(',') for r in raw.splitlines() if r.strip()]
    assert len(rows) == 1 and rows[0][0].strip().startswith('GPU-'), 'ambiguous GPU telemetry'
    return float(rows[0][1]) > 0


def run_child(cmd, kind, deadline):
    proc = subprocess.Popen(cmd, start_new_session=True)
    started = last_gpu = time.time()
    limit = min(deadline, started + (3 * 3600 if kind == 'cpu' else 4 * 3600))
    try:
        while proc.poll() is None and time.time() < limit:
            try:
                proc.wait(timeout=min(30, max(1, limit - time.time())))
            except subprocess.TimeoutExpired:
                if kind != 'cpu':
                    if active_gpu():
                        last_gpu = time.time()
                    if time.time() - last_gpu > 1200:
                        raise RuntimeError('no GPU compute observed for 20 minutes')
        if proc.poll() is None:
            raise TimeoutError('allocation or task deadline; outputs remain checkpointed')
        return proc.returncode
    finally:
        stop(proc)


def main(kind):
    verify_sources()
    deadline = float(os.environ.get('SLURM_JOB_END_TIME', time.time() + 4 * 3600)) - 120
    worker = os.environ.get('SLURM_JOB_ID', 'local') + '_' + os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    for d in ('claims', 'errors', 'status'):
        (RUN / d).mkdir(exist_ok=True)
    while time.time() < deadline - 60:
        queue = QUEUES[kind]
        try:
            paths = read(queue) if queue.exists() else []
        except Exception as exc:
            if not transient(exc):
                raise
            print('READ_RETRY', type(exc).__name__, str(exc), flush=True)
            time.sleep(30)
            continue
        worked = False
        ranked = []
        for name in paths:
            try:
                ranked.append((priority(kind, read(name)), name))
            except Exception as exc:
                if not transient(exc):
                    raise
        paths = [name for _, name in sorted(ranked, key=lambda x: x[0])]   # stable: queue order within a tier
        for name in paths:
            if time.time() >= deadline - 60:
                return
            try:
                task = read(name)
                done = complete(kind, task)
            except Exception as exc:
                if not transient(exc):
                    raise
                print('READ_RETRY', type(exc).__name__, str(exc), flush=True)
                continue
            if done or not eligible(kind, task):
                continue
            ident = ('gpu' if kind == 'gpu_light' else 'bridge' if kind == 'bridge_light' else kind) + '_' + task_id(task)
            with (RUN / 'claims' / (ident + '.lock')).open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if complete(kind, task):
                    continue
                err = RUN / 'errors' / (ident + '.json')
                previous = read(err) if err.exists() else {}
                if time.time() - previous.get('last_failure', 0) < min(3600, 60 * 2 ** min(previous.get('attempts', 0), 6)):
                    continue
                atomic(RUN / 'status' / (worker + '.json'), dict(kind=kind, task=name, started_at=time.time()))
                print('TASK_START', kind, name, flush=True)
                try:
                    script = RUN / 'code' / ('propose.py' if kind == 'cpu' else 'gpu.py')
                    code = run_child([PY, '-B', '-u', str(script), name], kind, deadline)
                    if code == 75 and kind != 'cpu':
                        continue
                    assert code == 0, ('stage exit', code)
                    assert complete(kind, task), 'incomplete output'
                    atomic(RUN / 'status' / (worker + '.json'), dict(kind=kind, last_complete=name,
                                                                    completed_at=time.time()))
                except Exception:
                    detail = traceback.format_exc()
                    atomic(err, dict(task=name, attempts=previous.get('attempts', 0) + 1, last_failure=time.time(),
                                     error=detail))
                    print('TASK_UNRESOLVED', detail, flush=True)
                    if kind != 'cpu':
                        return
                worked = True
                break
        if not worked:
            print('RELEASE_NO_RUNNABLE_TASK', kind, flush=True)
            return


if __name__ == '__main__':
    main(sys.argv[1])
