"""Rotate one dedicated GPU participant; never change frozen scientific code."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

OPS = Path(__file__).resolve().parent
NAME = 'unified_gpu_participant'
TERMINAL = {'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED',
            'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'TIMEOUT'}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp' + str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Watchdog:
    def __init__(self, ops=OPS, runner=subprocess.run):
        self.ops, self.runner = Path(ops), runner
        self.path = self.ops / 'gpu_recovery_watchdog_state.json'
        self.state = read(self.path) if self.path.exists() else dict(
            schema='unified_gpu_recovery_v1', consecutive_errors=0,
            consecutive_worker_failures=0, terminal_jobs=[], submissions=[])

    def save(self, phase, **more):
        self.state.update(phase=phase, heartbeat_at=utc(), pid=os.getpid(),
                          watchdog_job_id=os.environ.get('SLURM_JOB_ID'), **more)
        atomic(self.path, self.state)

    def event(self, kind, **more):
        record = dict(at=utc(), kind=kind, **more)
        with (self.ops / 'gpu_recovery_events.jsonl').open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(record, sort_keys=True), flush=True)

    def call(self, argv):
        # Do not dump the environment or submission correlation marker.
        result = self.runner(argv, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=45)
        require(result.returncode == 0, '%s failed with exit %s: %s' %
                (argv[0], result.returncode, (result.stderr or '')[-1200:]))
        return result.stdout.strip()

    def queue(self):
        output = self.call(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T|%j|%k'])
        rows = []
        for line in output.splitlines():
            values = line.strip().split('|', 3)
            require(len(values) == 4, 'malformed squeue response')
            rows.append(dict(job_id=values[0], state=values[1], name=values[2], marker=values[3]))
        return rows

    def accounting(self, job=None, since=None):
        argv = ['sacct', '-n', '-P', '-X', '-o', 'JobID,State,JobName%100,ElapsedRaw,Comment%128']
        argv += ['-j', job] if job else ['-u', str(os.getuid()), '--starttime', since[:19]]
        rows = []
        for line in self.call(argv).splitlines():
            values = line.strip().split('|')
            require(len(values) >= 5, 'malformed sacct response')
            rows.append(dict(job_id=values[0], state=values[1].split()[0].rstrip('+'),
                             name=values[2], elapsed=int(values[3] or 0), marker=values[4]))
        return rows

    def main_complete(self):
        path = self.ops.parent / 'pipeline_audit.json'
        if not path.exists():
            return False
        audit = read(path)
        ready = (audit.get('passed') is True and audit.get('independent_final_test') is True
                 and audit.get('actual_latency_measurements_complete') is True
                 and audit.get('acceptance_protocol') == 'unified_raw_ba_v1'
                 and audit.get('original_requests') == 10000
                 and audit.get('arm_request_evaluations') == 20000)
        if not ready:
            return False
        require(bool(audit.get('evidence')), 'complete main audit lacks evidence')
        for path, sha in audit['evidence'].items():
            require(digest(path) == sha, 'main audit evidence changed: ' + path)
        return True

    def worker_record(self):
        record = read(self.ops / 'gpu_worker_job.json')
        require(re.fullmatch(r'\d+', str(record.get('job_id', ''))) is not None, 'invalid GPU job id')
        require(Path(record.get('script', '')).resolve() == (self.ops / 'gpu_worker.sbatch').resolve(),
                'GPU record names a different submission script')
        require(record.get('role') == 'independent_A100_participant', 'GPU record has a different role')
        sha = digest(self.ops / 'gpu_worker.sbatch')
        require(record.get('sbatch_sha256') == sha, 'GPU submission script hash differs from job record')
        pinned = self.state.setdefault('sbatch_sha256', sha)
        require(pinned == sha, 'GPU submission script changed during watchdog lifetime')
        return record

    def inspect_heartbeat(self, record):
        path = self.ops / 'local_worker_state.json'
        if not path.exists():
            self.state['participant'] = dict(phase='heartbeat_absent')
            return
        value = read(path)
        stamp = value.get('heartbeat_at')
        age = None if not stamp else max(0, time.time() - datetime.datetime.fromisoformat(stamp).timestamp())
        current = value.get('started_at', '') >= record['submitted_at']
        count = len(value.get('validated_task_keys', []))
        previous = self.state.setdefault('validated_tasks_seen', count)
        if current and count > previous:
            self.state['consecutive_worker_failures'] = 0
        self.state['validated_tasks_seen'] = max(previous, count)
        self.state['participant'] = dict(phase=value.get('phase'), heartbeat_at=stamp,
            heartbeat_age_seconds=age, current_allocation=current, validated_tasks=count,
            stale=current and (age is None or age > 900))

    def adopt(self, intent, job_id):
        require(re.fullmatch(r'\d+', job_id) is not None, 'ambiguous submitted GPU job id')
        record = dict(job_id=job_id, submitted_at=intent['submitted_at'], script=str(self.ops / 'gpu_worker.sbatch'),
                      sbatch_sha256=self.state['sbatch_sha256'], role='independent_A100_participant',
                      requested_gpu='A100 80GB', state='submitted', previous_job_id=intent['previous_job_id'])
        atomic(self.ops / 'gpu_worker_job.json', record)
        intent.update(job_id=job_id, resolved_at=utc())
        self.save('gpu_submitted', current_job_id=job_id)
        self.event('gpu_submitted', job_id=job_id, previous_job_id=intent['previous_job_id'])

    def reconcile(self, intent, queue):
        found = {r['job_id'] for r in queue if r['marker'] == intent['marker'] and r['name'] == NAME}
        found |= {r['job_id'] for r in self.accounting(since=intent['submitted_at'])
                  if r['marker'] == intent['marker'] and r['name'] == NAME}
        require(len(found) == 1, 'unresolved prior sbatch submission; no new submission is permitted')
        self.adopt(intent, found.pop())

    def submit(self, previous):
        # Persist before sbatch. A timeout/crash is reconciled, never blindly retried.
        intent = dict(submitted_at=utc(), previous_job_id=previous,
                      marker='unified-gpu-recovery-' + uuid.uuid4().hex, job_id=None)
        self.state['submissions'].append(intent)
        self.save('submitting_gpu')
        result = self.call(['sbatch', '--parsable', '--comment', intent['marker'],
                            str(self.ops / 'gpu_worker.sbatch')])
        self.adopt(intent, result.split(';')[0])

    def tick(self):
        record = self.worker_record()
        self.inspect_heartbeat(record)
        queue = self.queue()  # A communication failure cannot become an empty queue.
        pending = [x for x in self.state['submissions'] if x['job_id'] is None]
        require(len(pending) <= 1, 'multiple unresolved submissions')
        if pending:
            self.reconcile(pending[0], queue)
            record = self.worker_record()
        job = str(record['job_id'])
        active = [r for r in queue if r['job_id'] == job]
        require(len(active) <= 1, 'duplicate GPU queue records')
        if active:
            require(active[0]['name'] == NAME, 'GPU queue name differs from owned participant')
        if self.main_complete():
            if active:
                # Only the recorded, named dedicated participant may be cancelled.
                self.call(['scancel', job])
                self.event('main_complete_gpu_cancel_requested', job_id=job)
                self.save('waiting_gpu_release', current_job_id=job)
                return False
            rows = [r for r in self.accounting(job=job) if r['job_id'] == job]
            require(len(rows) == 1 and rows[0]['name'] == NAME,
                    'cannot confirm owned GPU termination after main completed')
            if rows[0]['state'] not in TERMINAL:
                self.save('waiting_gpu_release', current_job_id=job)
                return False
            self.save('complete', current_job_id=job)
            return True
        if active:
            participant = self.state.get('participant', {})
            if active[0]['state'] == 'RUNNING' and participant.get('stale'):
                cancelled = self.state.setdefault('stale_cancelled_jobs', [])
                if job not in cancelled:
                    alert = dict(at=utc(), status='stalled_owned_gpu', job_id=job,
                                 heartbeat_at=participant.get('heartbeat_at'),
                                 heartbeat_age_seconds=participant.get('heartbeat_age_seconds'),
                                 action='cancel_owned_participant_then_wait_for_accounting_terminal')
                    atomic(self.ops / 'gpu_recovery_stall_alert.json', alert)
                    self.event('stalled_owned_gpu', job_id=job,
                               heartbeat_age_seconds=participant.get('heartbeat_age_seconds'))
                    self.call(['scancel', job])
                    cancelled.append(job)
                self.save('waiting_stalled_gpu_release', current_job_id=job)
                return False
            self.save('monitoring_gpu', current_job_id=job, gpu_state=active[0]['state'])
            return False
        rows = [r for r in self.accounting(job=job) if r['job_id'] == job]
        require(len(rows) == 1, 'GPU job absent from queue without unique accounting evidence')
        terminal = rows[0]
        require(terminal['name'] == NAME, 'GPU accounting name differs from owned participant')
        if terminal['state'] not in TERMINAL:
            self.save('waiting_accounting_terminal', current_job_id=job, gpu_state=terminal['state'])
            return False
        if job not in self.state['terminal_jobs']:
            normal = terminal['state'] == 'TIMEOUT' or (
                terminal['state'] == 'COMPLETED' and terminal['elapsed'] >= 23 * 3600)
            self.state['consecutive_worker_failures'] = 0 if normal else self.state['consecutive_worker_failures'] + 1
            self.state['terminal_jobs'].append(job)
            self.event('gpu_terminal', job_id=job, state=terminal['state'], elapsed=terminal['elapsed'],
                       normal_rotation=normal, consecutive_worker_failures=self.state['consecutive_worker_failures'])
            self.save('gpu_terminal', current_job_id=job)
        require(self.state['consecutive_worker_failures'] < 3, 'three consecutive GPU worker failures')
        self.submit(job)
        return False

    def run(self, poll):
        if self.state.get('phase') == 'error':
            raise RuntimeError('watchdog previously stopped with error; inspect alert before restarting')
        while True:
            self.save('checking_gpu')
            try:
                done = self.tick()
                self.state['consecutive_errors'] = 0
                self.save(self.state['phase'])
                if done:
                    return 0
            except Exception as error:
                self.state['consecutive_errors'] += 1
                stopped = (self.state['consecutive_errors'] >= 3 or
                           self.state['consecutive_worker_failures'] >= 3)
                self.event('recovery_error', error=str(error), consecutive_errors=self.state['consecutive_errors'])
                self.save('error' if stopped else 'retrying_observation', error=str(error))
                if stopped:
                    atomic(self.ops / 'gpu_recovery_alert.json', dict(at=utc(), error=str(error),
                           status='error', consecutive_errors=self.state['consecutive_errors'],
                           consecutive_worker_failures=self.state['consecutive_worker_failures'],
                           current_job_id=self.state.get('current_job_id')))
                    return 2
            time.sleep(poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll', type=float, default=30)
    args = parser.parse_args()
    require(1 <= args.poll <= 60, 'poll must be 1 to 60 seconds')
    with (OPS / 'gpu_recovery_watchdog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return Watchdog().run(args.poll)


if __name__ == '__main__':
    raise SystemExit(main())
