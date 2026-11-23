"""Keep the single-fragment CPU controller alive, preserving its exact arguments.

Source mutation is a hard, explicit stop. Unexpected controller exits restart
the same recorded workload; Slurm GPU arrays remain protected by the controller's
journal, source hashes, locks, and queue/accounting reconciliation.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rigor_protocol import B, PY, atomic, digest, read
from baseline_campaign import METHODS, utc, verify_sources


def all_complete():
    for method in METHODS:
        path = B / method / 'pipeline_audit.json'
        if not path.exists():
            return False
        audit = read(path)
        if audit.get('passed') is not True or audit.get('complete') is not True:
            return False
        for source, sha in audit['evidence'].items():
            if digest(source) != sha:
                raise ValueError(f'completed audit evidence changed: {source}')
    return True


def heartbeat(stage, **fields):
    value = dict(timestamp=utc(), stage=stage, complete=stage == 'complete',
                 supervisor_pid=os.getpid(), supervisor_job=os.environ.get('SLURM_JOB_ID'), **fields)
    atomic(B / 'supervisor_progress.json', value)


def stop_child(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def supervise(controller_args, poll=15):
    args = [PY, '-B', '-u', str(B / 'code/baseline_controller.py'), *controller_args]
    state_path = B / 'supervisor_state.json'
    state = read(state_path) if state_path.exists() else dict(restarts=0, controller_argv=args)
    if state['controller_argv'] != args:
        raise ValueError('supervisor resume cannot silently change controller arguments')
    child = None
    with (B / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (B / 'logs').mkdir(exist_ok=True)
        try:
            while True:
                try:
                    verify_sources()
                    complete = all_complete()
                except Exception as error:
                    stop_child(child)
                    heartbeat('source_integrity_failure', error=repr(error), restarts=state['restarts'])
                    atomic(B / 'alerts.json', dict(timestamp=utc(), status='source_integrity_failure',
                                                  error=repr(error), automatically_restarted=False))
                    return 2
                if complete:
                    stop_child(child)
                    heartbeat('complete', restarts=state['restarts'])
                    return 0
                if child is None or child.poll() is not None:
                    previous_code = None if child is None else child.returncode
                    if child is not None:
                        state['restarts'] += 1
                        state['previous_exit_code'] = previous_code
                        delay = min(300, 15 * state['restarts'])
                        until = time.time() + delay
                        while time.time() < until:
                            heartbeat('controller_restart_wait', previous_exit_code=previous_code,
                                      restarts=state['restarts'], retry_after_epoch=until)
                            time.sleep(min(poll, max(.1, until - time.time())))
                        # Revalidate immediately before launch, not only before
                        # the restart backoff interval.
                        try:
                            verify_sources()
                        except Exception as error:
                            heartbeat('source_integrity_failure', error=repr(error), restarts=state['restarts'])
                            atomic(B / 'alerts.json', dict(timestamp=utc(), status='source_integrity_failure',
                                                          error=repr(error), automatically_restarted=False))
                            return 2
                    with (B / 'logs/controller_supervised.log').open('a') as handle:
                        child = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT,
                                                 start_new_session=True, env=dict(os.environ))
                    state.update(child_pid=child.pid, child_started_at=utc(), updated_at=utc())
                    atomic(state_path, state)
                status_path = B / 'progress.json'
                status = read(status_path) if status_path.exists() else None
                heartbeat('controller_running', child_pid=child.pid, restarts=state['restarts'],
                          controller_progress=status, controller_argv=args)
                time.sleep(poll)
        finally:
            stop_child(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll', type=int, default=15)
    args, controller_args = parser.parse_known_args()
    if controller_args[:1] == ['--']:
        controller_args = controller_args[1:]
    if args.poll < 5:
        raise ValueError('supervisor poll must be at least five seconds')
    raise SystemExit(supervise(controller_args, args.poll))


if __name__ == '__main__':
    main()
