"""Durable two-GPU rolling scheduling of isolated single-fragment campaigns.

Serial arrays occupy one slot each; finished arrays are replenished without
waiting for a slow task in the other slot. A held submission is journaled before
release, and its unique Slurm comment makes interrupted submissions recoverable.
"""
import argparse
import collections
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from rigor_protocol import B, PY, atomic, digest, read
from baseline_campaign import METHODS, verify_sources, verify_smoke_preflight
from slurm_recovery import reconcile, reconcile_known

CODE = B / 'code'
STATE = B / 'controller_state.json'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(argv, **kwargs):
    return subprocess.check_output(argv, text=True, timeout=90, **kwargs).strip()


def queue():
    output = command(['squeue', '-h', '-r', '-u', str(os.getuid()), '-o', '%i|%F|%K|%T|%k|%r'])
    rows = []
    for line in output.splitlines():
        job, parent, index, status, comment, reason = line.split('|', 5)
        if status in ACTIVE:
            rows.append(dict(job=job, parent=parent, index=index, status=status, comment=comment, reason=reason))
    return rows


def save(state):
    state['updated_at'] = utc()
    atomic(STATE, state)


def new_state():
    return dict(schema='baseline_rolling_controller_v1', created_at=utc(), records=[], events=[])


def record_event(state, kind, **values):
    row = dict(timestamp=utc(), kind=kind, **values)
    state['events'].append(row)
    state['events'] = state['events'][-1000:]
    with (B / 'controller_events.jsonl').open('a') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n')


def recover(state, rows):
    """Reconcile ambiguous submits and missing arrays without freeing work early."""
    def persist_intent(path, intent):
        atomic(path, intent)
        save(state)
    for rec in state['records']:
        if rec['state'] in ('submission_absent', 'terminal'):
            continue
        if rec.get('job') is None and rec['state'] == 'submitting':
            found = {x['parent'] for x in rows if x['comment'] == rec['token']}
            if len(found) > 1:
                raise ValueError('duplicate Slurm jobs for one submission token')
            if found:
                rec.update(job=found.pop(), state='held')
                record_event(state, 'recovered_held_submission', job=rec['job'], token=rec['token'])
                save(state)
            else:
                proof = reconcile(rec, command, persist_intent, B / 'intents' / (rec['token'] + '.json'))
                if proof.get('absent'):
                    rec.update(state='submission_absent', reconciliation=proof)
                else:
                    rec.update(job=proof['job'], state='terminal', reconciliation=proof)
                record_event(state, 'accounting_reconciled_submission', token=rec['token'], proof=proof)
                save(state)
        if rec.get('job') and rec['state'] != 'terminal':
            live = [x for x in rows if x['parent'] == rec['job']]
            if not live:
                proof = reconcile_known(rec['job'], command)
                rec.update(state='terminal', reconciliation=proof)
                record_event(state, 'accounting_confirmed_terminal', job=rec['job'], proof=proof)
                save(state)
                continue
            if rec['state'] == 'held':
                if any(x['reason'] == 'JobHeldAdmin' for x in live):
                    raise RuntimeError(f'array {rec["job"]} is administrator-held; cannot release')
                if any(x['reason'] == 'JobHeldUser' for x in live):
                    command(['scontrol', 'release', rec['job']])
                rec['state'] = 'released'
                record_event(state, 'released_or_observed_released', job=rec['job'])
            save(state)


def completed_indices(plan, cache):
    """Scheduling hint only. Campaign audit validates all evidence before use."""
    return {i for i, task in enumerate(plan['tasks']) if (cache / task['task_id'] / 'result.json').exists()}


def inflight(state, rows, method, role):
    indices = set()
    for rec in state['records']:
        if (rec['method'] == method and rec['role'] == role
                and rec['state'] not in ('terminal', 'submission_absent')):
            indices.update(rec['indices'])
    return indices


def attempt_counts(state, method, role):
    counts = collections.Counter()
    for rec in state['records']:
        if rec['method'] == method and rec['role'] == role and rec.get('job'):
            counts.update(rec['indices'])
    return counts


def candidates(total, completed, running, attempts, limit):
    return [i for i in range(total) if i not in completed and i not in running and attempts[i] < limit]


def submit(state, method, role, plan_path, indices):
    token = 'baseline-' + uuid.uuid4().hex
    rec = dict(token=token, method=method, role=role, plan=str(plan_path.resolve()),
        plan_sha256=digest(plan_path), indices=list(indices), job=None,
        state='submitting', created_at=utc())
    state['records'].append(rec)
    save(state)
    logdir = B / 'logs'
    logdir.mkdir(exist_ok=True)
    args = ['sbatch', '--parsable', '--hold', '--comment', token,
        '--job-name', f'bl_{method}_{role}', '--partition', 'gpu',
        '--gres', 'gpu:1', '--constraint', 'a100-80g',
        '--cpus-per-task', '6', '--mem', '64G', '--time', '1-00:00:00',
        '--array', ','.join(map(str, indices)) + '%1',
        '--output', str(logdir / '%x_%A_%a.out'),
        '--error', str(logdir / '%x_%A_%a.err'),
        str(CODE / 'baseline_gpu.sbatch'), str(B / method), role, str(plan_path)]
    try:
        response = command(args)
    except Exception:
        # Keep the intent unresolved: the next queue poll either adopts the
        # held token or marks it absent. Never silently submit a second copy.
        save(state)
        raise
    job = response.split(';', 1)[0]
    if not job.isdigit():
        raise ValueError(f'invalid sbatch response: {response}')
    rec.update(job=job, state='held')
    record_event(state, 'submitted_held', job=job, method=method, role=role, indices=indices)
    save(state)
    command(['scontrol', 'release', job])
    rec['state'] = 'released'
    record_event(state, 'released', job=job)
    save(state)
    return rec


def campaign(method, action, extra=()):
    env = dict(os.environ, BASELINE_RUN_ROOT=str(B / method))
    # One method transition may audit thousands of PNGs. It must not look like
    # a dead controller; the audit child has its own explicit current stage.
    atomic(B / 'progress.json', dict(timestamp=utc(), complete=False,
        stage='validating_' + action, method=method, controller_pid=os.getpid()))
    log = B / 'logs' / f'{method}_{action}.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('a') as handle:
        proc = subprocess.Popen([PY, '-B', '-u', str(CODE / 'baseline_campaign.py'), action, *extra],
                                env=env, stdout=handle, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            atomic(B / 'progress.json', dict(timestamp=utc(), complete=False,
                stage='validating_' + action, method=method,
                controller_pid=os.getpid(), child_pid=proc.pid))
            time.sleep(15)
        if proc.returncode:
            raise RuntimeError(f'{method} {action} failed ({proc.returncode}); see {log}')


def method_stage(method):
    root = B / method
    if (root / 'pipeline_audit.json').exists():
        audit = read(root / 'pipeline_audit.json')
        if not audit.get('passed') or not audit.get('complete'):
            raise ValueError('invalid completed baseline audit')
        return 'complete'
    return 'final' if (root / 'selection/complete.json').exists() else 'selection'


def cycle(state, slots, chunk, max_attempts):
    verify_sources()
    verify_smoke_preflight()
    rows = queue()  # Query failures raise; they NEVER mean no active jobs.
    recover(state, rows)
    occupied = sum(rec['state'] not in ('terminal', 'submission_absent') for rec in state['records'])
    status = {}
    available = []
    for method in METHODS:
        role = method_stage(method)
        if role == 'complete':
            status[method] = dict(stage=role, complete=True)
            continue
        plan_path = B / method / 'plans' / f'{role}_tasks.json'
        if not plan_path.exists():
            raise ValueError(f'prepare method first: {plan_path}')
        plan = read(plan_path)
        campaign(method, 'inspect', ('--role', role))
        done = completed_indices(plan, B / method / role / 'cache')
        running = inflight(state, rows, method, role)
        attempts = attempt_counts(state, method, role)
        eligible = candidates(len(plan['tasks']), done, running, attempts, max_attempts)
        exhausted = [i for i in range(len(plan['tasks']))
                     if i not in done and i not in running and attempts[i] >= max_attempts]
        status[method] = dict(stage=role, complete=False, total=len(plan['tasks']),
            result_files=len(done), inflight=len(running - done), pending=len(eligible),
            retry_exhausted=exhausted, file_counts_are_not_scientific_audit=True)
        if len(done) == len(plan['tasks']):
            try:
                campaign(method, 'select' if role == 'selection' else 'audit')
            except Exception:
                # Full transition audit may discover an evidence file changed
                # while result.json itself stayed unchanged. Force the next
                # inspection to verify every task and quarantine bad artifacts.
                campaign(method, 'inspect', ('--role', role, '--force'))
                raise
            record_event(state, 'stage_completed', method=method, role=role)
            save(state)
            return False
        if eligible:
            available.append((method, role, plan_path, eligible))
    # Fair refill: each method gets a serial slot when both need measurement.
    cursor = int(state.get('refill_cursor', 0))
    while occupied < slots and available:
        position = cursor % len(available)
        method, role, path, eligible = available[position]
        take = eligible[:chunk]
        submit(state, method, role, path, take)
        occupied += 1
        eligible = eligible[len(take):]
        if eligible:
            available[position] = (method, role, path, eligible)
        else:
            available.pop(position)
        cursor += 1
    state['refill_cursor'] = cursor
    save(state)
    complete = all(v['complete'] for v in status.values())
    exhausted_by_method = {m: v['retry_exhausted'] for m, v in status.items() if v.get('retry_exhausted')}
    if exhausted_by_method:
        alert = dict(timestamp=utc(), status='attention_required', retry_exhausted=exhausted_by_method,
                     scientific_failure=False, automatic_submissions_stopped_for_these_indices=True)
        atomic(B / 'alerts.json', alert)
        if state.get('last_exhausted_alert') != exhausted_by_method:
            record_event(state, 'retry_exhausted', methods=exhausted_by_method)
            state['last_exhausted_alert'] = exhausted_by_method
            save(state)
    stage = 'complete' if complete else ('attention_required' if exhausted_by_method else 'measuring')
    progress = dict(timestamp=utc(), complete=complete, stage=stage,
        methods=status, gpu_slots=slots, active_arrays=occupied, controller_pid=os.getpid(),
        controller_job=os.environ.get('SLURM_JOB_ID'), auto_retry_limit=max_attempts)
    atomic(B / 'progress.json', progress)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slots', type=int, default=2)
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--poll', type=int, default=30)
    parser.add_argument('--max-attempts', type=int, default=4)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.slots <= 2 and 1 <= args.chunk <= 128 and args.poll >= 5):
        raise ValueError('baseline allocation is limited to two GPU slots')
    B.mkdir(exist_ok=True)
    with (B / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(STATE) if STATE.exists() else new_state()
        failures = 0
        while True:
            try:
                if cycle(state, args.slots, args.chunk, args.max_attempts):
                    return
                failures = 0
            except Exception as error:
                failures += 1
                record_event(state, 'controller_error', error=repr(error), consecutive_errors=failures)
                save(state)
                atomic(B / 'progress.json', dict(timestamp=utc(), complete=False,
                    stage='controller_retry', error=repr(error), consecutive_errors=failures,
                    controller_pid=os.getpid(), next_retry_seconds=min(300, args.poll * failures)))
                if args.once:
                    raise
            if args.once:
                return
            time.sleep(min(300, args.poll * max(1, failures)))


if __name__ == '__main__':
    main()
