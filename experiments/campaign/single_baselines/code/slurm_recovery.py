"""Queue/accounting reconciliation derived from the reviewed shared helper.

JobID (array display identity), not JobIDRaw (individual numeric child job),
groups multiple accounting rows belonging to the same submitted array.
"""
import datetime
import os
import time

TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def parent_job(value):
    parent = value.strip().split('_', 1)[0].split('.', 1)[0]
    if not parent.isdigit():
        raise RuntimeError('invalid accounting array identity')
    return parent


def state_name(value):
    return value.split()[0].rstrip('+')


def reconcile(intent, command, atomic, path, clock=time.time):
    """Resolve a held submit with unknown job id, after a successful queue read."""
    created = datetime.datetime.fromisoformat(intent['created_at'])
    start = (created - datetime.timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%S')
    raw = command(['sacct', '-X', '-n', '-P', '-u', str(os.getuid()), '-S', start,
                   '--format=JobID,State,Comment%100'])
    found = {}
    for line in raw.splitlines():
        parts = line.split('|')
        if len(parts) >= 3 and parts[2].strip() == intent['token']:
            found.setdefault(parent_job(parts[0]), set()).add(state_name(parts[1]))
    if len(found) > 1:
        raise RuntimeError('multiple accounting arrays for held submission token')
    if found:
        job, states = next(iter(found.items()))
        if not states <= TERMINAL:
            raise RuntimeError(f'accounting sees {job} {states}, absent from queue; retry fresh snapshot')
        return dict(job=job, terminal_states=sorted(states))
    current = clock()
    intent.setdefault('absence_first_epoch', current)
    intent['absence_last_epoch'] = current
    atomic(path, intent)
    if current - created.timestamp() >= 300 and current - intent['absence_first_epoch'] >= 120:
        return dict(absent=True, proof='two successful queue/accounting absence observations, >=120s apart, held intent >=300s old')
    raise RuntimeError('held submission absent; awaiting second accounting observation and minimum age')


def reconcile_known(job, command):
    """A disappeared released array remains in flight until accounting is terminal."""
    raw = command(['sacct', '-X', '-n', '-P', '-j', str(job), '--format=JobID,State'])
    states = []
    for line in raw.splitlines():
        parts = line.split('|')
        if len(parts) >= 2 and parts[0].strip() and parent_job(parts[0]) == str(job):
            states.append(state_name(parts[1]))
    if not states:
        raise RuntimeError(f'array {job} absent from queue and accounting; retain in-flight reservation')
    if not set(states) <= TERMINAL:
        raise RuntimeError(f'array {job} accounting remains active {sorted(set(states))}; retain reservation')
    return dict(job=str(job), terminal_states=sorted(set(states)), accounting_rows=len(states))
