"""Rolling CPU proposals and GPU live queue for the main controls; missing evidence stays unresolved.

Per request and variant: three distinct-structure candidates at the initial margin (0.06, 0.02), one
request-local corrected candidate (rank 4) after every candidate was completely measured and failed,
then the frozen backoff stages 04 (0.04, 0.02), 02 (0.02, 0.01) and 00 (0, 0), each with its own
correction. Pending measurements never advance the search. GPU work is dispatched only after the
bridging check has decided which frozen cells may be reused.
"""
import collections
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
import common
from common import (CP, FROZEN_CELLS, POLICY, ROOT, RUN, STAGES, atomic, bridging_decided, cell_name, cfgkey,
                    judge, load_cell, output, read, sha, task_id, transient, verify_sources)

_native_load_cell = common.load_cell
_native_combine = CP.combine
_native_valid = CP.valid_cell
_reductions, _validations = {}, {}
DEFAULT_CAPACITY = dict(cpu=64, gpu=4, gpu_light=0)
WALLTIME = dict(cpu='12:00:00', gpu='08:00:00', gpu_light='08:00:00')
# Worker job names identify this run's workers in squeue (capacity accounting); a second controlled run
# sets TAILOR_JOB_PREFIX so the two controllers never count each other's workers.
_PREFIX = os.environ.get('TAILOR_JOB_PREFIX', 'exp_mc')
JOB = dict(cpu=f'{_PREFIX}_cpu', gpu=f'{_PREFIX}_gpu', gpu_light=f'{_PREFIX}_gpul')


def _combined(cell, order, tau, det_min=.9):
    key = id(cell), tuple(order), float(tau), float(det_min)
    if key not in _reductions:
        _reductions[key] = (cell, _native_combine(cell, order, tau, det_min))
    return _reductions[key][1]


def _valid(cell, order, n, cfg=None, alphas=CP.REQUEST_ALPHAS):
    key = id(cell), tuple(order), n, json.dumps(cfg, sort_keys=True), tuple(alphas)
    if key not in _validations:
        _validations[key] = (cell, _native_valid(cell, order, n, cfg=cfg, alphas=alphas))
    return _validations[key][1]


@lru_cache(maxsize=60000)
def _cycle_cell(order, fe, strengths, attack):
    return _native_load_cell(dict(order=list(order), fe_on=list(fe), s=dict(zip(order, strengths))), attack)


def _cached_cell(c, attack, frozen=True):
    assert frozen
    order, fe, strengths = cfgkey(c)
    return _cycle_cell(order, fe, strengths, attack)


def _walk_step(stage, c, verdict):
    origins = sorted({cell.get('origin') for cell in [common.load_cell(c, a) for a in verdict['attacks']] if cell})
    return dict(stage=stage, rank=c['rank'], cfg_id=CP.exact_cfg_id(cfgkey(c)), passed=verdict['pass'],
                pending=verdict['pending'], origins=origins)


def _queue_gpu(gpu, t, c, verdict):
    e = gpu.setdefault(cfgkey(c), dict(candidate=c, attacks=set(), cls=t['cls']))
    e['attacks'].update(verdict['pending'])


def inspect_target(t, cpu, gpu):
    """Next action or terminal status of one (request, variant) target."""
    walks = []
    for stage in STAGES:
        task = dict(t, stage=stage)
        if not output(task).exists():
            cpu.append(task)
            return dict(status='pending_proposal' if stage == 'initial' else 'pending_fallback', walk=walks)
        cs = read(output(task))['candidates']
        if not cs:
            continue
        for c in cs:
            verdict = judge(t['request'], c)
            walks.append(_walk_step(stage, c, verdict))
            if verdict['pass']:
                return dict(status='accepted', stage=stage, rank=c['rank'], candidate=c, verdict=verdict, walk=walks)
            if verdict['pending']:
                _queue_gpu(gpu, t, c, verdict)
                return dict(status='pending_live', walk=walks)
        patched = dict(t, stage=stage + '_patch')
        if not output(patched).exists():
            cpu.append(patched)
            return dict(status='pending_refinement', walk=walks)
        for c in read(output(patched))['candidates']:
            verdict = judge(t['request'], c)
            walks.append(_walk_step(stage, c, verdict))
            if verdict['pass']:
                return dict(status='accepted', stage=stage, rank=4, candidate=c, verdict=verdict, walk=walks)
            if verdict['pending']:
                _queue_gpu(gpu, t, c, verdict)
                return dict(status='pending_live', walk=walks)
    return dict(status='live_exhausted' if walks else 'model_unsat', walk=walks)


def has_frozen_cell(candidate, attacks):
    """True when the frozen cache holds a cell this configuration could reuse after the bridging decision."""
    key = cfgkey(candidate)
    return any((FROZEN_CELLS / cell_name(key, a)).exists() for a in attacks)


def split_until_bridged(groups, decided):
    """Before the bridging decision, configurations with a reusable frozen cell wait; the rest start now."""
    if decided:
        return list(groups), []
    ready, deferred = [], []
    for g in groups:
        (deferred if has_frozen_cell(g['candidate'], g['attacks']) else ready).append(g)
    return ready, deferred


def task_file(kind, task):
    p = RUN / 'tasks' / (kind + '_' + task_id(task) + '.json')
    if not p.exists():
        atomic(p, task)
    return str(p)


def fair_cpu_tasks(tasks):
    """Corrections and backoff stages first (they finish requests), then initial proposals by request."""
    tiers = collections.defaultdict(lambda: collections.defaultdict(collections.deque))
    for task in sorted(tasks, key=lambda t: (t['i'], t['stage'], t['variant'])):
        tier = 0 if task['stage'].endswith('_patch') else (1 if task['stage'] != 'initial' else 2)
        tiers[tier][task['variant']].append(task)
    out = []
    for tier in sorted(tiers):
        groups = tiers[tier]
        names = sorted(groups)
        while any(groups.values()):
            for name in names:
                if groups[name]:
                    out.append(groups[name].popleft())
    return out


def sweep():
    _cycle_cell.cache_clear()
    _reductions.clear()
    _validations.clear()
    CP.combine, CP.valid_cell = _combined, _valid
    common.load_cell = _cached_cell
    try:
        manifest = read(RUN / 'manifest.json')
        cpu, gpu, results = [], {}, []
        for t in manifest['targets']:
            result = inspect_target(t, cpu, gpu)
            # Per-image arrays stay in the cells, not in every copy of the periodic report.
            if 'verdict' in result:
                result['verdict'] = {k: v for k, v in result['verdict'].items() if k != 'attacks'}
            results.append(dict(variant=t['variant'], arm=t['arm'], spec_id=t['spec_id'], cls=t['cls'], i=t['i'],
                                **result))
        cpu = fair_cpu_tasks(cpu)
        atomic(RUN / 'queue_cpu.json', [task_file('cpu', t) for t in cpu])
        groups = []
        for _, g in sorted(gpu.items(), key=lambda x: CP.exact_cfg_id(x[0])):
            g['attacks'] = sorted(g['attacks'])
            groups.append(g)
        decided = bridging_decided()
        ready, deferred = split_until_bridged(groups, decided)
        jobs = [task_file('gpu', g) for g in ready]
        atomic(RUN / 'queue_gpu.json', jobs)
        counts = collections.defaultdict(collections.Counter)
        for r in results:
            counts[r['variant'] + '_' + r['arm']][r['status']] += 1
        status = dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), policy=POLICY,
                      counts={k: dict(v) for k, v in counts.items()}, pending_cpu=len(cpu),
                      pending_gpu_configs=len(jobs), deferred_gpu_configs=len(deferred), bridging_decided=decided,
                      target_evaluations=len(results),
                      complete=not any(r['status'].startswith('pending') for r in results))
        atomic(RUN / 'results.json', dict(status=status, rows=results))
        from report import publish
        publish(results, status)
        atomic(RUN / 'progress.json', status)
        print('PROGRESS', json.dumps(status), flush=True)
        return status
    finally:
        CP.combine, CP.valid_cell = _native_combine, _native_valid
        common.load_cell = _native_load_cell


def active_jobs():
    try:
        rows = subprocess.check_output(['squeue', '-u', str(os.getuid()), '-h', '-r', '-o', '%i|%j'], text=True,
                                       timeout=30)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        # Unknown queue state is never an empty queue: defer submissions to the next sweep.
        print('SCHEDULER_QUERY_RETRY', type(exc).__name__, str(exc), flush=True)
        return None
    return [dict(zip(('id', 'name'), r.split('|'))) for r in rows.splitlines() if r.strip()]


def _backoff(err_path):
    previous = read(err_path) if err_path.exists() else {}
    return time.time() - previous.get('last_failure', 0) < min(3600, 60 * 2 ** min(previous.get('attempts', 0), 6))


def runnable_gpu_count(light=False):
    """Unique configurations with missing evidence and a free lock; attack variants share one GPU.
    Light (L40S) workers take only configurations whose still-missing attacks a cross-GPU check approved."""
    from common import light_eligible
    paths = read(RUN / 'queue_gpu.json') if (RUN / 'queue_gpu.json').exists() else []
    seen, count = set(), 0
    for name in paths:
        task = read(name)
        c = task['candidate']
        digest = CP.exact_cfg_id(cfgkey(c))
        if digest in seen:
            continue
        missing = [a for a in task['attacks'] if load_cell(c, a) is None]
        if not missing:
            continue
        if light and not light_eligible(missing):
            continue
        seen.add(digest)
        if _backoff(RUN / 'errors' / ('gpu_' + task_id(task) + '.json')):
            continue
        with (ROOT / 'fixed/checkpoints' / ('cfg_' + digest + '.lock')).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            count += 1
    return count


def runnable_cpu_count(limit):
    runnable = 0
    for name in read(RUN / 'queue_cpu.json'):
        task = read(name)
        if output(task).exists() or _backoff(RUN / 'errors' / ('cpu_' + task_id(task) + '.json')):
            continue
        (RUN / 'claims').mkdir(exist_ok=True)
        with (RUN / 'claims' / ('cpu_' + task_id(task) + '.lock')).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            runnable += 1
            if runnable >= limit:
                break
    return runnable


def capacity():
    """Worker caps; RUN/capacity.json (operational, not part of the source binding) may lower or raise them."""
    caps = dict(DEFAULT_CAPACITY)
    p = RUN / 'capacity.json'
    if p.exists():
        caps.update({k: int(v) for k, v in read(p).items() if k in caps})
    return caps


# Partition names are site-specific: set them here for your cluster.
# CPU proposals may also run preemptible: each proposal is written atomically, so a preempted worker loses at
# most the proposal in flight (the cpu queue stalled the A3 proposals on 2026-11-09 evening).
PARTITION = dict(cpu='cpu', gpu='gpu', gpu_light='gpu')
GRES = dict(gpu=['--gres=gpu:1', '--constraint=a100-80g'], gpu_light=['--gres=gpu:l40s:1'])


def submit_command(kind, n):
    cmd = ['sbatch', '--parsable', '-p', PARTITION[kind],
           '-c', '1' if kind == 'cpu' else '6', '--mem=4G' if kind == 'cpu' else '--mem=64G',
           '-t', WALLTIME[kind], f'--array=0-{n - 1}%{n}', '-J', JOB[kind],
           '-o', str(RUN / f'logs/{kind}_%A_%a.log'), '--chdir=' + str(RUN), '--export=ALL']
    return cmd + GRES.get(kind, []) + [str(RUN / 'code/worker.sbatch'), kind]


def ensure_workers(state, status):
    active = active_jobs()
    if active is None:
        return
    for kind, cap in capacity().items():
        # GPU work starts before the bridging decision: the queue then holds only configurations without a
        # reusable frozen cell (split_until_bridged); the rest join it once the decision exists.
        occupied = sum(j['name'] == JOB[kind] for j in active)
        pending = status['pending_cpu' if kind == 'cpu' else 'pending_gpu_configs']
        if occupied >= cap or not pending:
            continue
        n = min(cap - occupied, pending)
        n = min(n, runnable_cpu_count(n) if kind == 'cpu' else runnable_gpu_count(light=kind == 'gpu_light'))
        if not n:
            continue
        job = subprocess.check_output(submit_command(kind, n), text=True, timeout=60).strip().split(';')[0]
        state[kind] = job
        with (RUN / 'jobs.jsonl').open('a') as f:
            f.write(json.dumps(dict(kind=kind, job=job, workers=n, at=time.time())) + '\n')
        atomic(RUN / 'worker_jobs.json', state)


def main():
    verify_sources()
    (RUN / 'logs').mkdir(exist_ok=True)
    with (RUN / 'controller.lock').open('a') as h:
        fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(RUN / 'worker_jobs.json') if (RUN / 'worker_jobs.json').exists() else {}
        deadline = float(os.environ.get('SLURM_JOB_END_TIME', time.time() + 10 * 3600)) - 120
        while time.time() < deadline:
            try:
                status = sweep()
            except Exception as exc:
                if not transient(exc):
                    raise
                print('SWEEP_RETRY', type(exc).__name__, str(exc), flush=True)
                time.sleep(60)
                continue
            if status['complete']:
                atomic(RUN / 'pipeline_audit.json', dict(source_binding=sha(RUN / 'sources.json'), **status))
                return
            ensure_workers(state, status)
            time.sleep(45)


if __name__ == '__main__':
    if sys.argv[1:] == ['sweep']:
        verify_sources()
        sweep()
    else:
        main()
