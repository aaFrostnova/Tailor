"""Resolve initially infeasible requests with margin backoff and full live gates.

This follow-up starts only after the paired fixed-margin campaign completes.
The original four inputs, image population, live gates and fixed results stay
frozen. Completed stages can be replayed without rerunning their measurements.
"""
import collections
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import campaign as C
from margin_fallback_policy import POLICY, STAGES, assert_same_request, outcome, make_attempt, apply_attempt
from pipeline_audit import audit_row

AUDITS = {'fixed_groups': {}, 'fallback_groups': {}}


def progress(phase, **kwargs):
    data = dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                phase=phase, policy=POLICY, **kwargs)
    C.atomic(C.FROOT / 'progress.json', data)
    print('FALLBACK_PROGRESS', json.dumps(data), flush=True)


def fixed_outdir(k, arm):
    return C.FROOT.parent / 'live_topk' / f'{k}_unified_ba_out{arm}n2000_20260914'


def audit_walk(k, path, active_ids=None, stage=None, arm=None):
    rows = C.read(path / 'walk.json')['rows']
    refs = C.read(C.INPUTS / (k + '.json'))['records']
    assert [r['i'] for r in rows] == list(range(2000)), (path, 'request coverage')
    active = set(range(2000) if active_ids is None else active_ids)
    totals = collections.Counter()
    for row in rows:
        assert_same_request(row, refs[row['i']])
        if row['i'] not in active:
            assert not row.get('fallback_active', True)
            continue
        if stage is not None:
            assert row['fallback_active'] and row['stage'] == stage
            assert (row['rate_margin'], row['mean_margin']) == STAGES[stage]
            if not row['candidates']:
                assert row['solver_outcome'] == 'unsat'
        else:
            assert arm in C.ARMS
            assert (row['rate_margin'], row['mean_margin']) == (int(arm)/100, .02)
        assert outcome(row) != 'pending', (path, row['i'], 'live incomplete')
        assert all(c['certified'] for c in row['candidates'])
        for verdict in row['walk']:
            for cell in verdict['attacks'].values():
                assert cell['n'] == len(cell['image_pass']) == len(cell['image_bit_accuracy'])
                assert cell['passed_images'] + cell['failed_images'] == cell['n']
        if outcome(row) == 'live_exhausted':
            assert {v['rank'] for v in row['walk']} == {c['rank'] for c in row['candidates']}
            patch = path / f"patched_s{row['i']:02d}.json"
            assert patch.exists() and str(row['i']) in C.read(patch)['evaluated'], (path, row['i'], 'rank4 skipped')
        totals.update(audit_row(row, path, C.FROOT.parent / 'live_topk/cells'))
    key = str(path / 'walk.json')
    AUDITS['fixed_groups' if stage is None else 'fallback_groups'][key] = dict(
        passed=True, active_requests=len(active), stage=stage, origin_arm=arm,
        walk_sha256=hashlib.sha256((path / 'walk.json').read_bytes()).hexdigest(), **totals)
    return rows


def initial_state(row, arm, path):
    status = outcome(row)
    assert status != 'pending'
    state = {f: row[f] for f in ('i',) + C.REQUEST_FIELDS}
    state.update(cls=row['cls'], stress=row['stress'], protocol=row['protocol'], origin_arm=arm,
                 initial_outcome=status, initial_margins=[int(arm)/100, .02],
                 initial_source=str(path / 'walk.json'), attempts=[], selected=None, selected_margins=None,
                 status=dict(live_pass='fixed_live_pass', live_exhausted='fixed_live_exhausted',
                             no_candidate='pending_fallback')[status])
    if status == 'live_pass':
        v = next(v for v in row['walk'] if v['rank'] == row['deployed_rank'])
        state['selected'] = {f: v[f] for f in ('rank', 'cfg_id', 'psnr_live', 'capacity_bits_estimate')}
        state['selected_margins'] = state['initial_margins']
    return state


def save_states(states):
    for key, rows in states.items():
        C.atomic(C.FROOT / 'requests' / (key + '.json'), dict(policy=POLICY, rows=rows))


def verify_sources():
    sources = C.read(C.FROOT / 'source_hashes.json')
    for name, digest in sources.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, ('frozen input/code changed', name)


def load_fixed():
    verify_sources()
    C.boot('04')
    assert C.read(C.BASE_ROOT / 'progress.json')['phase'] == 'complete', 'fixed campaign still running'
    summaries = C.read(C.BASE_ROOT / 'full_results.json')
    assert len(summaries) == 10 and sum(x['completed_requests'] for x in summaries) == 20000
    assert {(r['cls'], r['arm']) for r in summaries} == {(k, a) for k in C.CLASSES for a in C.ARMS}
    states, fixed = {}, {}
    for k in C.CLASSES:
        for arm in C.ARMS:
            path = fixed_outdir(k, arm)
            rows = audit_walk(k, path, arm=arm)
            summary = next(s for s in summaries if (s['cls'], s['arm']) == (k, arm))
            statuses = collections.Counter(outcome(r) for r in rows)
            assert summary['total_requests'] == summary['completed_requests'] == 2000 and summary['open'] == 0
            assert summary['live_pass'] == statuses['live_pass']
            assert summary['offline_unsat'] == statuses['no_candidate']
            assert summary['live_exhausted'] == statuses['live_exhausted']
            fixed[k + '_' + arm] = rows
            states[k + '_' + arm] = [initial_state(r, arm, path) for r in rows]
        # Paired .04 is an already completed, identical-input first backoff for .06.
        for state in states[k + '_06']:
            if state['status'] != 'pending_fallback':
                continue
            row = fixed[k + '_04'][state['i']]
            assert_same_request(state, row)
            apply_attempt(state, make_attempt(row, '04', fixed_outdir(k, '04') / 'walk.json'))
    save_states(states)
    return states


def array_controller():
    spec = importlib.util.spec_from_file_location('_fallback_arrays_' + C.STAGE, C.BASE_CODE / 'controller.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_stage(stage, states):
    C.configure(stage)
    active = {key: [r['i'] for r in rows if r['status'] == 'pending_fallback'] for key, rows in states.items()}
    count = sum(map(len, active.values()))
    progress('stage_running', stage=stage, active_requests=count)
    C.atomic(C.ROOT / 'active.json', active)
    if not count:
        return
    pairs = [tuple(key.split('_')) for key, ids in active.items() if ids]
    tasks = [[k, arm, i] for k, arm in pairs for i in active[k + '_' + arm]]
    enum = C.ROOT / 'plans/enumeration.json'
    C.atomic(enum, [tasks[j:j+10] for j in range(0, len(tasks), 10)])
    ctrl = array_controller()
    missing = [j for j, batch in enumerate(C.read(enum)) if any(not C.checkpoint(k, a, i).exists() for k, a, i in batch)]
    ctrl.run_array('enum_batch', enum, missing)
    for k, arm in pairs:
        C.assemble(k, arm)
    for rank in (1, 2, 3, 4):
        if rank == 4:
            patches = []
            for k, arm in pairs:
                C.driver(k, arm, 'walk', 100)
                rows = C.read(C.outdir(k, arm) / 'walk.json')['rows']
                assert not any(v['pending'] for r in rows for v in r.get('walk', []) if v['rank'] < 4), 'lower ranks remain unmeasured'
                patches += [dict(cls=k, arm=arm, i=r['i']) for r in rows
                            if r['i'] in active[k + '_' + arm] and C.needs_patch(r)]
            manifest = C.ROOT / 'plans/patches.json'
            C.atomic(manifest, patches)
            ctrl.run_array('patch_task', manifest)
        measurements = []
        for k, arm in pairs:
            path, configs = C.plan_rank(k, arm, rank)
            measurements += [dict(plan=str(path), index=i) for i in range(len(configs))]
        manifest = C.ROOT / f'plans/measure_rank{rank}.json'
        C.atomic(manifest, measurements)
        ctrl.run_array('measure_task', manifest)
        xenv = []
        for k, arm in pairs:
            xenv += C.xenv_plan(k, arm, rank)
        manifest = C.ROOT / f'plans/xenv_rank{rank}.json'
        C.atomic(manifest, xenv)
        ctrl.run_array('xenv_task', manifest)
        for k, arm in pairs:
            C.driver(k, arm, 'xenv_mark')
        progress('stage_rank_completed', stage=stage, rank=rank, active_requests=count)
    for k, arm in pairs:
        C.driver(k, arm, 'walk', 100)
        path = C.outdir(k, arm)
        rows = audit_walk(k, path, active[k + '_' + arm], stage, arm)
        for i in active[k + '_' + arm]:
            assert_same_request(states[k + '_' + arm][i], rows[i])
            apply_attempt(states[k + '_' + arm][i], make_attempt(rows[i], stage, path / 'walk.json'))
    save_states(states)
    progress('stage_complete', stage=stage, active_requests=count)


def summarize(states):
    results = []
    for key, rows in states.items():
        assert len(rows) == 2000 and [r['i'] for r in rows] == list(range(2000))
        counts = collections.Counter(r['status'] for r in rows)
        assert not counts['pending_fallback']
        assert set(counts) <= {'fixed_live_pass', 'fixed_live_exhausted', 'fallback_live_pass',
                               'zero_margin_model_unsat', 'fallback_live_exhausted'}
        delivered = [r['selected'] for r in rows if r['selected'] is not None]
        assert len(delivered) == counts['fixed_live_pass'] + counts['fallback_live_pass']
        rescued = [r for r in rows if r['status'] == 'fallback_live_pass']
        result = dict(cls=rows[0]['cls'], arm=rows[0]['origin_arm'], origin_arm=rows[0]['origin_arm'], policy=POLICY,
                      protocol=rows[0]['protocol'], total_requests=2000, completed_requests=2000,
                      open=0, live_n=100, unmarker_n=30, min_live_rate=.9,
                      latency_basis='surrogate/model', zero_margin_unsat_scope='current frozen surrogate and configuration search space only',
                      status_counts=dict(counts), live_pass=len(delivered), fallback_rescued=len(rescued),
                      rescue_by_stage=dict(collections.Counter(r['attempts'][-1]['stage'] for r in rescued)),
                      pass_rate=len(delivered)/2000, denominator='all 2000 original requests, including stress',
                      request_inputs=list(C.REQUEST_FIELDS), capacity_role='output_only',
                      capacity_kind='BSC estimate from live bit accuracy',
                      mean_live_psnr_db=sum(v['psnr_live'] for v in delivered)/len(delivered) if delivered else None,
                      mean_capacity_bits_estimate=sum(v['capacity_bits_estimate'] for v in delivered)/len(delivered) if delivered else None,
                      min_capacity_bits_estimate=min((v['capacity_bits_estimate'] for v in delivered), default=None))
        C.atomic(C.FROOT / 'summaries' / (key + '.json'), result)
        results.append(result)
    assert sum(r['completed_requests'] for r in results) == 20000
    return results


def main():
    # A controller restart must not rewrite indexed manifests while children
    # from the previous controller are still consuming them.
    with (C.FROOT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recorded = set()
        for path in (C.FROOT / 'stages').glob('*/jobs.jsonl'):
            for line in path.read_text().splitlines():
                recorded.add(str(json.loads(line)['job']))
        running = {line.split('_')[0] for line in subprocess.check_output(
            ['squeue', '-u', '3529', '-h', '-r', '-o', '%i'], text=True).splitlines()}
        survivors = sorted(recorded & running)
        if survivors:
            progress('waiting_previous_arrays', jobs=survivors)
            ctrl = array_controller()
            for job in survivors:
                ctrl.wait_array(job)
        run_campaign()


def run_campaign():
    progress('checking_fixed_results')
    states = load_fixed()
    for stage in ('02', '00'):
        run_stage(stage, states)
    results = summarize(states)
    C.atomic(C.FROOT / 'full_results.json', results)
    C.atomic(C.FROOT / 'pipeline_audit.json', dict(
        passed=True, original_requests=10000, arm_request_evaluations=20000,
        completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        policy=POLICY, live_n=100, unmarker_n=30, min_live_rate=.9,
        latency_basis='surrogate/model', capacity_kind='BSC estimate from live bit accuracy',
        fixed_results=str(C.BASE_ROOT / 'full_results.json'), **AUDITS))
    progress('complete', original_requests=10000, arm_request_evaluations=20000,
             fallback_rescued=sum(r['fallback_rescued'] for r in results))


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        raise  # Another active controller owns progress; do not overwrite it.
    except Exception as exc:
        progress('failed', error=repr(exc))
        raise
