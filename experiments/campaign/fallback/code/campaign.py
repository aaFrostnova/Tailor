"""Reuse the fixed campaign's measured pipeline at progressively smaller margins.

The fixed campaign is never edited. Stage manifests and request verdicts live in
this sibling run; immutable four-input requests and raw image cells are shared.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

FROOT = Path(__file__).resolve().parent.parent
CODE = FROOT / 'code'
BASE_ROOT = FROOT.parent / 'fixed'
BASE_CODE = BASE_ROOT / 'code'
sys.path.insert(1, str(BASE_CODE))
spec = importlib.util.spec_from_file_location('_fixed_campaign', BASE_CODE / 'campaign.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from margin_fallback_policy import POLICY, STAGES, assert_same_request, strict_optimize_checks
from capacity_protocol import PROTOCOL, REQUEST_FIELDS, solver_scenario, valid_cell, needs_patch, exact_cfg_key, exact_cfg_id

INPUTS, SC, PY = base.INPUTS, base.SC, base.PY
CLASSES, ARMS = base.CLASSES, base.ARMS
atomic, read, same_request = base.atomic, base.read, base.same_request
_original_environment, _original_boot = base.environment, base.boot
STAGE = os.environ.get('FALLBACK_STAGE', '02')
ROOT = FROOT / 'stages' / STAGE


def configure(stage):
    global STAGE, ROOT
    assert stage in ('02', '00')
    STAGE = stage
    ROOT = FROOT / 'stages' / stage
    os.environ['FALLBACK_STAGE'] = stage
    for name in ('plans', 'logs', 'checkpoints'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    base.ROOT = ROOT


def tag(arm):
    return f'unified_ba_fallback{STAGE}_from{arm}n2000_20260914'


def environment(arm):
    _original_environment(arm)
    rate, mean = STAGES[STAGE]
    os.environ.update(RATE_MARGIN=str(rate), MEAN_MARGIN=str(mean), TAG=tag(arm))


def boot(arm):
    _original_boot(arm)
    strict_optimize_checks()


def checkpoint(k, arm, i):
    return ROOT / 'checkpoints' / f'{k}_{arm}' / f'q{i:04d}.json'


def verify_checkpoint(d, ref, arm):
    assert d['protocol'] == PROTOCOL and d['fallback_policy'] == POLICY
    assert_same_request(d, ref)
    assert (d['rate_margin'], d['mean_margin']) == STAGES[STAGE]
    assert d['stage'] == STAGE and d['origin_arm'] == arm
    assert d['stress'] == ref['stress']
    assert all(c['certified'] for c in d['candidates'])
    assert [c['rank'] for c in d['candidates']] == list(range(1, len(d['candidates']) + 1))
    assert len({json.dumps(c['skeleton'], sort_keys=True) for c in d['candidates']}) == len(d['candidates'])
    psnr = [c['psnr_db'] for c in d['candidates']]
    assert all(a >= b - 1e-6 for a, b in zip(psnr, psnr[1:]))
    if not d['candidates']:
        assert d['solver_outcome'] == 'unsat', 'unknown is not an empty enumeration'


def enumerate_one(k, arm, i):
    boot(arm)
    import watermark_smt_topk as T
    import solver_eval_continuous as SEC
    request = read(INPUTS / (k + '.json'))['records'][i]
    shared = ROOT / 'checkpoints' / 'shared' / k / f'q{i:04d}.json'
    shared.parent.mkdir(parents=True, exist_ok=True)
    with shared.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if shared.exists():
            data = read(shared)
            assert_same_request(data, request)
            assert (data['rate_margin'], data['mean_margin']) == STAGES[STAGE]
        else:
            rate, mean = STAGES[STAGE]
            scen = solver_scenario(request, margin=mean)
            started = time.time()
            print('FALLBACK_ENUM_START', k, arm, i, STAGE, rate, mean, flush=True)
            cs = T.enumerate_topk(scen, k=3, enable_order=True, continuous_strength=True,
                                 surrogate=SEC.sg, rate_margin=rate)
            data = dict(request, protocol=PROTOCOL, min_ba=scen["min_ba"], fallback_policy=POLICY, stage=STAGE, fallback_active=True,
                        rate_margin=rate, mean_margin=mean, margin_run=None,
                        solver_outcome='sat' if cs else 'unsat', sec=time.time()-started,
                        candidates=[dict(rank=c['rank'], skeleton=c['skeleton'].as_dict(),
                            order=c['cfg']['order'], fe=c['cfg']['fe'], fe_on=c['cfg']['fe_on'],
                            s=c['cfg']['s'], psnr_db=c['psnr_db'], ms=c['cfg']['ms'], certified=c['certified'],
                              **{k: c['cfg'][k] for k in ('live_threshold', 'threshold_count', 'max_tests', 'surrogate_protocol')})
                            for c in cs])
            assert all(c['certified'] for c in data['candidates'])
            atomic(shared, data)
        data = dict(data, origin_arm=arm)
        verify_checkpoint(data, request, arm)
        atomic(checkpoint(k, arm, i), data)
        print('FALLBACK_ENUM_VERIFIED', k, arm, i, STAGE, len(data['candidates']), round(data['sec'], 1), flush=True)


def enum_batch(index):
    failures = []
    for k, arm, i in read(ROOT / 'plans/enumeration.json')[index]:
        p = checkpoint(k, arm, i)
        if p.exists():
            verify_checkpoint(read(p), read(INPUTS / (k + '.json'))['records'][i], arm)
            continue
        result = subprocess.run([PY, '-u', str(CODE / 'campaign.py'), 'enum_one', k, arm, str(i)])
        if result.returncode:
            failures.append([k, arm, i, result.returncode])
    if failures:
        raise RuntimeError(f'fallback enumeration failed: {failures}')


def assemble(k, arm):
    active = set(read(ROOT / 'active.json').get(k + '_' + arm, []))
    rows = []
    rate, mean = STAGES[STAGE]
    for request in read(INPUTS / (k + '.json'))['records']:
        if request['i'] in active:
            row = read(checkpoint(k, arm, request['i']))
            verify_checkpoint(row, request, arm)
        else:
            # Keep the driver's exact 2000-ID coverage audit. Inactive entries are
            # placeholders only and are never counted as fallback UNSAT results.
            row = dict(request, protocol=PROTOCOL, candidates=[], fallback_active=False, stage=STAGE,
                       rate_margin=rate, mean_margin=mean, margin_run=None, origin_arm=arm)
        rows.append(row)
    assert [r['i'] for r in rows] == list(range(2000))
    atomic(outdir(k, arm) / 'requests_s00.json', dict(cls=k, tag=tag(arm), rows=rows,
           active_ids=sorted(active), stage=STAGE, rate_margin=rate, mean_margin=mean))


base.environment = environment
base.boot = boot
base.tag = tag
base.checkpoint = checkpoint
base.verify_checkpoint = verify_checkpoint
configure(STAGE)
outdir, driver, plan_rank = base.outdir, base.driver, base.plan_rank
measure_task, xenv_plan = base.measure_task, base.xenv_plan


def patch_task(manifest, index):
    item = read(manifest)[index]
    request = read(checkpoint(item['cls'], item['arm'], item['i']))
    verify_checkpoint(request, read(INPUTS / (item['cls'] + '.json'))['records'][item['i']], item['arm'])
    path = outdir(item['cls'], item['arm']) / f"patched_s{item['i']:02d}.json"
    if path.exists() and str(item['i']) in read(path).get('evaluated', {}):
        data = read(path)
        cand = data['candidates'].get(str(item['i']))
        assert data['evaluated'][str(item['i'])] == bool(cand)
        if cand:
            assert cand['certified'] and cand['rank'] == 4
        return
    base.patch_task(manifest, index)


def task_status(manifest, index):
    # A measure manifest points at a rank plan. On resume that plan may shrink
    # after cache hits; hash its contents too so an old index cannot skip a new
    # config that moved into that position.
    digest = hashlib.sha256(Path(manifest).read_bytes())
    item = read(manifest)[index]
    if isinstance(item, dict) and 'plan' in item:
        digest.update(Path(item['plan']).read_bytes())
    return ROOT / 'checkpoints/tasks' / f'{digest.hexdigest()[:16]}_{index:05d}.json'


def xenv_task(manifest, index):
    """Serialize identical cross-environment configs across the two origin arms."""
    item = read(manifest)[index]
    g = read(item['path'])
    key = exact_cfg_key(g['order'], g['fe'], [g['s'][f] for f in g['order']])
    cfgid = exact_cfg_id(key)
    assert cfgid == g['cfg_id'], 'xenv group has a different exact execution configuration'
    detector_cfg = dict(order=g['order'], fe_on=g['fe'])
    lockpath = ROOT / 'checkpoints' / ('xenv_cfg_' + cfgid + '.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for attack in item['required']:
            p = FROOT.parent / 'live_topk/cells' / f'{cfgid}_{attack}_pool300_n100.json'
            cached = read(p) if p.exists() else None
            if cached and valid_cell(cached.get('cell'), g['order'], 30 if attack == 'unmarker' else 100, cfg=detector_cfg):
                g['attacks'][attack] = cached['cell']
        atomic(item['path'], g)
        missing = [a for a in item['required']
                   if not valid_cell(g['attacks'].get(a), g['order'], 30 if a == 'unmarker' else 100, cfg=detector_cfg)]
        if missing:
            partial = ROOT / 'plans' / ('xenv_missing_' + cfgid + '.json')
            atomic(partial, [dict(item, required=missing)])
            base.xenv_task(partial, 0)
            g = read(item['path'])
        for attack in item['required']:
            c = g['attacks'][attack]
            assert valid_cell(c, g['order'], 30 if attack == 'unmarker' else 100, cfg=detector_cfg)
            p = FROOT.parent / 'live_topk/cells' / f'{cfgid}_{attack}_pool300_n100.json'
            existing = read(p) if p.exists() else None
            if existing and valid_cell(existing.get('cell'), g['order'], 30 if attack == 'unmarker' else 100, cfg=detector_cfg):
                continue  # Preserve the original PSNR and raw record used by fixed verdicts.
            atomic(p, dict(cell=c, psnr_db=g.get('psnr_db'), source=item['path'], n=c['n'], key=list(map(list,key)), protocol=PROTOCOL))


def dispatch_task(mode, manifest, index):
    status = task_status(manifest, index)
    status.parent.mkdir(parents=True, exist_ok=True)
    with status.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if status.exists() and read(status).get('complete'):
            return
        started = time.time()
        dict(measure_task=measure_task, patch_task=patch_task, xenv_task=xenv_task)[mode](manifest, index)
        atomic(status, dict(complete=True, stage=STAGE, mode=mode, index=index, seconds=time.time()-started))


def finish(k, arm):
    raise RuntimeError('use fallback controller aggregation; inactive placeholders are not UNSAT')


if __name__ == '__main__':
    args = sys.argv[1:]; sys.argv = [sys.argv[0]]
    mode = args.pop(0)
    if mode == 'enum_one':
        enumerate_one(args[0], args[1], int(args[2]))
    elif mode == 'enum_batch':
        enum_batch(int(args[0]))
    elif mode in ('measure_task', 'patch_task', 'xenv_task'):
        dispatch_task(mode, args[0], int(args[1]))
    else:
        raise SystemExit(mode)
