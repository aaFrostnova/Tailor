"""Full 5 x 2000 request top-k/live experiment, both rate margins, including stress.

Every request has a durable enumeration checkpoint. GPU tasks consume immutable
configuration manifests. An absent measurement or failed job is never UNSAT.
"""
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROJECT, PYTHON, RUN_ROOT, WORKSPACE

ROOT = Path(RUN_ROOT)
CODE, INPUTS = ROOT / 'code', ROOT / 'inputs'
SC = Path(WORKSPACE)
CF = Path(PROJECT)
PY = PYTHON
CLASSES = ('C1', 'C2', 'C3', 'C4', 'C5')
ARMS = ('04', '06')
from capacity_protocol import PROTOCOL, REQUEST_FIELDS, solver_scenario, valid_cell, needs_patch
FIELDS = ('i',) + REQUEST_FIELDS


def read(p):
    return json.loads(Path(p).read_text())


def atomic(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(data, indent=1) + '\n')
    os.replace(tmp, p)


def tag(arm):
    return f'capacity_out{arm}n2000_20260910'


def outdir(k, arm):
    return SC / 'live_topk' / f'{k}_{tag(arm)}'


def checkpoint(k, arm, i):
    return ROOT / 'checkpoints' / f'{k}_{arm}' / f'q{i:04d}.json'


def environment(arm):
    os.environ.update(TAG=tag(arm), RATE_MARGIN=str(int(arm) / 100), MEAN_MARGIN='.02',
                      ALLOW_XENV='1', SEED='0', SAMPLER_V='3',
                      FEAS_MATRIX=str(INPUTS / 'request_feasibility_matrix.json'),
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      PYTHONDONTWRITEBYTECODE='1',
                      HF_HOME=os.environ.get('HF_HOME', str(SC / 'hf_cache')),
                      HUGGINGFACE_HUB_CACHE=os.environ.get(
                          'HUGGINGFACE_HUB_CACHE', str(SC / 'hf_cache/hub')))


def boot(arm):
    environment(arm)
    sys.path[:0] = [str(CODE), str(CF / 'solver'), str(CF), str(SC)]
    import importlib.util
    # Preload frozen solver modules; legacy drivers later prepend repository paths.
    for name in ('surrogate_model', 'watermark_smt_v2', 'live_calibration', 'watermark_smt_topk', 'class_defs'):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, CODE / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    from c4_topk_tiebreak_repair import install
    install()
    if 'solver_eval_continuous' not in sys.modules:
        import types
        from surrogate_model import Surrogate
        sys.modules['solver_eval_continuous'] = types.SimpleNamespace(
            sg=Surrogate.from_dict(read(INPUTS / 'surrogate_canonical.json')))


def same_request(a, b):
    return all(a[f] == b[f] for f in FIELDS)


def verify_checkpoint(d, ref, arm):
    assert d.get('protocol') == PROTOCOL and 'min_bits' not in d
    assert same_request(d, ref), ('request changed', ref['i'])
    assert d['rate_margin'] == int(arm) / 100 and d['mean_margin'] == .02
    assert d['stress'] == ref['stress']
    assert all(c['certified'] for c in d['candidates']), ('uncertified', ref['i'])
    assert len({json.dumps(c['skeleton'], sort_keys=True) for c in d['candidates']}) == len(d['candidates'])
    vals = [c['psnr_db'] for c in d['candidates']]
    assert all(a >= b - 1e-6 for a, b in zip(vals, vals[1:]))


def enumerate_one(k, arm, i):
    boot(arm)
    import watermark_smt_topk as T
    import solver_eval_continuous as SEC
    data = read(INPUTS / (k + '.json'))
    r, sc = data['records'][i], data['scenarios'][i]
    assert r['i'] == i and same_request(r, dict(sc, i=i))
    scen = solver_scenario(r, margin=.02)
    start = time.time()
    print('ENUM_START', k, arm, i, 'stress', r['stress'], flush=True)
    candidates = T.enumerate_topk(scen, k=3, enable_order=True, continuous_strength=True,
                                 surrogate=SEC.sg, rate_margin=int(arm) / 100)
    d = {f: r[f] for f in FIELDS}
    d.update(protocol=PROTOCOL, cls=k, min_ba=scen['min_ba'], k_req=r['k'], stress=r['stress'], stress_kind=r.get('stress_kind'),
             margin_run=None, rate_margin=int(arm)/100, mean_margin=.02, sec=time.time()-start,
             candidates=[dict(rank=c['rank'], skeleton=c['skeleton'].as_dict(), order=c['cfg']['order'],
                              fe=c['cfg']['fe'], fe_on=c['cfg']['fe_on'], s=c['cfg']['s'],
                              psnr_db=c['psnr_db'], ms=c['cfg']['ms'], certified=c['certified']) for c in candidates])
    verify_checkpoint(d, r, arm)
    atomic(checkpoint(k, arm, i), d)
    print('ENUM_VERIFIED', k, arm, i, len(candidates), round(d['sec'], 1), flush=True)


def enum_batch(index):
    batch = read(ROOT / 'plans/enumeration.json')[index]
    failed = []
    for k, arm, i in batch:
        p = checkpoint(k, arm, i)
        if p.exists():
            verify_checkpoint(read(p), read(INPUTS / (k + '.json'))['records'][i], arm)
            continue
        result = subprocess.run([PY, '-u', __file__, 'enum_one', k, arm, str(i)])
        if result.returncode:
            failed.append([k, arm, i, result.returncode])
    atomic(ROOT / f'checkpoints/enum_task_{index:04d}.json', {'task': index, 'failed': failed})
    if failed:
        raise RuntimeError(f'enumeration task failures: {failed}')
    print('ENUM_BATCH_VERIFIED', index, flush=True)


def assemble(k, arm):
    ref = read(INPUTS / (k + '.json'))['records']
    rows = []
    for i in range(2000):
        d = read(checkpoint(k, arm, i))
        verify_checkpoint(d, ref[i], arm)
        rows.append(d)
    assert [r['i'] for r in rows] == list(range(2000))
    atomic(outdir(k, arm) / 'requests_s00.json', dict(cls=k, tag=tag(arm), n_req=2000,
           shard=0, n_shards=1, rate_margin=int(arm)/100, mean_margin=.02, k=3, seed=0, rows=rows))


def driver(k, arm, mode, *args):
    boot(arm)
    sys.argv = [str(CODE / 'live_topk_full.py'), mode, str(CLASSES.index(k)), *map(str, args)]
    return runpy.run_path(sys.argv[0], run_name='__main__')


def plan_rank(k, arm, rank):
    ns = driver(k, arm, 'walk', 100)
    need = {}
    for r in ns['rows']:
        if r['deployed_rank'] is not None:
            continue
        cand = next((c for c in r['candidates'] if c['rank'] == rank), None)
        if cand is None:
            continue
        prev = [c['rank'] for c in r['candidates'] if c['rank'] < rank]
        if prev and not any(v['rank'] == max(prev) and not v['pending'] for v in r['walk']):
            continue
        key = ns['cand_key'](cand)
        entry = need.setdefault(key, {'cand': cand, 'attacks': set()})
        entry['attacks'].update(r['attacks'])
    configs = []
    for key in sorted(need, key=lambda key: json.dumps(key)):
        entry = need[key]
        if any(ns['load_cell'](key, a) is None for a in entry['attacks']):
            configs.append({'key': list(map(list, key)), 'cand': entry['cand'], 'attacks': sorted(entry['attacks'])})
    path = ROOT / f'plans/{k}_{arm}_rank{rank}.json'
    atomic(path, {'cls': k, 'arm': arm, 'rank': rank, 'configs': configs})
    print('RANK_PLAN', k, arm, rank, len(configs), flush=True)
    return path, configs


def measure_task(manifest, index):
    item = read(manifest)[index]
    plan = read(item['plan'])
    os.environ['FULL_RANK_PLAN'] = item['plan']
    if item.get('prefetch_rows'):
        os.environ['FULL_PREFETCH_ROWS'] = item['prefetch_rows']
    cfg = plan['configs'][item['index']]
    digest = hashlib.sha1(json.dumps(cfg['key']).encode()).hexdigest()[:16]
    lock = ROOT / 'checkpoints' / ('cfg_' + digest + '.lock')
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        driver(plan['cls'], plan['arm'], 'measure_rank', plan['rank'], item['index'], len(plan['configs']), 100)
    print('MEASURE_TASK_VERIFIED', index, digest, flush=True)


def patch_task(manifest, index):
    item = read(manifest)[index]
    os.environ['FULL_PATCH_IDS'] = str(item['i'])
    driver(item['cls'], item['arm'], 'patch', item['i'], 1, 100)
    path = outdir(item['cls'], item['arm']) / f"patched_s{item['i']:02d}.json"
    assert str(item['i']) in read(path)['evaluated'], ('missing patch verdict', item)
    print('PATCH_TASK_VERIFIED', item, flush=True)


def xenv_plan(k, arm, rank):
    ns = driver(k, arm, 'walk', 100)
    driver(k, arm, 'xenv_mark')
    groups = {}
    for r in ns['rows']:
        for v in r['walk']:
            if v['rank'] != rank:
                continue
            missing = [a for a in v['pending'] if a in ns['XENV']]
            if missing:
                groups.setdefault(v['cfg_id'], set()).update(missing)
    tasks = []
    for cfgid, missing in sorted(groups.items()):
        gi = int(cfgid[:12], 16)
        path = Path(ns['XDIR']) / f'g{gi:04d}.json'
        g = read(path)
        assert g['cfg_id'] == cfgid
        g['xenv_needed'] = sorted(set(g.get('xenv_needed', [])) | missing)
        atomic(path, g)
        tasks.append(dict(cls=k, arm=arm, gidx=gi, domain=f'topk_{k}_{tag(arm)}',
                          required=g['xenv_needed'], path=str(path)))
    return tasks


def xenv_task(manifest, index):
    item = read(manifest)[index]
    environment(item['arm'])
    steps = ','.join(sorted({'0.' + a[-1] for a in item['required'] if a.startswith('ctrlregen')})) or '-'
    os.environ.update(CERT_DOMAIN=item['domain'], FULL_GIDX=str(item['gidx']), FULL_STEPS=steps,
                      FULL_UNM='1' if 'unmarker' in item['required'] else '0')
    subprocess.run(['bash', str(CODE / 'xenv_static.sh'), str(CLASSES.index(item['cls'])), '100', '30'], check=True)
    g = read(item['path'])
    for a in item['required']:
        c = g.get('attacks', {}).get(a)
        assert valid_cell(c, g['order'], 30 if a == 'unmarker' else 100), (item, a, 'incomplete')
    print('XENV_TASK_VERIFIED', item, flush=True)


def finish(k, arm):
    driver(k, arm, 'walk', 100)
    rows = read(outdir(k, arm) / 'walk.json')['rows']
    assert [r['i'] for r in rows] == list(range(2000))
    assert not any(r['open'] or any(v['pending'] for v in r['walk']) for r in rows), 'missing live cells'
    no_candidates = sum(not r['candidates'] for r in rows)
    passed = [r for r in rows if r['deployed_rank'] is not None]
    result = dict(cls=k, arm=arm, total_requests=2000, completed_requests=2000, offline_unsat=no_candidates,
                  live_pass=len(passed), live_exhausted=2000-no_candidates-len(passed), open=0,
                  stress_requests=sum(r['stress'] for r in rows),
                  pass_at={rank: sum(r['deployed_rank'] <= rank for r in passed)/2000 for rank in (1, 2, 3, 4)},
                  pass_at_denominator='all 2000 original requests, including stress and offline UNSAT')
    delivered = [next(v for v in r['walk'] if v['pass']) for r in passed]
    result.update(protocol=PROTOCOL, request_inputs=list(REQUEST_FIELDS),
                  capacity_role='output_only', capacity_kind='BSC estimate from live bit accuracy',
                  mean_live_psnr_db=sum(v['psnr_live'] for v in delivered)/len(delivered) if delivered else None,
                  mean_capacity_bits_estimate=sum(v['capacity_bits_estimate'] for v in delivered)/len(delivered) if delivered else None,
                  min_capacity_bits_estimate=min((v['capacity_bits_estimate'] for v in delivered), default=None),
                  rank4_rescued=sum(r['deployed_rank'] == 4 for r in passed))
    for row in rows:
        assert 'min_bits' not in row
        for verdict in row['walk']:
            assert not verdict['pending']
            for attack, cell in verdict['attacks'].items():
                assert cell['n'] == len(cell['image_pass']) == len(cell['image_bit_accuracy'])
                assert cell['passed_images'] + cell['failed_images'] == cell['n']
        if row['candidates'] and row['deployed_rank'] is None:
            patch = outdir(k, arm) / f"patched_s{row['i']:02d}.json"
            assert patch.exists() and str(row['i']) in read(patch)['evaluated'], ('rank4 was skipped', row['i'])
    assert result['offline_unsat'] + result['live_pass'] + result['live_exhausted'] == 2000
    atomic(outdir(k, arm) / 'full_summary.json', result)
    return result


def task_status(manifest, index):
    digest = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()[:16]
    return ROOT / 'checkpoints/tasks' / f'{digest}_{index:05d}.json'


def dispatch_task(mode, manifest, index):
    status = task_status(manifest, index)
    status.parent.mkdir(parents=True, exist_ok=True)
    with status.with_suffix('.lock').open('a') as handle:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if os.environ.get('FULL_LOCAL_WORKER') else 0)
        try:
            fcntl.flock(handle, flags)
        except BlockingIOError:
            return
        if status.exists() and read(status).get('complete'):
            return
        start = time.time()
        functions = {'measure_task': measure_task, 'patch_task': patch_task, 'xenv_task': xenv_task}
        functions[mode](manifest, index)
        atomic(status, {'complete': True, 'mode': mode, 'index': index, 'seconds': time.time()-start,
                        'worker': 'local' if os.environ.get('FULL_LOCAL_WORKER') else os.environ.get('SLURM_JOB_ID')})


if __name__ == '__main__':
    args = sys.argv[1:]
    sys.argv = [sys.argv[0]]
    mode = args.pop(0)
    if mode == 'enum_one': enumerate_one(args[0], args[1], int(args[2]))
    elif mode == 'enum_batch': enum_batch(int(args[0]))
    elif mode == 'assemble': assemble(*args)
    elif mode == 'plan': plan_rank(args[0], args[1], int(args[2]))
    elif mode in ('measure_task', 'patch_task', 'xenv_task'): dispatch_task(mode, args[0], int(args[1]))
    elif mode == 'finish': finish(*args)
    else: raise SystemExit(mode)
