"""Verify the accelerated top-k (fast_topk.py) against the original, and time both.

Each solver runs in a fresh interpreter per case, so neither inherits the other's z3 state (z3 breaks exact ties
by its internal state). Per rank the candidates are compared at four levels:
  identical      every field equal, floats bit for bit;
  equivalent     same skeleton (fragments, order, stages), same certified value bit for bit, strengths within
                 1e-6 (the eps band of the objective; such strengths embed to identical pixels, checked separately);
  tie choice     same certified value, another skeleton of the exact tie set (the original picks arbitrarily);
  discrepancy    a different certified value: the acceleration changed the answer (must never happen).
The fresh original is also compared with the stored record (September checkpoint for TAILOR-F, the main-control
proposal for a variant), which checks that the original itself is reproducible across processes.

Cases (JSON list): {"kind": "tailor", "cls": "C1", "i": 0}  (fixed checkpoint q<i>.json, arm 06 margins)
                   {"kind": "variant", "variant": "AlwaysAllThree", "spec_id": ..., "stage": "initial"}
Two fast variants are run: fast_original keeps the original final pin step (re-solve at ps >= best - eps with
the fewest front-ends) and only seeds the proposal; fast_certified also returns the certified model directly when
it has no front-end on (no final re-solve).
Usage: TAILOR_MAIN_CONTROLS_RUN=RUN python verify_fast.py CASES.json OUT_DIR SHARD NSHARD
       (internally: verify_fast.py one CASE_JSON original|fast_original|fast_certified)
"""
import os
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN_CKPT = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/rigor_editing_only_20260911/unified_ba_20260914/fixed/checkpoints')


def one(case, solver):
    """Run one solver on one case in this (fresh) interpreter; print a JSON result."""
    sys.path[:0] = [str(HERE), str(HERE.parent / 'main_controls'), str(HERE.parent)]
    import common as M
    M.C.boot(M.ARM)
    import watermark_smt_topk as T
    from surrogate_model import Surrogate
    sg = Surrogate.from_dict(M.read(M.ROOT / 'fixed/inputs/surrogate_canonical.json'))
    if case['kind'] == 'tailor':
        q = json.loads((FROZEN_CKPT / f"{case['cls']}_06" / f"q{int(case['i']):04d}.json").read_text())
        req = dict(attacks=q['attacks'], fpr=q['fpr'], min_psnr=q['min_psnr'], max_ms=q['max_ms'])
        rate, mean, no_pass, sizes = q['rate_margin'], q['mean_margin'], True, None
        stored = q['candidates']
    else:
        M.install_variant(case['variant'], M.ARM)
        rec = M.read(M.RUN / 'proposals' / case['variant'] / M.ARM / case['spec_id'] / f"{case['stage']}.json")
        req = {r['spec_id']: r for r in M.read(M.RUN / 'inputs/requests.json')}[case['spec_id']]['request']
        rate, mean = rec['rate_margin'], rec['mean_margin']
        no_pass = M.NO_PASSENGERS[case['variant']]
        sizes = {1} if case['variant'] == 'SingleOptimized' else {3}
        stored = rec['candidates']
    scen = M.CP.solver_scenario(req, margin=mean)
    kw = dict(enable_order=True, continuous_strength=True, surrogate=sg, rate_margin=rate)
    stats, t_seed = [], 0.0
    if solver == 'original':
        t0 = time.perf_counter()
        cands = T.enumerate_topk(scen, k=3, no_passengers=no_pass, **kw)
        seconds = time.perf_counter() - t0
    else:
        from fast_topk import enumerate_topk_fast, grid_seeds
        from grid import frozen
        from grid.tables import GridTables
        tables = GridTables(mode='fitted', ns=frozen.boot())          # once per process, like the surrogate load
        t0 = time.perf_counter()
        seeds = grid_seeds(tables, req, rate_margin=rate, mean_margin=mean, no_passengers=no_pass, sizes=sizes)
        t_seed = time.perf_counter() - t0
        cands = enumerate_topk_fast(scen, k=3, seeds=seeds, stats=stats, no_passengers=no_pass,
                                    final=solver.split('_', 1)[1], **kw)
        seconds = time.perf_counter() - t0
    view = [dict(rank=c['rank'], skeleton=[list(c['skeleton'].frags), list(c['skeleton'].order),
                                            list(c['skeleton'].fe)],
                 cfg=c['cfg'], value=c['psnr_proxy'], certified=c['certified']) for c in cands]
    stored_view = [dict(order=c.get('order'), fe_on=sorted(c.get('fe_on') or []), s=c.get('s')) for c in stored]
    print('RESULT ' + json.dumps(dict(solver=solver, seconds=seconds, seeds_s=t_seed, stats=stats, candidates=view,
                                      stored=stored_view, booted_pin_step=bool(getattr(T, '_c4_tiebreak_repaired',
                                                                                       False))),
                                 default=str), flush=True)


def run(case, solver):
    p = subprocess.run([sys.executable, '-B', '-W', 'ignore', __file__, 'one', json.dumps(case), solver],
                       capture_output=True, text=True, timeout=7200)
    line = next((l for l in p.stdout.splitlines() if l.startswith('RESULT ')), None)
    if line is None:
        raise RuntimeError(f'{solver} failed on {case}: {p.stderr[-2000:]}')
    return json.loads(line[len('RESULT '):])


def compare(a, b):
    levels = []
    for x, y in zip(a, b):
        if json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True):
            levels.append('identical')
        elif x['value'] != y['value']:
            levels.append('discrepancy')
        elif x['skeleton'] != y['skeleton']:
            levels.append('tie_choice')
        elif max(abs(x['cfg']['s'][f] - y['cfg']['s'][f]) for f in x['cfg']['s']) <= 1e-6:
            levels.append('equivalent')
        else:
            levels.append('tie_choice')
    if len(a) != len(b):
        levels.append('discrepancy')
    order = ['discrepancy', 'tie_choice', 'equivalent', 'identical']
    return levels, min(levels, key=order.index) if levels else 'identical'


def main(cases_path, out_dir, shard, nshard):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for case in json.loads(Path(cases_path).read_text())[shard::nshard]:
        name = (f"tailor_{case['cls']}_{case['i']}" if case['kind'] == 'tailor'
                else f"{case['variant']}_{case['spec_id'][:12]}_{case['stage']}")
        dest = out / f'{name}.json'
        if dest.exists():
            continue
        o = run(case, 'original')
        reproducible = [dict(order=c['cfg']['order'], fe_on=sorted(c['cfg']['fe_on']), s=c['cfg']['s']) == s
                        for c, s in zip(o['candidates'], o['stored'])]
        rec = dict(case=case, original_s=o['seconds'], original_reproduces_stored=reproducible,
                   booted_pin_step=o['booted_pin_step'], original=o['candidates'])
        line = dict(case=name, original_s=round(o['seconds'], 1), reproducible=reproducible)
        for solver in ('fast_original', 'fast_certified'):
            f = run(case, solver)
            levels, overall = compare(o['candidates'], f['candidates'])
            rec[solver] = dict(seconds=f['seconds'], speedup=o['seconds'] / f['seconds'], levels=levels,
                               overall=overall, rank_stats=f['stats'], candidates=f['candidates'])
            line[solver] = dict(overall=overall, levels=levels, s=round(f['seconds'], 1))
        tmp = dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(rec, indent=1, default=str))
        tmp.replace(dest)
        print(json.dumps(line), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'one':
        one(json.loads(sys.argv[2]), sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
