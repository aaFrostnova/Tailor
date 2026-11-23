"""One measurement of the solver-scaling experiment, in a fresh process (module patches are process-wide).

  grid        C2 grid search over the n fragments' knots (grid_n.GridSearchN), FITTED mode: the solver's own
              model at the knots, so both methods search the same model. The clock covers what C2's clock covers
              (request pruning, ranking, the visit); the per-process tables are timed separately.
  smt         TAILOR's certified top-3 (watermark_smt_topk.enumerate_topk, booted solver) on the same model.
  grid_frozen, smt_frozen   the unmodified grid / solver (n = 3 only), for the identity check.
  pincheck    n >= 4: the solver's model pinned to on-knot configurations agrees with the grid's evaluation
              (feasibility, distortion), as experiments/tests/test_grid_solver_equivalence.py checks for n = 3.

Usage: TAILOR_MAIN_CONTROLS_RUN=RUN python run_scaling.py N CASE_JSON METHOD OUT_JSON
CASE_JSON: {"id": ..., "path": checkpoint q-file with attacks/fpr/min_psnr/max_ms/rate_margin/mean_margin}
"""
import datetime
import json
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / 'main_controls')]


def _atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def _cand_smt(c):
    cfg = c['cfg']
    return dict(rank=c['rank'], order=list(cfg['order']), fe_on=sorted(cfg['fe_on']), s=cfg['s'],
                value=c['psnr_proxy'], psnr_db=c['psnr_db'], certified=c['certified'])


def _cand_grid(c):
    return dict(rank=c['rank'], order=c['order'], fe_on=c['fe_on'], s=c['s'], D=c['D_pred'], psnr_db=c['psnr_db'])


def run(n, case, method, out):
    from grid import frozen
    ns = frozen.boot()
    q = json.loads(Path(case['path']).read_text())
    req = dict(attacks=q['attacks'], fpr=q['fpr'], min_psnr=q['min_psnr'], max_ms=q['max_ms'])
    rate, mean = float(q['rate_margin']), float(q['mean_margin'])
    rec = dict(schema='tailor_exp_scaling_v1', n=n, case=case, method=method, request=req, rate_margin=rate,
               mean_margin=mean, host=platform.node(), job=os.environ.get('SLURM_JOB_ID'),
               started=datetime.datetime.now(datetime.timezone.utc).isoformat())
    frozen_method = method.endswith('_frozen')
    if frozen_method:
        if n != 3:
            raise ValueError('the frozen methods exist for n = 3 only')
        sg, _ = frozen.load_surrogate(ns)
    else:
        from scaling import virtual as V
        V.install(ns, n)
        sg, _ = V.load(ns, n)
    rec['fragments'] = list(sg.fragments)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    progress = Path(str(out) + '.progress.jsonl')
    if method in ('grid', 'grid_frozen', 'pincheck'):
        if frozen_method:
            from grid.grid_enumerator import GridSearch as GS
        else:
            from scaling import grid_n as G
            G.install(n)
            GS = G.GridSearchN
        from grid.tables import GridTables
        t0 = time.perf_counter()
        tables = GridTables(sg, mode='fitted', ns=ns, sha='virtual' if not frozen_method else None)
        rec['tables_s'] = time.perf_counter() - t0
        if method == 'pincheck':
            rec.update(pincheck(ns, sg, tables, GS, req, rate, mean, seed=int(case.get('seed', 11))))
        else:
            t0 = time.perf_counter()
            gs = GS(tables, req, rate_margin=rate, mean_margin=mean, t0=t0)
            res = gs.run(k=3)
            rec.update(seconds=res['total_s'], setup_s=res['setup_s'], search_s=res['search_s'], status=res['status'],
                       n_space=res['n_space'], n_ranked=res['n_ranked'], pruned=res['pruned'],
                       examined=res['examined'], evaluated=res['evaluated'],
                       first_feasible_s=res['first_feasible_s'],
                       candidates=[_cand_grid(c) for c in res['candidates']])
    elif method in ('smt', 'smt_frozen'):
        T = ns.T
        original = T.solve_exact_blocked
        t_start = time.perf_counter()

        def timed(*args, **kw):                        # per-rank progress, so a timed-out run still reports
            t = time.perf_counter()
            result = original(*args, **kw)
            with progress.open('a') as fh:
                fh.write(json.dumps(dict(rank_seconds=time.perf_counter() - t, elapsed=time.perf_counter() - t_start,
                                         value=result[2], certified=result[3])) + '\n')
            return result
        T.solve_exact_blocked = timed
        scen = ns.CP.solver_scenario(req, margin=mean)
        kw = dict(enable_order=True, continuous_strength=True, surrogate=sg, rate_margin=rate)
        t0 = time.perf_counter()
        cands = T.enumerate_topk(scen, k=3, no_passengers=True, **kw)
        rec['seconds'] = time.perf_counter() - t0
        rec['status'] = 'sat' if cands else 'model_unsat'
        rec['candidates'] = [_cand_smt(c) for c in cands]
        rec['booted_pin_step'] = bool(getattr(T, '_c4_tiebreak_repaired', False))
    else:
        raise ValueError(method)
    rec['maxrss_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    rec['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _atomic(out, rec)
    print(json.dumps({k: rec.get(k) for k in ('n', 'method', 'status', 'seconds', 'maxrss_gb')}), flush=True)


def pincheck(ns, sg, tables, GS, req, rate, mean, seed=11, n_feasible=10, n_random=20):
    """Pin the solver's model to on-knot configurations; it must agree with the grid (feasible, D, latency)."""
    import numpy as np
    import z3
    from grid import grid_enumerator as GE
    W, T, CP = ns.W, ns.T, ns.CP
    scen = CP.solver_scenario(req, margin=mean)
    built = W.build(**scen, enable_order=True, continuous_strength=True, surrogate=sg, rate_margin=rate)
    solver = z3.Solver()
    solver.add(built[0].assertions())
    solver.add(T.no_passenger_clause(built, scen['attacks']))
    gs = GS(tables, req, rate_margin=rate, mean_margin=mean)
    F = GE.FRAGMENTS
    assert gs.fe_resp == built[0]._fe_responsible, (gs.fe_resp, built[0]._fe_responsible)
    assert {f: gs.lo[f] for f in F} == built[0]._clean_floors
    rng = random.Random(seed)
    comparable = [s for s in GE.structures(fragments=F) if s.stage is None or s.stage in gs.fe_gain]
    configs = []
    feasible = np.flatnonzero(gs.evaluate_positions(0, min(gs.n_ranked, 2_000_000)))
    for pos in rng.sample(list(feasible), min(n_feasible, len(feasible))):
        s = gs.structures[int(gs.c_sid[pos])]
        kn = gs.c_kn[pos]
        configs.append((s, {f: tables.knots[f][int(kn[F.index(f)])] for f in s.order}))
    for _ in range(n_random):
        s = rng.choice(comparable)
        configs.append((s, {f: rng.choice(tables.knots[f]) for f in s.order}))
    agree = disagree = sat_seen = 0
    issues = []
    for s, strengths in configs:
        ev = gs.evaluate(s.order, s.stage, strengths, details=False)
        o, u = built[0], built[1]
        solver.push()
        for f in F:
            solver.add(u[f] == z3.BoolVal(f in s.order))
        for f in s.order:
            solver.add(o._svars[f] == z3.RealVal(strengths[f]))
        pos_ = {f: i for i, f in enumerate(s.order)}
        for (f, g), v in o._pvars.items():
            if f in pos_ and g in pos_:
                solver.add(v == z3.BoolVal(pos_[f] < pos_[g]))
        for name, v in o._fevars.items():
            solver.add(v == z3.BoolVal(name == s.stage))
        res = solver.check()
        ok = res == z3.sat
        D = -float(solver.model().eval(built[6]).as_fraction()) if ok else None
        solver.pop()
        same = ok == ev['feasible'] and (not ok or abs(D - ev['D_pred']) <= 1e-9 * max(1.0, D))
        sat_seen += ok
        if same:
            agree += 1
        else:
            disagree += 1
            issues.append(dict(structure=repr(s), strengths=strengths, smt_sat=ok, smt_D=D,
                               grid_feasible=ev['feasible'], grid_D=ev['D_pred'], violations=ev['violations']))
    return dict(pincheck=dict(configs=len(configs), agree=agree, disagree=disagree, smt_sat=sat_seen,
                              issues=issues[:10]), status='checked')


if __name__ == '__main__':
    run(int(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3], sys.argv[4])
