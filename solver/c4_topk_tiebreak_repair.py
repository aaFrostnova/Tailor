"""C4 recovery override: exact front-end tie breaking with bounded SAT queries.

The original top-k solver's main objective, blocking clauses and certification
are unchanged. Only reconstruction of a model at the certified objective is
replaced. Four front-end Booleans need at most five ordinary SAT queries.
"""
import inspect
import z3


def model_with_fewest_frontends(built, lower_bound):
    opt, objective = built[0], built[6]
    constraints = list(opt.assertions()) + [objective >= lower_bound]
    frontends = list((getattr(opt, '_fevars', None) or {}).values())
    count = z3.Sum([z3.If(v, 1, 0) for v in frontends]) if frontends else z3.IntVal(0)
    solver = z3.Solver()
    solver.add(*constraints)
    for n in range(len(frontends) + 1):
        solver.push()
        solver.add(count <= n)
        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            assert all(z3.is_true(model.eval(c, model_completion=True)) for c in constraints)
            return model
        if result == z3.unknown:
            raise RuntimeError('front-end tie-break returned unknown: ' + solver.reason_unknown())
        solver.pop()
    raise RuntimeError('certified objective is infeasible even without a front-end limit')


def install():
    import watermark_smt_topk as module
    if getattr(module, '_c4_tiebreak_repaired', False):
        return
    source = inspect.getsource(module.solve_exact_blocked)
    old = '''    pinned = _inst()
    o, ps = pinned[0], pinned[6]
    o.add(ps >= best - eps)
    assert o.check() == z3.sat, "the certified optimum became unsatisfiable when pinned"
    fev = getattr(o, "_fevars", None)
    if fev:
        o.minimize(z3.Sum([z3.If(v, 1, 0) for v in fev.values()]))
        assert o.check() == z3.sat, "pinning the optimum and minimising front-ends became unsat"
    return pinned, o.model(), best, certified'''
    new = '''    pinned = _inst()
    model = _c4_model_with_fewest_frontends(pinned, best - eps)
    return pinned, model, best, certified'''
    assert source.count(old) == 1, 'top-k source changed; review the recovery override'
    module._c4_model_with_fewest_frontends = model_with_fewest_frontends
    exec(compile(source.replace(old, new), __file__, 'exec'), module.__dict__)
    module._c4_tiebreak_repaired = True


if __name__ == '__main__':
    import runpy
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.paths import PROJECT, WORKSPACE
    cf, sc = Path(PROJECT), Path(WORKSPACE)
    sys.path[:0] = [str(cf / 'solver'), str(cf), str(sc)]
    install()
    sys.argv[0] = str(sc / 'live_topk.py')
    runpy.run_path(sys.argv[0], run_name='__main__')
