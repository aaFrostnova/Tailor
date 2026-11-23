"""Exact-output acceleration of TAILOR's certified top-k search (watermark_smt_topk.enumerate_topk).

Goal: the same candidates as the original, faster. Everything that defines the answer is the original code:
the instances (watermark_smt_topk._instance: the same build, no-passenger clause and blocking clauses), the
certification probes and the final pin step that picks the returned configuration. The original is the solver
as booted (campaign.boot): boot installs c4_topk_tiebreak_repair, which replaces the frozen source's Optimize
pin step by bounded SAT queries (_c4_model_with_fewest_frontends); every TAILOR run used the booted solver, so
the final step here calls that same function. One thing is added: before
the optimizer proposes a rank's optimum, its instance receives the bound ps >= LB, where LB is the objective of
a feasible configuration of that very instance (a seed, typically the grid's best on-knot configuration among
the unblocked skeletons, pinned and evaluated by z3 itself). A feasible configuration cannot beat the optimum,
so the bound removes no optimal configuration; it only spares the optimizer the region below the seed. The
certification loop and the final pin step are unchanged, so the returned configuration is the original's
whenever the certified value is the same float; verify_fast.py checks this request by request.

Usage (inside a process that has booted the frozen solver, as experiments/main_controls/common.py does):
    seeds = grid_seeds(tables, request, rate_margin=..., mean_margin=..., no_passengers=...)
    cands = enumerate_topk_fast(scen, k=3, seeds=seeds, enable_order=True, continuous_strength=True,
                                surrogate=sg, rate_margin=..., no_passengers=...)
"""
import time

import z3


def _frozen():
    import watermark_smt_topk as T
    import watermark_smt_v2 as W
    return T, W


def seed_value(scen, blocks, seed, no_passengers, build_kw):
    """Exact objective (z3 rational) of `seed` = (order, stage, strengths) in the blocked instance, or None when
    the seed is infeasible there (another skeleton is blocked, a variant restriction, offsets, ...)."""
    T, W = _frozen()
    built = T._instance(scen, blocks, (), build_kw, no_passengers)
    o, u = built[0], built[1]
    order, stage, strengths = seed
    s = z3.Solver()
    s.add(o.assertions())
    for f in W.FR:
        s.add(u[f] == z3.BoolVal(f in order))
    for f in order:
        s.add(o._svars[f] == z3.RealVal(strengths[f]))
    pos = {f: i for i, f in enumerate(order)}
    for (f, g), v in o._pvars.items():
        if f in pos and g in pos:
            s.add(v == z3.BoolVal(pos[f] < pos[g]))
    for name, v in (getattr(o, '_fevars', None) or {}).items():
        s.add(v == z3.BoolVal(name == stage))
    if s.check() != z3.sat:
        return None
    return s.model().eval(built[6])


def _blocked(seed, blocks):
    order, stage, _ = seed
    fe = [stage] if stage else []
    return any(list(b.order) == list(order) and sorted(b.fe) == fe for b in blocks)


def solve_exact_blocked_fast(scen, blocks=(), lb=None, eps=1e-9, max_rounds=32, no_passengers=True,
                             final='original', **build_kw):
    """watermark_smt_topk.solve_exact_blocked with the proposal instance carrying ps >= lb.

    final='original' then runs the booted original's final pin step (a fresh instance, ps >= best - eps, the
    fewest front-ends by bounded SAT queries). final='certified' returns the certified model itself when it
    switches on no front-end; otherwise it falls back to the original final step. Among configurations that tie
    exactly on the objective the pin step's choice is the one the original makes; the certified model may be
    another member of the tie set, so final='certified' is not output-preserving (verify_fast.py measures it)."""
    T, W = _frozen()
    _inst = lambda: T._instance(scen, blocks, (), build_kw, no_passengers)
    built = _inst()
    o, ps = built[0], built[6]
    if lb is not None:
        o.add(ps >= lb)                                  # attained by a feasible configuration: removes no optimum
    o.maximize(ps)
    if o.check() != z3.sat:
        if lb is not None:
            raise AssertionError('an instance with a feasible seed became unsatisfiable')
        return None, None, None, True
    m_best, inst_best = o.model(), built
    best = float(m_best.eval(ps).as_fraction())
    # ---- certification, line for line the original (watermark_smt_topk.py:127-135) ----
    certified = False
    for _ in range(max_rounds):
        probe = _inst()
        o2, ps2 = probe[0], probe[6]
        o2.add(ps2 > best + eps)
        if o2.check() != z3.sat:
            certified = True; break
        o2.maximize(ps2); o2.check()
        m_best, inst_best = o2.model(), probe
        best = float(m_best.eval(ps2).as_fraction())
    if final == 'certified' and certified:
        fev = getattr(inst_best[0], '_fevars', None) or {}
        if not any(z3.is_true(m_best.eval(v, model_completion=True)) for v in fev.values()):
            return inst_best, m_best, best, certified
    # ---- final pin step: the booted original's. campaign.boot installs c4_topk_tiebreak_repair, which replaces
    # the frozen source's Optimize pin step (watermark_smt_topk.py:136-144, kept below for an unpatched module)
    # by bounded SAT queries on a fresh instance; the call is the same function on the same arguments ----
    pinned = _inst()
    if getattr(T, '_c4_tiebreak_repaired', False):
        return pinned, T._c4_model_with_fewest_frontends(pinned, best - eps), best, certified
    o, ps = pinned[0], pinned[6]
    o.add(ps >= best - eps)
    assert o.check() == z3.sat, "the certified optimum became unsatisfiable when pinned"
    fev = getattr(o, "_fevars", None)
    if fev:
        o.minimize(z3.Sum([z3.If(v, 1, 0) for v in fev.values()]))
        assert o.check() == z3.sat, "pinning the optimum and minimising front-ends became unsat"
    return pinned, o.model(), best, certified


def enumerate_topk_fast(scen, k, eps=1e-9, max_rounds=32, no_passengers=True, seeds=(), stats=None,
                        final='certified', **build_kw):
    """watermark_smt_topk.enumerate_topk with seeded proposals; same arguments plus `seeds` (best first)."""
    T, W = _frozen()
    out, blocks = [], []
    for rank in range(1, int(k) + 1):
        t0 = time.perf_counter()
        lb, tried = None, 0
        for seed in seeds:
            if _blocked(seed, blocks):
                continue
            tried += 1
            lb = seed_value(scen, blocks, seed, no_passengers, build_kw)
            if lb is not None:
                break
        t1 = time.perf_counter()
        built, m, value, certified = solve_exact_blocked_fast(scen, blocks=blocks, lb=lb, eps=eps,
                                                              max_rounds=max_rounds, no_passengers=no_passengers,
                                                              final=final, **build_kw)
        if stats is not None:
            stats.append(dict(rank=rank, seeds_tried=tried, seeded=lb is not None, seed_s=t1 - t0,
                              solve_s=time.perf_counter() - t1,
                              lb=None if lb is None else float(lb.as_fraction()), value=value))
        if built is None:
            break
        c = T.candidate_of(scen, built, m, value, certified, rank)
        out.append(c)
        blocks.append(c["skeleton"])
    return out


def grid_seeds(tables, request, *, rate_margin, mean_margin, no_passengers=True, sizes=None):
    """Best on-knot configuration of every structure, best first, from the grid baseline in FITTED mode (the
    solver's own model at the knots, experiments/tests/test_grid_solver_equivalence.py). `sizes` keeps only skeletons
    with that many fragments (a variant restriction). Seeds are only candidates: seed_value re-evaluates them."""
    import numpy as np
    from grid.grid_enumerator import FRAGMENTS, GridSearch
    gs = GridSearch(tables, request, rate_margin=rate_margin, mean_margin=mean_margin, no_passengers=no_passengers)
    ok = np.flatnonzero(gs.evaluate_positions(0, gs.n_ranked))
    seen, seeds = set(), []
    for pos in ok:
        sid = int(gs.c_sid[pos])
        if sid in seen:
            continue
        seen.add(sid)
        st = gs.structures[sid]
        if sizes is not None and len(st.order) not in sizes:
            continue
        kn = gs.c_kn[pos]
        strengths = {f: tables.knots[f][int(kn[FRAGMENTS.index(f)])] for f in st.order}
        seeds.append((list(st.order), st.stage, strengths))
    return seeds
