"""Greedy fragment selection using the frozen SMT model, including its constraints.

Each greedy trial pins the fragment subset, then optimizes coverage, distortion,
order, one frontend, and continuous strengths in the SAME symbolic model. This
is a greedy-subset/solver-local hybrid, not a solver-free speed baseline.
Only attack-coverage requirements are temporarily relaxed to rank additions.
The emitted candidate must satisfy every original SMT constraint, including
the fragment-contribution restriction. No live/final observation is read here.
"""
from pathlib import Path
import functools
import itertools
import json
import os
import sys
import time

MAIN = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(MAIN / 'fixed/code'), str(MAIN / 'code')]
import campaign
campaign.environment('04')
import z3
import capacity_protocol as CP
import watermark_smt_v2 as W
import watermark_smt_topk as T
from surrogate_model import Surrogate
from live_calibration import read_config
from unified_detector import threshold_for_config, threshold_count, budget_for_config

VERSION = 'shared_table_greedy_subset_v1'
EPS = 1e-9


@functools.lru_cache(maxsize=1)
def surrogate():
    return Surrogate.from_dict(json.loads((MAIN / 'fixed/inputs/surrogate_canonical.json').read_text()))


def checked(solver):
    outcome = solver.check()
    if outcome == z3.unknown:
        raise RuntimeError('solver unresolved: ' + solver.reason_unknown())
    return outcome


def partial_model(request, rate_margin, mean_margin, sg=None):
    """Preserve all full-model assertions except one coverage OR per attack."""
    sg = surrogate() if sg is None else sg
    scenario = CP.solver_scenario(request, margin=mean_margin)
    built = W.build(**scenario, enable_order=True, continuous_strength=True,
                    surrogate=sg, rate_margin=rate_margin)
    opt, selected = built[:2]
    covers = {a: z3.Or(*[z3.And(selected[f], opt._cover[f, a])
                         for f in sg.fragments]) for a in request['attacks']}
    wanted = {v.sexpr(): a for a, v in covers.items()}
    retained, removed = [], []
    for clause in opt.assertions():
        if clause.sexpr() in wanted:
            removed.append(wanted[clause.sexpr()])
        else:
            retained.append(clause)
    if sorted(removed) != sorted(request['attacks']):
        raise RuntimeError('full-model coverage encoding changed; review adapter')
    retained.append(T.no_passenger_clause(built, request['attacks']))
    return built, retained, covers


def optimize_trial(built, retained, covers, subset, preserve=(), timeout_ms=120000):
    """Max coverage, then minimum distortion, with strict local certificates."""
    selected = built[1]
    constraints = list(retained)
    constraints += [v == (f in subset) for f, v in selected.items()]
    constraints += [covers[a] for a in preserve]
    score = z3.Sum([z3.If(v, 1, 0) for v in covers.values()])
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(*constraints)
    if checked(solver) == z3.unsat:
        return None
    lo, hi = 0, len(covers)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        solver.push()
        solver.add(score >= mid)
        feasible = checked(solver) == z3.sat
        solver.pop()
        if feasible:
            lo = mid
        else:
            hi = mid - 1
    constraints.append(score == lo)
    opt = z3.Optimize()
    opt.set(timeout=timeout_ms)
    opt.add(*constraints)
    objective = built[6]
    opt.maximize(objective)
    probe = z3.Solver()
    probe.set(timeout=timeout_ms)
    probe.add(*constraints)
    certified = False
    for _ in range(32):
        if checked(opt) != z3.sat:
            raise RuntimeError('a feasible greedy trial became infeasible')
        model = opt.model()
        value = model.eval(objective).as_fraction()
        probe.push()
        probe.add(objective > z3.RealVal(str(value)) + z3.RealVal(str(EPS)))
        better = checked(probe)
        probe.pop()
        if better == z3.unsat:
            certified = True
            break
        opt.add(objective > z3.RealVal(str(value)) + z3.RealVal(str(EPS)))
    if not certified:
        raise RuntimeError('uncertified local distortion optimum; not a negative result')
    # Deterministic structural ties, within the main campaign's objective tolerance.
    constraints.append(objective >= z3.RealVal(str(value)) - z3.RealVal(str(EPS)))
    tie = z3.Solver()
    tie.set(timeout=timeout_ms)
    tie.add(*constraints)
    fe_vars = built[0]._fevars
    for stages in [()] + [(x,) for x in ('resync', 'scale', 'angle', 'tile')]:
        tie.push()
        tie.add(*[v == (name in stages) for name, v in fe_vars.items()])
        if checked(tie) == z3.sat:
            for order in itertools.permutations([f for f in W.FR if f in subset]):
                tie.push()
                positions = {f: i for i, f in enumerate(order)}
                tie.add(*[v == (positions[f] < positions[g])
                          for (f, g), v in built[0]._pvars.items()
                          if f in positions and g in positions])
                if checked(tie) == z3.sat:
                    model = tie.model()
                    covered = [a for a, v in covers.items()
                               if z3.is_true(model.eval(v, model_completion=True))]
                    if len(covered) != lo or not set(preserve) <= set(covered):
                        raise RuntimeError('greedy coverage certificate mismatch')
                    cfg = read_config(built, model)
                    # All full recovery constraints must hold before emitting a candidate.
                    all_full = all(z3.is_true(model.eval(v, model_completion=True))
                                   for v in built[0].assertions())
                    return dict(cfg=cfg, covered=covered, coverage_count=lo,
                                objective=float(model.eval(objective).as_fraction()),
                                full_constraints_satisfied=all_full, local_optimum_certified=True)
                tie.pop()
        tie.pop()
    raise RuntimeError('local optimum could not be reconstructed')


def greedy_walk(fragments, attack_count, trial):
    """Injectable greedy control; stops are heuristic, never global UNSAT."""
    chosen, covered, trace = [], [], []
    while len(chosen) < len(fragments):
        options = []
        for f in fragments:
            if f in chosen:
                continue
            value = trial(tuple(chosen + [f]), tuple(covered))
            if value is not None:
                options.append((f, value))
        improving = [(f, v) for f, v in options if v['coverage_count'] > len(covered)]
        trace.append(dict(chosen_before=list(chosen), covered_before=list(covered),
                          trials=[dict(added_fragment=f, coverage=v['coverage_count'],
                                       objective=v['objective']) for f, v in options]))
        if not improving:
            return dict(status='greedy_stalled', candidate=None, trace=trace,
                        global_unsat_claimed=False)
        greatest = max(v['coverage_count'] for _, v in improving)
        winners = [(f, v) for f, v in improving if v['coverage_count'] == greatest]
        best_objective = max(v['objective'] for _, v in winners)
        winners = [(f, v) for f, v in winners if v['objective'] >= best_objective - EPS]
        f, best = min(winners, key=lambda x: (len(x[1]['cfg']['fe_on']), fragments.index(x[0])))
        chosen.append(f)
        covered = best['covered']
        trace[-1]['selected_fragment'] = f
        if len(covered) == attack_count:
            if not best['full_constraints_satisfied']:
                raise RuntimeError('candidate violates the original model')
            return dict(status='candidate_ready_for_live', candidate=best,
                        trace=trace, global_unsat_claimed=False)
    return dict(status='greedy_stalled', candidate=None, trace=trace,
                global_unsat_claimed=False)


def propose(request, arm='04', timeout_ms=120000):
    if arm not in ('04', '06'):
        raise ValueError('unsupported margin arm')
    if 'min_bits' in request:
        raise ValueError('payload is an output, not a request input')
    campaign.environment(arm)
    started = time.monotonic()
    built, constraints, covers = partial_model(request, int(arm) / 100, .02)
    result = greedy_walk(list(W.FR), len(request['attacks']),
        lambda subset, preserve: optimize_trial(built, constraints, covers, subset, preserve, timeout_ms))
    result.update(policy=VERSION, arm=arm, request={k:request[k] for k in CP.REQUEST_FIELDS},
                  rate_margin=int(arm)/100, mean_margin=.02, seconds=time.monotonic()-started,
                  same_smt_model=True, maximum_frontends=1, live_candidate_budget=1,
                  no_live_or_final_feedback=True, search_scope='greedy_subset_exact_local_parameters')
    if result['candidate'] is not None:
        cfg = result['candidate']['cfg']
        detection = dict(order=cfg['order'], fe_on=cfg['fe_on'])
        result['candidate'].update(tau=threshold_for_config(detection, request['fpr']),
            threshold_count=threshold_count(detection, request['fpr']),
            max_tests=budget_for_config(detection)['max_tests'], psnr_db=T.db(-cfg['psnr_proxy']))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-name', choices=['C1','C2','C3','C4','C5'], required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--arm', choices=['04','06'], default='04')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout-ms', type=int, default=120000)
    args = parser.parse_args()
    request = json.loads((MAIN / 'fixed/inputs' / (args.class_name + '.json')).read_text())['records'][args.index]
    result = propose(request, args.arm, args.timeout_ms)
    result.update(class_name=args.class_name, request_index=args.index)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text, flush=True)
