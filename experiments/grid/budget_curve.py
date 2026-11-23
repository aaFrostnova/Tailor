"""PR2 budget-matched comparison: grid baseline C2 versus TAILOR-F on the 7,321 cohort.

Offline time per request:
  C2        exhaustive grid search (run_budgeted_search.py rows, budget inf), summed over the margin stages a
            request walks through (initial, 04, 02, 00) up to the stage it was accepted at; all four stages
            for model-UNSAT and live-exhausted requests. Budget-limited rows (0.01/0.03/0.1/0.3 s) give the
            model-level curve below the exhaustive time.
  TAILOR-F  the frozen initial-stage SMT top-3 solve: fixed/checkpoints/<cls>_06/q<source_i>.json `sec` for the
            original requests, editing_request_extension proposals Full/06/<spec>/initial.json `seconds` for
            the editing requests. Later-stage solves (request-local correction, backoff margins) were not kept,
            so a request accepted after the initial stage gets its initial time as a LOWER bound; this favours
            TAILOR-F in the curve.
Satisfaction at budget B = accepted live with offline time <= B. Live acceptance: TAILOR-F from the frozen
record chain (gate SMT06 index); C2 from the C2 live run (RUNL/results.json); C2 requests still pending live
are reported as a bracket (counted as failures / as successes).

Usage: python experiments/grid/budget_curve.py RUNG RUNL MAIN_RUN
Outputs: RUNL/analysis/{search_budget_curve.csv, search_budget_curve.pdf, grid_search_summary.json}
"""
import collections
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'experiments/main_controls'), str(REPO / 'experiments')]
from tailor_exp import gate  # noqa: E402
import rejudge  # noqa: E402

FROZEN = Path(rejudge.FROZEN)
STAGES = ('initial', '04', '02', '00')
FINITE = (0.01, 0.03, 0.1, 0.3)
BUDGETS = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 60, 120, 300, 600, 1200, 3000, math.inf)


def read(p):
    return json.loads(Path(p).read_text())


def tailor_records():
    inputs = gate._Inputs(REPO, rejudge.FROZEN)
    frozen = inputs.read('paper', gate.COHORT_FILE)
    inputs.pin_run_inputs(frozen['source_sha256'])
    manifest = {r['spec_id']: r for r in gate.cohort.main_cohort_rows(REPO)}
    index = gate._index('SMT06', gate._assemble(inputs)['SMT06'], manifest)
    return manifest, index


def tailor_stage(rec, fb):
    """'initial' when TAILOR-F was accepted from its initial top-3, else 'later' (time is a lower bound)."""
    if rec.get('cohort') == 'editing':
        return 'initial' if rec.get('stage') == 'initial' else 'later'
    if rec.get('previous_status') == 'fixed_live_pass' and fb is not None and not fb.get('attempts'):
        return 'initial'
    return 'later'


def tailor_times(requests):
    out = {}
    fb = {}
    for cls in ('C1', 'C2', 'C3', 'C4', 'C5'):
        for r in read(FROZEN / 'fallback/requests' / f'{cls}_06.json')['rows']:
            fb[(cls, str(r['i']))] = r
    for r in requests:
        if r['source_cohort'] == 'original':
            q = read(FROZEN / 'fixed/checkpoints' / f"{r['cls']}_06" / f"q{int(r['source_i']):04d}.json")
            rq = r['request']
            if (sorted(q['attacks']) != sorted(rq['attacks']) or q['min_psnr'] != rq['min_psnr']
                    or q['max_ms'] != rq['max_ms'] or abs(q['fpr'] - rq['fpr']) > 1e-15):
                raise SystemExit(f"checkpoint does not match request {r['spec_id']}")
            out[r['spec_id']] = dict(sec=float(q['sec']), outcome=q['solver_outcome'],
                                     fb=fb.get((r['cls'], str(r['source_i']))))
        else:
            p = FROZEN / 'editing_request_extension_20260921/proposals/editing/Full/06' / r['spec_id'] / 'initial.json'
            q = read(p)
            out[r['spec_id']] = dict(sec=float(q['seconds']), outcome=q['outcome'], fb=None)
    return out


def c2_offline(rung):
    exh, fin = {}, collections.defaultdict(dict)
    for p in sorted((Path(rung) / 'offline').glob('*.jsonl')):
        for line in open(p):
            r = json.loads(line)
            key = (r['spec_id'], r['margin_stage'])
            if r.get('budget_s') is None:
                exh[key] = r
            else:
                fin[key][float(r['budget_s'])] = r
    return exh, fin


def frac(xs, n):
    return sum(xs) / n if n else None


def main(rung, runl, main_run):
    rung, runl, main_run = Path(rung), Path(runl), Path(main_run)
    requests = read(main_run / 'inputs/requests.json')
    manifest, tindex = tailor_records()
    if {r['spec_id'] for r in requests} != set(manifest):
        raise SystemExit('request file and cohort manifest disagree')
    ttime = tailor_times(requests)
    exh, fin = c2_offline(rung)
    live = {r['spec_id']: r for r in read(runl / 'results.json')['rows']}
    n = len(requests)
    per = []
    for r in requests:
        sid = r['spec_id']
        t = tindex[sid]
        tt = ttime[sid]
        c = live[sid]
        c_status = c['status']
        stage = c.get('stage') if c_status == 'accepted' else None
        upto = STAGES[:STAGES.index(stage) + 1] if stage else STAGES
        c_time = sum(exh[(sid, s)]['total_s'] for s in upto)
        c_first = exh[(sid, 'initial')].get('first_feasible_s')
        t_acc = t['_status'] == 'accepted'
        per.append(dict(
            spec_id=sid, cls=r['cls'], source_cohort=r['source_cohort'],
            tailor_status=t['_status'], tailor_accepted=t_acc, tailor_seconds=tt['sec'],
            tailor_initial_outcome=tt['outcome'],
            tailor_time_kind=('exact_initial' if t_acc and tailor_stage(t, tt['fb']) == 'initial'
                              else 'initial_lower_bound'),
            tailor_psnr=gate._psnr(t) if t_acc else None,
            c2_status=c_status, c2_stage=stage, c2_seconds=c_time, c2_first_feasible_s=c_first,
            c2_initial_status=exh[(sid, 'initial')]['status'],
            c2_examined=exh[(sid, 'initial')].get('examined'), c2_evaluated=exh[(sid, 'initial')].get('evaluated'),
            c2_space=exh[(sid, 'initial')].get('n_space'),
            c2_finite={str(b): fin[(sid, 'initial')][b]['status'] for b in FINITE if b in fin[(sid, 'initial')]},
            c2_psnr=(c.get('verdict') or {}).get('psnr_live') if c_status == 'accepted' else None,
            c2_walk=len(c.get('walk') or []), tailor_walk=len(t.get('walk') or []) if isinstance(t.get('walk'), list) else None))
    curve = []
    pending = sum(p['c2_status'].startswith('pending') for p in per)
    for b in BUDGETS:
        t_sat = frac([p['tailor_accepted'] and p['tailor_seconds'] <= b for p in per], n)
        t_model = frac([p['tailor_initial_outcome'] == 'sat' and p['tailor_seconds'] <= b for p in per], n)
        c_acc = [p['c2_status'] == 'accepted' and p['c2_seconds'] <= b for p in per]
        c_pend = [p['c2_status'].startswith('pending') and p['c2_seconds'] <= b for p in per]
        if b in FINITE:
            c_model = frac([p['c2_finite'].get(str(b)) == 'sat' for p in per], n)
        else:
            c_model = frac([p['c2_initial_status'] == 'sat' and exh[(p['spec_id'], 'initial')]['total_s'] <= b
                            for p in per], n)
        curve.append(dict(budget_s=b, tailor_live_satisfaction=t_sat, tailor_initial_model_sat=t_model,
                          c2_live_satisfaction_low=frac(c_acc, n),
                          c2_live_satisfaction_high=frac([a or q for a, q in zip(c_acc, c_pend)], n),
                          c2_initial_model_sat=c_model))
    joint = [p for p in per if p['tailor_accepted'] and p['c2_status'] == 'accepted' and p['c2_psnr'] is not None]
    dpsnr = [p['c2_psnr'] - p['tailor_psnr'] for p in joint]

    def dist(xs):
        xs = sorted(xs)
        if not xs:
            return None
        q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
        return dict(n=len(xs), median=statistics.median(xs), mean=statistics.mean(xs), p90=q(0.9), max=xs[-1],
                    min=xs[0], total_hours=sum(xs) / 3600)

    by_cls = {}
    for cls in sorted({p['cls'] for p in per}):
        part = [p for p in per if p['cls'] == cls]
        by_cls[cls] = dict(n=len(part),
                           tailor_accepted=sum(p['tailor_accepted'] for p in part),
                           c2_accepted=sum(p['c2_status'] == 'accepted' for p in part),
                           c2_pending=sum(p['c2_status'].startswith('pending') for p in part),
                           tailor_seconds=dist([p['tailor_seconds'] for p in part]),
                           c2_seconds=dist([p['c2_seconds'] for p in part]))
    summary = dict(
        policy='exp_c2_budget_v1', n=n, c2_pending_live=pending, complete=pending == 0,
        timing_note=('TAILOR-F = frozen initial SMT top-3 solve; later-stage solves were not kept, so requests '
                     'accepted after the initial stage carry a lower bound. C2 = exhaustive grid time summed '
                     'over the margin stages walked.'),
        tailor=dict(accepted=sum(p['tailor_accepted'] for p in per),
                    status=dict(collections.Counter(p['tailor_status'] for p in per)),
                    time_kind=dict(collections.Counter(p['tailor_time_kind'] for p in per)),
                    seconds=dist([p['tailor_seconds'] for p in per])),
        c2=dict(accepted=sum(p['c2_status'] == 'accepted' for p in per),
                status=dict(collections.Counter(p['c2_status'] for p in per)),
                accepted_stage=dict(collections.Counter(p['c2_stage'] for p in per if p['c2_stage'])),
                seconds=dist([p['c2_seconds'] for p in per]),
                first_feasible_s=dist([p['c2_first_feasible_s'] for p in per if p['c2_first_feasible_s'] is not None]),
                examined_initial=dist([p['c2_examined'] for p in per if p['c2_examined'] is not None]),
                evaluated_initial=dist([p['c2_evaluated'] for p in per if p['c2_evaluated'] is not None]),
                space=sorted({p['c2_space'] for p in per if p['c2_space'] is not None}),
                live_walk=dist([p['c2_walk'] for p in per if p['c2_status'] != 'model_unsat'])),
        speed_ratio_median=(statistics.median(p['tailor_seconds'] for p in per)
                            / statistics.median(p['c2_seconds'] for p in per)),
        paired_psnr=dict(joint=len(joint), mean_c2_minus_tailor=statistics.mean(dpsnr) if dpsnr else None,
                         median=statistics.median(dpsnr) if dpsnr else None,
                         c2_higher=sum(d > 0 for d in dpsnr), tailor_higher=sum(d < 0 for d in dpsnr)),
        by_scenario=by_cls, curve=curve)
    out = runl / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'grid_search_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    with open(out / 'grid_search_per_request.jsonl', 'w') as fh:
        for p in per:
            fh.write(json.dumps(p) + '\n')
    cols = list(curve[0])
    with open(out / 'search_budget_curve.csv', 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for row in curve:
            fh.write(','.join('inf' if v == math.inf else f'{v:.6g}' for v in (row[c] for c in cols)) + '\n')
    plot(curve, summary, out / 'search_budget_curve.pdf')
    print(json.dumps({k: summary[k] for k in ('n', 'c2_pending_live', 'speed_ratio_median', 'paired_psnr')}, indent=1))
    print('TAILOR-F seconds', summary['tailor']['seconds'])
    print('C2 seconds', summary['c2']['seconds'])
    print('TAILOR-F time kinds', summary['tailor']['time_kind'])
    for row in curve:
        print(f"B={row['budget_s']:>7}: TAILOR-F live {row['tailor_live_satisfaction']:.3f} "
              f"(model {row['tailor_initial_model_sat']:.3f}) | C2 live [{row['c2_live_satisfaction_low']:.3f}, "
              f"{row['c2_live_satisfaction_high']:.3f}] (model {row['c2_initial_model_sat']:.3f})")


def plot(curve, summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blue, orange, ink, grid = '#2a78d6', '#eb6834', '#3a3a38', '#e6e5e0'
    xs = [r['budget_s'] for r in curve if r['budget_s'] != math.inf]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    t = [r['tailor_live_satisfaction'] * 100 for r in curve if r['budget_s'] != math.inf]
    lo = [r['c2_live_satisfaction_low'] * 100 for r in curve if r['budget_s'] != math.inf]
    hi = [r['c2_live_satisfaction_high'] * 100 for r in curve if r['budget_s'] != math.inf]
    ax.step(xs, t, where='post', color=blue, lw=2, label='TAILOR-F (initial SMT solve time)')
    ax.step(xs, lo, where='post', color=orange, lw=2, ls='--', label='Grid C2 (exhaustive, measured knots)')
    if any(h != l for h, l in zip(hi, lo)):
        ax.fill_between(xs, lo, hi, step='post', color=orange, alpha=0.15, lw=0, label='C2 requests still pending live')
    ax.set_xscale('log')
    ax.set_ylim(0, 100)
    ax.set_xlabel('offline search budget per request (s)', fontsize=9, color=ink)
    ax.set_ylabel('requests satisfied live (%)', fontsize=9, color=ink)
    ax.grid(True, color=grid, lw=0.8)
    ax.tick_params(colors=ink, labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == '__main__':
    main(*sys.argv[1:4])
