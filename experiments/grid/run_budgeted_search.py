"""Run C2 (offline part) for a list of requests at given wall-clock budgets and margin stages; write JSONL.

One row per (request, margin stage, budget) with spec_id, status (sat / model_unsat / budget_exhausted), the
top-3 distinct structures (order, fe_on, s, psnr_pred, ms, threshold_count, max_tests, flags, ...) and timings.
CPU only, no GPU, no live measurement; an exhaustive request takes well under a second.

  PY=$TAILOR_PYTHON
  $PY -B experiments/grid/run_budgeted_search.py OUT.jsonl --spread 20 --budgets inf,0.05 --margins initial
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grid import frozen  # noqa: E402
from grid.grid_enumerator import INF, MARGIN_STAGES, SCHEMA, search  # noqa: E402
from grid.tables import GridTables  # noqa: E402


def code_sha256():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).resolve().parent.glob('*.py')):
        h.update(p.name.encode() + b'\0' + p.read_bytes())
    return h.hexdigest()


def parse_budgets(text):
    out = []
    for x in text.split(','):
        x = x.strip().lower()
        out.append(INF if x in ('inf', 'none', 'exhaustive') else float(x))
        if out[-1] < 0:
            raise ValueError('budgets must be non-negative')
    return out


def select(rows, args):
    if args.indices:
        wanted = [int(x) for x in args.indices.split(',')]
        by_i = {r['i']: r for r in rows}
        return [by_i[i] for i in wanted]
    if args.range:
        a, b = (int(x) for x in args.range.split(':'))
        return rows[a:b]
    if args.spread:
        n = args.spread
        return [rows[(j * len(rows)) // n] for j in range(n)]
    if args.spec_ids:
        src = Path(args.spec_ids)
        ids = json.loads(src.read_text()) if src.exists() else args.spec_ids.split(',')
        by_id = {r['spec_id']: r for r in rows}
        return [by_id[x] for x in ids]
    return rows


def compact(candidate, details):
    if not details:
        candidate = {k: v for k, v in candidate.items() if k != 'cells'}
    return candidate


def summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['margin_stage'], r['budget_s']), []).append(r)
    lines = []
    for (stage, budget), rs in sorted(groups.items(), key=lambda t: (t[0][0], INF if t[0][1] is None else t[0][1])):
        tot = sorted(r['total_s'] for r in rs)
        q = lambda p: tot[min(len(tot) - 1, int(math.ceil(p * len(tot))) - 1)]
        status = {}
        for r in rs:
            status[r['status']] = status.get(r['status'], 0) + 1
        ff = [r['first_feasible_s'] for r in rs if r['first_feasible_s'] is not None]
        lines.append(f'margin={stage} budget={budget}: n={len(rs)} status={status} '
                     f'total_s median={statistics.median(tot):.4f} p90={q(.9):.4f} max={tot[-1]:.4f} '
                     f'setup_s median={statistics.median(r["setup_s"] for r in rs):.4f} '
                     f'first_feasible_s median={statistics.median(ff) if ff else None} '
                     f'examined median={statistics.median(r["examined"] for r in rs)} '
                     f'complete={sum(r["complete"] for r in rs)} exhausted={sum(r["exhausted"] for r in rs)}')
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out', help='output JSONL path')
    ap.add_argument('--requests', default=None, help='requests.json (default RUN/inputs/requests.json)')
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument('--indices', help='comma-separated request ids (field i)')
    pick.add_argument('--range', help='row slice a:b')
    pick.add_argument('--spread', type=int, help='n evenly spaced rows')
    pick.add_argument('--spec-ids', help='JSON list file or comma-separated spec ids')
    ap.add_argument('--budgets', default='inf', help='comma-separated seconds; inf = exhaustive')
    ap.add_argument('--margins', default='initial', help=f'comma-separated margin stages from {MARGIN_STAGES}')
    ap.add_argument('--mode', choices=('raw', 'fitted'), default='raw')
    ap.add_argument('--rate-rule', choices=('raw', 'solver'), default=None)
    ap.add_argument('--raw-distortion', default=None, help='raw single-fragment distortion table (JSON)')
    ap.add_argument('--no-early-stop', action='store_true', help='visit every ranked candidate')
    ap.add_argument('--no-warmup', action='store_true')
    ap.add_argument('--details', action='store_true', help='keep per-cell details of returned candidates')
    ap.add_argument('--k', type=int, default=3)
    args = ap.parse_args(argv)

    budgets = parse_budgets(args.budgets)
    stages = [x.strip() for x in args.margins.split(',')]
    if any(s not in MARGIN_STAGES for s in stages):
        raise SystemExit(f'margin stages must be in {MARGIN_STAGES}')
    ns = frozen.boot()
    t_load = time.perf_counter()
    tables = GridTables(mode=args.mode, raw_distortion=args.raw_distortion, ns=ns)
    t_load = time.perf_counter() - t_load
    req_path = Path(args.requests) if args.requests else frozen.requests_path()
    req_bytes = req_path.read_bytes()
    rows = select(json.loads(req_bytes), args)
    meta = dict(schema=SCHEMA, requests_path=str(req_path), requests_sha256=hashlib.sha256(req_bytes).hexdigest(),
                surrogate_sha256=tables.surrogate_sha256, raw_distortion_sha256=tables.raw_distortion_sha256,
                distortion_source='raw_table' if tables.raw_distortion else 'fitted_curves',
                grid_code_sha256=code_sha256(), knots={f: list(v) for f, v in tables.knots.items()},
                stop_when_complete=not args.no_early_stop, arm=ns.arm)
    if rows and not args.no_warmup:                       # first-call overheads are not part of any request
        search(tables, rows[0]['request'], margin_stage=stages[0], rate_rule=args.rate_rule)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + f'.partial{os.getpid()}')
    written = []
    with partial.open('w') as fh:
        for r in rows:
            for stage in stages:
                for budget in budgets:
                    res = search(tables, r['request'], margin_stage=stage, budget_s=budget,
                                 stop_when_complete=not args.no_early_stop, k=args.k, rate_rule=args.rate_rule)
                    res['candidates'] = [compact(c, args.details) for c in res['candidates']]
                    row = dict(spec_id=r['spec_id'], i=r['i'], cls=r['cls'], request=r['request'], **res, **meta)
                    row['best_psnr_pred'] = res['candidates'][0]['psnr_pred'] if res['candidates'] else None
                    fh.write(json.dumps(row, sort_keys=True) + '\n')
                    written.append(row)
    os.replace(partial, out)
    print(f'tables built in {t_load:.2f}s; {len(written)} rows -> {out}')
    for line in summary(written):
        print(line)
    return written


if __name__ == '__main__':
    main()
