"""Summarize the solver-scaling experiment: time and memory per method and fragment count, agreement, crossover.

Usage: python experiments/scaling/analyze_scaling.py RUN_DIR   (RUN_DIR = runs/scaling_<date>)
Writes RUN_DIR/analysis/{scaling_summary.json, scaling_rows.csv, scaling_time.pdf}. Tasks without a result are
listed with their SLURM state (TIMEOUT, OUT_OF_MEMORY, ...) and, for the solver, the per-rank progress lines.
"""
import collections
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    i = (len(xs) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def cq(xs, p):
    """Quantile where unfinished tasks are +inf (slower than every finished one); None when it lands on one."""
    xs = sorted(xs)
    if not xs:
        return None
    i = (len(xs) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    a, b, frac = xs[lo], xs[hi], i - lo
    v = a if frac == 0 else (math.inf if math.isinf(b) else a + (b - a) * frac)
    return None if math.isinf(v) else v


def elapsed_s(text):
    """sacct Elapsed ([D-]HH:MM:SS) in seconds."""
    days, _, hms = text.rpartition('-')
    h, m, sec = (int(x) for x in hms.split(':'))
    return (int(days) if days else 0) * 86400 + h * 3600 + m * 60 + sec


def slurm_states(run):
    """{(method, n, case id): last SLURM state} from the job ids in jobs.txt and the task lists."""
    out = {}
    jobs = {}
    for line in (run / 'jobs.txt').read_text().split('\n') if (run / 'jobs.txt').exists() else []:
        for item in line.split():
            if '=' in item:
                k, v = item.split('=', 1)
                jobs[k] = v
    names = {'smt3': 'smt_n3', 'smt3_retry': 'smt_n3', 'smt4_retry': 'smt_n4', 'smt4': 'smt_n4', 'smt5': 'smt_n5', 'smt6': 'smt_n6', 'smt7': 'smt_n7', 'grid7': 'grid_n7', 'grid34': 'grid_n34',
             'grid5': 'grid_n5', 'grid6': 'grid_n6', 'pin4': 'pincheck_n4'}
    for key, job in jobs.items():
        tasks_file = run / 'tasks' / f'{names.get(key, key)}.json'
        if not tasks_file.exists():
            continue
        tasks = json.loads(tasks_file.read_text())
        try:
            text = subprocess.check_output(['sacct', '-j', job, '-X', '-n', '-P', '-o', 'JobID,State,Elapsed,MaxRSS'],
                                           text=True, timeout=60)
        except Exception:
            continue
        for row in text.strip().split('\n'):
            parts = row.split('|')
            if len(parts) < 3 or '_' not in parts[0]:
                continue
            idx = parts[0].split('_', 1)[1]
            if not idx.isdigit() or int(idx) >= len(tasks):
                continue
            t = tasks[int(idx)]
            out[(t['method'], t['n'], t['case']['id'])] = dict(state=parts[1].split()[0], elapsed=parts[2])
    return out


def main(run):
    run = Path(run)
    rows, missing = [], []
    states = slurm_states(run)
    cases = [c['id'] for c in json.loads((run / 'cases.json').read_text())]
    for method in ('grid', 'smt', 'pincheck'):
        for n in range(3, 8):
            for cid in cases:
                p = run / method / f'n{n}_{cid}.json'
                if p.exists():
                    rows.append(json.loads(p.read_text()))
                elif (method, n, cid) in states:
                    prog = Path(str(p) + '.progress.jsonl')
                    lines = [json.loads(l) for l in prog.read_text().split('\n') if l.strip()] if prog.exists() else []
                    missing.append(dict(method=method, n=n, case=cid, **states[(method, n, cid)],
                                        ranks_done=len(lines), elapsed_at_last_rank=lines[-1]['elapsed'] if lines else None))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['method'], r['n'])].append(r)
    summary = {}
    for (method, n), rs in sorted(by.items()):
        if method == 'pincheck':
            summary[f'pincheck_n{n}'] = dict(requests=len(rs), configs=sum(r['pincheck']['configs'] for r in rs),
                                            agree=sum(r['pincheck']['agree'] for r in rs),
                                            disagree=sum(r['pincheck']['disagree'] for r in rs))
            continue
        t = [r['seconds'] for r in rs]
        summary[f'{method}_n{n}'] = dict(
            done=len(rs), median_s=q(t, .5), p25_s=q(t, .25), p75_s=q(t, .75), max_s=max(t), mean_s=statistics.mean(t),
            maxrss_gb_max=max(r['maxrss_gb'] for r in rs), status=dict(collections.Counter(r['status'] for r in rs)),
            n_space=rs[0].get('n_space'),
            n_ranked_median=q([r['n_ranked'] for r in rs], .5) if method == 'grid' else None)
    for (method, n), ms in sorted(collections.Counter((m['method'], m['n']) for m in missing).items()):
        summary.setdefault(f'{method}_n{n}', {})['not_finished'] = dict(collections.Counter(
            m['state'] for m in missing if m['method'] == method and m['n'] == n))
    # censoring: a task without a result (out of memory, still running, timed out) is slower than every finished one
    for method in ('grid', 'smt'):
        for n in range(3, 8):
            done = [r['seconds'] for r in rows if r['method'] == method and r['n'] == n]
            unf = [m for m in missing if m['method'] == method and m['n'] == n]
            if not done and not unf:
                continue
            allv = done + [math.inf] * len(unf)
            running = [elapsed_s(m['elapsed']) for m in unf if m['state'] == 'RUNNING' and m.get('elapsed')]
            summary.setdefault(f'{method}_n{n}', {}).update(
                tasks=len(allv), median_s_censored=cq(allv, .5), p25_s_censored=cq(allv, .25),
                p75_s_censored=cq(allv, .75), running_lower_bound_s=max(running) if running else None)
    # paired: same request, both methods
    pairs = collections.defaultdict(list)
    res = {(r['method'], r['n'], r['case']['id']): r for r in rows}
    agreement = collections.Counter()
    for (method, n, cid), r in res.items():
        if method != 'smt' or ('grid', n, cid) not in res:
            continue
        g = res[('grid', n, cid)]
        pairs[n].append(r['seconds'] / max(g['seconds'], 1e-9))
        gs, ss = g['status'], r['status']
        if gs == 'sat' and ss != 'sat':
            agreement['VIOLATION grid sat, smt unsat'] += 1
        elif gs == 'sat':
            gb = g['candidates'][0]['psnr_db']
            sb = r['candidates'][0]['psnr_db']
            agreement['both sat'] += 1
            agreement['smt >= grid psnr' if sb >= gb - 1e-9 else 'VIOLATION smt < grid psnr'] += 1
        else:
            agreement[f'grid {gs}, smt {ss}'] += 1
    summary['paired_ratio_smt_over_grid'] = {f'n{n}': dict(pairs=len(v), median=q(v, .5), min=min(v), max=max(v))
                                             for n, v in sorted(pairs.items())}
    summary['agreement'] = dict(agreement)
    summary['not_finished'] = missing
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'scaling_summary.json').write_text(json.dumps(summary, indent=1, default=str))
    with (out / 'scaling_rows.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['method', 'n', 'case', 'status', 'seconds', 'maxrss_gb', 'n_space', 'n_ranked', 'examined',
                    'best_psnr_db'])
        for r in sorted(rows, key=lambda r: (r['method'], r['n'], r['case']['id'])):
            if r['method'] == 'pincheck':
                continue
            w.writerow([r['method'], r['n'], r['case']['id'], r['status'], r['seconds'], r['maxrss_gb'],
                        r.get('n_space'), r.get('n_ranked'), r.get('examined'),
                        r['candidates'][0]['psnr_db'] if r.get('candidates') else None])
    try:
        plot(summary, out / 'scaling_time.pdf')
    except Exception as exc:                          # the summary is the result; the figure is a convenience
        print('plot skipped:', exc)
    print(json.dumps({k: v for k, v in summary.items() if k != 'not_finished'}, indent=1, default=str))
    print('not finished:', len(missing))


def plot(summary, path):
    """Censored medians (unfinished tasks count as slowest) with the interquartile band; an unfinished method at n
    with no finite median is drawn as a lower bound (the longest elapsed time of its running tasks)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for method, label, color in (('grid', 'Grid over measured knots', '#4C72B0'), ('smt', 'TAILOR (SMT)', '#DD8452')):
        ns, med, lo, hi = [], [], [], []
        for n in range(3, 8):
            s = summary.get(f'{method}_n{n}') or {}
            m = s.get('median_s_censored')
            if m is not None:
                ns.append(n); med.append(m)
                lo.append(s.get('p25_s_censored') or m)
                hi.append(s.get('p75_s_censored') or m)
            elif s.get('running_lower_bound_s'):
                lb = s['running_lower_bound_s']
                ax.plot([n], [lb], marker='^', color=color, markersize=8, linestyle='none')
                ax.annotate(f'unfinished\n>{lb / 3600:.1f} h', (n, lb), textcoords='offset points', xytext=(-6, 8),
                            ha='right', fontsize=7, color=color)
            oom = (s.get('not_finished') or {}).get('OUT_OF_MEMORY')
            if oom and m is not None:
                ax.annotate(f'{oom}/{s.get("tasks")} >64 GB', (n, m), textcoords='offset points', xytext=(6, -10),
                            fontsize=7, color=color)
        # a point resting on fewer than 5 requests (a probe) is drawn open, joined by a dashed segment, and labelled
        tasks = {n: (summary.get(f'{method}_n{n}') or {}).get('tasks', 0) for n in ns}
        full = [i for i, n in enumerate(ns) if tasks[n] >= 5]
        if full:
            ax.plot([ns[i] for i in full], [med[i] for i in full], marker='o', color=color, label=label)
            ax.fill_between([ns[i] for i in full], [lo[i] for i in full], [hi[i] for i in full], color=color,
                            alpha=0.2, linewidth=0)
        for i, n in enumerate(ns):
            if tasks[n] >= 5:
                continue
            if i > 0:
                ax.plot([ns[i - 1], n], [med[i - 1], med[i]], linestyle='--', color=color)
            ax.plot([n], [med[i]], marker='o', markerfacecolor='white', color=color, linestyle='none')
            ax.annotate(f'{tasks[n]} request{"s" if tasks[n] > 1 else ""}\n(probe)', (n, med[i]),
                        textcoords='offset points', xytext=(-8, -2), ha='right', va='center', fontsize=7, color=color)
    ax.set_yscale('log')
    ax.set_xlabel('watermarks (3 measured, the rest virtual)')
    ax.set_ylabel('seconds per request (median)')
    ax.set_xticks(range(3, 8))
    ax.set_ylim(top=max(ax.get_ylim()[1], 1e5))
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'), dpi=200)


if __name__ == '__main__':
    main(sys.argv[1])
