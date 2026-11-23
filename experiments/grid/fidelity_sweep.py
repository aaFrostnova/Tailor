"""Fidelity sweep: share of requests served with at least q dB, TAILOR-F (SMT) versus the knot grid (C2).

A request counts as served at level q when its method accepted it in live validation and the deployed
configuration's live PSNR (gate._psnr, the mean over the 100 calibration images) is at least q. This is a property
of the deployed solutions, read from the existing live results; it is not a re-run under tightened PSNR floors
(a re-run could re-plan, e.g. fall back to a smaller margin, and would need new live measurements).

Usage: python experiments/grid/fidelity_sweep.py RUNL OUT_DIR
Inputs: RUNL/analysis/grid_search_per_request.jsonl (budget_curve.py).
Outputs: OUT_DIR/fidelity_sweep.{csv,json,pdf,png}
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def main(runl, out):
    runl, out = Path(runl), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (runl / 'analysis/grid_search_per_request.jsonl').read_text().splitlines()
            if line.strip()]
    n = len(rows)
    smt = np.array([r['tailor_psnr'] if r['tailor_accepted'] else -np.inf for r in rows], float)
    grid = np.array([r['c2_psnr'] if r['c2_status'] == 'accepted' else -np.inf for r in rows], float)
    qs = np.round(np.arange(30.0, 50.0001, 0.1), 1)
    served_smt = np.array([(smt >= q).mean() for q in qs])
    served_grid = np.array([(grid >= q).mean() for q in qs])
    only_smt = np.array([((smt >= q) & ~(grid >= q)).sum() for q in qs])
    only_grid = np.array([((grid >= q) & ~(smt >= q)).sum() for q in qs])
    gap = served_smt - served_grid
    i = int(np.argmax(gap))
    summary = dict(
        n_requests=n, definition='served at q = accepted in live validation and deployed live PSNR >= q dB',
        not_a_rerun='tightened floors are not re-planned; see module docstring',
        overall=dict(smt=float((smt > -np.inf).mean()), grid=float((grid > -np.inf).mean())),
        max_gap=dict(q_db=float(qs[i]), smt=float(served_smt[i]), grid=float(served_grid[i]),
                     gap_points=float(100 * gap[i]), only_smt=int(only_smt[i]), only_grid=int(only_grid[i])),
        grid_last_ahead_db=float(max(q for q, g in zip(qs, gap) if g < 0)) if (gap < 0).any() else None,
        at=dict((f'{q:.0f}', dict(smt=float(served_smt[k]), grid=float(served_grid[k]), only_smt=int(only_smt[k]),
                                 only_grid=int(only_grid[k])))
                for k, q in enumerate(qs) if abs(q - round(q)) < 1e-9 and 34 <= q <= 47))
    (out / 'fidelity_sweep.json').write_text(json.dumps(summary, indent=1) + '\n')
    with (out / 'fidelity_sweep.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['psnr_floor_db', 'served_smt', 'served_grid', 'only_smt', 'only_grid'])
        for k, q in enumerate(qs):
            w.writerow([f'{q:.1f}', f'{served_smt[k]:.6f}', f'{served_grid[k]:.6f}', int(only_smt[k]), int(only_grid[k])])
    floors = [r.get('min_psnr') for r in json.loads((runl.parent / 'main_controls_20261109/inputs/requests.json')
                                                    .read_text()) for r in [r['request']]]
    max_floor = max(f for f in floors if f is not None)
    summary['benchmark_max_floor_db'] = max_floor
    (out / 'fidelity_sweep.json').write_text(json.dumps(summary, indent=1) + '\n')
    plot(qs, served_smt, served_grid, summary, out, smt=smt, grid=grid, max_floor=max_floor)
    print(json.dumps({k: v for k, v in summary.items() if k != 'at'}, indent=1))
    for q, v in summary['at'].items():
        print(q, 'dB', 'SMT %.2f%%' % (100 * v['smt']), 'grid %.2f%%' % (100 * v['grid']), 'only SMT', v['only_smt'],
              'only grid', v['only_grid'])


def plot(qs, s, g, summary, out, smt=None, grid=None, max_floor=None):
    """Top: share of requests retained at each PSNR threshold, the gap shaded, the benchmark's highest floor marked.
    Bottom: per 1 dB threshold, requests retained only by TAILOR (up) and only by the grid (down).
    The axis wording follows the paper's Figure 8 caption ("retained at threshold q")."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blue, grey = '#2f62ad', '#8a8a8a'
    # Times-style text, as in the paper's other figures: Times New Roman where it is installed, otherwise
    # Nimbus Roman, the metric-compatible clone that the paper body itself is set in. Sizes are about one
    # point above the former sans-serif ones, because Times glyphs are smaller at equal point size.
    import matplotlib.font_manager as fm
    family = next((n for n in ('Times New Roman', 'Nimbus Roman')
                   if any(f.name == n for f in fm.fontManager.ttflist)), 'serif')
    plt.rcParams.update({'font.family': family, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    lo, hi = 35.0, 47.0
    m = (qs >= lo) & (qs <= hi)
    x = qs[m]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.4, 3.9), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    ax.fill_between(x, 100 * g[m], 100 * s[m], where=s[m] >= g[m], step='post', color=blue, alpha=0.28, linewidth=0)
    ax.step(x, 100 * g[m], where='post', color=grey, linewidth=1.3, label='Grid over measured knots')
    ax.step(x, 100 * s[m], where='post', color=blue, linewidth=1.6, label='TAILOR (SMT)')
    if max_floor is not None:
        ax.axvline(max_floor, color='#555555', linewidth=0.7, linestyle=(0, (3, 2)))
        ax.text(max_floor - 0.15, 4, 'highest floor\nin the benchmark', ha='right', va='bottom', fontsize=7.8,
                color='#555555')
    mg = summary['max_gap']
    ax.annotate('', xy=(mg['q_db'], 100 * mg['grid']), xytext=(mg['q_db'], 100 * mg['smt']),
                arrowprops=dict(arrowstyle='<->', color='#222222', linewidth=0.7, shrinkA=0, shrinkB=0))
    ax.annotate(f"+{mg['gap_points']:.1f} pts\n{100 * mg['smt']:.1f}% vs {100 * mg['grid']:.1f}%",
                xy=(mg['q_db'] + 0.05, 100 * (mg['smt'] + mg['grid']) / 2), xytext=(43.9, 52), fontsize=8, va='center',
                arrowprops=dict(arrowstyle='-', color='#444444', linewidth=0.5, shrinkA=1, shrinkB=1))
    ax.set_ylim(0, 100)
    ax.set_ylabel('Requests retained (%)')
    ax.legend(frameon=False, loc='upper right', fontsize=8.2)
    levels = np.arange(36, 47)
    only_s = np.array([int(((smt >= q) & ~(grid >= q)).sum()) for q in levels])
    only_g = np.array([int(((grid >= q) & ~(smt >= q)).sum()) for q in levels])
    bx.bar(levels, only_s, width=0.62, color=blue, label='Retained only by TAILOR')
    bx.bar(levels, -only_g, width=0.62, color=grey, label='Retained only by Grid')
    bx.axhline(0, color='#333333', linewidth=0.6)
    k = int(np.argmax(only_s))
    bx.text(levels[k], only_s[k] + 12, f'{only_s[k]}', ha='center', va='bottom', fontsize=8, color=blue)
    bx.text(levels[k], -only_g[k] - 12, f'{only_g[k]}', ha='center', va='top', fontsize=8, color='#555555')
    bx.set_ylabel('Requests')
    bx.set_xlabel('PSNR threshold (dB)')
    bx.set_xlim(lo, hi)
    bx.set_xticks(range(36, 47, 2))
    bx.set_ylim(-140, max(only_s) + 430)   # headroom so that the legend clears the label of the tallest bar
    bx.legend(frameon=False, loc='upper left', fontsize=8.2, borderaxespad=0.1)
    fig.align_ylabels((ax, bx))
    fig.tight_layout(h_pad=0.3)
    fig.savefig(out / 'fidelity_sweep.pdf')
    fig.savefig(out / 'fidelity_sweep.png', dpi=220)


if __name__ == '__main__':
    main(*sys.argv[1:3])
