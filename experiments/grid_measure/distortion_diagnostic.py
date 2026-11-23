"""PR7 distortion diagnostic: the deployed distortion model against measured PSNR.

The deployed model (surrogate_canonical.json, used by the solver) predicts MSE = sum_f d_f(s_f) + sum_{f<g}
e_fg(s_f + s_g) (+ a geometric-stage term inside the solver), PSNR = 10 log10(255^2 / MSE). Three test sets:
  deployed   every configuration TAILOR-F (held-out manifest) or a main control accepted: the solver's predicted
             PSNR (candidate psnr_db) against the live PSNR measured on the 100 selection images;
  profiling  the 87 PR3/PR7 configurations (9 single, 54 ordered pairs, 24 ordered triples covering all six
             orders) measured with per-image records: the curve prediction against the measured mean-MSE PSNR,
             and, for triples, the pairwise model refit on the measured singles and pairs (fit/test split);
  appendix   the 15 database points behind the appendix figures (additivity blocks), as stored.
Points whose pair strength sum lies outside the measured range of e_fg, or whose strength lies outside d_f's
knots, are flagged as extrapolation.

Usage: TAILOR_MAIN_CONTROLS_RUN=RUN python distortion_diagnostic.py RUNI RUNH OUT_DIR
Outputs: OUT_DIR/distortion_diagnostic.csv, distortion_diagnostic.pdf, distortion_diagnostic.json
"""
import collections
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'main_controls')]
from common import CP, ROOT, RUN, cfgkey, read  # noqa: E402


def psnr(mse):
    return 10 * math.log10(255 ** 2 / mse)


class Curves:
    def __init__(self, path):
        d = read(path)
        self.d = {f: (np.array(v['xs']), np.array(v['ys'])) for f, v in d['d'].items()}
        self.e = {tuple(k.split('|')): (np.array(v['xs']), np.array(v['ys'])) for k, v in d['e'].items()}
        self.raw = d

    def pwl(self, xy, x):
        xs, ys = xy
        return float(np.interp(x, xs, ys)), bool(x < xs[0] - 1e-9 or x > xs[-1] + 1e-9)

    def predict(self, s):
        mse, extra = 0.0, []
        for f, v in s.items():
            y, out = self.pwl(self.d[f], v)
            mse += y
            if out:
                extra.append(f'd_{f}')
        for f, g in itertools.combinations(sorted(s), 2):
            y, out = self.pwl(self.e[(f, g)], s[f] + s[g])
            mse += y
            if out:
                extra.append(f'e_{f}|{g}')
        return psnr(mse), extra


def deployed():
    out = {}
    for r in read(RUN / 'results.json')['rows']:
        if r['status'] != 'accepted':
            continue
        c = r['candidate']
        k = CP.exact_cfg_id(cfgkey(c))
        out.setdefault(k, dict(source=r['variant'], order=c['order'], fe_on=sorted(c['fe_on']), s=c['s'],
                               predicted=float(c['psnr_db']), measured=float(r['verdict']['psnr_live'])))
    return out


def tailor(runh):
    out = {}
    for a in read(Path(runh) / 'manifest.json')['assignments']:
        c = a['candidate']
        out.setdefault(a['cfg_id'], dict(source='TAILOR-F', order=c['order'], fe_on=sorted(c['fe_on']), s=c['s'],
                                         predicted=float(c['psnr_db']), measured=float(a['calibration_psnr'])))
    return out


def profiling(runi, curves):
    rows = []
    recs = {}
    for folder in ('matrix', 'triples'):
        for p in sorted((Path(runi) / folder).glob('*.json')):
            d = read(p)
            recs[d['task']['name']] = d
    for name, d in recs.items():
        t = d['task']
        pred, extra = curves.predict(t['s'])
        kind = {1: 'single', 2: 'pair', 3: 'triple'}[len(t['order'])]
        rows.append(dict(set='profiling', source=f"profiling {kind}", name=name, order=t['order'], fe_on=[],
                         s=t['s'], predicted=pred, measured=psnr(d['mean_mse']), extrapolated=extra))
    # pairwise model refit on the measured singles and pairs (fit), applied to the triples (test)
    single, pair = {}, {}
    for n, r in recs.items():
        t = r['task']
        if len(t['order']) == 1:
            single[(t['order'][0], t['levels'][t['order'][0]])] = r['mean_mse']
        elif len(t['order']) == 2:
            f, g = t['order']
            pair[(f, t['levels'][f], g, t['levels'][g])] = r['mean_mse']
    refit = []
    for n, r in recs.items():
        t = r['task']
        if len(t['order']) != 3:
            continue
        o, L = t['order'], t['levels']
        mse = sum(single[(f, L[f])] for f in o)
        for i, f in enumerate(o):
            for g in o[i + 1:]:
                mse += pair[(f, L[f], g, L[g])] - single[(f, L[f])] - single[(g, L[g])]
        refit.append(dict(name=n, order=o, predicted=psnr(mse), measured=psnr(r['mean_mse'])))
    return rows, refit


def appendix(curves):
    rows = []
    # the five strength blocks x three cyclic orders (the top-level 'additivity' block repeats 'mixed')
    blocks = [v for v in curves.raw['additivity_multi'].values() if 'per_perm' in v]
    for b in blocks:
        for p in b['per_perm']:
            rows.append(dict(set='appendix', source=f"appendix {b['label']}", name=b['label'], order=p['perm'], fe_on=[],
                             s=b['strengths'], predicted=b['pred_psnr'], measured=p['true_psnr'], extrapolated=[],
                             stored_dpsnr=p['dpsnr']))
    return rows


def stats(rows):
    err = [r['predicted'] - r['measured'] for r in rows]
    if not err:
        return None
    a = [abs(x) for x in err]
    return dict(n=len(err), mean=statistics.mean(err), mean_abs=statistics.mean(a), max_abs=max(a),
                p95_abs=sorted(a)[min(len(a) - 1, int(0.95 * len(a)))], over_0_3=sum(x > 0.3 for x in a))


def main(runi, runh, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    curves = Curves(ROOT / 'fixed/inputs/surrogate_canonical.json')
    dep = dict(tailor(runh))
    for k, v in deployed().items():
        dep.setdefault(k, v)
    rows = []
    for k, v in dep.items():
        _, extra = curves.predict(v['s'])
        rows.append(dict(set='deployed', name=k[:12], extrapolated=extra, **v))
    prof, refit = profiling(runi, curves)
    rows += prof + appendix(curves)
    summary = dict(model='surrogate_canonical d_f + e_fg (+ stage term for deployed predictions)', sets={})
    for key, sel in (('deployed, all', lambda r: r['set'] == 'deployed'),
                     ('deployed, TAILOR-F', lambda r: r['set'] == 'deployed' and r['source'] == 'TAILOR-F'),
                     ('deployed, Always-All-Three', lambda r: r['set'] == 'deployed' and r['source'] == 'AlwaysAllThree'),
                     ('deployed, one fragment', lambda r: r['set'] == 'deployed' and len(r['order']) == 1),
                     ('deployed, two fragments', lambda r: r['set'] == 'deployed' and len(r['order']) == 2),
                     ('deployed, three fragments', lambda r: r['set'] == 'deployed' and len(r['order']) == 3),
                     ('deployed, with a geometric stage', lambda r: r['set'] == 'deployed' and r['fe_on']),
                     ('deployed, extrapolated', lambda r: r['set'] == 'deployed' and r['extrapolated']),
                     ('profiling, single', lambda r: r['source'] == 'profiling single'),
                     ('profiling, pair', lambda r: r['source'] == 'profiling pair'),
                     ('profiling, triple', lambda r: r['source'] == 'profiling triple'),
                     ('appendix points', lambda r: r['set'] == 'appendix')):
        summary['sets'][key] = stats([r for r in rows if sel(r)])
    summary['sets']['profiling triples, pairwise refit on measured singles and pairs'] = stats(refit)
    with open(out / 'distortion_diagnostic.csv', 'w') as fh:
        fh.write('set,source,name,order,fe_on,strengths,predicted_psnr,measured_psnr,error_db,extrapolated\n')
        for r in rows:
            fh.write(','.join([r['set'], r['source'], r['name'], '>'.join(r['order']), '+'.join(r['fe_on']) or 'none',
                               ' '.join(f"{f}={r['s'][f]:.4g}" for f in r['order']),
                               f"{r['predicted']:.4f}", f"{r['measured']:.4f}", f"{r['predicted'] - r['measured']:+.4f}",
                               '+'.join(r['extrapolated']) or 'no']) + '\n')
    (out / 'distortion_diagnostic.json').write_text(json.dumps(summary, indent=1) + '\n')
    plot(rows, out / 'distortion_diagnostic.pdf')
    for k, v in summary['sets'].items():
        if v:
            print(f"{k:62s} n={v['n']:4d} mean {v['mean']:+.3f} mean|e| {v['mean_abs']:.3f} p95 {v['p95_abs']:.3f} "
                  f"max {v['max_abs']:.3f} dB, >0.3 dB: {v['over_0_3']}")


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, grid = '#3a3a38', '#e6e5e0'
    style = {1: ('#2a78d6', 'o'), 2: ('#eb6834', 's'), 3: ('#1baf7a', '^')}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, sel, title in ((axes[0], 'deployed', 'Deployed configurations (live PSNR)'),
                           (axes[1], 'profiling', 'Profiling configurations (all six orders)')):
        part = [r for r in rows if r['set'] == sel]
        for k, (color, marker) in style.items():
            pts = [r for r in part if len(r['order']) == k]
            if pts:
                ax.scatter([r['measured'] for r in pts], [r['predicted'] - r['measured'] for r in pts], s=12,
                           color=color, marker=marker, alpha=0.75, linewidths=0,
                           label=f"{k} fragment{'s' if k > 1 else ''} (n={len(pts)})")
        ax.axhline(0, color=ink, lw=1)
        for y in (-0.3, 0.3):
            ax.axhline(y, color=ink, lw=0.8, ls=':')
        ax.set_title(title, fontsize=9.5, color=ink)
        ax.set_xlabel('measured PSNR (dB)', fontsize=9, color=ink)
        ax.grid(True, color=grid, lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False)
    axes[0].set_ylabel('predicted minus measured PSNR (dB)', fontsize=9, color=ink)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main(*sys.argv[1:4])
