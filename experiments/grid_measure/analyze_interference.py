"""Ordered pairwise interference matrix (PR3) and triple residuals of the pairwise model (PR7).

Inputs: RUNI/matrix/*.json (9 single and 54 ordered-pair configurations) and RUNI/triples/*.json (24), each
measured on the 100 calibration images by measure_configs.py. Per image, the host fragment's bit accuracy is
read from the unified cell (per_fragment_ba of the lowest-threshold bucket; without a stage every bucket holds
the same per-fragment values).

PR3: drop(f <- g) = BA_f(single f) - BA_f(f then g), per image, at the same knots; mean and a 95% bootstrap
interval over images; asymmetry = drop(f <- g) - drop(g <- f) at matched levels.
PR7: pairwise prediction of a triple's per-fragment BA = BA_f(single) - sum over later fragments g of drop(f <- g)
(each drop measured in the pair at the same knots); residual = measured - predicted. Distortion: predicted MSE =
sum of single MSEs + sum over pairs of the pair's excess MSE (pair MSE minus its two single MSEs).

Usage: python analyze_interference.py RUNI
"""
import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

LEVELS = ('low', 'mid', 'high')
RNG = np.random.default_rng(20261110)


def load(folder):
    out = {}
    for p in sorted(Path(folder).glob('*.json')):
        d = json.loads(p.read_text())
        out[d['task']['name']] = d
    return out


def fragment_ba(rec, attack, f):
    cell = rec['cells'][attack]['cell']
    bucket = cell['by_threshold'][min(cell['by_threshold'], key=int)]
    return np.asarray(bucket['per_fragment_ba'][f], float)


def boot_ci(x, n=2000):
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def single_name(f, lv):
    return f'{f}-{lv}'


def pair_name(f, lf, g, lg):
    return f'{f}-{lf}__{g}-{lg}'


def matrix(recs):
    rows = []
    for name, rec in recs.items():
        t = rec['task']
        if t['kind'] != 'pair':
            continue
        f, g = t['order']
        lf, lg = t['levels'][f], t['levels'][g]
        sf, sg_ = recs[single_name(f, lf)], recs[single_name(g, lg)]
        for a in t['attacks']:
            host_alone, host_pair = fragment_ba(sf, a, f), fragment_ba(rec, a, f)
            over_alone, over_pair = fragment_ba(sg_, a, g), fragment_ba(rec, a, g)
            d_host = host_alone - host_pair
            rows.append(dict(first=f, second=g, level_first=lf, level_second=lg, attack=a,
                             host_alone=float(host_alone.mean()), host_in_pair=float(host_pair.mean()),
                             host_drop=float(d_host.mean()), host_drop_ci95=boot_ci(d_host),
                             second_alone=float(over_alone.mean()), second_in_pair=float(over_pair.mean()),
                             second_drop=float((over_alone - over_pair).mean()),
                             psnr_pair=rec['psnr_db'], psnr_first_alone=sf['psnr_db'], psnr_second_alone=sg_['psnr_db'],
                             mse_excess=float(rec['mean_mse'] - sf['mean_mse'] - sg_['mean_mse'])))
    return rows


def asymmetry(rows):
    key = {(r['first'], r['second'], r['level_first'], r['level_second'], r['attack']): r for r in rows}
    out = []
    for (f, g, lf, lg, a), r in key.items():
        rev = key.get((g, f, lg, lf, a))
        if rev and f < g:
            # host f overwritten by g, versus host g overwritten by f, at the same strengths
            out.append(dict(pair=f'{f}/{g}', levels=f'{lf}/{lg}', attack=a, drop_f_by_g=r['host_drop'],
                            drop_g_by_f=rev['host_drop'], asymmetry=r['host_drop'] - rev['host_drop']))
    return out


def triples(recs, trecs):
    out = []
    for name, rec in trecs.items():
        t = rec['task']
        order, lv = t['order'], t['levels']
        for i, f in enumerate(order):
            single = recs[single_name(f, lv[f])]
            for a in t['attacks']:
                alone = fragment_ba(single, a, f)
                pred = alone.copy()
                for g in order[i + 1:]:
                    pair = recs[pair_name(f, lv[f], g, lv[g])]
                    pred = pred - (alone - fragment_ba(pair, a, f))
                meas = fragment_ba(rec, a, f)
                res = meas - pred
                out.append(dict(triple=name, combo=t['combo'], order='>'.join(order), fragment=f, position=i,
                                attack=a, measured=float(meas.mean()), predicted=float(pred.mean()),
                                residual=float(res.mean()), residual_ci95=boot_ci(res)))
    return out


def triple_distortion(recs, trecs):
    out = []
    for name, rec in trecs.items():
        t = rec['task']
        order, lv = t['order'], t['levels']
        singles = {f: recs[single_name(f, lv[f])]['mean_mse'] for f in order}
        pred = sum(singles.values())
        for i, f in enumerate(order):
            for g in order[i + 1:]:
                pair = recs[pair_name(f, lv[f], g, lv[g])]
                pred += pair['mean_mse'] - singles[f] - singles[g]
        psnr = lambda m: 10 * np.log10(255 ** 2 / m)
        out.append(dict(triple=name, combo=t['combo'], order='>'.join(order), mse_measured=rec['mean_mse'],
                        mse_predicted=float(pred), psnr_measured=float(psnr(rec['mean_mse'])),
                        psnr_predicted=float(psnr(pred)), psnr_error_db=float(psnr(pred) - psnr(rec['mean_mse']))))
    return out


def main(runi):
    runi = Path(runi)
    recs, trecs = load(runi / 'matrix'), load(runi / 'triples')
    assert len(recs) == 63 and len(trecs) == 24, (len(recs), len(trecs))
    m = matrix(recs)
    asym = asymmetry(m)
    tr = triples(recs, trecs)
    td = triple_distortion(recs, trecs)
    out = runi / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'pair_interference_matrix.json').write_text(json.dumps(m, indent=1) + '\n')
    (out / 'pair_asymmetry.json').write_text(json.dumps(asym, indent=1) + '\n')
    (out / 'triple_residuals.json').write_text(json.dumps(dict(ba=tr, distortion=td), indent=1) + '\n')
    with open(out / 'pair_interference_matrix.csv', 'w') as fh:
        cols = ['first', 'second', 'level_first', 'level_second', 'attack', 'host_alone', 'host_in_pair', 'host_drop',
                'second_drop', 'psnr_pair', 'mse_excess']
        fh.write(','.join(cols) + '\n')
        for r in m:
            fh.write(','.join(str(round(r[c], 5)) if isinstance(r[c], float) else str(r[c]) for c in cols) + '\n')
    # console summary
    print('PR3 ordered interference, mid/mid strengths: mean host BA drop by attack')
    mid = [r for r in m if r['level_first'] == 'mid' and r['level_second'] == 'mid']
    for f, g in sorted({(r['first'], r['second']) for r in mid}):
        cells = {r['attack']: r['host_drop'] for r in mid if (r['first'], r['second']) == (f, g)}
        print(f'  {f:9s} then {g:9s}: ' + '  '.join(f'{a} {v:+.3f}' for a, v in cells.items()))
    worst = sorted(m, key=lambda r: -r['host_drop'])[:5]
    print('largest host drops:', [(r['first'], r['second'], r['level_first'], r['level_second'], r['attack'],
                                  round(r['host_drop'], 3)) for r in worst])
    big_asym = sorted(asym, key=lambda r: -abs(r['asymmetry']))[:4]
    print('largest asymmetries:', [(r['pair'], r['levels'], r['attack'], round(r['asymmetry'], 3)) for r in big_asym])
    res = np.array([r['residual'] for r in tr])
    print(f'PR7 triple BA residual (measured - pairwise prediction) over {len(res)} fragment-attack cells: '
          f'mean {res.mean():+.4f}, median {np.median(res):+.4f}, max |r| {np.abs(res).max():.4f}, '
          f'share |r| > 0.02: {(np.abs(res) > 0.02).mean():.2f}')
    pe = np.array([r['psnr_error_db'] for r in td])
    print(f'PR7 triple distortion: PSNR(pred) - PSNR(measured) mean {pe.mean():+.3f} dB, max |err| {np.abs(pe).max():.3f} dB')


if __name__ == '__main__':
    main(sys.argv[1])
