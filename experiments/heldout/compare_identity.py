"""Fresh-identity transfer: the same frozen TAILOR-F configurations on the same held-out images, embedded and
verified under calibration identities (img_00000..img_00099, the held-out run) versus fresh deployment
identities (img_{offset+i}, the fresh-identity run). Only the identity differs, so every difference is the
effect of drawing new keyed permutations, masks and payloads.

Reports, over the accepted requests both runs judged:
  - request level: pass counts, the 2x2 agreement table and an exact McNemar test;
  - attack level: per request-attack pair and per distinct (configuration, attack) cell, the shift
    fresh_rate - calibration_identity_rate in the per-image acceptance rate and in the mean score,
    with a 95% interval that resamples configurations (requests sharing a configuration share its cells);
  - embed PSNR shift on the held-out images.

Usage: python compare_identity.py HELDOUT_REPORT_DIR FRESH_REPORT_DIR OUT_DIR
(the report dirs hold heldout_results.jsonl and heldout_summary.json from report_heldout.py)
"""
import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path


def rows(report_dir):
    return {r['spec_id']: r for r in (json.loads(line) for line in
                                      (Path(report_dir) / 'heldout_results.jsonl').read_text().splitlines() if line)}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cluster_ci(values_by_cfg, b=2000, seed=20261113):
    """95% percentile interval of the pooled mean, resampling configurations."""
    cfgs = list(values_by_cfg)
    if not cfgs:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(b):
        pick = [values_by_cfg[rng.choice(cfgs)] for _ in cfgs]
        flat = [v for vs in pick for v in vs]
        stats.append(sum(flat) / len(flat))
    stats.sort()
    return [stats[int(0.025 * b)], stats[int(0.975 * b) - 1]]


def main(heldout_dir, fresh_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    old, new = rows(heldout_dir), rows(fresh_dir)
    common = sorted(set(old) & set(new))
    fresh_summary = json.loads((Path(fresh_dir) / 'heldout_summary.json').read_text())
    a = b = c = d = 0
    for s in common:
        po, pn = old[s]['heldout_pass'], new[s]['heldout_pass']
        a += po and pn
        b += po and not pn
        c += (not po) and pn
        d += (not po) and (not pn)
    by_scenario = {}
    for sc in sorted({old[s]['scenario'] for s in common}) + ['all']:
        part = [s for s in common if sc == 'all' or old[s]['scenario'] == sc]
        by_scenario[sc] = dict(n=len(part), pass_calibration_ids=sum(old[s]['heldout_pass'] for s in part),
                               pass_fresh_ids=sum(new[s]['heldout_pass'] for s in part))
    # attack level: per request-attack pair, and per distinct configuration-attack cell
    pair_shift = collections.defaultdict(list)
    cell_shift = {}
    cell_cfg = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in common:
        cfg = old[s]['cfg_id']
        for att, v in old[s]['attacks'].items():
            w = new[s]['attacks'][att]
            pair_shift[att].append(w['held_rate'] - v['held_rate'])
            key = (cfg, att)
            if key not in cell_shift:
                cell_shift[key] = dict(rate=w['held_rate'] - v['held_rate'], mean=w['held_mean'] - v['held_mean'],
                                       old_rate=v['held_rate'], new_rate=w['held_rate'])
                cell_cfg['all'][cfg].append(cell_shift[key]['rate'])
                cell_cfg[att][cfg].append(cell_shift[key]['rate'])
    attacks = {}
    for att in sorted(pair_shift):
        cells = [v for (cfg, x), v in cell_shift.items() if x == att]
        attacks[att] = dict(
            request_pairs=len(pair_shift[att]), cells=len(cells),
            rate_shift_mean=statistics.mean(v['rate'] for v in cells),
            rate_shift_cluster95=cluster_ci(cell_cfg[att]),
            mean_score_shift=statistics.mean(v['mean'] for v in cells),
            old_rate_mean=statistics.mean(v['old_rate'] for v in cells),
            new_rate_mean=statistics.mean(v['new_rate'] for v in cells),
            cells_below_rho_old=sum(v['old_rate'] < 0.9 for v in cells),
            cells_below_rho_new=sum(v['new_rate'] < 0.9 for v in cells))
    allcells = list(cell_shift.values())
    dpsnr = [new[s]['psnr_heldout'] - old[s]['psnr_heldout'] for s in common
             if new[s].get('psnr_heldout') is not None and old[s].get('psnr_heldout') is not None]
    summary = dict(
        policy='exp_fresh_identity_transfer_v1',
        design='same configurations, same held-out images, same attacks and code; identities img_{i} versus '
               'img_{offset+i}',
        fresh_identity_offset=(json.loads((Path(fresh_dir).parent / 'manifest.json').read_text()).get('identity') or {}).get('offset')
        if (Path(fresh_dir).parent / 'manifest.json').exists() else None,
        requests_compared=len(common), fresh_pending=fresh_summary.get('pending'),
        fresh_complete=fresh_summary.get('complete'),
        request_agreement=dict(both_pass=a, only_calibration_ids_pass=b, only_fresh_ids_pass=c, both_fail=d,
                               mcnemar_exact_p=mcnemar_exact(b, c)),
        scenarios=by_scenario,
        cells=dict(n=len(allcells), rate_shift_mean=statistics.mean(v['rate'] for v in allcells) if allcells else None,
                   rate_shift_cluster95=cluster_ci(cell_cfg['all']),
                   abs_rate_shift_mean=statistics.mean(abs(v['rate']) for v in allcells) if allcells else None,
                   below_rho_old=sum(v['old_rate'] < 0.9 for v in allcells),
                   below_rho_new=sum(v['new_rate'] < 0.9 for v in allcells)),
        attacks=attacks,
        psnr_shift=dict(n=len(dpsnr), mean=statistics.mean(dpsnr) if dpsnr else None,
                        min=min(dpsnr) if dpsnr else None, max=max(dpsnr) if dpsnr else None))
    (out_dir / 'fresh_identity_transfer.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'attacks'}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
