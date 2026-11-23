"""PR1 table rows: TAILOR-Single (SingleOptimized) and Always-All-Three next to TAILOR-F on the 7,321 requests.

Reads RUN/results.json (main controls), the frozen TAILOR-F record chain (gate SMT06 index) and the SSIM cells
(frozen SSIM campaign, else measure_ssim.py reproductions in SSIM_DIR). Per variant: overall and macro
satisfaction with Wilson 95% intervals (pending requests counted as not yet accepted, and the bracket with them
counted as accepted), S1-S5 rates, request-weighted PSNR and SSIM over own accepted requests, and the paired
comparison with TAILOR-F on joint successes (PSNR and SSIM per request; the difference is TAILOR-F minus the
row, as in tab/paired_quality.tex). Rows are labelled
partial while any request of the variant is pending.

Usage: TAILOR_MAIN_CONTROLS_RUN=RUN python export_pr1.py SSIM_DIR OUT_DIR
Outputs: OUT_DIR/pr1_summary.json, OUT_DIR/pr1_main_rows.tex, OUT_DIR/pr1_paired_rows.tex
"""
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import CP, RUN, atomic, cfgkey, read  # noqa: E402
from measure_ssim import cell_for  # noqa: E402

NAMES = {'SingleOptimized': 'TAILOR-Single', 'AlwaysAllThree': 'Always-All-Three'}
SCEN = ('S1', 'S2', 'S3', 'S4', 'S5')


def wilson(k, n, z=1.959964):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def tailor_f():
    from tailor_exp import gate
    import rejudge
    inputs = gate._Inputs(HERE.parents[1], rejudge.FROZEN)
    frozen = inputs.read('paper', gate.COHORT_FILE)
    inputs.pin_run_inputs(frozen['source_sha256'])
    manifest = {r['spec_id']: r for r in gate.cohort.main_cohort_rows(HERE.parents[1])}
    index = gate._index('SMT06', gate._assemble(inputs)['SMT06'], manifest)
    cands = rejudge.candidate_index()
    out = {}
    for sid, r in index.items():
        rec = dict(accepted=r['_status'] == 'accepted', cls=r['cls'])
        if rec['accepted']:
            c = r.get('candidate') or cands.get(r.get('cfg_id'))
            rec.update(psnr=gate._psnr(r), cfg_id=CP.exact_cfg_id(cfgkey(c)))
        out[sid] = rec
    return out


def summarize(rows, tf, ssim_dir):
    n = len(rows)
    acc = [r for r in rows if r['status'] == 'accepted']
    pend = [r for r in rows if r['status'].startswith('pending')]
    scen = {}
    for s in SCEN:
        part = [r for r in rows if r['cls'] == 'C' + s[1:]]
        k = sum(r['status'] == 'accepted' for r in part)
        p = sum(r['status'].startswith('pending') for r in part)
        scen[s] = dict(n=len(part), accepted=k, pending=p, percent=100 * k / len(part),
                       percent_if_pending_pass=100 * (k + p) / len(part))
    ssim, missing = {}, 0
    for r in acc:
        rec, _ = cell_for(ssim_dir, CP.exact_cfg_id(cfgkey(r['candidate'])))
        if rec is None:
            missing += 1
        else:
            ssim[r['spec_id']] = rec['ssim_mean']
    joint = []
    for r in acc:
        t = tf[r['spec_id']]
        if not t['accepted']:
            continue
        trec, _ = cell_for(ssim_dir, t['cfg_id'])
        joint.append(dict(psnr=float(r['verdict']['psnr_live']), tf_psnr=t['psnr'], ssim=ssim.get(r['spec_id']),
                          tf_ssim=trec['ssim_mean'] if trec else None))
    js = [j for j in joint if j['ssim'] is not None and j['tf_ssim'] is not None]
    mean = lambda xs: statistics.mean(xs) if xs else None
    return dict(
        n=n, accepted=len(acc), pending=len(pend), complete=not pend,
        percent=100 * len(acc) / n, wilson95=[100 * x for x in wilson(len(acc), n)],
        percent_if_pending_pass=100 * (len(acc) + len(pend)) / n,
        macro=mean([v['percent'] for v in scen.values()]),
        macro_if_pending_pass=mean([v['percent_if_pending_pass'] for v in scen.values()]),
        scenarios=scen,
        psnr=mean([float(r['verdict']['psnr_live']) for r in acc]),
        ssim=mean(list(ssim.values())), ssim_missing=missing,
        paired=dict(n=len(joint), psnr=mean([j['psnr'] for j in joint]), tf_psnr=mean([j['tf_psnr'] for j in joint]),
                    delta_psnr=mean([j['tf_psnr'] - j['psnr'] for j in joint]),   # paper convention: TAILOR-F minus row
                    ssim_n=len(js), ssim=mean([j['ssim'] for j in js]), tf_ssim=mean([j['tf_ssim'] for j in js])),
        label='final' if not pend else 'partial: pending requests remain')


def fmt(x, d=2):
    return '-' if x is None else f'{x:.{d}f}'


def main(ssim_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read(RUN / 'results.json')['rows']
    tf = tailor_f()
    tf_rows = [dict(cls=v['cls'], status='accepted' if v['accepted'] else 'rejected') for v in tf.values()]
    summary = dict(policy='exp_pr1_export_v1', results_updated_at=read(RUN / 'summary.json')['progress']['updated_at'],
                   variants={v: summarize([r for r in rows if r['variant'] == v], tf, ssim_dir) for v in NAMES},
                   tailor_f=dict(accepted=sum(v['accepted'] for v in tf.values()), n=len(tf),
                                 percent=100 * sum(v['accepted'] for v in tf.values()) / len(tf),
                                 macro=statistics.mean(
                                     100 * sum(r['status'] == 'accepted' for r in tf_rows if r['cls'] == 'C' + s[1:])
                                     / sum(r['cls'] == 'C' + s[1:] for r in tf_rows) for s in SCEN)))
    atomic(out / 'pr1_summary.json', summary)
    main_rows, paired_rows = [], []
    for v, name in NAMES.items():
        s = summary['variants'][v]
        mark = '' if s['complete'] else ' % partial'
        main_rows.append(f"{name} & 0.06 & {fmt(s['percent'])} & {fmt(s['macro'])} & "
                         + ' & '.join(fmt(s['scenarios'][k]['percent']) for k in SCEN) + f' \\\\{mark}')
        p = s['paired']
        paired_rows.append(f"{name} & {p['n']:,} & {fmt(p['psnr'])} & {fmt(p['tf_psnr'])} & {p['delta_psnr']:+.2f} & "
                           f"{fmt(p['ssim'], 4)} & {fmt(p['tf_ssim'], 4)} \\\\{mark}")
    (out / 'pr1_main_rows.tex').write_text('\n'.join(main_rows) + '\n')
    (out / 'pr1_paired_rows.tex').write_text('\n'.join(paired_rows) + '\n')
    for v, name in NAMES.items():
        s = summary['variants'][v]
        print(f"{name}: {s['accepted']}/{s['n']} = {s['percent']:.2f}% (Wilson {s['wilson95'][0]:.2f}-{s['wilson95'][1]:.2f}; "
              f"if pending pass {s['percent_if_pending_pass']:.2f}%), macro {fmt(s['macro'])}, PSNR {fmt(s['psnr'])}, "
              f"SSIM {fmt(s['ssim'], 4)} (missing {s['ssim_missing']}), {s['label']}")
        print('   ' + ', '.join(f"{k} {fmt(x['percent'])}" for k, x in s['scenarios'].items()))
        p = s['paired']
        print(f"   paired with TAILOR-F on {p['n']}: PSNR {fmt(p['psnr'])} vs {fmt(p['tf_psnr'])} ({p['delta_psnr']:+.2f}), "
              f"SSIM {fmt(p['ssim'], 4)} vs {fmt(p['tf_ssim'], 4)} on {p['ssim_n']}")
    print(f"TAILOR-F: {summary['tailor_f']['percent']:.2f}%, macro {summary['tailor_f']['macro']:.2f}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
