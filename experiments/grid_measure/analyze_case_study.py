"""PR3 case study analysis: the frozen high-interference pair on the held-out images, three versions.

Inputs: RUNI/case_study/{selection.json, tasks.json, measure/*.json} (make_case_study.py, measure_configs.py) and,
for the full-TAILOR version, the held-out run (PR5): RUNH/manifest.json assignments joined with the held-out
report rows (report_heldout.py heldout_results.jsonl).

Per version (unit = direct composition at strength 1.0, order = reversed order, strength = adjusted strengths)
and attack: each fragment's bit accuracy in the pair against the same fragment alone at the same strength on the
same images (paired per image, 95% bootstrap interval), the detection rate at the configuration's own
threshold for alpha 1e-4 and 1e-9 (T(c) = 3 for a pair, 1 alone), and the embed PSNR. Full TAILOR: TAILOR-F's
deployments containing both fragments, split by order, with their held-out pass counts; deployments without the
pair are the reference. The selection chronology is copied from selection.json.

Usage: TAILOR_MAIN_CONTROLS_RUN=RUNI python analyze_case_study.py RUNI RUNH HELDOUT_REPORT_DIR
Output: RUNI/case_study/interference_case_study.json
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'main_controls')]
from common import CP  # noqa: E402
import unified_detector as UD  # noqa: E402

RNG = np.random.default_rng(20261110)
ALPHAS = {'1e-4': 1e-4, '1e-9': 1e-9}


def load(folder):
    return {json.loads(p.read_text())['task']['name']: json.loads(p.read_text())
            for p in sorted(Path(folder).glob('*.json'))}


def per_fragment(rec, attack, f):
    cell = rec['cells'][attack]['cell']
    bucket = cell['by_threshold'][min(cell['by_threshold'], key=int)]
    return np.asarray(bucket['per_fragment_ba'][f], float)


def detection(rec, attack, alpha):
    t = rec['task']
    h = UD.threshold_count(dict(order=t['order'], fe_on=t['fe_on']), alpha)
    b = rec['cells'][attack]['cell']['by_threshold'][str(h)]
    return float(np.mean(b['detected'])), h


def boot(x, n=2000):
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def single_name(f, s):
    return f'{f}-{s:g}'


def versions(recs, tasks):
    out = {}
    for t in tasks:
        if t['version'] == 'single reference':
            continue
        rec = recs[t['name']]
        rows = {}
        for a in t['attacks']:
            frag = {}
            for f in t['order']:
                alone = recs[single_name(f, t['s'][f])]
                pa, pp = per_fragment(alone, a, f), per_fragment(rec, a, f)
                d = pa - pp
                frag[f] = dict(position=t['order'].index(f), alone=float(pa.mean()), in_pair=float(pp.mean()),
                               drop=float(d.mean()), drop_ci95=boot(d))
            det = {k: dict(zip(('rate', 'threshold_count'), detection(rec, a, v))) for k, v in ALPHAS.items()}
            alone_det = {f: {k: detection(recs[single_name(f, t['s'][f])], a, v)[0] for k, v in ALPHAS.items()}
                         for f in t['order']}
            rows[a] = dict(fragments=frag, pair_detection=det, single_detection=alone_det)
        out[t['name']] = dict(version=t['version'], order=t['order'], s=t['s'], psnr_db=rec['psnr_db'],
                              single_psnr_db={f: recs[single_name(f, t['s'][f])]['psnr_db'] for f in t['order']},
                              attacks=rows, device=rec['device'], images=rec['images'])
    return out


def full_tailor(runh, report_dir, first, second):
    manifest = json.loads((Path(runh) / 'manifest.json').read_text())
    path = Path(report_dir) / 'heldout_results.jsonl'
    judged = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            judged[r['spec_id']] = r
    groups = collections.defaultdict(lambda: dict(deployments=0, configurations=set(), judged=0, heldout_pass=0,
                                                  strengths=[]))
    for a in manifest['assignments']:
        o = a['candidate']['order']
        if first in o and second in o:
            key = f'{first} before {second}' if o.index(first) < o.index(second) else f'{second} before {first}'
            groups[key]['strengths'].append((a['candidate']['s'][first], a['candidate']['s'][second]))
        else:
            key = 'pair not used'
        g = groups[key]
        g['deployments'] += 1
        g['configurations'].add(a['cfg_id'])
        if a['spec_id'] in judged:
            g['judged'] += 1
            g['heldout_pass'] += bool(judged[a['spec_id']]['heldout_pass'])
    out = {}
    for k, g in groups.items():
        s = np.array(g['strengths']) if g['strengths'] else None
        out[k] = dict(deployments=g['deployments'], configurations=len(g['configurations']), judged=g['judged'],
                      heldout_pass=g['heldout_pass'],
                      median_strengths=(None if s is None else {first: float(np.median(s[:, 0])),
                                                                 second: float(np.median(s[:, 1]))}))
    return dict(groups=out, heldout_complete=len(judged) == len(manifest['assignments']),
                heldout_judged=len(judged), deployments=len(manifest['assignments']))


def main(runi, runh, report_dir):
    cs = Path(runi) / 'case_study'
    sel = json.loads((cs / 'selection.json').read_text())
    tasks = json.loads((cs / 'tasks.json').read_text())
    recs = load(cs / 'measure')
    missing = [t['name'] for t in tasks if t['name'] not in recs]
    if missing:
        raise SystemExit(f'unmeasured case-study configurations: {missing}')
    if any(r['images'] != sel['roles']['heldout_images_sha256'] for r in recs.values()):
        raise SystemExit('a case-study measurement used other images than the frozen held-out set')
    first, second = sel['selection']['first'], sel['selection']['second']
    result = dict(policy='exp_pr3_case_study_v1', selection=sel['selection'], chronology=sel['chronology'],
                  frozen_at=sel['frozen_at'], profiling_matrix_sha256=sel['profiling_matrix_sha256'],
                  heldout_images_sha256=sel['roles']['heldout_images_sha256'], roles=sel['roles'],
                  versions=versions(recs, tasks), full_tailor=full_tailor(runh, report_dir, first, second))
    (cs / 'interference_case_study.json').write_text(json.dumps(result, indent=1) + '\n')
    print(f'case study {first} -> {second} on held-out images {result["heldout_images_sha256"][:8]}')
    for name, v in result['versions'].items():
        print(f"  {name:8s} {'>'.join(v['order'])} s={v['s']} PSNR {v['psnr_db']:.2f} dB")
        for a, row in v['attacks'].items():
            fr = '  '.join(f"{f} {x['alone']:.3f}->{x['in_pair']:.3f} (drop {x['drop']:+.3f})"
                           for f, x in row['fragments'].items())
            print(f"    {a:7s} {fr}  det@1e-4 {row['pair_detection']['1e-4']['rate']:.2f}")
    for k, g in result['full_tailor']['groups'].items():
        print(f"  TAILOR-F {k}: {g['deployments']} deployments, {g['configurations']} configurations, "
              f"held-out {g['heldout_pass']}/{g['judged']} judged, strengths {g['median_strengths']}")


if __name__ == '__main__':
    main(*sys.argv[1:4])
