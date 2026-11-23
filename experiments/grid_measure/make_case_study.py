"""PR3 case study: the high-interference ordered pair, chosen from profiling measurements only, verified on the
held-out images in three versions.

Selection reads only the profiling matrix (RUNI/analysis/pair_interference_matrix.json, measured on pool offset
0 before any of these held-out measurements) and is frozen in OUT_DIR/selection.json with that file's sha256:
  pair      the ordered pair (first, second) with the largest mean host drop over the non-clean attacks and the
            9 profiled strength combinations;
  unit      that order at strength 1.0 for both fragments (the library default), i.e. direct composition;
  order     the reversed order at strength 1.0 (order adjustment);
  strength  the same order at the profiled knot combination that maximizes the smaller of the two fragments'
            mean in-pair bit accuracy over the non-clean attacks, ties broken by higher PSNR (strength
            adjustment);
plus each fragment alone at every strength used, so the held-out host drop is measured rather than borrowed.
The third version, full TAILOR, is read from the held-out run (PR5): TAILOR-F's deployed configurations that
contain both fragments, with their order, strengths and held-out outcome; no new measurement.
Attacks: the profiling subset (clean, jpeg25, crop75, rot9, regen). Images: the frozen held-out set; the script
asserts that it shares no path and no pixel hash with the profiling images.

Usage: TAILOR_MAIN_CONTROLS_RUN=RUNI python make_case_study.py RUNI HELDOUT_IMAGE_SET OUT_DIR
Then:  TAILOR_MAIN_CONTROLS_RUN=RUNI python measure_configs.py OUT_DIR/tasks.json OUT_DIR/measure 0 1 \
           --image-set HELDOUT_IMAGE_SET
"""
import collections
import datetime
import hashlib
import json
import statistics
import sys
from pathlib import Path

ATTACKS = ['clean', 'jpeg25', 'crop75', 'rot9', 'regen']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def select(rows):
    noisy = [r for r in rows if r['attack'] != 'clean']
    drop = collections.defaultdict(list)
    for r in noisy:
        drop[(r['first'], r['second'])].append(r['host_drop'])
    ranking = sorted(((statistics.mean(v), k) for k, v in drop.items()), reverse=True)
    first, second = ranking[0][1]
    combos = collections.defaultdict(lambda: dict(host=[], second=[], psnr=None))
    for r in noisy:
        if (r['first'], r['second']) == (first, second):
            c = combos[(r['level_first'], r['level_second'])]
            c['host'].append(r['host_in_pair'])
            c['second'].append(r['second_in_pair'])
            c['psnr'] = r['psnr_pair']
    scored = sorted(((min(statistics.mean(c['host']), statistics.mean(c['second'])), c['psnr'], lv)
                     for lv, c in combos.items()), reverse=True)
    best = scored[0][2]
    return dict(first=first, second=second,
                ranking=[dict(order=list(k), mean_host_drop=m) for m, k in ranking],
                strength_levels=dict(zip((first, second), best)),
                strength_scores=[dict(levels=list(lv), min_mean_ba=s, psnr=p) for s, p, lv in scored])


def tasks(sel, knots):
    f, g = sel['first'], sel['second']
    lv = sel['strength_levels']
    s_adj = {f: knots[f][lv[f]], g: knots[g][lv[g]]}
    out = [dict(name='unit', order=[f, g], s={f: 1.0, g: 1.0}, version='unit/default direct composition'),
           dict(name='order', order=[g, f], s={f: 1.0, g: 1.0}, version='order adjusted'),
           dict(name='strength', order=[f, g], s=s_adj, version='strength adjusted')]
    singles = {(x, t['s'][x]) for t in out for x in t['order']}
    for x, v in sorted(singles):
        out.append(dict(name=f'{x}-{v:g}', order=[x], s={x: v}, version='single reference'))
    for t in out:
        t.update(fe_on=[], attacks=ATTACKS, kind='case_study')
    return out


def disjoint(heldout_set):
    """The held-out images share no path or pixel hash with the profiling images (pool offset 0)."""
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'main_controls'), str(Path(__file__).resolve().parent)]
    from measure_distortion import boot  # puts the relocated root and the project on sys.path
    boot()
    from src.image_pool import sample
    from PIL import Image
    doc = json.loads(Path(heldout_set).read_text())
    held_paths = {r['path'] for r in doc['images']}
    prof = sample(100, offset=0)
    if held_paths & set(prof):
        raise SystemExit('held-out and profiling images share a path')

    def pix(p):
        return hashlib.sha256(Image.open(p).convert('RGB').resize((512, 512), Image.BICUBIC).tobytes()).hexdigest()

    held_pix = {r.get('pixel_sha256') or pix(r['path']) for r in doc['images']}
    if held_pix & {pix(p) for p in prof}:
        raise SystemExit('held-out and profiling images share pixels')
    return dict(profiling='pool offset 0, n=100', heldout_images_sha256=doc['images_sha256'],
                shared_paths=0, shared_pixel_hashes=0)


def main(runi, heldout_set, out_dir):
    runi, out = Path(runi), Path(out_dir)
    if (out / 'selection.json').exists():
        raise SystemExit('selection exists; a frozen case study is never re-selected')
    src = runi / 'analysis/pair_interference_matrix.json'
    rows = json.loads(src.read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_interference_tasks import KNOTS, LEVELS
    knots = {f: dict(zip(LEVELS, v)) for f, v in KNOTS.items()}
    sel = select(rows)
    roles = disjoint(heldout_set)
    ts = tasks(sel, knots)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'tasks.json').write_text(json.dumps(ts, indent=1) + '\n')
    (out / 'selection.json').write_text(json.dumps(dict(
        policy='exp_pr3_case_study_v1', frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        profiling_matrix=str(src), profiling_matrix_sha256=sha(src), selection=sel, knots=knots,
        heldout_image_set=str(heldout_set), heldout_image_set_sha256=sha(heldout_set), roles=roles,
        tasks_sha256=sha(out / 'tasks.json'),
        chronology=('pair, order and strength chosen from the profiling matrix only; no held-out measurement of '
                    'these configurations existed when this file was written')), indent=1) + '\n')
    print('CASE STUDY', sel['first'], '->', sel['second'], 'strength', sel['strength_levels'], len(ts), 'tasks')


if __name__ == '__main__':
    main(*sys.argv[1:4])
