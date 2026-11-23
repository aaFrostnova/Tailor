"""Freeze a set of pool images that no recorded experiment role has used (held-out positives, negatives).

Exclusion: every image named by the run's split manifests (all roles), by their historical exclusion
manifests (the earlier rigor splits and the superseded FPR calibration plan) and by any manifest passed
with --exclude (for example an earlier held-out set). An image is excluded by path, by file sha256 and by
pixel sha256 (RGB, bicubic 512 x 512, the hash the split manifests use), so a duplicate picture under
another file name is excluded too. Pixel hashes are computed for the whole pool and cached.

Usage: python freeze_images.py OUT_JSON --role heldout --counts A=25,B=25,C=25,D=15,E=10 --seed 20261109
       [--exclude other_manifest.json ...] [--hash-cache POOL_HASHES_JSON]
"""
import os
import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image

POOL = Path(os.environ.get('TAILOR_WORKSPACE', 'workspace') + '/pool')
FROZEN = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/'
              'rigor_editing_only_20260911/unified_ba_20260914')
BASE_MANIFESTS = (
    FROZEN / 'plans/image_splits.json',
    FROZEN / 'plans/reserved_unseen_image_splits.json',
    Path(os.environ.get('TAILOR_WORKSPACE', 'workspace') + '/rigor_editing_only_20260911/plans/image_splits.json'),
    FROZEN.parent / 'fpr_calibration_20260914/plan.json',
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_sha256(path):
    with Image.open(path) as im:
        image = im.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
        return hashlib.sha256(image.tobytes()).hexdigest()


def canonical(path):
    """Pool paths appear under the /home outputs symlink and under /scratch; compare by pool-relative name."""
    text = str(path)
    return text.split('/pool/', 1)[1] if '/pool/' in text else text


def manifest_images(doc):
    """(pool-relative path, sha256 or None, pixel sha256 or None) for every image a manifest names."""
    out = []

    def walk(x):
        if isinstance(x, dict):
            p = x.get('path')
            if isinstance(p, str) and p.endswith('.png'):
                out.append((canonical(p), x.get('sha256'), x.get('pixel_sha256')))
            for v in x.values():
                if isinstance(v, (dict, list)):
                    walk(v)
                elif isinstance(v, str) and v.endswith('.png') and '/pool/' in v and v is not p:
                    out.append((canonical(v), None, None))
        elif isinstance(x, list):
            for v in x:
                if isinstance(v, str) and v.endswith('.png') and '/pool/' in v:
                    out.append((canonical(v), None, None))
                else:
                    walk(v)
    walk(doc)
    return out


def pool_files():
    files = {}
    for source in sorted(p.name for p in POOL.iterdir() if p.is_dir()):
        for p in sorted((POOL / source / 'img').glob('*.png')):
            files[canonical(p)] = p
    return files


def pool_hashes(files, cache):
    known = json.loads(Path(cache).read_text()) if cache and Path(cache).exists() else {}
    changed = False
    for rel, p in files.items():
        if rel not in known:
            known[rel] = dict(sha256=file_sha256(p), pixel_sha256=pixel_sha256(p))
            changed = True
    if cache and changed:
        Path(cache).write_text(json.dumps(known, sort_keys=True) + '\n')
    return known


def freeze(out, role, counts, seed, extra_manifests=(), cache=None):
    out = Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists; an image set is frozen once')
    manifests = [Path(p) for p in (*BASE_MANIFESTS, *extra_manifests)]
    excluded = []
    for m in manifests:
        excluded += manifest_images(json.loads(m.read_text()))
    files = pool_files()
    hashes = pool_hashes(files, cache)
    # The pixel hash here must be the manifests' hash: recompute it for recorded images and compare.
    checked = 0
    for rel, sha, pix in excluded:
        if pix and rel in hashes:
            if hashes[rel]['pixel_sha256'] != pix or (sha and hashes[rel]['sha256'] != sha):
                raise SystemExit(f'pool image {rel} no longer matches its manifest record')
            checked += 1
    if not checked:
        raise SystemExit('no manifest pixel hash could be checked against the pool')
    bad_paths = {rel for rel, _, _ in excluded}
    bad_sha = {sha for _, sha, _ in excluded if sha} | {hashes[r]['sha256'] for r in bad_paths if r in hashes}
    bad_pix = {pix for _, _, pix in excluded if pix} | {hashes[r]['pixel_sha256'] for r in bad_paths if r in hashes}
    chosen, seen_pix = [], set()
    for k, (source, n) in enumerate(sorted(counts.items())):
        names = sorted(r for r in files if r.startswith(source + '/'))
        rng = random.Random(f'{seed}|{role}|{source}')
        rng.shuffle(names)
        picked = 0
        for rel in names:
            h = hashes[rel]
            if rel in bad_paths or h['sha256'] in bad_sha or h['pixel_sha256'] in bad_pix or h['pixel_sha256'] in seen_pix:
                continue
            seen_pix.add(h['pixel_sha256'])
            chosen.append(dict(source=source, path=str(files[rel]), name=rel, sha256=h['sha256'],
                               pixel_sha256=h['pixel_sha256']))
            picked += 1
            if picked == n:
                break
        if picked < n:
            raise SystemExit(f'source {source}: only {picked} unused images, {n} requested')
    # Index order matches the measurement convention: sources in order, images in sampled order.
    for i, row in enumerate(chosen):
        row['index'] = i
    images_sha256 = hashlib.sha256(json.dumps([[r['name'], r['sha256'], r['pixel_sha256']] for r in chosen],
                                              separators=(',', ':')).encode()).hexdigest()
    doc = dict(schema='tailor_exp_image_set_v1', role=role, seed=seed, counts=counts, n=len(chosen),
               resolution=[512, 512], pixel_hash='sha256 of RGB bicubic 512x512 bytes', images_sha256=images_sha256,
               exclusion=dict(manifests={str(m): file_sha256(m) for m in manifests}, images=len(excluded),
                              distinct_paths=len(bad_paths), pixel_hashes_checked_against_pool=checked),
               images=chosen)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + '\n')
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('out')
    ap.add_argument('--role', required=True)
    ap.add_argument('--counts', required=True, help='e.g. A=25,B=25,C=25,D=15,E=10')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--hash-cache')
    a = ap.parse_args()
    counts = {k: int(v) for k, v in (x.split('=') for x in a.counts.split(','))}
    doc = freeze(a.out, a.role, counts, a.seed, a.exclude, a.hash_cache)
    print(json.dumps({k: v for k, v in doc.items() if k != 'images'}, indent=1))


if __name__ == '__main__':
    main()
