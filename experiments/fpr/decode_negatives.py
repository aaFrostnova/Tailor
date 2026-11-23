"""PR4 negatives: run the deployed unified verifier once per (frozen configuration, negative image).

Two negative kinds, never used for any selection:
  clean  -- a never-watermarked image queried under identity img_{j mod 100:05d};
  wrong  -- the image watermarked by the configuration under identity (j+1) mod 100, queried under j.
With FPR_FRESH_IDENTITY_BASE=B set, every query uses a deployment identity that calibration never evaluated,
one per (task k = manifest index, image j), as each request in deployment has its own identity:
  clean  -- image j queried under identity B + 1000 k + j;
  wrong  -- image j watermarked under identity B + 1000 k + 500 + j, queried under B + 1000 k + j.
This is the condition of the FPR theorem (the queried identity's keyed permutation and mask are fresh PRF outputs,
independent of the calibration transcript; calibration and selection used identities 0..99 only). Distinct
identities also make the trials independent given the images, which the mod-100 scheme does not.
Every path of the configuration is evaluated (fragments, fused, geometric search), so the stored per-threshold
decisions are the deployed decisions at every request alpha; a false acceptance at alpha is `detected` at
threshold_count(T(c), alpha). The composite is built exactly as the frozen driver builds it.

Usage (one array task processes its shard of the task list; finished outputs are skipped):
  TAILOR_MAIN_CONTROLS_RUN=$RUNF python decode_negatives.py SHARD NSHARD [--reverse]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'main_controls'),
                str(Path(__file__).resolve().parents[1] / 'grid_measure')]
from common import CP, RUN, atomic, cfgkey, read  # noqa: E402
from measure_distortion import boot, composite_for  # noqa: E402

FRESH = int(os.environ['FPR_FRESH_IDENTITY_BASE']) if os.environ.get('FPR_FRESH_IDENTITY_BASE') else None
if FRESH is not None and FRESH < 100000:
    raise SystemExit('FPR_FRESH_IDENTITY_BASE must lie far above the calibration identities 0..99')


def open_images(paths):
    from PIL import Image
    return [Image.open(p).convert('RGB').resize((512, 512), Image.BICUBIC) for p in paths]


def decode_set(comp, order, images, identities, embed_identities=None):
    from unified_detector import decode_thresholds_detailed
    from unified_measurement import cell_of_detailed
    rows, t0 = [], time.time()
    for j, img in enumerate(images):
        query = comp.secret_for(identities[j])
        if embed_identities is not None:
            img, _ = comp.embed(img, embed_identities[j])
            img = img if img.size == (512, 512) else img.resize((512, 512))
        rows.append(decode_thresholds_detailed(comp, img, query, CP.REQUEST_ALPHAS))
    cell = cell_of_detailed(order, rows)
    return cell, (time.time() - t0) / max(1, len(images))


def main(shard, nshard, reverse=False):
    """reverse: walk the task list from its end (a helper GPU that meets the forward shards only at the end)."""
    boot()
    import torch
    from unified_detector import install
    install()
    manifest = read(RUN / 'manifest.json')
    clean_paths = [r['path'] for r in read(RUN / 'inputs/negatives.json')['images']]
    images = open_images(clean_paths)
    device = torch.cuda.get_device_name(0)
    order = list(enumerate(manifest['tasks']))
    if reverse:
        order.reverse()
    for k, task in order:
        if k % nshard != shard:
            continue
        out = RUN / 'negatives' / f"{task['cfg_id']}_{task['kind']}.json"
        if out.exists():
            continue
        c = task['candidate']
        key = cfgkey(c)
        assert CP.exact_cfg_id(key) == task['cfg_id']
        comp = composite_for(list(key[0]), list(key[1]), list(key[2]))
        n = task['n']
        if FRESH is None:
            ids = [j % 100 for j in range(n)]
            embed_ids = [(j + 1) % 100 for j in range(n)] if task['kind'] == 'wrong' else None
            scheme = dict(kind='calibration_mod100')
        else:
            assert n <= 500, n                                  # B + 1000 k + [0, 1000) stays inside task k
            ids = [FRESH + 1000 * k + j for j in range(n)]
            embed_ids = [FRESH + 1000 * k + 500 + j for j in range(n)] if task['kind'] == 'wrong' else None
            scheme = dict(kind='fresh', base=FRESH, task_index=k, query='B+1000k+j',
                          embed=('B+1000k+500+j' if embed_ids else None))
        cell, sec = decode_set(comp, list(key[0]), images[:n], ids, embed_ids)
        atomic(out, dict(task=task, cell=cell, seconds_per_image=sec, device=device, host=os.uname().nodename,
                         protocol=CP.PROTOCOL, identity_scheme=scheme))
        print('NEGATIVES_DONE', task['cfg_id'][:12], task['kind'], n, f'{sec:.2f}s/img', flush=True)
        del comp
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), reverse='--reverse' in sys.argv[3:])
