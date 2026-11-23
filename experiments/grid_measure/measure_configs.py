"""Measure given configurations on a given image set: embed, attack (or not), decode with the unified verifier.

Used by the interference matrix (PR3) and the triple-residual diagnostic (PR7). It follows the frozen driver's
measurement exactly (live_topk_full.gpu_measurer.ensure: composite_for, embed at index i, r512, attack_pil_any,
decode_thresholds_detailed at every request alpha, cell_of_detailed) and adds the attack 'clean' (no attack),
which the frozen driver cannot measure. In-process attacks only.

Task file: JSON list of {name, order, fe_on, s: {fragment: strength}, attacks: [...]}.
Images: --offset N (pool sample, as the frozen driver draws it) or --image-set FILE (frozen image set JSON).
Usage: TAILOR_MAIN_CONTROLS_RUN=<run with a relocated root> python measure_configs.py TASKS OUT_DIR SHARD NSHARD
       (--offset 0 | --image-set FILE)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'main_controls'), str(Path(__file__).resolve().parent)]
from common import CP, atomic  # noqa: E402
from measure_distortion import boot, composite_for  # noqa: E402

IN_PROCESS = {'clean', 'jpeg25', 'blur', 'noise', 'bright', 'contrast', 'crop75', 'crop50', 'rot9', 'rs256', 'hflip',
              'crop_jpeg', 'border20', 'vaeB', 'vaeC', 'regen', 'rinse2x'}


def r512(img):
    return img if img.size == (512, 512) else img.resize((512, 512))


def load_images(offset=None, image_set=None):
    from PIL import Image
    if image_set:
        doc = json.loads(Path(image_set).read_text())
        files, ident = [r['path'] for r in doc['images']], doc['images_sha256']
    else:
        from src.image_pool import sample
        files, ident = sample(100, offset=offset), f'pool_offset_{offset}_n100'
    return files, ident, [Image.open(f).convert('RGB').resize((512, 512), Image.BICUBIC) for f in files]


def measure(task, covers, dev='cuda'):
    from src.attacks import attack_pil_any
    from unified_detector import decode_thresholds_detailed
    from unified_measurement import cell_of_detailed
    order = list(task['order'])
    comp = composite_for(order, list(task['fe_on']), [task['s'][f] for f in order], dev=dev)
    embeds, mse = [], []
    for i, cov in enumerate(covers):
        emb, sec = comp.embed(cov, i)
        emb = r512(emb)
        embeds.append((emb, sec))
        mse.append(float(np.mean((np.asarray(cov, np.float64) - np.asarray(emb, np.float64)) ** 2)))
    psnr = [10 * np.log10(255.0 ** 2 / max(m, 1e-9)) for m in mse]
    cells = {}
    for a in task['attacks']:
        if a not in IN_PROCESS:
            raise ValueError(f'{a} is not an in-process attack')
        t0 = time.time()
        rows = []
        for emb, (iid, tx) in embeds:
            v = emb if a == 'clean' else r512(attack_pil_any(a, emb, dev=dev))
            rows.append(decode_thresholds_detailed(comp, v, (iid, tx), CP.REQUEST_ALPHAS))
        cells[a] = dict(cell=cell_of_detailed(order, rows), seconds=time.time() - t0)
    return dict(mse=mse, mean_mse=float(np.mean(mse)), psnr_db=float(np.mean(psnr)), cells=cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tasks')
    ap.add_argument('out')
    ap.add_argument('shard', type=int)
    ap.add_argument('nshard', type=int)
    ap.add_argument('--offset', type=int)
    ap.add_argument('--image-set')
    a = ap.parse_args()
    if (a.offset is None) == (a.image_set is None):
        raise SystemExit('give exactly one of --offset or --image-set')
    boot()
    import torch
    tasks = json.loads(Path(a.tasks).read_text())
    files, ident, covers = load_images(a.offset, a.image_set)
    device = torch.cuda.get_device_name(0)
    out = Path(a.out)
    for k, task in enumerate(tasks):
        if k % a.nshard != a.shard:
            continue
        p = out / f"{task['name']}.json"
        if p.exists():
            continue
        t0 = time.time()
        result = measure(task, covers)
        atomic(p, dict(task=task, images=ident, device=device, host=os.uname().nodename, protocol=CP.PROTOCOL,
                       seconds=time.time() - t0, **result))
        print('MEASURED', task['name'], round(result['psnr_db'], 3), f'{time.time() - t0:.0f}s', flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
