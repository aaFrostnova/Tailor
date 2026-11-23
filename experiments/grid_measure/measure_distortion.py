"""Raw single-fragment embedding distortion at the measured strength knots (input to the grid baseline C2).

The frozen surrogate keeps distortion only as fitted curves, so C2's single-fragment distortion is measured
here: every fragment at every knot, with every stage that changes what is embedded (none and resync for all;
scale for VINE; tile for TrustMark), embedded on the 100 calibration images (pool offset 0, the images the
surrogate was profiled on). Nothing is attacked or decoded. The composite is built exactly as the frozen
measurement driver builds it (live_topk_full.composite_for); `check` re-measures one frozen configuration on
the selection images and compares its PSNR with the frozen cell before any knot is measured.

Usage: TAILOR_MAIN_CONTROLS_RUN=<run with a relocated root> python measure_distortion.py OUT_JSON
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'main_controls'))
from common import ARM, C, CP, FROZEN, read  # noqa: E402

FKEY = {'VINE': 'vine', 'TrustMark': 'trustmark', 'VideoSeal': 'videoseal'}
KNOTS = {'VINE': (0.14, 0.17, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
         'TrustMark': (0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0),
         'VideoSeal': (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.75, 2.0, 2.25, 2.5, 2.8, 3.2)}
STAGES = {'VINE': (None, 'resync', 'scale'), 'TrustMark': (None, 'resync', 'tile'), 'VideoSeal': (None, 'resync')}
CHECK_KEY = (('TrustMark',), ('tile',), (1.0,))          # frozen cell psnr_db 41.44711670843185


def boot():
    C.boot(ARM)
    for p in (str(C.CF), f'{C.CF}/scripts', f'{C.CF}/solver', f'{C.CF}/external/WatermarkAttacker', str(C.SC)):
        if p not in sys.path:
            sys.path.insert(0, p)


def composite_for(order, fe_on, strengths, dev='cuda'):
    """Copy of the frozen driver's composite_for (live_topk_full.py, gpu_measurer)."""
    import watermark_smt_v2 as W
    from eval_matrix import OursComposite
    fe = {k: (k in fe_on) for k in ('resync', 'scale', 'angle', 'tile')}
    c = OursComposite(dev, tm_variant='B', vine_variant='R',
                      config={'frags': [FKEY[f] for f in order], 'order': [FKEY[f] for f in order], 'strengths': {},
                              **W.frontend_config(fe)})
    c.strength = dict(c.DEFAULT_STRENGTH)
    c.strength.update({FKEY[f]: float(v) for f, v in zip(order, strengths)})
    return c


def covers(offset):
    from PIL import Image
    from src.image_pool import sample
    files = sample(100, offset=offset)
    return files, [Image.open(f).convert('RGB').resize((512, 512), Image.BICUBIC) for f in files]


def distortion(comp, images):
    mse = []
    for i, cov in enumerate(images):
        emb, _ = comp.embed(cov, i)
        emb = emb if emb.size == (512, 512) else emb.resize((512, 512))
        a, b = np.asarray(cov, np.float64), np.asarray(emb, np.float64)
        mse.append(float(np.mean((a - b) ** 2)))
    psnr = [10 * np.log10(255.0 ** 2 / max(m, 1e-9)) for m in mse]
    return dict(mse=mse, mean_mse=float(np.mean(mse)), mean_psnr=float(np.mean(psnr)),
                psnr_of_mean_mse=float(10 * np.log10(255.0 ** 2 / max(np.mean(mse), 1e-9))))


def main(out):
    out = Path(out)
    boot()
    import torch
    _, selection = covers(300)
    order, fe_on, strengths = CHECK_KEY
    frozen = read(FROZEN / 'live_topk/cells' / f'{CP.exact_cfg_id(CHECK_KEY)}_blur_pool300_n100.json')
    check = distortion(composite_for(order, fe_on, strengths), selection)
    gap = abs(check['mean_psnr'] - frozen['psnr_db'])
    print('CHECK', check['mean_psnr'], frozen['psnr_db'], gap, flush=True)
    if gap > 0.01:
        raise SystemExit(f'composite replication differs from the frozen driver by {gap} dB')
    files, calibration = covers(0)
    rows = {}
    for f, knots in KNOTS.items():
        for stage in STAGES[f]:
            for s in knots:
                t0 = time.time()
                comp = composite_for([f], [stage] if stage else [], [s])
                r = distortion(comp, calibration)
                rows.setdefault(f, {}).setdefault(stage or 'none', {})[repr(s)] = r
                print(f, stage or 'none', s, round(r['mean_mse'], 4), round(r['mean_psnr'], 3),
                      f'{time.time() - t0:.0f}s', flush=True)
                del comp
                torch.cuda.empty_cache()
    doc = dict(schema='tailor_exp_raw_distortion_v1', images='pool offset 0, n=100 (calibration role)',
               files=files, device=torch.cuda.get_device_name(0), check=dict(config=CHECK_KEY, measured=check['mean_psnr'],
               frozen=frozen['psnr_db'], gap_db=gap), distortion_domain='mse_255', table=rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc) + '\n')
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
