"""Where does each fragment's payload live? The structural measurement behind the geometric stages.

For the three plain single-fragment configs, over n images:
  (a) radial energy profile of the embedding residual (Chebyshev bands, so a square frame);
  (b) region ablation: revert either the outer ring or the interior to the cover, then decode.
Nothing is re-embedded; the frozen cached embeddings are reused.
"""
import os
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
B = OUT.parent.parent
U = B / 'results/rigor_editing_only_20260911/unified_ba_20260914'
CF = Path(os.environ.get('TAILOR_PROJECT', '.'))
sys.path[:0] = [str(B / 'qualitative_edge_20260925'), str(U / 'code'), str(CF),
                str(CF / 'scripts'), str(CF / 'solver'), str(CF / 'external/WatermarkAttacker')]

from image_sources import load_cover, load_watermarked, configuration_id   # noqa: E402
import final_measurement as FM                                             # noqa: E402
import unified_detector as UD                                              # noqa: E402

FRAGMENTS = [('VINE', 'vine'), ('TrustMark', 'trustmark'), ('VideoSeal', 'videoseal')]
N_IMAGES = 24
BANDS = 10
RING = 0.90          # outer 10% of the half-width


def radial(delta):
    """Mean |residual| per Chebyshev band, normalized so the bands sum to the image mean."""
    a = np.abs(delta).mean(2)
    n = a.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.maximum(np.abs(xx - (n - 1) / 2), np.abs(yy - (n - 1) / 2)) / ((n - 1) / 2)
    return [float(a[(r >= lo) & (r < lo + 1.0 / BANDS)].mean()) for lo in np.arange(0, 1, 1.0 / BANDS)]


def mask_ring(n=512, thr=RING):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.maximum(np.abs(xx - (n - 1) / 2), np.abs(yy - (n - 1) / 2)) / ((n - 1) / 2)
    return r >= thr


def main():
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    ring = mask_ring()[..., None]
    ns = FM.measurement_namespace()
    out = {}
    for name, key in FRAGMENTS:
        cfg = dict(order=[name], fe_on=[], s={name: 1.0})
        cfg_id = configuration_id(cfg)
        comp = FM.composite(ns, cfg)
        rows = []
        for i in range(N_IMAGES):
            cover = load_cover(i).astype(np.float64)
            wm = load_watermarked(cfg_id, i).astype(np.float64)
            iid, tx = comp.secret_for(i)
            variants = {
                'full': wm,
                'ring_only': np.where(ring, wm, cover),        # keep only the outer 10% ring
                'interior_only': np.where(ring, cover, wm),    # keep everything but that ring
            }
            rec = dict(image=i, radial=radial(wm - cover))
            for vname, arr in variants.items():
                pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
                llr = UD._llrs(comp._frag_llr(key, pil, iid))
                rec[vname] = UD._score(llr, tx)['accuracy']
            rows.append(rec)
            print(name, i, {k: round(rec[k], 3) for k in ('full', 'ring_only', 'interior_only')}, flush=True)
        out[name] = dict(cfg_id=cfg_id, rows=rows)
    (OUT / 'structure.json').write_text(json.dumps(
        dict(n_images=N_IMAGES, bands=BANDS, ring_threshold=RING, results=out), indent=1) + '\n')
    print('WROTE', OUT / 'structure.json', flush=True)


if __name__ == '__main__':
    main()
