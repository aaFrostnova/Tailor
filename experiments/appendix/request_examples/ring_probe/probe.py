"""Which nested VINE ring actually carries which attack?

Sweeps the deployed 133-point scale grid on the frozen nested-VINE embeddings and
records VINE bit accuracy per scale for clean, crop75 and crop50.  Nothing is
re-embedded: the cached watermarked images of the deployed config are reused.
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

from image_sources import load_watermarked                      # noqa: E402
import final_measurement as FM                                  # noqa: E402
import unified_detector as UD                                   # noqa: E402
from src.attacks import center_crop_area                        # noqa: E402

CFG_NESTED = dict(order=['VINE'], fe_on=['scale'], s={'VINE': 1.0})
CFG_PLAIN = dict(order=['VINE'], fe_on=[], s={'VINE': 1.0})
N_IMAGES = 12
CONDITIONS = [('clean', None), ('crop75', 0.75), ('crop50', 0.50)]


def scores_by_scale(comp, image, iid, tx, scales):
    """VINE bit accuracy at each centered scale, exactly as the decoder crops."""
    out = []
    for s in scales:
        if s >= .999:
            view = image
        else:
            size = int(round(512 * float(s)))
            off = (512 - size) // 2
            view = image.crop((off, off, off + size, off + size))
        llr = UD._llrs(comp._frag_llr('vine', view, iid))
        out.append(UD._score(llr, tx)['accuracy'])
    return out


def main():
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    scales = np.arange(.34, 1.0001, .005)
    assert len(scales) == 133
    ns = FM.measurement_namespace()
    result = {}
    for tag, cfg in [('nested', CFG_NESTED), ('plain', CFG_PLAIN)]:
        comp = FM.composite(ns, cfg)
        cfg_id = UD.config_for_comp(comp) and __import__('image_sources').configuration_id(cfg)
        rows = []
        for i in range(N_IMAGES):
            image = Image.fromarray(load_watermarked(cfg_id, i))
            iid, tx = comp.secret_for(i)
            for name, frac in CONDITIONS:
                view = image if frac is None else center_crop_area(image, frac)
                rows.append(dict(image=i, condition=name,
                                 accuracy=scores_by_scale(comp, view, iid, tx, scales)))
                print(tag, i, name, 'max', round(max(rows[-1]['accuracy']), 3), flush=True)
        result[tag] = dict(cfg_id=cfg_id, rows=rows)
    (OUT / 'ring_scan.json').write_text(json.dumps(
        dict(scales=[round(float(s), 3) for s in scales], n_images=N_IMAGES,
             conditions=[c for c, _ in CONDITIONS], results=result), indent=1) + '\n')
    print('WROTE', OUT / 'ring_scan.json', flush=True)


if __name__ == '__main__':
    main()
