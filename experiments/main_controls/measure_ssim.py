"""SSIM of the configurations the main controls (PR1) accepted, measured as the frozen SSIM campaign measured it.

For each distinct accepted configuration: reuse the frozen SSIM campaign's cell when one exists (same cfg_id, i.e.
the same order, stages and exact strengths, on the same 100 selection images); otherwise embed the 100 selection
images with the frozen driver's composite (index i, 512 x 512) and compute per-image SSIM with
skimage.metrics.structural_similarity (channel_axis=-1, data_range=255, gaussian_weights=True, sigma=1.5,
win_size=11, use_sample_covariance=False, float64; the RGB channel mean), then the mean over images. Images
come from the campaign's frozen manifest and are verified by file and pixel sha256. The reproduced mean PSNR is
compared with the configuration's live PSNR; the campaign's tolerance is 0.001 dB on an A100, so the device and
the deviation are recorded and `summarize` reports them.

`summarize` gives request-weighted means (every accepted request votes with its configuration's mean) and the
paired comparison with TAILOR-F on joint successes (TAILOR-F's SSIM from the frozen campaign cells).

Usage:
  TAILOR_MAIN_CONTROLS_RUN=RUN python measure_ssim.py measure OUT_DIR [--variants V,...] [--shard k --nshards n]
  TAILOR_MAIN_CONTROLS_RUN=RUN python measure_ssim.py summarize OUT_DIR
"""
import os
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'grid_measure')]
from common import CP, RUN, atomic, cfgkey, read  # noqa: E402

CAMPAIGN = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/ssim_preview_20260924')
IMAGES = CAMPAIGN / 'reproduction/a100_probe_manifest.json'
METRIC = dict(name='SSIM', implementation='skimage.metrics.structural_similarity', color='RGB channel mean',
              data_range=255, gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False,
              precision='float64', size=[512, 512])
TOLERANCE_DB = 0.001
VARIANTS = ('SingleOptimized', 'AlwaysAllThree')


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=-1, data_range=255, gaussian_weights=True, sigma=1.5,
                                       win_size=11, use_sample_covariance=False))


def accepted(variants):
    """cfg_id -> (candidate, live PSNRs of the requests that deployed it)."""
    out = {}
    for r in read(RUN / 'results.json')['rows']:
        if r['variant'] in variants and r['status'] == 'accepted':
            c = r['candidate']
            e = out.setdefault(CP.exact_cfg_id(cfgkey(c)), dict(candidate=c, psnr=set()))
            e['psnr'].add(float(r['verdict']['psnr_live']))
    return out


def covers():
    from PIL import Image
    doc = read(IMAGES)['images']
    imgs = []
    for row in doc['images']:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise SystemExit(f"cover changed: {row['path']}")
        im = Image.open(row['path']).convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
        if hashlib.sha256(im.tobytes()).hexdigest() != row['pixel_sha256']:
            raise SystemExit(f"cover pixels differ: {row['path']}")
        imgs.append(im)
    return doc['images_sha256'], imgs


def measure(out, variants, shard, nshards):
    import fcntl
    import torch
    from measure_distortion import boot, composite_for
    out = Path(out)
    (out / 'cells').mkdir(parents=True, exist_ok=True)
    (out / 'locks').mkdir(exist_ok=True)
    todo = sorted(accepted(variants).items())
    todo = [(k, e) for k, e in todo if not (CAMPAIGN / 'cells' / f'{k}.json').exists()][shard::nshards]
    boot()
    images_sha, imgs = covers()
    cov = [np.asarray(im, np.float64) for im in imgs]
    gpu = torch.cuda.get_device_name(0)
    for k, e in todo:
        cell = out / 'cells' / f'{k}.json'
        with (out / 'locks' / k).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if cell.exists():
                continue
            t0 = time.time()
            c = e['candidate']
            comp = composite_for(list(c['order']), list(c['fe_on']), [c['s'][f] for f in c['order']])
            ss, ps = [], []
            for i, im in enumerate(imgs):
                emb, _ = comp.embed(im, i)
                emb = emb if emb.size == (512, 512) else emb.resize((512, 512))
                b = np.asarray(emb.convert('RGB'), np.float64)
                ss.append(ssim(cov[i], b))
                ps.append(float(10 * np.log10(255.0 ** 2 / max(float(np.mean((cov[i] - b) ** 2)), 1e-9))))
            del comp
            torch.cuda.empty_cache()
            pm = float(np.mean(ps))
            dev = max(abs(pm - p) for p in e['psnr'])
            atomic(cell, dict(complete=True, cfg_id=k, cfg=dict(order=c['order'], fe_on=sorted(c['fe_on']), s=c['s']),
                              images_sha256=images_sha, metric=METRIC, ssim=ss, ssim_mean=float(np.mean(ss)),
                              psnr_db=ps, psnr_mean=pm, live_psnr=sorted(e['psnr']), psnr_deviation_db=dev,
                              within_campaign_tolerance=dev <= TOLERANCE_DB, device=gpu,
                              source_kind='native_reproduction', seconds=time.time() - t0))
            print(json.dumps(dict(cfg_id=k[:12], ssim=round(float(np.mean(ss)), 5), psnr_dev=round(dev, 5),
                                  seconds=round(time.time() - t0, 1))), flush=True)


def cell_for(out, k):
    for p in (CAMPAIGN / 'cells' / f'{k}.json', Path(out) / 'cells' / f'{k}.json'):
        if p.exists():
            return read(p), ('frozen_campaign' if p.parent.parent == CAMPAIGN else 'pr1_reproduction')
    return None, None


def summarize(out):
    out = Path(out)
    sys.path.insert(0, str(HERE.parent))
    from tailor_exp import gate
    import rejudge
    rows = read(RUN / 'results.json')['rows']
    inputs = gate._Inputs(HERE.parents[1], rejudge.FROZEN)
    frozen = inputs.read('paper', gate.COHORT_FILE)
    inputs.pin_run_inputs(frozen['source_sha256'])
    manifest = {r['spec_id']: r for r in gate.cohort.main_cohort_rows(HERE.parents[1])}
    tailor = gate._index('SMT06', gate._assemble(inputs)['SMT06'], manifest)
    cands = rejudge.candidate_index()
    summary = dict(metric=METRIC, variants={})
    for v in VARIANTS:
        acc = [r for r in rows if r['variant'] == v and r['status'] == 'accepted']
        per, missing, sources, devs = {}, set(), dict(frozen_campaign=0, pr1_reproduction=0), []
        for r in acc:
            k = CP.exact_cfg_id(cfgkey(r['candidate']))
            rec, src = cell_for(out, k)
            if rec is None:
                missing.add(k)
                continue
            per[r['spec_id']] = rec['ssim_mean']
            sources[src] += 1
            if src == 'pr1_reproduction':
                devs.append(rec['psnr_deviation_db'])
        joint = []
        for sid, s in per.items():
            t = tailor[sid]
            if t['_status'] != 'accepted':
                continue
            c = t.get('candidate') or cands.get(t.get('cfg_id'))
            trec, _ = cell_for(out, CP.exact_cfg_id(cfgkey(c)))
            if trec is not None:
                joint.append((s, trec['ssim_mean']))
        summary['variants'][v] = dict(
            accepted=len(acc), with_ssim=len(per), configurations_missing=len(missing),
            ssim=statistics.mean(per.values()) if per else None, sources=sources,
            reproduction_psnr_deviation_db=dict(n=len(devs), max=max(devs) if devs else None,
                                               within_0_001=sum(d <= TOLERANCE_DB for d in devs)),
            paired_with_tailor_f=dict(n=len(joint), variant=statistics.mean(a for a, _ in joint) if joint else None,
                                      tailor_f=statistics.mean(b for _, b in joint) if joint else None),
            complete=not missing and all(not r['status'].startswith('pending') for r in rows if r['variant'] == v))
    atomic(out / 'summary.json', summary)
    print(json.dumps(summary['variants'], indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('measure', 'summarize'))
    p.add_argument('out')
    p.add_argument('--variants', default=','.join(VARIANTS))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()
    if a.cmd == 'measure':
        measure(a.out, tuple(a.variants.split(',')), a.shard, a.nshards)
    else:
        summarize(a.out)
