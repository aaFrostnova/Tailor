"""Render three frozen qualitative requests without running any experiment.

The verdict uses fresh decoder observations, encoder PSNR, and measured latency.
All request budgets and selected configurations remain unchanged.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / 'qualitative_edge_20260925'))
from image_sources import load_cover, load_watermarked, image_metrics
sys.path.insert(0, str(BASE / 'results/rigor_editing_only_20260911/unified_ba_20260914/code'))
from unified_detector import budget_for_config, threshold_count

INK = MUTED = '#000000'
GREEN, RED = '#246556', '#9c4844'
GOOD_BG, BAD_BG, NEUTRAL = '#e8f1eb', '#faeae5', '#f1f4f7'
COLORS = ['#56a4b7', '#ae80aa', '#879b83', '#c39452', '#27847d', '#234f80']
NAMES = {'Fixed1': 'Enumeration', 'Greedy06': 'TAILOR-G', 'SMT06': 'TAILOR-F'}


def scientific(value, digits=2):
    e = int(math.floor(math.log10(value)))
    m = value / 10**e
    return rf'${m:.{digits}f}\!\times\!10^{{{e}}}$'


def requirement_fpr(value):
    e = math.log10(value)
    if abs(e - round(e)) < 1e-9:
        return rf'$10^{{{int(round(e))}}}$'
    return scientific(value)


def pill(fig, x, y, width, height, text, passed=None, bold=False, fontsize=20.5):
    color = INK
    fill = NEUTRAL if passed is None else GOOD_BG if passed else BAD_BG
    fig.add_artist(FancyBboxPatch((x, y), width, height, transform=fig.transFigure,
        boxstyle='round,pad=0.0018,rounding_size=0.004', linewidth=0,
        facecolor=fill, zorder=0))
    fig.text(x + width / 2, y + height / 2, text, ha='center', va='center',
             fontsize=fontsize, color=color, weight='bold' if bold else 'normal')


def prepare(data, image_index, gain):
    req = data['request']
    cover = load_cover(image_index)
    rows = []
    for m in data['methods']:
        wm = np.array(np.load(m['fresh_embeddings_npy'], mmap_mode='r', allow_pickle=False)[image_index], copy=True)
        delta = wm.astype(np.float64) - cover.astype(np.float64)
        cfg = {k: m['cfg'][k] for k in ['order', 's', 'fe_on']}
        threshold = threshold_count(cfg, req['fpr'])
        tests = budget_for_config(cfg)['max_tests']
        assert abs(threshold / 100 - m['tau']) < 1e-9
        bound = tests * sum(math.comb(100, k) for k in range(threshold, 101)) / 2**100
        assert bound <= req['fpr']
        passed = []
        for attack in req['attacks']:
            av = m['attacks'][attack]
            ok = av['rate'] >= .90 - 1e-12 and av['mean_ba'] >= m['tau'] - 1e-12
            assert av['passed'] == ok and av['n'] > 0
            passed.append(ok)
        quality = m['psnr_mean'] >= req['psnr_min'] - 1e-9
        latency = m['measured_latency_ms'] <= req['latency_max_ms'] + 1e-9
        assert (all(passed) and quality and latency) == m['accepted'], m['method']
        rows.append(dict(m, image=wm, residual=np.clip(np.abs(delta)*gain/255, 0, 1),
            passing_attacks=sum(passed), quality_ok=quality, latency_ok=latency,
            fpr_bound=bound, displayed_image_metrics=dict(psnr=float(10*np.log10(255**2/max(float(np.mean(delta**2)),1e-9))), pixel_sha256=hashlib.sha256(wm.tobytes()).hexdigest(), source=m['fresh_embeddings_npy'], cfg_id=m['cfg_id'], image_index=image_index),
            residual_clipped_fraction=float(np.mean(np.abs(delta)*gain > 255))))
    return cover, rows


def make_figure(data, cover, rows):
    req = data['request']
    assert len(rows) == 6
    fig = plt.figure(figsize=(14.3, 6.55), facecolor='white')   # scaled to 5.5 in in the paper
    title = (f"{req['scenario']} · {len(req['attacks'])} Attacks     |     FPR ≤ "
        + requirement_fpr(req['fpr'])
        + f"     |     PSNR ≥ {req['psnr_min']:g} dB     |     Latency ≤ {req['latency_max_ms']:,.0f} ms")
    fig.text(.025, .962, title, fontsize=24.1, color=INK)
    x0, gap = .025, .012
    w = (.95-6*gap)/7
    xs = [x0+i*(w+gap) for i in range(7)]
    image_h = w*16.8/7.7
    image_y, residual_y = .588, .306
    for j, (x, title) in enumerate(zip(xs, ['Original'] + [NAMES.get(m['method'], m['method']) for m in rows])):
        fig.text(x+w/2, .907, title, ha='center', fontsize=23.8, weight='bold')
        ax=fig.add_axes([x, image_y, w, image_h])
        ax.imshow(cover if j == 0 else rows[j-1]['image']); ax.axis('off')
        if j:
            ax=fig.add_axes([x, residual_y, w, image_h])
            ax.imshow(rows[j-1]['residual']); ax.axis('off')
    fig.text(xs[0]+w/2, residual_y+image_h/2, 'Watermark\nResidual', ha='center',
             va='center', fontsize=26.7, weight='bold', linespacing=1.3)
    ys = [.240, .194, .148, .102, .056]
    rh = .036
    labels = ['Attacks Passed', f"PSNR ≥ {req['psnr_min']:g} dB", 'Latency',
              'FPR Bound', 'Request Outcome']
    for y, label in zip(ys, labels):
        # the longest label, Request Outcome in bold, has to clear the first pill
        fig.text(xs[0], y+rh/2, label, va='center', fontsize=17.4,
                 weight='bold' if label == 'Request Outcome' else 'normal')
    for j, m in enumerate(rows):
        values=[(f"{m['passing_attacks']} / {len(req['attacks'])}", m['passing_attacks'] == len(req['attacks'])),
                (f"{m['psnr_mean']:.2f} dB", m['quality_ok']),
                (f"{m['measured_latency_ms']:,.1f} ms", m['latency_ok']),
                (scientific(m['fpr_bound']), True),
                ('Accepted' if m['accepted'] else 'Rejected', m['accepted'])]
        for k, (y, (text, ok)) in enumerate(zip(ys, values)):
            pill(fig, xs[j+1], y, w, rh, text, ok, bold=k == 4,
                 fontsize=19.5 if k == 3 else 15.8)
    return fig


def main():
    plt.rcParams.update({'font.family':'Nimbus Roman', 'mathtext.fontset':'stix',
        'text.color':INK, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
    overview = []
    with PdfPages(ROOT/'three_edge_requests.pdf') as bundle:
        for i, folder in enumerate(sorted(ROOT.glob('case_0[123]')), 1):
            path = folder/'request_evidence.json'
            data = json.loads(path.read_text())
            image_index = data['displayed_image']['index']
            cover, rows = prepare(data, image_index, 12)
            fig = make_figure(data, cover, rows)
            for ext in ['pdf','png']:
                fig.savefig(folder/f'edge_request.{ext}', dpi=240, facecolor='white', bbox_inches='tight', pad_inches=.07)
            fig.savefig(folder/'edge_request_preview.png', dpi=125, facecolor='white', bbox_inches='tight', pad_inches=.07)
            bundle.savefig(fig, facecolor='white', bbox_inches='tight', pad_inches=.07)
            plt.close(fig)
            audit = dict(request_id=data['request']['id'], image_index=image_index, residual_gain=12,
                cover_pixel_sha256=hashlib.sha256(cover.tobytes()).hexdigest(),
                evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                verdict_latency_source='fresh_synchronized_A100_measurement',
                measured_latency_source='fresh_live_image_subset_three_repetitions',
                methods=[{k:v for k,v in m.items() if k not in ['image','residual']} for m in rows])
            (folder/'figure_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
            attackrows = [dict(method=m['method'], attack=a, **{k:v for k,v in av.items() if k in ['n','rate','mean_ba','passed']})
                          for m in rows for a,av in m['attacks'].items()]
            with (folder/'attack_gate_values.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(attackrows[0])); writer.writeheader();writer.writerows(attackrows)
            overview.append(dict(case=i, request=data['request'], image_index=image_index,
                methods=[{k:v for k,v in m.items() if k in ['method','cfg_id','accepted','passing_attacks','psnr_mean','measured_latency_ms','fresh_timing']} for m in rows]))
            print(f"Rendered case {i}: {data['request']['id'][:12]}, image {image_index}",flush=True)
    assert len(overview)==3
    (ROOT/'cases_summary.json').write_text(json.dumps(overview,indent=2)+'\n')


if __name__ == '__main__':
    main()
