"""Render request -> TAILOR solution examples (one figure per example).

Inputs: selected_examples.json (frozen verdicts), cached native embeddings, and
fresh timing results in timing/tasks/<cfg_id>/result.json (model latency is shown,
labeled as such, only while the fresh timing is missing).
"""
import os
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
from PIL import Image

ROOT = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/request_solution_examples_20260925')  # frozen inputs
OUT = Path(__file__).resolve().parent / 'figures'
B = ROOT.parent
sys.path.insert(0, str(B / 'qualitative_edge_20260925'))
from image_sources import load_cover, load_watermarked

# Palette shared with the appendix qualitative figures (Figures 8-10).  The fragment
# hues are low-contrast, so every colored mark is also named in ink text.
INK = MUTED = '#000000'
RULE, TRACK = '#c9d1da', '#eceff3'
GREEN, GOOD_BG = '#246556', '#e8f1eb'
CAP_BG = '#e9eef5'
CARD_BG, CARD_EDGE = '#f7f9fb', '#dde3ea'
FRAG = {'VINE': '#56a4b7', 'TrustMark': '#ae80aa', 'VideoSeal': '#879b83'}
TAILOR, TAILOR_LIGHT = '#234f80', '#9fb4cf'
GAIN = 12
W = 15.0

FAMILIES = [
    ('Photometric', ['jpeg25', 'blur', 'noise', 'bright', 'contrast']),
    ('Geometric', ['crop75', 'crop50', 'rot9', 'crop_jpeg', 'rs256', 'hflip', 'border20']),
    ('Neural Compression', ['vaeB', 'vaeC']),
    ('Regeneration and Editing', ['regen', 'rinse2x', 'ctrlregen_s03', 'ctrlregen_s05', 'editing_ip2p_s20_v1']),
    ('Watermark Removal', ['unmarker']),
]
NAME = {'jpeg25': 'JPEG', 'blur': 'Blur', 'noise': 'Noise', 'bright': 'Brightness', 'contrast': 'Contrast',
        'crop75': 'Crop 75%', 'crop50': 'Crop 50%', 'rot9': 'Rotation', 'crop_jpeg': 'Crop + JPEG',
        'rs256': 'Resize 256', 'hflip': 'Horizontal Flip', 'border20': 'Border-20', 'vaeB': 'VAE-B', 'vaeC': 'VAE-C',
        'regen': 'Regen', 'rinse2x': 'Rinse-2x', 'ctrlregen_s03': 'CtrlRegen+ 0.3',
        'ctrlregen_s05': 'CtrlRegen+ 0.5', 'editing_ip2p_s20_v1': 'Image Editing', 'unmarker': 'UnMarker'}
SCENARIO = {'S1': 'Signal Processing and Re-encoding', 'S2': 'Geometric Transformations',
            'S3': 'Regeneration and Editing', 'S4': 'Watermark Removal', 'S5': 'Broad Attack Coverage'}


def sci(x):
    e = int(math.floor(math.log10(x))); m = x / 10 ** e
    return rf'${m:.1f}\times10^{{{e}}}$'


def sci_bold(x):
    e = int(math.floor(math.log10(x))); m = x / 10 ** e
    return rf'$\mathbf{{{m:.1f}\times10^{{{e}}}}}$'


def fpr_label(x):
    if abs(x - 2 ** -37) / 2 ** -37 < 1e-9:
        return r'$2^{-37}$'
    e = math.log10(x)
    if abs(e - round(e)) < 1e-9:
        return rf'$10^{{{int(round(e))}}}$'
    return sci(x)


def seconds_or_ms(ms):
    return f'{ms / 1000:.2f} s' if ms >= 1000 else f'{ms:,.0f} ms'


def budget(ms):
    return f'{ms / 1000:g} s' if ms >= 1000 else f'{ms:,.0f} ms'


def measured_latency(cfg_id, attacks):
    p = ROOT / 'timing/tasks' / cfg_id / 'result.json'
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if not r.get('complete'):
        return None
    return max(next(iter(r['decoders'][a].values()))['mean_ms'] for a in attacks)


class Canvas:
    """All drawing in inch coordinates on one full-figure axes, so rounded corners stay isotropic."""
    def __init__(self, H, width=W):
        self.H, self.W = H, width
        self.fig = plt.figure(figsize=(width, H), facecolor='white')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width); self.ax.set_ylim(0, H); self.ax.axis('off')
        self.fig.canvas.draw()
        self.r = self.fig.canvas.get_renderer()

    def text(self, x, y, s, **kw):
        kw.setdefault('va', 'center'); kw.setdefault('color', INK)
        return self.ax.text(x, y, s, **kw)

    def width(self, s, **kw):
        t = self.ax.text(0, 0, s, **kw)
        w = t.get_window_extent(renderer=self.r).width / self.fig.dpi
        t.remove()
        return w

    def fit(self, s, max_w, fontsize, **kw):
        while fontsize > 11 and self.width(s, fontsize=fontsize, **kw) > max_w:
            fontsize -= 0.25
        return fontsize

    def box(self, x, y, w, h, r=0.10, **kw):  # (x, y) = lower-left, inches
        p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', **kw)
        self.ax.add_patch(p); return p

    def rect(self, x, y, w, h, **kw):
        p = Rectangle((x, y), w, h, **kw); self.ax.add_patch(p); return p

    def line(self, xs, ys, **kw):
        self.ax.plot(xs, ys, **kw)

    def image_axes(self, x, y, w, h):
        return self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H])


def card(c, x, y0, w, y1):
    c.box(x, y0, w, y1 - y0, r=0.14, facecolor=CARD_BG, edgecolor=CARD_EDGE, linewidth=1.2, zorder=0)


def request_panel(c, ex, x0, width, top, pad):
    req = ex['request']
    y = top - 0.40
    c.text(x0 + pad, y, 'User Request', fontsize=29.2, weight='bold')
    y -= 0.36
    sub = f"{ex['scenario']} · {SCENARIO[ex['scenario']]}"
    if c.width(sub, fontsize=20.6) > width - 2 * pad:      # wrap at the last space that fits
        words, line = sub.split(' '), ''
        while words and c.width((line + ' ' + words[0]).strip(), fontsize=20.6) <= width - 2 * pad:
            line = (line + ' ' + words.pop(0)).strip()
        c.text(x0 + pad, y, line, fontsize=20.6, color=MUTED)
        y -= 0.28
        sub = ' '.join(words)
    c.text(x0 + pad, y, sub, fontsize=20.6, color=MUTED)
    y -= 0.52
    c.text(x0 + pad, y, 'Constraints', fontsize=23.2, weight='bold')
    for label, value in [('FPR', r'$\leq$ ' + fpr_label(req['fpr'])),
                         ('PSNR', rf"$\geq$ {req['psnr_min']:g} dB"),
                         ('Latency', r'$\leq$ ' + budget(req['latency_max_ms']))]:
        y -= 0.38
        c.text(x0 + pad + 0.10, y, label, fontsize=22.1)
        c.text(x0 + pad + 1.20, y, value, fontsize=22.1)
    y -= 0.55
    c.text(x0 + pad, y, f"Attacks ({len(req['attacks'])})", fontsize=23.2, weight='bold')
    y -= 0.08
    fs = 13.6
    for fam, members in FAMILIES:
        present = [a for a in members if a in req['attacks']]
        if not present:
            continue
        y -= 0.32
        c.text(x0 + pad, y, fam, fontsize=18.9, color=MUTED, style='italic')
        y -= 0.37
        x = x0 + pad + 0.08
        for a in present:
            w = c.width(NAME[a], fontsize=fs)
            if x + w + 0.10 > x0 + width - 0.12:
                x = x0 + pad + 0.08; y -= 0.40
            c.text(x, y, NAME[a], fontsize=fs, bbox=dict(boxstyle='round,pad=0.28,rounding_size=0.55',
                   facecolor='white', edgecolor=RULE, linewidth=1.0))
            x += w + 0.22
    return y - 0.25


def solution_panel(c, ex, x0, width, top, pad, attacks):
    req, cfg, v = ex['request'], ex['cfg'], ex['verdict']
    y = top - 0.40
    c.text(x0 + pad, y, 'TAILOR Solution', fontsize=29.2, weight='bold')
    y -= 0.46
    c.text(x0 + pad, y, 'Embedding Order and Strength', fontsize=20.6, color=MUTED)
    y -= 0.44
    bh, bx, bw = 0.42, x0 + pad, width - 2 * pad
    for k, f in enumerate(cfg['order'], 1):
        c.box(bx, y - bh / 2, bw, bh, r=0.07, facecolor='white', edgecolor=FRAG[f], linewidth=1.8)
        c.rect(bx + 0.03, y - bh / 2 + 0.03, 0.13, bh - 0.06, facecolor=FRAG[f], edgecolor='none')
        c.text(bx + 0.32, y, f'{k}.  {f}', fontsize=22.8, weight='bold')
        c.text(bx + bw - 0.18, y, rf'$\lambda$ = {cfg["s"][f]:.2f}', fontsize=22.8, ha='right')
        y -= 0.52
    y += 0.02
    geo = cfg['fe_on'][0] if cfg['fe_on'] else None
    def row(y, label, value, bold):
        c.text(x0 + width - pad, y, value, fontsize=22.1, ha='right', weight='bold' if bold else 'normal')
        room = width - 2 * pad - c.width(value, fontsize=22.1, weight='bold' if bold else 'normal') - 0.16
        c.text(x0 + pad, y, label, fontsize=c.fit(label, room, 20.6), color=MUTED)
    row(y, 'Geometric Recovery', {'scale': 'Scale Search', None: 'None'}.get(geo, str(geo).title()), bool(geo))
    y -= 0.40
    row(y, 'Acceptance Threshold', f"{v['threshold_count']} / 100 bits", True)
    # images: original, watermarked, residual (common 12x display gain, as in Figures 8-10)
    i = ex['display_image_index']
    cover = load_cover(i); wm = load_watermarked(ex['cfg_id'], i)
    res = np.clip(np.abs(wm.astype(np.float64) - cover.astype(np.float64)) * GAIN / 255, 0, 1)
    gap = 0.14
    iw = (width - 2 * pad - 2 * gap) / 3
    iy = y - 0.30 - iw
    for k, (im, cap) in enumerate([(cover, 'Original'), (wm, 'Watermarked'), (res, 'Residual')]):
        x = x0 + pad + k * (iw + gap)
        ax = c.image_axes(x, iy, iw, iw)
        if im.dtype != np.uint8:
            im = (im * 255).round().astype(np.uint8)
        ax.imshow(np.asarray(Image.fromarray(im).resize((384, 384), Image.Resampling.LANCZOS)))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(RULE); s.set_linewidth(0.8)
        c.text(x + iw / 2, iy - 0.20, cap, ha='center', fontsize=18.9, color=MUTED)
    # outcome tiles: requested bound (small label) above the achieved value (large)
    lat = measured_latency(ex['cfg_id'], attacks)
    lat_txt = seconds_or_ms(lat) if lat is not None else seconds_or_ms(v['latency_model_ms']) + ' (model)'
    n = len(attacks)
    tiles = [('Attacks Passed', f'{n} / {n}'),
             (rf"PSNR  $\geq$ {req['psnr_min']:g} dB", f"{v['psnr_live']:.2f} dB"),
             (r'Latency  $\leq$ ' + budget(req['latency_max_ms']), lat_txt),
             (r'FPR Bound  $\leq$ ' + fpr_label(req['fpr']), sci_bold(v['fpr_bound']))]
    pw, ph = (width - 2 * pad - 0.12) / 2, 0.70
    py = iy - 0.46 - ph
    value_fs = min(c.fit(val, pw - 0.20, 24.7, weight='bold') for _, val in tiles)
    label_fs = min(c.fit(label, pw - 0.24, 18.9) for label, _ in tiles)
    for k, (label, val) in enumerate(tiles):
        x = x0 + pad + (k % 2) * (pw + 0.12)
        yy = py - (k // 2) * (ph + 0.10)
        c.box(x, yy, pw, ph, r=0.08, facecolor=GOOD_BG, edgecolor='none', zorder=1)
        c.text(x + 0.14, yy + ph - 0.19, label, fontsize=label_fs, color=MUTED, zorder=2)
        c.text(x + 0.14, yy + 0.23, val, fontsize=value_fs, weight='bold', zorder=2)
    return py - (len(tiles) // 2 - 1) * (ph + 0.10) - 0.02, lat


def verification_panel(c, ex, x0, width, top, pad, attacks):
    cfg, v = ex['cfg'], ex['verdict']
    frags = list(cfg['order'])
    multi = len(frags) > 1
    y = top - 0.40
    c.text(x0 + pad, y, 'Live Check on User Images', fontsize=29.2, weight='bold')
    y -= 0.36
    c.text(x0 + pad, y, 'Image Pass Rate per Attack (100 Images Each)', fontsize=20.6, color=MUTED)
    cols = (frags if multi else []) + ['TAILOR']
    label_w = 1.95
    gx0 = x0 + pad + label_w
    cw = (x0 + width - 0.50 - gx0) / len(cols)
    bar_w = min(cw * 0.80, 1.9)
    if multi:
        y -= 0.50
        span = gx0 + (len(frags) - 1) * cw + bar_w
        tx = gx0 + len(frags) * cw
        group_fs = min(c.fit('Individual Fragment Readouts', span - gx0 + 0.10, 18.9, style='italic'),
                       c.fit('Full Verifier', min(bar_w + 2 * (tx - span) - 0.10, x0 + width - pad - tx), 18.9, style='italic'))
        c.text((gx0 + span) / 2, y, 'Individual Fragment Readouts', ha='center', fontsize=group_fs, color=MUTED, style='italic')
        c.line([gx0, span], [y - 0.16, y - 0.16], color=RULE, linewidth=0.9)
        tx = gx0 + len(frags) * cw
        c.text(tx + bar_w / 2, y, 'Full Verifier', ha='center', fontsize=group_fs, color=MUTED, style='italic')
        c.line([tx, tx + bar_w], [y - 0.16, y - 0.16], color=RULE, linewidth=0.9)
        y -= 0.40
    else:
        y -= 0.52
    head_fs = min(c.fit(col, cw - 0.08, 20.4, weight='bold') for col in cols)
    for j, col in enumerate(cols):
        x = gx0 + j * cw
        c.text(x + bar_w / 2, y, col, ha='center', fontsize=head_fs, weight='bold')
        c.line([x, x + bar_w], [y - 0.17, y - 0.17], color=TAILOR if col == 'TAILOR' else FRAG[col],
               linewidth=3.2, solid_capstyle='butt')
    row_h = 0.34
    y0 = y - 0.45
    fam = {a: f for f, m in FAMILIES for a in m}
    geo_used = False
    for r, a in enumerate(attacks):
        yy = y0 - r * row_h
        if r and fam[a] != fam[attacks[r - 1]]:
            c.line([x0 + pad, x0 + width - pad], [yy + row_h / 2] * 2, color=RULE, linewidth=0.8)
        c.text(x0 + pad, yy, NAME[a], fontsize=20.4)
        av = v['attacks'][a]
        for j, col in enumerate(cols):
            x = gx0 + j * cw
            c.rect(x, yy - row_h * 0.32, bar_w, row_h * 0.64, facecolor=TRACK, edgecolor='none')
            if col == 'TAILOR':
                rate = av['rate']
                geo_share = av['geometry_images'] / av['n']
                prim = rate - geo_share
                c.rect(x, yy - row_h * 0.32, bar_w * prim, row_h * 0.64, facecolor=TAILOR, edgecolor='none')
                if geo_share > 0:
                    geo_used = True
                    c.rect(x + bar_w * prim, yy - row_h * 0.32, bar_w * geo_share, row_h * 0.64,
                           facecolor=TAILOR_LIGHT, edgecolor='white', linewidth=0.6, hatch='////')
                c.text(x + bar_w + 0.06, yy, f'{rate * 100:.0f}', fontsize=19.4, weight='bold')
            else:
                rate = av['fragment_rates'][col]
                c.rect(x, yy - row_h * 0.32, bar_w * rate, row_h * 0.64, facecolor=FRAG[col], edgecolor='none',
                       alpha=1.0 if rate >= .9 - 1e-9 else 0.32)
            c.line([x + bar_w * 0.9] * 2, [yy - row_h * 0.46, yy + row_h * 0.46], color=INK, linewidth=0.9,
                   linestyle=(0, (2, 1.6)))
    # legend row
    ly = y0 - (len(attacks) - 1) * row_h - row_h * 0.6 - 0.24
    lx = x0 + pad
    c.line([lx + 0.10] * 2, [ly - 0.14, ly + 0.14], color=INK, linewidth=0.9, linestyle=(0, (2, 1.6)))
    c.text(lx + 0.24, ly, '90% Gate', fontsize=18.9, color=MUTED)
    lx += 0.24 + c.width('90% Gate', fontsize=18.9) + 0.40
    if multi:
        for f in frags:
            c.rect(lx, ly - 0.09, 0.16, 0.18, facecolor=FRAG[f], alpha=0.32, edgecolor='none')
            lx += 0.20
        c.text(lx + 0.06, ly, 'Fragment Below Gate', fontsize=18.9, color=MUTED)
        lx += 0.06 + c.width('Fragment Below Gate', fontsize=18.9) + 0.40
    if geo_used:
        label = 'Recovered by Scale Search'
        if lx + 0.40 + c.width(label, fontsize=18.9) > x0 + width - pad:   # wrap to a second legend row
            lx, ly = x0 + pad, ly - 0.34
        c.rect(lx, ly - 0.09, 0.30, 0.18, facecolor=TAILOR_LIGHT, edgecolor='white', hatch='////')
        c.text(lx + 0.40, ly, label, fontsize=18.9, color=MUTED)
    return ly - 0.20


def draw(ex, H=None, check=True):
    req = ex['request']
    attacks = [a for _, fam in FAMILIES for a in fam if a in req['attacks']]
    assert sorted(attacks) == sorted(req['attacks'])
    multi = len(ex['cfg']['order']) > 1
    if H is None:
        H = 4.5 + 0.34 * len(attacks) + (0.30 if multi else 0)   # generous first pass
    c = Canvas(H)
    top, bot, pad = H - 0.10, 0.10, 0.22
    rx, rw = 0.05, 3.45
    sx, sw = rx + rw + 0.55, 4.40
    vx = sx + sw + 0.15
    vw = W - 0.05 - vx
    for x, w in [(rx, rw), (sx, sw), (vx, vw)]:
        card(c, x, bot, w, top)
    request_floor = request_panel(c, ex, rx, rw, top, pad)
    ay = (top + bot) / 2
    c.ax.add_patch(FancyArrowPatch((rx + rw + 0.08, ay), (sx - 0.08, ay),
                   arrowstyle='-|>,head_length=0.8,head_width=0.5', mutation_scale=18, linewidth=2.6, color=TAILOR))
    solution_floor, lat = solution_panel(c, ex, sx, sw, top, pad, attacks)
    verify_floor = verification_panel(c, ex, vx, vw, top, pad, attacks)
    layout = dict(H=round(H, 3), request_floor=round(request_floor, 3), solution_floor=round(solution_floor, 3),
                  verify_floor=round(verify_floor, 3), card_bottom=bot)
    if check:
        assert min(request_floor, solution_floor, verify_floor) > bot, layout
    return c.fig, dict(latency_measured_ms=lat, latency_model_ms=ex['verdict']['latency_model_ms'], layout=layout)


def main():
    plt.rcParams.update({'font.family': 'Nimbus Roman', 'mathtext.fontset': 'stix',
                         'text.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'hatch.linewidth': 0.8})
    sel = json.loads((ROOT / 'selected_examples.json').read_text())
    out = OUT; out.mkdir(exist_ok=True)
    report = []
    for ex in sel['examples']:
        fig, info = draw(ex, check=False)         # pass 1: generous height, measure content floors
        lay = info['layout']
        plt.close(fig)
        slack = min(lay['request_floor'], lay['solution_floor'], lay['verify_floor']) - lay['card_bottom']
        fig, info = draw(ex, H=lay['H'] - slack + 0.18)   # pass 2: top-anchored layout shifted up
        stem = f"request_solution_{ex['scenario'].lower()}"
        fig.savefig(out / f'{stem}.pdf', facecolor='white')
        fig.savefig(out / f'{stem}.png', dpi=160, facecolor='white')
        plt.close(fig)
        report.append(dict(example=ex['example'], scenario=ex['scenario'], spec_id=ex['spec_id'], **info))
        print(stem, json.dumps(info), flush=True)
    (out / 'render_report.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
