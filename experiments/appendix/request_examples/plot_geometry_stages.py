"""Diagram the four optional geometric stages: what each adds at embedding and searches at decoding.

The residual and attacked panels are real images from the frozen cached embeddings; the right
panel is a schematic of the view enumeration in unified_detector.py.  Each schematic draws
inside its own (x, y, w, h) box, so panels cannot collide.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Polygon
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from plot_examples import Canvas, INK, MUTED, RULE, CARD_BG, CARD_EDGE, FRAG, TAILOR, TAILOR_LIGHT, GAIN

PROBE = ROOT / 'geo_probe'
W = 8.0
ACCENT = '#b8601f'           # the view the search verifies
CAP = 12.6                   # schematic label size

STAGES = [
    dict(key='scale', title='Scale', serves='VINE',
         residual='res_scale.png', attack=('att_scale_crop50.png', 'Center Crop 50%'),
         ),
    dict(key='tile', title='Tile', serves='TrustMark',
         residual='res_tile.png', attack=('att_tile_border20.png', 'Border Translation'),
         ),
    dict(key='resync', title='Resync', serves='Every Fragment',
         residual='res_resync.png', attack=('att_resync_rot9.png', 'Rotation'),
         ),
    dict(key='angle', title='Angle', serves='Every Fragment',
         residual='res_plain_TM.png', attack=('att_angle_rot9.png', 'Rotation'),
         ),
]


def load(name):
    return np.asarray(Image.open(PROBE / name).convert('RGB'))


def panel(c, arr, x, y, s, caption, edge=RULE):
    ax = c.image_axes(x, y, s, s)
    ax.imshow(np.asarray(Image.fromarray(arr).resize((360, 360), Image.Resampling.LANCZOS)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(edge); sp.set_linewidth(0.9)
    c.text(x + s / 2, y - 0.16, caption, ha='center', fontsize=11.8, color=MUTED)


def wrap(c, text, max_w, fontsize):
    words, lines, line = text.split(), [], ''
    for w in words:
        trial = (line + ' ' + w).strip()
        if line and c.width(trial, fontsize=fontsize) > max_w:
            lines.append(line); line = w
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


# ---- schematics: each draws inside its own (x, y, w, h) box ----------------------

def sch_scale(c, x, y, w, h):
    side = h * 0.60
    cx, by = x + 0.08 + side / 2, y + 0.28
    for frac, style in [(1.0, '-'), (0.72, (0, (3.5, 2))), (0.46, '-')]:
        s = side * frac
        c.rect(cx - s / 2, by + (side - s) / 2, s, s, facecolor='none',
               edgecolor=FRAG['VINE'], linewidth=1.9, linestyle=style)
    c.text(cx, by + side + 0.13, 'Nested Rings', ha='center', fontsize=CAP)
    rx0, rx1 = x + 0.08 + side + 0.34, x + w - 0.06
    ry = by + side * 0.50
    c.line([rx0, rx1], [ry, ry], color=RULE, linewidth=1.3)
    for t in np.linspace(0, 1, 27):
        c.line([rx0 + t * (rx1 - rx0)] * 2, [ry - 0.05, ry + 0.05], color=MUTED, linewidth=0.6)
    hx = rx0 + 0.42 * (rx1 - rx0)
    c.line([hx, hx], [ry - 0.11, ry + 0.11], color=ACCENT, linewidth=2.3)
    c.text((rx0 + rx1) / 2, ry + 0.20, 'Verified Scale', ha='center', fontsize=CAP, color=ACCENT)
    c.text(rx0, ry - 0.17, '0.34', ha='left', va='top', fontsize=CAP, color=MUTED)
    c.text(rx1, ry - 0.17, '1.00', ha='right', va='top', fontsize=CAP, color=MUTED)


def sch_tile(c, x, y, w, h):
    fs = h * 0.64
    fx, fy = x + 0.10, y + 0.26
    c.rect(fx, fy, fs, fs, facecolor='none', edgecolor=RULE, linewidth=1.2)
    c.line([fx + fs / 2] * 2, [fy, fy + fs], color=FRAG['TrustMark'], linewidth=1.5)
    c.line([fx, fx + fs], [fy + fs / 2] * 2, color=FRAG['TrustMark'], linewidth=1.5)
    step = fs / 16
    for gy in range(9):
        for gx in range(9):
            c.ax.plot([fx + gx * step], [fy + fs - gy * step], marker='o', markersize=1.5,
                      color=MUTED, zorder=3)
    c.text(fx + fs / 2, fy + fs + 0.13, '2 $\\times$ 2 Cells', ha='center', fontsize=CAP)
    wx, wy = fx + 3 * step, fy + fs - 2 * step - fs / 2
    c.rect(wx, wy, fs / 2, fs / 2, facecolor='none', edgecolor=ACCENT, linewidth=2.1)
    c.text(fx + fs + 0.26, fy + fs * 0.52, 'Verified Window', fontsize=CAP, color=ACCENT)


def sch_resync(c, x, y, w, h):
    q = h * 0.48
    by = y + 0.32
    warp = [(0.10, 0.04), (0.95, 0.00), (1.00, 0.88), (0.16, 0.98)]
    pts = [(x + 0.12 + a * q, by + b * q) for a, b in warp]
    c.ax.add_patch(Polygon(pts, closed=True, facecolor='none', edgecolor=MUTED, linewidth=1.3))
    for px, py in pts:
        c.ax.plot([px], [py], marker='o', markersize=4.2, color=ACCENT, zorder=4)
    c.text(x + 0.12 + q / 2, by + q + 0.15, 'Predicted Corners', ha='center', fontsize=CAP, color=ACCENT)
    ax0 = x + 0.12 + q + 0.12
    c.ax.add_patch(FancyArrowPatch((ax0, by + q / 2), (ax0 + 0.28, by + q / 2),
                   arrowstyle='-|>,head_length=0.6,head_width=0.38', mutation_scale=14,
                   linewidth=1.8, color=TAILOR))
    rx = ax0 + 0.46
    for k, a in enumerate((6, 3, -3, -6)):
        c.rect(rx + 0.030 * (k + 1), by + 0.030 * (k + 1), q, q, facecolor='white',
               edgecolor=TAILOR_LIGHT, linewidth=1.0, angle=a * 0.6)
    c.rect(rx, by, q, q, facecolor='white', edgecolor=TAILOR, linewidth=1.9)
    c.text(rx + q / 2, by + q + 0.15, 'Rectified', ha='center', fontsize=CAP)
    c.text(rx + q / 2, by - 0.17, 'Plus 4 Tilts', ha='center', va='top', fontsize=CAP, color=MUTED)


def sch_angle(c, x, y, w, h):
    ws = h * 0.48
    wx, wy = x + 0.16, y + 0.32
    c.rect(wx, wy, ws, ws, facecolor='none', edgecolor=RULE, linewidth=1.2)
    for cx0, cy0, sx, sy in [(0, 0, 1, 1), (1, 0, -1, 1), (0, 1, 1, -1), (1, 1, -1, -1)]:
        px, py = wx + cx0 * ws, wy + cy0 * ws
        c.ax.add_patch(Polygon([(px, py), (px + sx * ws * 0.26, py), (px, py + sy * ws * 0.34)],
                       closed=True, facecolor='#2b3440', edgecolor='none'))
    c.line([wx, wx + ws * 0.26], [wy + ws * 0.66, wy + ws], color=ACCENT, linewidth=2.0)
    c.text(wx + ws / 2, wy + ws + 0.15, 'Fill Wedges', ha='center', fontsize=CAP, color=ACCENT)
    c.text(wx + ws / 2, wy - 0.17, 'Slope Estimate', ha='center', va='top', fontsize=CAP, color=MUTED)
    r = h * 0.24
    cx, cy = wx + ws + 0.40 + r, wy + ws / 2
    c.ax.add_patch(Circle((cx, cy), r, facecolor='none', edgecolor=RULE, linewidth=1.2))
    for a in range(-180, 180, 10):
        t = np.deg2rad(a)
        c.line([cx + r * 0.82 * np.cos(t), cx + r * np.cos(t)],
               [cy + r * 0.82 * np.sin(t), cy + r * np.sin(t)], color=MUTED, linewidth=0.6)
    t = np.deg2rad(62)
    c.ax.add_patch(FancyArrowPatch((cx, cy), (cx + r * 0.90 * np.cos(t), cy + r * 0.90 * np.sin(t)),
                   arrowstyle='-|>,head_length=0.55,head_width=0.36', mutation_scale=13,
                   linewidth=2.0, color=ACCENT))
    c.text(cx, cy + r + 0.15, 'Angle Grid', ha='center', fontsize=CAP, color=MUTED)


SCH = dict(scale=sch_scale, tile=sch_tile, resync=sch_resync, angle=sch_angle)


def main():
    plt.rcParams.update({'font.family': 'Nimbus Roman', 'mathtext.fontset': 'stix',
                         'text.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    pad, gap, img, sch_w, arrow = 0.20, 0.26, 1.15, 2.34, 0.34
    label_w = W - 0.10 - 2 * pad - 2 * img - sch_w - 3 * gap - arrow
    rh = img + 0.86
    H = len(STAGES) * rh + 0.10
    c = Canvas(H, width=W)
    x_res = 0.05 + pad + label_w
    x_att = x_res + img + gap
    x_sch = x_att + img + gap + arrow
    assert x_sch + sch_w <= W - 0.05 + 1e-9, 'schematic column runs past the card'
    for i, st in enumerate(STAGES):
        y1 = H - 0.05 - i * rh
        y0 = y1 - rh + 0.12
        c.box(0.05, y0, W - 0.10, rh - 0.12, r=0.11, facecolor=CARD_BG, edgecolor=CARD_EDGE,
              linewidth=1.1, zorder=0)
        # The mechanism and the parameters are in the prose; the panel keeps the name,
        # what the stage serves, and the size of the search it pays for.
        lx = 0.05 + pad
        serves = wrap(c, f"Serves {st['serves']}", label_w - 0.10, 13.4)
        block = 0.36 + 0.30 * len(serves)
        yy = y0 + (rh - 0.12 + block) / 2 - 0.18          # the block sits centred in the card
        c.text(lx, yy, st['title'], fontsize=18.5, weight='bold')
        yy -= 0.36
        for ln in serves:
            c.text(lx, yy, ln, fontsize=13.4, color=MUTED)
            yy -= 0.30
        assert yy > y0 - 0.02, ('label text overflows the card', st['key'])
        iy = y0 + (rh - 0.12 - img) / 2 + 0.07
        unchanged = st['key'] == 'angle'
        panel(c, load(st['residual']), x_res, iy, img,
              'Residual, Unchanged' if unchanged else 'Residual',
              edge=RULE if unchanged else TAILOR)
        panel(c, load(st['attack'][0]), x_att, iy, img, st['attack'][1])
        c.ax.add_patch(FancyArrowPatch((x_sch - arrow + 0.04, iy + img / 2), (x_sch - 0.06, iy + img / 2),
                       arrowstyle='-|>,head_length=0.6,head_width=0.38', mutation_scale=14,
                       linewidth=1.8, color=TAILOR))
        SCH[st['key']](c, x_sch, iy, sch_w, img)
    out = ROOT / 'figures'
    c.fig.savefig(out / 'geometric_stages.pdf', facecolor='white')
    c.fig.savefig(out / 'geometric_stages.png', dpi=170, facecolor='white')
    print('wrote', out / 'geometric_stages.pdf', f'{W:.2f}x{H:.2f} in')


if __name__ == '__main__':
    main()
