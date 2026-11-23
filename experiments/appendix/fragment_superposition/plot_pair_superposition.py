"""A single ordered 3x3 matrix with both independent readouts in each pair cell."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

from plot_fragment_superposition import FRAGMENTS, NAMES, INK, MUTED, validate

COLOR_BOUNDS = [-3.0, -1.0, -.20, -.05, .05]


def draw(data, out):
    rows, standalone = validate(data)
    plt.rcParams.update({
        'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif',
        'font.size': 12, 'text.color': INK, 'axes.labelcolor': INK,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })
    # Explicit bins improve contrast without exaggerating changes within +/-0.05 pp.
    cmap = ListedColormap(['#193f65', '#4c80ac', '#98bfd9', '#e8f0f6'])
    norm = BoundaryNorm(COLOR_BOUNDS, cmap.N)
    fig = plt.figure(figsize=(6.7, 4.9))
    ax = fig.add_axes([.205, .035, .63, .81])
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(2.5, -.5)
    ax.set_xticks(range(3), NAMES)
    ax.set_yticks(range(3), NAMES)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Second embedded (lower)', fontsize=12.5, labelpad=12)
    ax.set_ylabel('First embedded (upper)', fontsize=12.5, labelpad=12)
    ax.tick_params(axis='both', length=0, pad=9, labelsize=11.5, colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    records = []
    for i, first in enumerate(FRAGMENTS):
        for j, second in enumerate(FRAGMENTS):
            if first == second:
                ax.add_patch(Rectangle((j-.48, i-.48), .96, .96,
                    facecolor='#f0f0f0', edgecolor='none'))
                ax.text(j, i, '—', ha='center', va='center',
                    fontsize=22, color='#b1b9c1')
                continue
            row = rows[(first, second)]
            for position, fragment in enumerate((first, second)):
                center = i + (-.245 if position == 0 else .245)
                measured = row['decoders'][fragment]
                delta = measured['mean_delta_pp']
                color = 'white' if delta < -.20 else INK
                ax.add_patch(Rectangle((j-.48, center-.235), .96, .47,
                    facecolor=cmap(norm(delta)), edgecolor='none'))
                change = '0.00' if delta == 0 else f'{delta:+.2f}'.replace('-', '−')
                ax.text(j, center, change, ha='center', va='center',
                    fontsize=19, weight='bold', color=color)
                records.append(dict(first=first, second=second, decoder=fragment,
                    position=position+1, bit_accuracy_percent=measured['mean_ba_percent'],
                    delta_pp=delta))
    cax = fig.add_axes([.87, .13, .027, .62])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, spacing='uniform', ticks=COLOR_BOUNDS, drawedges=True)
    cb.ax.set_yticklabels(['−3', '−1', '−0.20', '−0.05', '+0.05'])
    cb.ax.tick_params(length=0, pad=5, labelsize=10, colors=MUTED)
    cb.dividers.set_color('white')
    cb.dividers.set_linewidth(1.2)
    cb.outline.set_visible(False)
    cb.set_label(r'$\Delta$BA (pp)', fontsize=12, labelpad=8)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'pair_superposition_matrix.{ext}', dpi=400,
            facecolor='white', bbox_inches='tight', pad_inches=.055)
    fig.savefig(out / 'pair_superposition_matrix_preview.png', dpi=180,
        facecolor='white', bbox_inches='tight', pad_inches=.055)
    plt.close(fig)
    assert len(records) == 12
    assert all(COLOR_BOUNDS[0] <= r['delta_pp'] < COLOR_BOUNDS[-1] for r in records)
    return standalone, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path,
        default=Path(__file__).resolve().parents[1]/'data/fragment_superposition_20260924.json')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.data.read_text())
    standalone, records = draw(data, args.output_dir)
    (args.output_dir/'pair_matrix_validation.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'n_images_per_readout': 100, 'n_pair_orders': 6,
        'n_pair_readouts': 1200, 'standalone_ba_percent': standalone,
        'diagonal': 'masked; same-watermark pair was not measured',
        'color_quantity': 'composition minus standalone BA, percentage points',
        'color_boundaries_pp': COLOR_BOUNDS,
        'scale': 'explicit discrete intervals; shared by both readouts and all pairs',
        'annotations': 'signed delta only; upper=first watermark, lower=second',
        'cells': records,
    }, indent=2)+'\n')
    print(args.output_dir/'pair_superposition_matrix.pdf')


if __name__ == '__main__':
    main()
