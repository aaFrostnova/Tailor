"""Plot all measured orders; export large previews to an explicit scratch directory."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

FRAGMENTS = ['vine', 'trustmark', 'videoseal']
NAMES = ['VINE', 'TrustMark', 'VideoSeal']
LETTERS = dict(zip(FRAGMENTS, 'VTS'))
INK = MUTED = '#000000'


def validate(data):
    assert data['complete'] and data['n_images'] == 100
    rows = {tuple(r['order']): r for r in data['results']}
    expected = {p for k in (1, 2, 3) for p in itertools.permutations(FRAGMENTS, k)}
    assert set(rows) == expected and len(data['results']) == 15
    assert data['settings']['attacks'] == ['clean']
    assert all(v == 1 for v in data['settings']['strengths'].values())
    assert not any(data['settings'][k] for k in ('geometry', 'fusion', 'ecc_decoding'))
    standalone = {f: rows[(f,)]['decoders'][f]['mean_ba_percent'] for f in FRAGMENTS}
    readouts = 0
    for order, row in rows.items():
        assert set(row['decoders']) == set(order)
        for f, m in row['decoders'].items():
            assert m['n'] == 100
            assert abs(m['mean_ba_percent'] - standalone[f] - m['mean_delta_pp']) < 1e-8
            readouts += m['n']
    assert readouts == data['fragment_readouts'] == 3300
    return rows, standalone


def draw(rows, standalone, out):
    plt.rcParams.update({
        'font.family': 'Nimbus Roman', 'mathtext.fontset': 'stix',
        'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })
    # One linear scale across both panels: small gains are not visually amplified.
    cmap = LinearSegmentedColormap.from_list('change', ['#164d78', '#6392b5', '#b8d2e2', '#f3f7f9'])
    norm = Normalize(-4.10, 0.05)
    fig = plt.figure(figsize=(7.6, 3.08))
    cell_records = []
    for size, left, title in [(2, .105, 'Two watermarks'), (3, .625, 'Three watermarks')]:
        ax = fig.add_axes([left, .205, .355, .650])
        orders = list(itertools.permutations(FRAGMENTS, size))
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(5.5, -.5)
        ax.axis('off')
        fig.text(left + .1775, .994, title, ha='center', va='top', fontsize=11.0, weight='bold')
        for j, (f, name) in enumerate(zip(FRAGMENTS, NAMES)):
            ax.text(j, -.84, f'{name} ({LETTERS[f]})', ha='center', va='center',
                    fontsize=8.6, clip_on=False)
        for i, order in enumerate(orders):
            ax.text(-.58, i, r'$' + r'\!\to\!'.join(LETTERS[f] for f in order) + '$',
                    ha='right', va='center', fontsize=9.5, clip_on=False)
            for j, f in enumerate(FRAGMENTS):
                m = rows[order]['decoders'].get(f)
                if m is None:
                    ax.add_patch(Rectangle((j-.477, i-.47), .954, .94,
                                           facecolor='#f1f1f1', edgecolor='none'))
                    ax.text(j, i, '—', ha='center', va='center', color='#adb4bb', fontsize=12)
                    continue
                delta = m['mean_delta_pp']
                color = 'white' if delta < -2.3 else INK
                ax.add_patch(Rectangle((j-.477, i-.47), .954, .94,
                                       facecolor=cmap(norm(delta)), edgecolor='none'))
                ax.text(j, i-.125, f"{m['mean_ba_percent']:.2f}",
                        ha='center', va='center', fontsize=10.4, weight='bold', color=color)
                signed = '0.00' if delta == 0 else f'{delta:+.2f}'.replace('-', '−')
                ax.text(j, i+.195, f'({signed})', ha='center', va='center',
                        fontsize=9.6, color=color if delta < -2.3 else MUTED)
                cell_records.append(dict(order=list(order), fragment=f,
                                         ba_percent=m['mean_ba_percent'], delta_pp=delta))
    cax = fig.add_axes([.255, .089, .47, .036])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cb.set_ticks([-4, -3, -2, -1, 0])
    cb.ax.tick_params(length=0, pad=3, labelsize=8.5, colors=MUTED)
    cb.outline.set_visible(False)
    fig.text(.748, .107, r'$\Delta$BA (pp)', fontsize=9.2, ha='left', va='center')
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'fragment_superposition.{ext}', dpi=400, facecolor='white', bbox_inches='tight', pad_inches=.04)
    fig.savefig(out / 'fragment_superposition_preview.png', dpi=180,
                facecolor='white', bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    return cell_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'data/fragment_superposition_20260924.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.data.read_text())
    rows, standalone = validate(data)
    records = draw(rows, standalone, args.output_dir)
    (args.output_dir / 'figure_validation.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'standalone_ba_percent': standalone, 'n_images_per_cell': 100,
        'n_pair_orders': 6, 'n_triple_orders': 6, 'n_plotted_readouts': 3000,
        'color_quantity': 'composition minus standalone BA, percentage points',
        'color_scale': [-4.10, .05], 'scale': 'linear; shared across all cells',
        'cells': records,
    }, indent=2) + '\n')
    print(args.output_dir / 'fragment_superposition.pdf')


if __name__ == '__main__':
    main()
