"""Stage the paper tables and figures from the audited analysis files; never writes into the paper sources.

Every table is generated from a run's analysis JSON. A result whose run is still incomplete is written as
Pending (never a partial number, never a number from another cohort). House style: no double dashes, single
hyphen for an undefined cell, no links.

Usage: python3 experiments/tools/export_staging.py [OUT_DIR]   (default experiments/staging)
Outputs: tab/fpr_retry.tex, tab/interference_case_study.tex, figures/interference_matrix.pdf,
         tab/search_budget.tex, figures/search_budget.pdf, tab/heldout_transfer.tex, tab/main_controls.tex,
         staging_manifest.json (source sha256 of every input)
"""
import os
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get('TAILOR_RUNS', 'runs'))
SRC = dict(
    fpr=BASE / 'fpr_fresh_identity_20261112/analysis/fpr_summary.json',   # fresh deployment identities (theorem)
    fpr_calibration_ids=BASE / 'fpr_retries_20261109/analysis/fpr_summary.json',
    matrix=BASE / 'interference_20261109/analysis/pair_interference_matrix.json',
    case=BASE / 'interference_20261109/case_study/interference_case_study.json',
    budget=BASE / 'grid_c2_live_20261110/analysis/grid_search_summary.json',
    heldout=BASE / 'heldout_20261109/report_final/heldout_summary.json',
    pr1=BASE / 'main_controls_20261109/report_final/pr1_summary.json',
    distortion=BASE / 'interference_20261109/distortion/distortion_diagnostic.json',
)
LABEL = ['1e-1', '1e-2', '1e-4', '1e-6', '1e-9', '2^-37']
TEXA = {'1e-1': '$10^{-1}$', '1e-2': '$10^{-2}$', '1e-4': '$10^{-4}$', '1e-6': '$10^{-6}$', '1e-9': '$10^{-9}$',
        '2^-37': '$2^{-37}$'}
FRAG = {'VINE': 'VINE', 'TrustMark': 'TrustMark', 'VideoSeal': 'VideoSeal'}
BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#2a78d6', '#1c5cab', '#104281', '#0d366b']
INK, GRID = '#3a3a38', '#e6e5e0'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sci(x):
    e = math.floor(math.log10(x))
    return f'${x / 10 ** e:.1f}\\times 10^{{{e}}}$'


def signed(x, d=3, math_mode=True):
    x = 0.0 if abs(x) < 0.5 * 10 ** -d else x
    t = f'{x:+.{d}f}'
    return f'${t}$' if math_mode else t


def pct(x, d=2):
    return '-' if x is None else f'{100 * x:.{d}f}'


def write(path, text):
    assert '--' not in text.replace('\\cmidrule', ''), f'double dash in {path.name}'
    assert 'http' not in text, f'link in {path.name}'
    path.write_text(text)


def table(caption, label, spec, header, rows, note=None):
    body = '\n'.join(rows)
    foot = f'\\par\\smallskip\n\\parbox{{\\linewidth}}{{\\footnotesize {note}}}\n' if note else ''
    return (f'\\begin{{table}}[t]\n\\centering\n\\small\n\\caption{{{caption}}}\n\\label{{{label}}}\n'
            f'\\setlength{{\\tabcolsep}}{{4pt}}\n\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\n\\toprule\n{header} \\\\\n'
            f'\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n{foot}\\end{{table}}\n')


def fpr_table(out):
    """Fresh-identity negatives: every (configuration, image) trial is queried under its own deployment identity that
    calibration never used, the condition of Appendix B, so trials are independent given the images."""
    s = read(SRC['fpr'])
    if not s.get('complete'):
        return 'pending'

    def cell(r):
        if not r['events']:
            return '0'
        return f"{100 * r['rate']:.3f}" if 100 * r['rate'] < 0.1 else pct(r['rate'])
    rows = []
    for a in LABEL:
        cells = [cell(s['by_alpha'][f'{kind}|{a}'][g]) for kind in ('clean', 'wrong') for g in ('first_try', 'retry')]
        model = s['by_alpha'][f'clean|{a}']['first_try']['union_bound_prediction']
        mtxt = f'{100 * model:.2f}' if model >= 1e-4 else sci(100 * model)
        rows.append(f"{TEXA[a]} & " + ' & '.join(cells) + f' & {mtxt} \\\\')
    n_clean = s['by_alpha']['clean|1e-1']['first_try']['trials']
    n_wrong = s['by_alpha']['wrong|1e-1']['first_try']['trials']
    tm = s['fragment_calibration']['clean|TrustMark|1e-1']
    per = tm.get('events_per_configuration') or [tm['events']]
    tm_events, tm_n = sum(per), len(per) * tm['images']
    note = (f"Rates in percent over {n_clean:,} (never watermarked) and {n_wrong:,} (wrong identity) configuration and "
            f"image trials per group, each queried under its own deployment identity that calibration never used "
            f"(Appendix~\\ref{{sec:fpr-selection}}), so the trials are independent given the images; zero means no "
            f"false acceptance. TrustMark alone: {tm_events}/{tm_n:,} at $10^{{-1}}$, model "
            f"{100 * tm['model_rate']:.2f}.")
    # strict alphas: report any acceptance together with the model's chance of seeing at least one in these cells
    strict = ['1e-6', '1e-9', '2^-37']
    cells = [s['by_alpha'][f'{kind}|{a}'][g] for a in strict for kind in ('clean', 'wrong') for g in ('first_try', 'retry')]
    hits = sum(c['events'] for c in cells)
    if hits:
        p_none = 1.0
        for c in cells:
            p_none *= (1 - min(1.0, c['union_bound_prediction'])) ** c['trials']
        note += (f" At $\\alpha\\le10^{{-6}}$ the groups record {hits} acceptance{'s' if hits > 1 else ''}, and the "
                 f"model gives a {100 * (1 - p_none):.0f}\\% chance of at least one.")
    text = table('False acceptance of the frozen \\sysfull{} deployments under fresh deployment identities, first-try '
                 'versus retry-selected configurations, evaluated at every request $\\alpha$ with each configuration\'s '
                 'own path count.', 'tab:fpr-retry',
                 'lrrrrr',
                 '& \\multicolumn{2}{c}{Never watermarked} & \\multicolumn{2}{c}{Wrong identity} & \\\\\n'
                 '\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n$\\alpha$ & First try & Retry & First try & Retry & Model bound',
                 rows, note)
    write(out / 'tab/fpr_retry.tex', text)
    return 'final'


def interference(out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    m = read(SRC['matrix'])
    attacks = ['jpeg25', 'crop75', 'rot9', 'regen']
    pairs = sorted({(r['first'], r['second']) for r in m})
    def grid(select):
        g = np.full((len(pairs), len(attacks)), np.nan)
        for i, p in enumerate(pairs):
            for j, a in enumerate(attacks):
                vals = [r['host_drop'] for r in m if (r['first'], r['second']) == p and r['attack'] == a and select(r)]
                if vals:
                    g[i, j] = max(vals) if select is ALL else vals[0]
        return g
    ALL = lambda r: True
    mid = grid(lambda r: r['level_first'] == 'mid' and r['level_second'] == 'mid')
    worst = grid(ALL)
    cmap = LinearSegmentedColormap.from_list('seq', BLUE)
    vmax = max(np.nanmax(mid), np.nanmax(worst))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, g, title in zip(axes, (mid, worst), ('Mid strengths', 'Largest over the 9 strength pairs')):
        im = ax.imshow(np.clip(g, 0, None), cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g[i, j]
                ax.text(j, i, signed(v, 3, math_mode=False).replace('-', '\u2212'), ha='center', va='center', fontsize=7.5,
                        color='white' if v > 0.6 * vmax else INK)
        ax.set_xticks(range(len(attacks)), attacks, fontsize=8, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(pairs)), [f'{FRAG[a]} then {FRAG[b]}' for a, b in pairs], fontsize=8, color=INK)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label('bit-accuracy drop of the first fragment', fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(out / 'figures/interference_matrix.pdf', bbox_inches='tight')
    fig.savefig(out / 'figures/interference_matrix_preview.png', bbox_inches='tight', dpi=110)
    plt.close(fig)
    c = read(SRC['case'])
    rows = []
    names = dict(unit='Direct composition', order='Reversed order', strength='Raised VideoSeal strength')
    for k in ('unit', 'order', 'strength'):
        v = c['versions'][k]
        vs = [v['attacks'][a]['fragments']['VideoSeal']['drop'] for a in attacks]
        det = min(v['attacks'][a]['pair_detection']['1e-4']['rate'] for a in ('jpeg25', 'crop75', 'rot9'))
        rows.append(f"{names[k]} & {' then '.join(v['order'])} & {v['psnr_db']:.2f} & "
                    + ' & '.join(signed(x) for x in vs) + f' & {100 * det:.0f} \\\\')
    ft = c['full_tailor']['groups']
    first = ft.get('VideoSeal before TrustMark', {})
    second = ft.get('TrustMark before VideoSeal', {})
    note = (f"\\sysfull{{}} deploys the pair in {first.get('deployments', 0) + second.get('deployments', 0):,} requests, "
            f"{second.get('deployments', 0):,} with TrustMark first; held-out pass "
            f"{first.get('heldout_pass', 0)}/{first.get('judged', 0)} and {second.get('heldout_pass', 0)}/{second.get('judged', 0)}"
            f"{'' if c['full_tailor'].get('heldout_complete') else ' (held-out run incomplete)'}.")
    text = table('High-interference pair chosen from the profiling matrix and verified on the held-out images: '
                 'VideoSeal bit-accuracy drop caused by TrustMark, and the lowest composite detection rate at '
                 '$\\alpha=10^{-4}$ over jpeg25, crop75 and rot9.',
                 'tab:interference-case', 'llrrrrrr',
                 'Version & Order & PSNR & jpeg25 & crop75 & rot9 & regen & Det. (\\%)', rows, note)
    write(out / 'tab/interference_case_study.tex', text)
    return 'final' if c['full_tailor'].get('heldout_complete') else 'final except the held-out counts'


def search_budget(out):
    s = read(SRC['budget']) if SRC['budget'].exists() else None
    if not s or not s.get('complete'):
        write(out / 'tab/search_budget.tex', '% Pending: the C2 live validation is incomplete.\n')
        return 'pending'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = [0.1, 1, 10, 60, 300, 1200, math.inf]
    rows = []
    for r in s['curve']:
        if r['budget_s'] in show:
            b = '$\\infty$' if r['budget_s'] == math.inf else f"{r['budget_s']:g}"
            rows.append(f"{b} & {pct(r['tailor_live_satisfaction'])} & {pct(r['c2_live_satisfaction_low'])} \\\\")
    p = s['paired_psnr']
    note = (f"Median offline time per request {s['tailor']['seconds']['median']:.1f} s (\\sysfull{{}}) and "
            f"{s['c2']['seconds']['median']:.2f} s (grid). Paired PSNR on {p['joint']:,} joint successes: grid minus "
            f"\\sysfull{{}} {signed(p['mean_c2_minus_tailor'], 2)} dB.")
    write(out / 'tab/search_budget.tex', table(
        'Live satisfaction (\\%) at matched offline search budgets: \\sysfull{} versus exhaustive grid enumeration '
        'over the measured strength knots.', 'tab:search-budget', 'lrr', 'Budget (s) & \\sysfull{} & Grid', rows, note))
    xs = [r['budget_s'] for r in s['curve'] if r['budget_s'] != math.inf]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.step(xs, [100 * r['tailor_live_satisfaction'] for r in s['curve'] if r['budget_s'] != math.inf], where='post',
            color='#2a78d6', lw=2, label='TAILOR-F')
    ax.step(xs, [100 * r['c2_live_satisfaction_low'] for r in s['curve'] if r['budget_s'] != math.inf], where='post',
            color='#eb6834', lw=2, ls='--', label='Grid enumeration')
    ax.set_xscale('log')
    ax.set_ylim(0, 100)
    ax.set_xlabel('offline search budget per request (s)', fontsize=9, color=INK)
    ax.set_ylabel('requests satisfied (%)', fontsize=9, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'figures/search_budget.pdf')
    fig.savefig(out / 'figures/search_budget_preview.png', dpi=110)
    plt.close(fig)
    return 'final'


def heldout(out):
    s = read(SRC['heldout']) if SRC['heldout'].exists() else None
    if not s or not s.get('complete'):
        write(out / 'tab/heldout_transfer.tex', '% Pending: the held-out run is incomplete.\n')
        return 'pending'
    rows = []
    for k in ('S1', 'S2', 'S3', 'S4', 'S5', 'all'):
        v = s['scenarios'][k]
        lo = v['heldout_cluster95'][0]
        rows.append(f"{'All' if k == 'all' else k} & {v['n']:,} & {v['configurations']} & {v['heldout_percent']:.2f} & "
                    f"{100 * lo:.2f} & {signed(v['psnr_shift_mean'], 2)} \\\\")
    write(out / 'tab/heldout_transfer.tex', table(
        'Held-out transfer: every accepted \\sysfull{} request judged again on 100 new images with its frozen '
        'configuration.', 'tab:heldout', 'lrrrrr',
        'Scenario & Requests & Configs & Pass (\\%) & Lower bound (\\%) & $\\Delta$PSNR (dB)', rows,
        'Lower bound: one-sided 95\\%, configurations resampled (requests sharing a configuration share its cells).'))
    return 'final'


def main_controls(out):
    s = read(SRC['pr1']) if SRC['pr1'].exists() else None
    if not s or not all(v['complete'] for v in s['variants'].values()):
        write(out / 'tab/main_controls.tex', '% Pending: the main-control rows are incomplete.\n')
        return 'pending'
    names = {'SingleOptimized': 'TAILOR-Single', 'AlwaysAllThree': 'Always-All-Three'}
    rows = []
    for k, name in names.items():
        v = s['variants'][k]
        rows.append(f"{name} & {v['percent']:.2f} & {v['macro']:.2f} & "
                    + ' & '.join(f"{v['scenarios'][x]['percent']:.2f}" for x in ('S1', 'S2', 'S3', 'S4', 'S5'))
                    + f" & {v['psnr']:.2f} & {v['ssim']:.4f} \\\\")
    write(out / 'tab/main_controls.tex', table(
        'Composition necessity on the 7,321 requests: one optimized fragment, and all three fragments forced.',
        'tab:main-controls', 'lrrrrrrrrr',
        'Method & Overall & Macro & S1 & S2 & S3 & S4 & S5 & PSNR & SSIM', rows))
    return 'final'


def distortion(out):
    import shutil
    s = read(SRC['distortion'])
    shutil.copy2(SRC['distortion'].with_suffix('.pdf'), out / 'figures/distortion_diagnostic.pdf')
    keys = [('deployed, all', 'Deployed configurations'), ('deployed, one fragment', '  one fragment'),
            ('deployed, two fragments', '  two fragments'), ('deployed, three fragments', '  three fragments'),
            ('deployed, with a geometric stage', '  with a geometric stage'),
            ('profiling, pair', 'Profiling pairs (both orders)'), ('profiling, triple', 'Profiling triples (six orders)'),
            ('profiling triples, pairwise refit on measured singles and pairs', 'Triples from measured pairs'),
            ('appendix points', 'Previous appendix points')]
    rows = []
    for k, name in keys:
        v = s['sets'][k]
        rows.append(f"{name.replace('  ', '\\quad ')} & {v['n']:,} & {v['mean_abs']:.3f} & {v['p95_abs']:.3f} & "
                    f"{v['max_abs']:.3f} & {v['over_0_3']} \\\\")
    write(out / 'tab/distortion_diagnostic.tex', table(
        'Distortion model against measured PSNR: absolute prediction error in dB.', 'tab:distortion-diagnostic',
        'lrrrrr', 'Set & $n$ & Mean & 95th pct. & Max & $>0.3$ dB', rows,
        'Deployed: the solver\'s predicted PSNR against the live PSNR on the 100 selection images. Profiling: the '
        'stored curves at the PR3 knots. Triples from measured pairs: the pairwise model refit on measured singles '
        'and pairs, applied to the 24 measured triples.'))
    return 'final'


def main(out_dir):
    out = Path(out_dir)
    (out / 'tab').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    status = dict(fpr=fpr_table(out), interference=interference(out), search_budget=search_budget(out),
                  heldout=heldout(out), main_controls=main_controls(out), distortion=distortion(out))
    manifest = dict(status=status, sources={k: (str(p), sha(p) if p.exists() else None) for k, p in SRC.items()})
    (out / 'staging_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(status, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else REPO / 'experiments/staging')
