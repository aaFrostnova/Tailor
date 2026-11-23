"""PR4 analysis: false acceptance of the frozen TAILOR-F deployments on fresh negatives.

Inputs (RUNF = TAILOR_MAIN_CONTROLS_RUN): manifest.json (tasks; per configuration the deployment kinds and
alphas of the requests that deployed it) and negatives/<cfg_id>_<kind>.json (decode_negatives.py: the deployed
verifier run once per configuration and negative image; per threshold bucket the accept decision and score).

For configuration c and alpha, the threshold is h = threshold_count(c, alpha) with the configuration's own path
count T(c); a false acceptance is `detected` in bucket h. Every configuration is evaluated at every request
alpha (identical alpha bins), and separately at the alphas it was deployed with.

Groups: `first_try` (the initial-margin rank-1 candidate passed; Top1 deploys only these) and `retry` (rank 2/3,
request-local correction or a backoff margin; Top3/Full also deploy these). All retry configurations were
measured; first-try-only configurations are a seeded sample, so policy-level rates use inverse-probability
weights (first_try_only / first_try_sampled) times the number of deployments.

Uncertainty: exact one-sided 95% Clopper-Pearson bounds treating each (configuration, image) trial as
independent (OPTIMISTIC: on a never-watermarked image the decisions depend on the decoders and paths, not on
the strengths, so configurations sharing fragments repeat the same decisions), plus cluster-aware intervals
that resample images and configurations; with zero events the cluster-aware bound uses the distinct images as
the effective sample. Per-fragment null calibration uses the distinct images as the sample. Null diagnostics: the empirical tail of the maximum 100-bit match count against the
union-bound model min(1, T(c) * P[Bin(100, 1/2) >= h]); single-path tails of configurations without geometry
against Bin(100, 1/2); dispersion across images and identities.

Usage: TAILOR_MAIN_CONTROLS_RUN=RUNF python experiments/fpr/analyze_fpr.py [--allow-partial]
Outputs: RUNF/analysis/{fpr_retry_results.jsonl, fpr_summary.json, fpr_retry_table.csv, null_tail.pdf}
"""
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'main_controls'))
from common import CP, RUN, atomic, cfgkey, read  # noqa: E402
import unified_detector as UD  # noqa: E402  (frozen copy; common puts the relocated root first on sys.path)

ALPHAS = tuple(CP.REQUEST_ALPHAS)
LABEL = {0.1: '1e-1', 0.01: '1e-2', 1e-4: '1e-4', 1e-6: '1e-6', 1e-9: '1e-9', 2 ** -37: '2^-37'}
GROUPS = ('first_try', 'retry')
KINDS = ('clean', 'wrong')
B = 2000
SEED = 20261110


def cp_upper(k, n, conf=0.95):
    """Exact one-sided Clopper-Pearson upper bound for k events in n Bernoulli trials."""
    if n <= 0:
        return float('nan')
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(conf, k + 1, n - k))


def cp_interval(k, n, conf=0.95):
    """Exact two-sided Clopper-Pearson interval."""
    lo = 0.0 if k == 0 else float(stats.beta.ppf((1 - conf) / 2, k, n - k + 1))
    hi = 1.0 if k >= n else float(stats.beta.ppf(1 - (1 - conf) / 2, k + 1, n - k))
    return [lo, hi]


def binom_tail(h, n=100):
    """P[Bin(n, 1/2) >= h]."""
    return float(stats.binom.sf(h - 1, n, 0.5))


def parse_alpha(text):
    a = float(text)
    for x in ALPHAS:
        if math.isclose(a, x, rel_tol=1e-9):
            return x
    raise ValueError(f'alpha {text} is not a request alpha')


def t_bin(t):
    return '1' if t == 1 else '3-4' if t <= 4 else '5-99' if t < 100 else '100-199' if t < 200 else '200+'


SCHEMES = set()          # identity scheme of the stored negatives (decode_negatives.py); must be uniform


def load(run, allow_partial=False):
    manifest = read(run / 'manifest.json')
    recs, missing = [], []
    for task in manifest['tasks']:
        path = run / 'negatives' / f"{task['cfg_id']}_{task['kind']}.json"
        if not path.exists():
            missing.append(path.name)
            continue
        d = read(path)
        if d['task']['cfg_id'] != task['cfg_id'] or d['task']['kind'] != task['kind']:
            raise SystemExit(f'{path.name}: task mismatch')
        SCHEMES.add((d.get('identity_scheme') or dict(kind='calibration_mod100'))['kind'])
        if len(SCHEMES) > 1:
            raise SystemExit(f'{path.name}: mixed identity schemes {sorted(SCHEMES)}')
        if d['cell']['protocol'] != CP.CELL_PROTOCOL or d['protocol'] != CP.PROTOCOL:
            raise SystemExit(f'{path.name}: protocol mismatch')
        cand = task['candidate']
        if CP.exact_cfg_id(cfgkey(cand)) != task['cfg_id']:
            raise SystemExit(f'{path.name}: configuration does not match its id')
        cfg = dict(order=cand['order'], fe_on=cand['fe_on'])
        buckets = d['cell']['by_threshold']
        fa, h = {}, {}
        for a in ALPHAS:
            k = UD.threshold_count(cfg, a)
            if str(k) not in buckets:
                raise SystemExit(f'{path.name}: no bucket for alpha {a} (threshold {k})')
            fa[a] = np.asarray(buckets[str(k)]['detected'], bool)
            h[a] = k
        if any(len(v) != task['n'] for v in fa.values()):
            raise SystemExit(f'{path.name}: wrong image count')
        top = buckets[max(buckets, key=int)]
        group = manifest['config_groups'][task['cfg_id']]
        recs.append(dict(
            cfg_id=task['cfg_id'], group=task['group'], kind=task['kind'], n=task['n'], order=cand['order'],
            fe_on=sorted(cand['fe_on']), T=UD.budget_for_config(cfg)['max_tests'], h=h, fa=fa,
            maxcount=np.rint(np.asarray(top['accepted_score'], float) * 100).astype(int),
            single=({f: np.asarray(top['per_fragment_ba'][f], float) for f in cand['order']}
                    if not cand['fe_on'] else {}),
            deploy_kinds=dict(group['kinds']),
            deploy_alphas={parse_alpha(k): v for k, v in group['alphas'].items()},
            device=d['device']))
    if missing and not allow_partial:
        raise SystemExit(f'{len(missing)} negative outputs missing; rerun with --allow-partial for an interim view')
    return manifest, recs, missing


def matrix(recs, kind, alpha, group=None):
    rows = [r for r in recs if r['kind'] == kind and (group is None or r['group'] == group)]
    if not rows:
        return rows, np.zeros((0, 0), bool)
    return rows, np.stack([r['fa'][alpha] for r in rows])


def cluster_interval(fa, rng, b=B):
    """95% percentile interval of the pooled rate, resampling configurations and images."""
    c, n = fa.shape
    if c == 0 or fa.sum() == 0:
        return None
    rates = np.empty(b)
    for i in range(b):
        rows = rng.integers(0, c, c)
        cols = rng.integers(0, n, n)
        rates[i] = fa[np.ix_(rows, cols)].mean()
    return [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]


def stratum_stats(fa, alpha, rng):
    c, n = fa.shape
    k, trials = int(fa.sum()), int(fa.size)
    images_hit = int(fa.any(axis=0).sum()) if c else 0
    out = dict(configurations=c, images=n, trials=trials, events=k, rate=k / trials if trials else None,
               cp_upper95=cp_upper(k, trials), nominal_alpha=alpha,
               images_with_any_event=images_hit)
    if k == 0:
        # cluster-aware: the distinct images are the effective sample when every configuration reuses them
        out['cluster_upper95'] = cp_upper(0, n)
    else:
        out['cluster_ci95'] = cluster_interval(fa, rng)
    return out


def weights(manifest, rec):
    conf = manifest['configurations']
    return conf['first_try_only'] / conf['first_try_sampled'] if rec['group'] == 'first_try' else 1.0


def policy_rates(manifest, rows, fa):
    """Deployment-weighted false-acceptance rate of Top1 (first-try deployments) and Full (all deployments)."""
    w = np.array([weights(manifest, r) for r in rows])
    n_first = np.array([r['deploy_kinds'].get('first_try', 0) for r in rows], float)
    n_all = np.array([sum(r['deploy_kinds'].values()) for r in rows], float)
    rate = fa.mean(axis=1)
    top1 = float((w * n_first * rate).sum() / (w * n_first).sum())
    full = float((w * n_all * rate).sum() / (w * n_all).sum())
    return top1, full


def policy_bootstrap(manifest, rows, fa, rng, b=B):
    if fa.sum() == 0:
        return None
    groups = np.array([r['group'] for r in rows])
    idx = {g: np.flatnonzero(groups == g) for g in GROUPS}
    diffs, top1s, fulls = [], [], []
    for _ in range(b):
        pick = np.concatenate([rng.choice(v, len(v)) for v in idx.values() if len(v)])
        cols = rng.integers(0, fa.shape[1], fa.shape[1])
        sub = fa[np.ix_(pick, cols)]
        t, f = policy_rates(manifest, [rows[i] for i in pick], sub)
        top1s.append(t)
        fulls.append(f)
        diffs.append(f - t)
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return dict(top1_ci95=pct(top1s), full_ci95=pct(fulls), full_minus_top1_ci95=pct(diffs))


def group_difference(recs, kind, alpha, rng, b=B):
    """Retry minus first-try pooled rate; images paired, configurations resampled within group."""
    _, f1 = matrix(recs, kind, alpha, 'first_try')
    _, f2 = matrix(recs, kind, alpha, 'retry')
    if not len(f1) or not len(f2):
        return None
    diff = float(f2.mean() - f1.mean())
    if f1.sum() + f2.sum() == 0:
        return dict(retry_minus_first_try=diff, ci95=None)
    d = np.empty(b)
    n = f1.shape[1]
    for i in range(b):
        cols = rng.integers(0, n, n)
        a = f1[np.ix_(rng.integers(0, len(f1), len(f1)), cols)].mean()
        r = f2[np.ix_(rng.integers(0, len(f2), len(f2)), cols)].mean()
        d[i] = r - a
    return dict(retry_minus_first_try=diff, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def predicted_rate(rows, alpha):
    """Mean over configurations of the union-bound value min(1, T * P[Bin >= h]) at their threshold."""
    return float(np.mean([min(1.0, r['T'] * binom_tail(r['h'][alpha])) for r in rows])) if rows else None


def null_tail(recs, kind, group=None):
    rows = [r for r in recs if r['kind'] == kind and (group is None or r['group'] == group)]
    if not rows:
        return None
    hs = list(range(50, 91))
    mx = np.concatenate([r['maxcount'] for r in rows])
    emp = [float((mx >= h).mean()) for h in hs]
    model = [float(np.mean([min(1.0, r['T'] * binom_tail(h)) for r in rows])) for h in hs]
    one = [r for r in rows if r['T'] == 1]
    one_emp = ([float((np.concatenate([r['maxcount'] for r in one]) >= h).mean()) for h in hs] if one else None)
    return dict(h=hs, empirical=emp, union_bound_model=model, trials=int(mx.size),
                single_path_configs=len(one), single_path_empirical=one_emp,
                single_path_binomial=[binom_tail(h) for h in hs])


def single_path(recs):
    out = {}
    for kind in KINDS:
        per = collections.defaultdict(list)
        for r in recs:
            if r['kind'] == kind:
                for f, v in r['single'].items():
                    per[f].append(v)
        rec = {}
        for f, vs in sorted(per.items()):
            v = np.concatenate(vs)
            counts = np.rint(v * 100)
            rec[f] = dict(n=int(v.size), mean_ba=float(v.mean()), sd_ba=float(v.std(ddof=1)),
                          binomial_sd=0.05,
                          tail={str(h): dict(empirical=float((counts >= h).mean()), binomial=binom_tail(h))
                                for h in (57, 60, 63, 65)})
        out[kind] = rec
    return out


def fragment_calibration(recs):
    """Single-fragment configurations without geometry. On a never-watermarked image the decoder output does not
    depend on the embedding strength, so every configuration of a fragment makes the same decisions: the distinct
    images are the sample. Exact binomial tests of the per-path null against the model tail P[Bin(100,1/2) >= h]
    and against the nominal alpha (one-sided, excess)."""
    out = {}
    for kind in KINDS:
        for f in ('VINE', 'TrustMark', 'VideoSeal'):
            sub = [r for r in recs if r['kind'] == kind and r['order'] == [f] and not r['fe_on']]
            if not sub:
                continue
            for a in (0.1, 0.01, 1e-4):
                fa = np.stack([r['fa'][a] for r in sub])
                n, h = fa.shape[1], sub[0]['h'][a]
                same = bool((fa == fa[0]).all())
                rec = dict(configurations=len(sub), images=n, identical_across_configurations=same, threshold=h,
                           model_rate=binom_tail(h), nominal_alpha=a)
                if same:
                    k = int(fa[0].sum())
                    rec.update(events=k, rate=k / n, cp95=cp_interval(k, n),
                               p_excess_over_model=float(stats.binom.sf(k - 1, n, binom_tail(h))) if k else 1.0,
                               p_excess_over_nominal=float(stats.binom.sf(k - 1, n, a)) if k else 1.0)
                else:
                    per = fa.sum(axis=1)
                    rec.update(events_per_configuration=sorted(int(x) for x in per), rate=float(fa.mean()),
                               cp95_pooled_by_image=cp_interval(int(fa.any(axis=0).sum()), n))
                out[f'{kind}|{f}|{LABEL[a]}'] = rec
    return out


def effective_sample(recs):
    """How many distinct decision patterns the configurations produce on the never-watermarked images."""
    pats = collections.defaultdict(list)
    for r in recs:
        if r['kind'] == 'clean':
            pats[(tuple(r['order']), tuple(r['fe_on']))].append(r)
    identical = sum(all((x['fa'][0.1] == g[0]['fa'][0.1]).all() for x in g) for g in pats.values())
    return dict(clean_configurations=sum(map(len, pats.values())), distinct_fragment_order_geometry=len(pats),
                groups_with_identical_decisions=identical, distinct_images=dict(clean=500, wrong=100),
                note=('decisions on a never-watermarked image depend on the decoders and paths, not the strengths; '
                      'trials across configurations are therefore not independent and the independent-trial bound '
                      'is optimistic; the distinct images give the conservative effective sample'))


def dispersion(recs, kind, alpha):
    """Variance of per-image (and per-identity) event counts relative to a common-rate binomial."""
    rows, fa = matrix(recs, kind, alpha)
    if not len(rows) or fa.sum() == 0:
        return None
    c, n = fa.shape
    p = fa.mean()
    per_image = fa.sum(axis=0)
    out = dict(per_image_dispersion=float(per_image.var(ddof=1) / (c * p * (1 - p))),
               max_events_one_image=int(per_image.max()), configurations=c)
    if kind == 'clean' and SCHEMES == {'calibration_mod100'}:      # fresh identities: one per image, nothing to pool
        ident = np.arange(n) % 100
        per_id = np.array([per_image[ident == i].sum() for i in range(100)])
        m = c * (n // 100)
        out['per_identity_dispersion'] = float(per_id.var(ddof=1) / (m * p * (1 - p)))
    return out


def plot(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blue, orange, ink, grid = '#2a78d6', '#eb6834', '#3a3a38', '#e6e5e0'
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, kind, title in zip(axes, KINDS, ('Never-watermarked images', 'Wrong-identity images')):
        for group, color in zip(GROUPS, (blue, orange)):
            t = summary['null_tail'][kind].get(group)
            if not t:
                continue
            emp = np.array(t['empirical'])
            ok = emp > 0
            ax.plot(np.array(t['h'])[ok], emp[ok], color=color, lw=2, label=f'{group.replace("_", "-")}: measured')
            ax.plot(t['h'], t['union_bound_model'], color=color, lw=2, ls='--', label=f'{group.replace("_", "-")}: union-bound model')
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1.2)
        ax.set_xlim(50, 80)
        ax.set_title(title, fontsize=10, color=ink)
        ax.set_xlabel('max matching bits h (of 100)', fontsize=9, color=ink)
        ax.grid(True, color=grid, lw=0.8)
        ax.tick_params(colors=ink, labelsize=8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('P(max matching bits >= h)', fontsize=9, color=ink)
    axes[1].legend(fontsize=7.5, frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(allow_partial):
    rng = np.random.default_rng(SEED)
    manifest, recs, missing = load(RUN, allow_partial)
    out = RUN / 'analysis'
    out.mkdir(exist_ok=True)
    summary = dict(policy='exp_fpr_retries_v1', complete=not missing, missing=len(missing),
                   outputs=len(recs), alphas={LABEL[a]: a for a in ALPHAS},
                   devices=dict(collections.Counter(f"{r['group']}|{r['device']}" for r in recs)),
                   by_alpha={}, deployed_alpha={}, policy_level={}, strata={}, null_tail={},
                   single_path=single_path(recs), dispersion={},
                   fragment_calibration=fragment_calibration(recs), effective_sample=effective_sample(recs))
    table = []
    for kind in KINDS:
        for a in ALPHAS:
            key = f'{kind}|{LABEL[a]}'
            rec = {}
            for g in (*GROUPS, 'all'):
                rows, fa = matrix(recs, kind, a, None if g == 'all' else g)
                if not len(rows):
                    continue
                s = stratum_stats(fa, a, rng)
                s['union_bound_prediction'] = predicted_rate(rows, a)
                rec[g] = s
                table.append(dict(kind=kind, alpha=LABEL[a], group=g, **{k: s.get(k) for k in (
                    'configurations', 'trials', 'events', 'rate', 'cp_upper95', 'union_bound_prediction',
                    'images_with_any_event')}))
            rec['retry_vs_first_try'] = group_difference(recs, kind, a, rng)
            summary['by_alpha'][key] = rec
            rows, fa = matrix(recs, kind, a)
            if len(rows):
                t1, full = policy_rates(manifest, rows, fa)
                summary['policy_level'][key] = dict(top1=t1, full=full, full_minus_top1=full - t1,
                                                    bootstrap=policy_bootstrap(manifest, rows, fa, rng))
            if a in (0.1, 0.01, 1e-4):
                strata = {}
                for name, fn in (('paths_T', lambda r: t_bin(r['T'])),
                                 ('geometry', lambda r: 'on' if r['fe_on'] else 'off'),
                                 ('fragments', lambda r: str(len(r['order'])))):
                    for g in GROUPS:
                        for level in sorted({fn(r) for r in recs if r['kind'] == kind}):
                            sub = [r for r in recs if r['kind'] == kind and r['group'] == g and fn(r) == level]
                            if not sub:
                                continue
                            f = np.stack([r['fa'][a] for r in sub])
                            strata[f'{name}={level}|{g}'] = dict(configurations=len(sub), trials=int(f.size),
                                                                 events=int(f.sum()), rate=float(f.mean()),
                                                                 cp_upper95=cp_upper(int(f.sum()), int(f.size)),
                                                                 union_bound_prediction=predicted_rate(sub, a))
                summary['strata'][key] = strata
                summary['dispersion'][key] = dispersion(recs, kind, a)
        # each configuration at the alphas its requests were deployed with
        for a in ALPHAS:
            for g in GROUPS:
                sub = [r for r in recs if r['kind'] == kind and r['group'] == g and a in r['deploy_alphas']]
                if not sub:
                    continue
                f = np.stack([r['fa'][a] for r in sub])
                s = stratum_stats(f, a, rng)
                s['deployments'] = int(sum(r['deploy_alphas'][a] for r in sub))
                summary['deployed_alpha'][f'{kind}|{LABEL[a]}|{g}'] = s
        summary['null_tail'][kind] = {g: null_tail(recs, kind, g) for g in GROUPS}
    with open(out / 'fpr_retry_results.jsonl', 'w') as fh:
        for r in recs:
            fh.write(json.dumps(dict(cfg_id=r['cfg_id'], group=r['group'], kind=r['kind'], n=r['n'], order=r['order'],
                                     fe_on=r['fe_on'], T=r['T'], device=r['device'],
                                     deploy_kinds=r['deploy_kinds'],
                                     deploy_alphas={LABEL[a]: v for a, v in r['deploy_alphas'].items()},
                                     threshold={LABEL[a]: r['h'][a] for a in ALPHAS},
                                     events={LABEL[a]: int(r['fa'][a].sum()) for a in ALPHAS},
                                     max_matching_bits=int(r['maxcount'].max()))) + '\n')
    cols = ['kind', 'alpha', 'group', 'configurations', 'trials', 'events', 'rate', 'cp_upper95',
            'union_bound_prediction', 'images_with_any_event']
    with open(out / 'fpr_retry_table.csv', 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for row in table:
            fh.write(','.join('' if row[c] is None else f'{row[c]:.6g}' if isinstance(row[c], float) else str(row[c])
                              for c in cols) + '\n')
    atomic(out / 'fpr_summary.json', summary)
    plot(summary, out / 'null_tail.pdf')
    print(f"FPR analysis: {len(recs)} outputs, missing {len(missing)}; devices {summary['devices']}")
    for kind in KINDS:
        for a in ALPHAS:
            rec = summary['by_alpha'][f'{kind}|{LABEL[a]}']
            parts = [f"{g} {rec[g]['events']}/{rec[g]['trials']} ({rec[g]['rate']:.4f}, CP95 {rec[g]['cp_upper95']:.2e},"
                     f" model {rec[g]['union_bound_prediction']:.2e})" for g in GROUPS if g in rec]
            d = rec['retry_vs_first_try']
            diff = '' if not d else f" | retry-first {d['retry_minus_first_try']:+.4f} {d['ci95']}"
            print(f'  {kind:5s} alpha {LABEL[a]:6s}: ' + '; '.join(parts) + diff)
    print('  per-fragment null calibration (distinct images as the sample):')
    for key, v in summary['fragment_calibration'].items():
        if 'events' in v:
            print(f"    {key:22s} {v['events']}/{v['images']} = {v['rate']:.4f} (model {v['model_rate']:.4f}, "
                  f"nominal {v['nominal_alpha']:g}) CI {[round(x, 4) for x in v['cp95']]} "
                  f"p(excess over nominal) {v['p_excess_over_nominal']:.3g}")
        else:
            print(f"    {key:22s} per-configuration events {v['events_per_configuration']} (strength-dependent)")
    print('  effective sample:', {k: v for k, v in summary['effective_sample'].items() if k != 'note'})
    for key, v in summary['policy_level'].items():
        print(f"  policy {key}: Top1 {v['top1']:.4f} Full {v['full']:.4f} diff {v['full_minus_top1']:+.4f} "
              f"{(v['bootstrap'] or {}).get('full_minus_top1_ci95')}")


if __name__ == '__main__':
    main('--allow-partial' in sys.argv[1:])
