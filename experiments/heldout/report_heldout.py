"""Held-out transfer report: each accepted TAILOR-F request judged once more on the frozen new images.

Same gate as calibration: every requested attack needs a per-image acceptance rate >= 0.90 and a mean
accepted score >= tau(T(c), alpha); the measured embed PSNR on the new images must meet the floor.
Latency is modeled from the configuration and is image-independent, so it keeps its calibration
outcome (passed for every accepted request). Calibration statistics come from the frozen cells.

Uncertainty: Wilson intervals over requests and a cluster-aware interval that resamples configurations
(requests sharing a configuration share its cells). Calibration-margin bins show how cells that only
just passed calibration moved on the new images (a check on how hard this image draw is).

Usage: TAILOR_MAIN_CONTROLS_RUN=$RUNH python report_heldout.py OUT_DIR
"""
import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'experiments/main_controls'), str(REPO / 'experiments')]
from common import CP, RUN, load_cell, read  # noqa: E402
import rejudge  # noqa: E402

SCENARIOS = ('S1', 'S2', 'S3', 'S4', 'S5')


def wilson(k, n, z=1.959964):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0., centre - half), min(1., centre + half)]


def cluster_interval(rows, key, b=2000, seed=20261110):
    """95% percentile interval of a request-level rate, resampling configurations: requests that share a
    configuration share its held-out cells, so they move together."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r['cfg_id']].append(bool(r[key]))
    groups = list(by.values())
    if not groups:
        return None
    if all(all(g) for g in groups) or not any(any(g) for g in groups):
        # no variation to resample: exact one-sided bound with the configurations as the effective sample
        from scipy import stats
        n = len(groups)
        lo = float(stats.beta.ppf(0.05, n, 1)) if all(all(g) for g in groups) else 0.0
        hi = 1.0 if lo else float(stats.beta.ppf(0.95, 1, n))
        return [lo, hi]
    rng = random.Random(seed)
    rates = []
    for _ in range(b):
        pick = [groups[rng.randrange(len(groups))] for _ in groups]
        rates.append(sum(map(sum, pick)) / sum(map(len, pick)))
    rates.sort()
    return [rates[int(0.025 * b)], rates[int(0.975 * b) - 1]]


def margin_bins(rows):
    """Request-attack pairs by their calibration margin to the gate, min(rate - 0.90, mean - tau): how the
    held-out outcome moved for cells that only just passed, counted per distinct (configuration, attack) cell."""
    edges = ((0.0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 2.0))
    cells = {}
    for r in rows:
        for att, v in r['attacks'].items():
            m = min(v['calib_rate'] - 0.9, v['calib_mean'] - r['tau'])
            cells[(r['cfg_id'], att, r['threshold_count'])] = (m, v)
    out = {}
    for lo, hi in edges:
        part = [v for m, v in cells.values() if lo - 1e-9 <= m < hi]
        if part:
            out[f'[{lo},{hi})'] = dict(cells=len(part), held_ok=sum(v['held_ok'] for v in part),
                                       mean_rate_shift=statistics.mean(v['held_rate'] - v['calib_rate'] for v in part),
                                       mean_score_shift=statistics.mean(v['held_mean'] - v['calib_mean'] for v in part),
                                       rose=sum(v['held_mean'] > v['calib_mean'] for v in part))
    return out


def judge_heldout(request, candidate, cells):
    c = dict(candidate, ms=0.0)          # modeled latency already passed in calibration; image-independent
    return CP.judge(request, c, cells, .9)


def main(out_dir):
    out_dir = Path(out_dir)
    manifest = read(RUN / 'manifest.json')
    frozen_cells = rejudge.FrozenCells()
    rows, pending = [], []
    for a in manifest['assignments']:
        request, cand = a['request'], dict(a['candidate'], rank=1)
        held = {att: load_cell(cand, att, frozen=False) for att in request['attacks']}
        if any(v is None for v in held.values()):
            pending.append(a['spec_id'])
            continue
        calib = {att: frozen_cells.load(cand, att) for att in request['attacks']}
        vh = judge_heldout(request, cand, held)
        vc = judge_heldout(request, cand, calib)
        per_attack = {att: dict(calib_rate=vc['attacks'][att]['rate'], held_rate=vh['attacks'][att]['rate'],
                                calib_mean=vc['attacks'][att]['mean'], held_mean=vh['attacks'][att]['mean'],
                                calib_ok=vc['attacks'][att]['ok'], held_ok=vh['attacks'][att]['ok'])
                      for att in request['attacks']}
        rows.append(dict(spec_id=a['spec_id'], scenario='S' + a['cls'][1:], cfg_id=a['cfg_id'],
                         calibration_pass=vc['pass'], heldout_pass=vh['pass'], heldout_cols_ok=vh['cols_ok'],
                         heldout_psnr_ok=vh['psnr_ok'], psnr_calibration=vc['psnr_live'], psnr_heldout=vh['psnr_live'],
                         psnr_floor=request['min_psnr'], tau=vh['tau'], threshold_count=vh['threshold_count'],
                         failing_attacks=sorted(att for att, s in per_attack.items() if not s['held_ok']),
                         attacks=per_attack))
    summary = dict(policy='exp_heldout_v1', image_set_sha256=manifest['image_set_sha256'],
                   accepted_requests=len(manifest['assignments']), judged=len(rows), pending=len(pending),
                   complete=not pending, scenarios={})
    for s in SCENARIOS + ('all',):
        part = [r for r in rows if s == 'all' or r['scenario'] == s]
        if not part:
            continue
        held = sum(r['heldout_pass'] for r in part)
        calib = sum(r['calibration_pass'] for r in part)
        flips = sum(r['calibration_pass'] and not r['heldout_pass'] for r in part)
        cfgs = collections.defaultdict(list)
        for r in part:
            cfgs[r['cfg_id']].append(r['heldout_pass'])
        dpsnr = [r['psnr_heldout'] - r['psnr_calibration'] for r in part]
        fails = collections.Counter(a for r in part for a in r['failing_attacks'])
        summary['scenarios'][s] = dict(
            n=len(part), calibration_pass=calib, heldout_pass=held, heldout_percent=100 * held / len(part),
            heldout_wilson95=wilson(held, len(part)),
            heldout_cluster95=cluster_interval(part, 'heldout_pass'), calibration_pass_heldout_fail=flips,
            psnr_only_failures=sum(r['heldout_cols_ok'] and not r['heldout_psnr_ok'] for r in part),
            configurations=len(cfgs), configurations_all_pass=sum(all(v) for v in cfgs.values()),
            psnr_shift_mean=statistics.mean(dpsnr), psnr_shift_min=min(dpsnr), psnr_shift_max=max(dpsnr),
            failing_attack_counts=dict(fails.most_common()))
    # attack-level shift over request-attack pairs
    shifts = collections.defaultdict(list)
    for r in rows:
        for att, v in r['attacks'].items():
            shifts[att].append(v['held_rate'] - v['calib_rate'])
    summary['attack_rate_shift'] = {a: dict(n=len(v), mean=statistics.mean(v), min=min(v), max=max(v))
                                    for a, v in sorted(shifts.items())}
    summary['calibration_margin_bins'] = margin_bins(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'heldout_results.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    (out_dir / 'heldout_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'attack_rate_shift'}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
