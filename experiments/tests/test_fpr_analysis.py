"""PR4 analysis (experiments/fpr/analyze_fpr.py): exact bounds, threshold lookup, reconciliation with the raw decisions.

Needs the relocated FPR run root for the frozen detector; skipped when it is absent.
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

RUNF = Path(os.environ.get('TAILOR_RUNS', 'runs') + '/fpr_retries_20261109')
os.environ.setdefault('TAILOR_MAIN_CONTROLS_RUN', str(RUNF))
AVAILABLE = (Path(os.environ['TAILOR_MAIN_CONTROLS_RUN']) / 'root/code/unified_detector.py').exists()
if AVAILABLE:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fpr'))
    import numpy as np  # noqa: E402
    import analyze_fpr as A  # noqa: E402


def config(order, fe_on, s):
    return dict(order=list(order), fe_on=list(fe_on), s=dict(zip(order, s)))


def write_output(run, cand, group, kind, n, events_at):
    """One negatives file whose buckets carry `events_at[alpha]` false acceptances on the first images."""
    cfg = dict(order=cand['order'], fe_on=cand['fe_on'])
    cfg_id = A.CP.exact_cfg_id(A.cfgkey(cand))
    buckets = {}
    for a in A.ALPHAS:
        h = A.UD.threshold_count(cfg, a)
        k = events_at.get(a, 0)
        det = [i < k for i in range(n)]
        score = [h / 100 if d else 0.5 for d in det]
        buckets[str(h)] = dict(threshold_count=h, detected=det, accepted_score=score, best_fragment_ba=score,
                               used_geometry=[False] * n, fused_ba=score,
                               per_fragment_ba={f: score for f in cand['order']})
    task = dict(cfg_id=cfg_id, group=group, kind=kind, n=n, candidate=cand)
    (run / 'negatives').mkdir(parents=True, exist_ok=True)
    (run / 'negatives' / f'{cfg_id}_{kind}.json').write_text(json.dumps(dict(
        task=task, cell=dict(protocol=A.CP.CELL_PROTOCOL, n=n, order=cand['order'], by_threshold=buckets),
        seconds_per_image=0.1, device='test', host='test', protocol=A.CP.PROTOCOL)))
    return task


@unittest.skipUnless(AVAILABLE, 'relocated FPR run root not present')
class Bounds(unittest.TestCase):
    def test_zero_events_upper_bound_is_exact(self):
        for n in (10, 500, 30000):
            self.assertAlmostEqual(A.cp_upper(0, n), 1 - 0.05 ** (1 / n), places=12)
        self.assertAlmostEqual(A.cp_upper(0, 30000) * 30000, 3.0, delta=0.01)   # the rule of three

    def test_bound_monotone_and_saturates(self):
        self.assertLess(A.cp_upper(1, 500), A.cp_upper(2, 500))
        self.assertGreater(A.cp_upper(5, 500), 5 / 500)
        self.assertEqual(A.cp_upper(500, 500), 1.0)

    def test_two_sided_interval_is_clopper_pearson(self):
        from scipy import stats
        for k, n in ((0, 500), (4, 500), (64, 500), (100, 100)):
            ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method='exact')
            lo, hi = A.cp_interval(k, n)
            self.assertAlmostEqual(lo, ci.low, places=10)
            self.assertAlmostEqual(hi, ci.high, places=10)

    def test_binomial_tail_matches_direct_sum(self):
        for h in (50, 57, 63, 80):
            direct = sum(math.comb(100, j) for j in range(h, 101)) / 2 ** 100
            self.assertAlmostEqual(A.binom_tail(h), direct, places=12)


@unittest.skipUnless(AVAILABLE, 'relocated FPR run root not present')
class Reconcile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        a = config(['TrustMark'], [], [0.85])
        b = config(['VINE', 'TrustMark'], ['tile'], [0.5, 0.7])
        tasks = [write_output(self.run, a, 'first_try', 'clean', 20, {0.1: 3, 0.01: 1}),
                 write_output(self.run, b, 'retry', 'clean', 20, {0.1: 1})]
        groups = {tasks[0]['cfg_id']: dict(kinds={'first_try': 4}, alphas={'0.1': 3, '1e-09': 1}, n_specs=4),
                  tasks[1]['cfg_id']: dict(kinds={'first_try': 1, 'retry': 2}, alphas={'0.01': 3}, n_specs=3)}
        (self.run / 'manifest.json').write_text(json.dumps(dict(
            tasks=tasks, config_groups=groups,
            configurations=dict(retry=1, first_try_only=4, first_try_sampled=1))))

    def tearDown(self):
        self.tmp.cleanup()

    def test_events_are_the_detected_flags_at_each_configurations_threshold(self):
        manifest, recs, missing = A.load(self.run)
        self.assertEqual(missing, [])
        by = {r['group']: r for r in recs}
        self.assertEqual(int(by['first_try']['fa'][0.1].sum()), 3)
        self.assertEqual(int(by['first_try']['fa'][0.01].sum()), 1)
        self.assertEqual(int(by['retry']['fa'][0.1].sum()), 1)
        self.assertEqual(by['retry']['T'], 3 + 81)          # two fragments + fused, plus the tile search
        self.assertEqual(by['first_try']['T'], 1)
        rows, fa = A.matrix(recs, 'clean', 0.1)
        self.assertEqual(int(fa.sum()), 4)
        s = A.stratum_stats(fa, 0.1, np.random.default_rng(0))
        self.assertEqual((s['events'], s['trials']), (4, 40))
        self.assertAlmostEqual(s['cp_upper95'], A.cp_upper(4, 40))

    def test_zero_events_use_the_distinct_images_as_cluster_sample(self):
        _, recs, _ = A.load(self.run)
        _, fa = A.matrix(recs, 'clean', 1e-9)
        s = A.stratum_stats(fa, 1e-9, np.random.default_rng(0))
        self.assertEqual(s['events'], 0)
        self.assertAlmostEqual(s['cluster_upper95'], A.cp_upper(0, 20))
        self.assertAlmostEqual(s['cp_upper95'], A.cp_upper(0, 40))

    def test_policy_rates_weight_sampled_first_try_configurations(self):
        manifest, recs, _ = A.load(self.run)
        rows, fa = A.matrix(recs, 'clean', 0.1)
        top1, full = A.policy_rates(manifest, rows, fa)
        r = {row['group']: fa[i].mean() for i, row in enumerate(rows)}
        w_first = 4 / 1                                       # first_try_only / first_try_sampled
        self.assertAlmostEqual(top1, (w_first * 4 * r['first_try'] + 1 * 1 * r['retry']) / (w_first * 4 + 1))
        self.assertAlmostEqual(full, (w_first * 4 * r['first_try'] + 1 * 3 * r['retry']) / (w_first * 4 + 3))

    def test_fragment_calibration_uses_the_distinct_images(self):
        _, recs, _ = A.load(self.run)
        cal = A.fragment_calibration(recs)
        rec = cal['clean|TrustMark|1e-1']
        self.assertTrue(rec['identical_across_configurations'])
        self.assertEqual((rec['events'], rec['images']), (3, 20))
        self.assertEqual(rec['cp95'], A.cp_interval(3, 20))
        self.assertNotIn('clean|VINE|1e-1', cal)              # VINE only appears with TrustMark and geometry

    def test_missing_outputs_stop_the_final_analysis(self):
        next((self.run / 'negatives').glob('*.json')).unlink()
        with self.assertRaises(SystemExit):
            A.load(self.run)
        _, recs, missing = A.load(self.run, allow_partial=True)
        self.assertEqual((len(recs), len(missing)), (1, 1))


if __name__ == '__main__':
    unittest.main()
