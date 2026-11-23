"""C2 thresholds: T(c) and tau of every structure equal unified_detector; impossible thresholds prune."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grid import frozen  # noqa: E402

AVAILABLE = frozen.available()
if AVAILABLE:
    from grid.grid_enumerator import GridSearch, search, structures  # noqa: E402
    from grid.tables import GridTables  # noqa: E402


@unittest.skipUnless(AVAILABLE, 'relocated run root not present')
class ThresholdTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = frozen.boot()
        cls.T = GridTables(mode='raw', ns=cls.ns)

    def request(self, fpr):
        return dict(attacks=['jpeg25', 'crop75', 'rot9', 'bright'], fpr=fpr, min_psnr=0.0, max_ms=1e12)

    def test_every_structure_matches_unified_detector(self):
        UD = self.ns.UD
        checked = 0
        for fpr in self.ns.CP.REQUEST_ALPHAS:
            gs = GridSearch(self.T, self.request(fpr), rate_margin=0.06, mean_margin=0.02)
            for s in structures():
                cfg = dict(order=list(s.order), fe_on=s.fe_on)
                count, tau, max_tests, mean_thr = gs.threshold(s)
                self.assertEqual(max_tests, UD.budget_for_config(cfg)['max_tests'])
                self.assertEqual(count, UD.threshold_count(cfg, fpr))
                self.assertEqual(tau, UD.threshold_for_config(cfg, fpr))
                # T(c) = primary (1, or m + 1 with fusion) + the declared geometric search budget.
                m = s.m
                geometry = {None: 0, 'resync': 5 * m, 'scale': 133, 'tile': 81, 'angle': 65 * m}[s.stage]
                self.assertEqual(max_tests, (1 if m == 1 else m + 1) + geometry)
                self.assertEqual(mean_thr, tau if tau > 1 else min(1.0, tau + 0.02))
                checked += 1
        self.assertEqual(checked, 67 * 6)

    def test_returned_candidates_carry_their_own_threshold(self):
        UD = self.ns.UD
        for fpr in (0.1, 1e-6, 2 ** -37):
            res = search(self.T, dict(self.request(fpr), min_psnr=30.0, max_ms=8000.0))
            self.assertEqual(res['status'], 'sat')
            for c in res['candidates']:
                cfg = dict(order=c['order'], fe_on=c['fe_on'])
                self.assertEqual(c['threshold_count'], UD.threshold_count(cfg, fpr))
                self.assertEqual(c['max_tests'], UD.budget_for_config(cfg)['max_tests'])
                self.assertEqual(c['live_threshold'], UD.threshold_for_config(cfg, fpr))

    def test_impossible_threshold_is_pruned_not_searched(self):
        gs = GridSearch(self.T, self.request(1e-40), rate_margin=0.06, mean_margin=0.02)
        for s in structures():
            count, tau, _, mean_thr = gs.threshold(s)
            self.assertEqual(count, 101)
            self.assertEqual(mean_thr, tau)                            # the margin never rescues count 101
            self.assertGreater(mean_thr, 1.0)
        res = gs.run()
        self.assertEqual(res['status'], 'model_unsat')
        self.assertEqual(res['n_ranked'], 0)
        self.assertGreater(res['pruned']['impossible_threshold'], 0)


if __name__ == '__main__':
    unittest.main()
