"""Verification arithmetic: T(c), the exact Binomial(100, 1/2) threshold, strict bit matching."""
import os
import importlib.util
import itertools
import unittest
from pathlib import Path

import numpy as np

from tailor_exp import protocol

FROZEN_DETECTOR = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/'
                       'rigor_editing_only_20260911/unified_ba_20260914/code/unified_detector.py')
FRAGMENTS = ('vine', 'trustmark', 'videoseal')
STAGES = ('resync', 'scale', 'tile', 'angle')
ALPHAS = (0.1, 0.01, 1e-4, 1e-6, 1e-9, 2 ** -37, 7.275957614183426e-12, 0.05, 3e-3)


def _frozen():
    spec = importlib.util.spec_from_file_location('frozen_unified_detector', FROZEN_DETECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BudgetTest(unittest.TestCase):
    def test_documented_examples(self):
        self.assertEqual(protocol.max_tests(['vine'], []), 1)
        self.assertEqual(protocol.max_tests(['videoseal', 'trustmark', 'vine'], ['scale']), 137)
        self.assertEqual(protocol.max_tests(['vine', 'videoseal'], ['resync']), 13)
        self.assertEqual(protocol.max_tests(['vine', 'trustmark', 'videoseal'], ['angle']), 199)
        self.assertEqual(protocol.max_tests(['trustmark', 'vine'], ['tile']), 84)

    def test_stage_host_rules(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.max_tests(['trustmark'], ['scale'])
        with self.assertRaises(protocol.ProtocolError):
            protocol.max_tests(['vine'], ['tile'])

    def test_deployable_allows_at_most_one_stage(self):
        protocol.check_deployable(['vine', 'trustmark'], ['resync'])
        with self.assertRaises(protocol.ProtocolError):
            protocol.check_deployable(['vine', 'trustmark'], ['resync', 'tile'])


class ThresholdTest(unittest.TestCase):
    def test_threshold_budget_matches_frozen_detector(self):
        frozen = _frozen()
        checked = 0
        for m in (1, 2, 3):
            for order in itertools.permutations(FRAGMENTS, m):
                for k in range(0, 3):
                    for stages in itertools.combinations(STAGES, k):
                        if 'scale' in stages and 'vine' not in order:
                            continue
                        if 'tile' in stages and 'trustmark' not in order:
                            continue
                        cfg = dict(order=list(order), fe_on=list(stages))
                        self.assertEqual(protocol.max_tests(order, stages),
                                         frozen.budget_for_config(cfg)['max_tests'])
                        for alpha in ALPHAS:
                            self.assertEqual(protocol.threshold_count(order, stages, alpha),
                                             frozen.threshold_count(cfg, alpha), (cfg, alpha))
                            checked += 1
        self.assertGreater(checked, 1000)

    def test_threshold_is_the_smallest_count_meeting_the_budget(self):
        from fractions import Fraction
        from math import comb
        for t_c, alpha in ((1, 0.01), (137, 1e-9), (13, 2 ** -37)):
            h = protocol.threshold_for_tests(t_c, alpha)
            tail = lambda c: Fraction(sum(comb(100, j) for j in range(c, 101)), 2 ** 100)
            self.assertLessEqual(t_c * tail(h), Fraction(alpha))
            self.assertGreater(t_c * tail(h - 1), Fraction(alpha))

    def test_impossible_threshold(self):
        # 137 * 2^-100 > 1e-30: even 100/100 matches cannot meet the budget.
        self.assertEqual(protocol.threshold_for_tests(137, 1e-30), 101)
        self.assertTrue(protocol.is_impossible(101))
        self.assertFalse(protocol.is_impossible(100))

    def test_alpha_must_be_in_open_unit_interval(self):
        for bad in (0.0, 1.0, -0.1, float('nan'), True):
            with self.assertRaises(protocol.ProtocolError):
                protocol.threshold_for_tests(1, bad)


class StrictMatchTest(unittest.TestCase):
    def test_zero_llr_is_a_miss(self):
        word = np.array([1, 0] * 50, dtype=np.uint8)
        llr = np.where(word == 1, 2.0, -2.0)
        self.assertEqual(protocol.strict_matches(llr, word), 100)
        llr[:10] = 0.0
        self.assertEqual(protocol.strict_matches(llr, word), 90)

    def test_strict_matches_agree_with_frozen_detector(self):
        frozen = _frozen()
        rng = np.random.default_rng(0)
        for _ in range(200):
            word = rng.integers(0, 2, 100).astype(np.uint8)
            llr = rng.normal(size=100)
            llr[rng.random(100) < 0.1] = 0.0
            self.assertEqual(protocol.strict_matches(llr, word), frozen.strict_matches(llr, word))

    def test_rejects_wrong_length(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.strict_matches(np.ones(64), np.ones(64, dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
