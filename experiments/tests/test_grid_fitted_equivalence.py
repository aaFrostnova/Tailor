"""Key correctness check of the C2 formula.

In FITTED mode (every quantity from the stored curves, rates by the solver's isotonic floor rule) C2's per-fragment
means, mean/rate/cover decisions, the curve each mean is read from, and attack coverage must equal
composite_prediction.predict_composite (the frozen numeric twin of the solver's coverage formula) at the same
on-knot configurations: both the scalar path (any configuration, pruned or not) and the vectorized path that the
search actually runs (ranked candidates).
"""
from pathlib import Path
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grid import frozen  # noqa: E402

AVAILABLE = frozen.available()
if AVAILABLE:
    from grid.grid_enumerator import FRAGMENTS, GridSearch, margins, structures  # noqa: E402
    from grid.tables import GridTables  # noqa: E402

SETTINGS = [(0.1, 'initial'), (1e-4, '04'), (1e-6, '02'), (2 ** -37, '00')]


def curve_key_of(source):
    kind, name = source.split(':', 1)
    if name.startswith('base:'):
        f, a = name[len('base:'):].split('|')
        return ('base', f, a)
    return ('frontend', name)


@unittest.skipUnless(AVAILABLE, 'relocated run root not present')
class FittedEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = frozen.boot()
        cls.TF = GridTables(mode='fitted', ns=cls.ns)
        cls.sg = cls.TF.sg
        cls.attacks = list(cls.sg.attacks)

    def predict(self, order, fe_on, strengths, attack, fpr, rate_margin, mean_margin):
        cfg = dict(order=list(order), fe_on=list(fe_on), s=dict(strengths))
        tau = self.ns.UD.threshold_for_config(dict(order=list(order), fe_on=list(fe_on)), fpr)
        return self.ns.CPD.predict_composite(self.sg, cfg, attack, tau=tau, mean_margin=mean_margin, det_min=0.9,
                                             rate_margin=rate_margin, pair_min_cascade=0.02)

    def compare_cell(self, cell, pf):
        self.assertLessEqual(abs(cell['mean'] - pf['mean']), 1e-12)
        self.assertEqual(cell['mean_pass'], pf['mean_pass'])
        self.assertEqual(cell['rate_pass'], pf['rate_pass'])
        self.assertEqual(cell['cover'], pf['cover_pass'])
        return cell['mean'] == pf['mean']

    def test_scalar_path_matches_predict_composite_on_random_knot_configurations(self):
        rng = random.Random(20261109)
        all_structures = structures()
        n_cfg = n_pairs = exact = cells = pair_cells = stage_cells = 0
        for fpr, stage in SETTINGS:
            rate_margin, mean_margin = margins(self.ns, stage)
            gs = GridSearch(self.TF, dict(attacks=self.attacks, fpr=fpr, min_psnr=0.0, max_ms=1e12),
                            rate_margin=rate_margin, mean_margin=mean_margin)
            for _ in range(60):
                s = rng.choice(all_structures)
                strengths = {f: rng.choice(self.TF.knots[f]) for f in s.order}
                ev = gs.evaluate(s.order, s.stage, strengths)
                n_cfg += 1
                for a in self.attacks:
                    p = self.predict(s.order, s.fe_on, strengths, a, fpr, rate_margin, mean_margin)
                    for f in s.order:
                        cell, pf = ev['cells'][a][f], p['fragments'][f]
                        exact += self.compare_cell(cell, pf)
                        cells += 1
                        self.assertEqual(curve_key_of(cell['mean_source']), tuple(pf['curve_key']))
                        pair_cells += cell['mean_source'].startswith('pair_fitted:')
                        stage_cells += 'base_fe_' in cell['mean_source']
                    self.assertEqual(bool(ev['coverage'][a]), p['attack_pass'])
                    n_pairs += 1
        self.assertEqual(n_cfg, 240)
        self.assertGreaterEqual(n_pairs, 200 * 21)
        self.assertGreater(pair_cells, 0)                                # pair-curve precedence exercised
        self.assertGreater(stage_cells, 500)
        self.assertEqual(exact, cells)                                   # same float operations: bit-identical

    def test_vectorized_path_matches_predict_composite_on_ranked_candidates(self):
        rng = np.random.default_rng(7)
        checked = 0
        for fpr, stage in SETTINGS:
            rate_margin, mean_margin = margins(self.ns, stage)
            gs = GridSearch(self.TF, dict(attacks=self.attacks, fpr=fpr, min_psnr=0.0, max_ms=1e12),
                            rate_margin=rate_margin, mean_margin=mean_margin)
            self.assertEqual({s.stage for s in gs.structures}, {None, 'resync', 'scale', 'angle', 'tile'})
            picks = np.sort(rng.choice(gs.n_ranked, size=60, replace=False))
            for pos in picks:
                feasible, cover = gs.evaluate_positions(int(pos), int(pos) + 1, return_cover=True)
                s = gs.structures[int(gs.c_sid[pos])]
                kn = gs.c_kn[pos]
                strengths = {f: self.TF.knots[f][int(kn[FRAGMENTS.index(f)])] for f in s.order}
                covered_all = True
                for j, a in enumerate(self.attacks):
                    p = self.predict(s.order, s.fe_on, strengths, a, fpr, rate_margin, mean_margin)
                    for f in s.order:
                        self.assertEqual(bool(cover[0, FRAGMENTS.index(f), j]), p['fragments'][f]['cover_pass'],
                                         (s, strengths, a, f))
                    self.assertEqual(bool(cover[0, :, j].any()), p['attack_pass'])
                    covered_all &= p['attack_pass']
                    checked += 1
                useful = all(any(self.predict(s.order, s.fe_on, strengths, a, fpr, rate_margin,
                                              mean_margin)['fragments'][f]['cover_pass'] for a in self.attacks)
                             for f in s.order)
                self.assertEqual(bool(feasible[0]), covered_all and useful)
        self.assertGreaterEqual(checked, 200 * 21)

    def test_vectorized_and_scalar_paths_agree_on_every_ranked_candidate_of_a_request(self):
        rate_margin, mean_margin = margins(self.ns, 'initial')
        request = dict(attacks=['jpeg25', 'crop75', 'rot9', 'border20', 'bright', 'regen'], fpr=1e-6,
                       min_psnr=0.0, max_ms=1e12)
        gs = GridSearch(self.TF, request, rate_margin=rate_margin, mean_margin=mean_margin)
        feasible = gs.evaluate_positions(0, gs.n_ranked)
        sample = np.unique(np.concatenate([np.flatnonzero(feasible)[:300],
                                           np.random.default_rng(3).choice(gs.n_ranked, 700, replace=False)]))
        for pos in sample:
            s = gs.structures[int(gs.c_sid[pos])]
            kn = gs.c_kn[pos]
            strengths = {f: self.TF.knots[f][int(kn[FRAGMENTS.index(f)])] for f in s.order}
            ev = gs.evaluate(s.order, s.stage, strengths, details=False)
            self.assertEqual(ev['feasible'], bool(feasible[pos]), (s, strengths, ev['violations']))
            self.assertEqual(ev['D_pred'], float(gs.c_D[pos]))
        self.assertGreater(int(feasible.sum()), 0)


if __name__ == '__main__':
    unittest.main()
