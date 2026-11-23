"""C2 search space: measured knot sets, structure rules and candidate counts (no interpolation anywhere)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grid import frozen  # noqa: E402
from grid import knots as K  # noqa: E402
from grid.grid_enumerator import STAGE_HOST, Structure, iter_space, structures  # noqa: E402

AVAILABLE = frozen.available()


class SpaceTest(unittest.TestCase):
    def test_toy_grid_count_is_3k_6k2_6k3_without_geometry(self):
        for k in range(1, 6):
            toy = {f: tuple(0.1 * (i + 1) + j for i in range(k)) for j, f in enumerate('ABC')}
            space = list(iter_space(toy, fragments=('A', 'B', 'C'), geometry=False))
            self.assertEqual(len(space), 3 * k + 6 * k ** 2 + 6 * k ** 3)
            self.assertEqual(len(set(space)), len(space))
            self.assertEqual(K.count_space(toy, structures(('A', 'B', 'C'), geometry=False)), len(space))
            for s, combo in space:
                for f, x in zip(s.order, combo):
                    self.assertIn(x, toy[f])

    def test_toy_grid_with_unequal_knot_counts(self):
        toy = {'A': (1.0, 2.0), 'B': (1.0, 2.0, 3.0), 'C': (1.0,)}
        n = len(list(iter_space(toy, fragments=('A', 'B', 'C'), geometry=False)))
        a, b, c = 2, 3, 1
        self.assertEqual(n, a + b + c + 2 * (a * b + a * c + b * c) + 6 * a * b * c)

    def test_67_structures_with_compatible_stages(self):
        ss = structures()
        self.assertEqual(len(ss), 67)
        self.assertEqual(len(set(ss)), 67)
        self.assertEqual(len(structures(geometry=False)), 15)
        orders = {s.order for s in ss}
        self.assertEqual(len(orders), 15)                          # every subset in every order
        for s in ss:
            self.assertLessEqual(len(s.fe_on), 1)
            if s.stage in STAGE_HOST:
                self.assertIn(STAGE_HOST[s.stage], s.order)
        self.assertEqual(sum(s.stage == 'scale' for s in ss), 11)
        self.assertEqual(sum(s.stage == 'tile' for s in ss), 11)
        for stage in (None, 'resync', 'angle'):
            self.assertEqual(sum(s.stage == stage for s in ss), 15)

    def test_58433_configurations_before_floors(self):
        self.assertEqual(K.count_space(K.EXPECTED_KNOTS, structures()), 58433)
        self.assertEqual(sum(1 for _ in iter_space(K.EXPECTED_KNOTS)), 58433)

    def test_structure_identity(self):
        s = Structure(('TrustMark', 'VINE'), 'tile')
        self.assertEqual(s.as_dict(), dict(frags=['TrustMark', 'VINE'], order=['TrustMark', 'VINE'], fe=['tile']))
        self.assertNotEqual(s, Structure(('VINE', 'TrustMark'), 'tile'))

    def test_knot_index_never_rounds(self):
        knots = K.EXPECTED_KNOTS['VINE']
        self.assertEqual(K.knot_index(knots, 0.3), 3)
        self.assertEqual(K.knot_index(knots, 0.30000000000000004), 3)
        self.assertIsNone(K.knot_index(knots, 0.31))
        self.assertIsNone(K.knot_index(knots, 0.2475))


@unittest.skipUnless(AVAILABLE, 'relocated run root not present')
class KnotTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = frozen.boot()
        cls.sg, _ = frozen.load_surrogate(cls.ns)

    def test_knots_are_derived_from_the_frozen_perimage_block(self):
        knots = K.knot_sets(self.sg._perimage)
        self.assertEqual(knots, K.EXPECTED_KNOTS)
        self.assertEqual({f: len(v) for f, v in knots.items()}, {'VINE': 11, 'TrustMark': 11, 'VideoSeal': 15})
        bad = dict(self.sg._perimage)
        shifted = [0.15] + list(K.EXPECTED_KNOTS['VINE'][1:])          # one knot moved: no longer the approved set
        for key, rec in self.sg._perimage.items():
            if K.parse_key(key)[1] == 'VINE' and tuple(rec['xs']) == K.EXPECTED_KNOTS['VINE']:
                bad[key] = dict(rec, xs=shifted)
        with self.assertRaises(AssertionError):
            K.knot_sets(bad)

    def test_availability_report(self):
        rep = K.availability(self.sg, K.EXPECTED_KNOTS)
        # The decode-side replacement curves of fragments a stage does not re-embed have no per-image record.
        fitted_only = rep['fitted_only_stage_curves']
        self.assertEqual(len(fitted_only), 64)
        self.assertTrue(all(c.startswith(('base_fe_scale_TrustMark', 'base_fe_scale_VideoSeal',
                                          'base_fe_tile_VINE', 'base_fe_tile_VideoSeal')) for c in fitted_only))
        # Records on other grids: editing (its own 9-knot grids) only.
        self.assertTrue(rep['records_off_grid'])
        self.assertTrue(all('editing_ip2p_s20_v1' in k for k in rep['records_off_grid']))
        partial = dict(rep['partial'])
        self.assertEqual(partial['VINE|unmarker'], (0.14, 0.17))
        self.assertEqual(partial['VideoSeal|unmarker'], (1.75, 2.25, 2.8))
        self.assertEqual(partial['VINE|editing_ip2p_s20_v1'],
                         tuple(k for k in K.EXPECTED_KNOTS['VINE'] if k not in (0.14, 1.0)))
        # Every plain cell the solver reads has a record.
        plain = [c for c in rep['cells'] if c['stage'] is None]
        self.assertTrue(all(c['present'] for c in plain))
        self.assertEqual(len(plain), 63)

    def test_records_map_exactly_onto_knots(self):
        for key, rec in self.sg._perimage.items():
            _, f, _ = K.parse_key(key)
            on = [K.knot_index(K.EXPECTED_KNOTS[f], x) for x in rec['xs']]
            hits = [i for i in on if i is not None]
            self.assertEqual(len(hits), len(set(hits)), key)
            if 'editing' not in key:
                self.assertTrue(all(i is not None for i in on), key)


if __name__ == '__main__':
    unittest.main()
