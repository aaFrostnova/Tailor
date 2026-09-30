"""CPU-only tests of the new rule and complete finite search accounting."""
import itertools
import ast
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

import unified_detector as UD


ALPHAS = [.1, .01, 1e-4, 1e-6, 1e-9, 2.0 ** -37]
WORD = np.zeros(100, dtype=np.uint8)


def vector(matches, magnitude=1.0):
    result = np.ones(100, dtype=np.float64) * magnitude
    result[:matches] *= -1
    return result


class Image:
    def __init__(self, label='att', size=(512, 512)):
        self.label, self.size = label, size

    def crop(self, box):
        return Image(self.label + '|crop:' + ','.join(map(str, box)),
                     (box[2] - box[0], box[3] - box[1]))

    def resize(self, size):
        return Image(self.label + '|resize', size)


class Tiled:
    def __init__(self, comp):
        self.comp, self.cell = comp, 256
        self.offsets = list(range(0, 257, 32))

    def _to512(self, image):
        return image if image.size == (512, 512) else image.resize((512, 512))

    def _win_llr(self, image, ox, oy, _perm, _mask):
        return np.clip(self.comp._frag_llr('trustmark', image.crop((ox, oy, ox + 256, oy + 256)), 'id'), -15, 15)


class Composite:
    def __init__(self, order=('vine',), stages=(), scorer=None):
        self.order = list(order)
        self.sb = types.SimpleNamespace(n=100)
        self.alpha = .01
        self.resync = 'resync' in stages
        self.fe_scale = 'scale' in stages
        self.fe_tile = 'tile' in stages
        self.fe_angle = 'angle' in stages
        self.geo = bool(stages)
        self._vss, self._rr, self._rs, self._bc = .005, 180.0, 3.0, 10.0
        self._sync, self.dev = object(), 'mock'
        self._tiled = Tiled(self)
        self.frag = {'trustmark': types.SimpleNamespace(get_perm_M=lambda iid: (np.arange(100), np.ones(100)))}
        self.scorer = scorer or (lambda name, image: np.zeros(100))
        self.reads, self.geometry_actions = [], []

    def _frag_llr(self, name, image, iid):
        self.reads.append((name, image.label))
        return self.scorer(name, image)

    def _sync_rectify(self, *_args):
        self.geometry_actions.append('rectify')
        return Image('rectified'), None

    def _rot(self, image, angle):
        self.geometry_actions.append(('rotate', angle))
        return Image(image.label + '|rotate:%g' % angle)

    def _cv(self, *_args):
        raise AssertionError('identity/BCH bypass was called')

    def decode(self, *_args, **_kwargs):
        return 1.0, True  # hostile legacy bypass; install must replace it

    def geo_cascade(self, *_args, **_kwargs):
        return True

    def embed(self, *_args):
        return 'unchanged embedding'


def angle_module(values):
    package = types.ModuleType('src')
    package.__path__ = []
    module = types.ModuleType('src.angle_probe')
    module.candidate_angles = lambda image, **kwargs: list(values)
    return patch.dict(sys.modules, {'src': package, 'src.angle_probe': module})


class UnifiedDetectorTests(unittest.TestCase):
    def test_actual_native_tile_preprocessing_matches_same_window_raw_route(self):
        # Execute the original lightweight method bodies without constructing
        # GPU models. The shared getter observes the real PIL pixels it would
        # receive, including resize, crop offsets and channel conversion.
        from PIL import Image as PILImage
        cf = Path('/data/tailor/project')
        spec = importlib.util.spec_from_file_location('_test_native_soft_fusion', cf / 'src/soft_fusion.py')
        soft = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(soft)
        def native_method(relative, class_name, method_name):
            tree = ast.parse((cf / relative).read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
            method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
            method.decorator_list = []
            namespace = {'np': np, 'method_soft_to_codeword_llr': soft.method_soft_to_codeword_llr}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(cf / relative), 'exec'), namespace)
            return namespace[method_name]
        to512 = native_method('src/tiled_trustmark.py', 'TiledTrustMark', '_to512')
        win = native_method('src/tiled_trustmark.py', 'TiledTrustMark', '_win_llr')
        frag_llr = native_method('scripts/defense/eval_matrix.py', 'OursComposite', '_frag_llr')
        calls = []
        def getter(pil):
            calls.append((pil.mode, pil.size, pil.tobytes()))
            pixels = np.asarray(pil.convert('RGB').resize((256, 256), PILImage.BILINEAR), dtype=np.float64)
            return pixels.reshape(-1)[:100] - 100
        perm, mask = np.arange(100)[::-1], np.where(np.arange(100) % 2, -1., 1.)
        tm = types.SimpleNamespace(raw_logits=getter, get_perm_M=lambda iid: (perm, mask))
        tiled = types.SimpleNamespace(cell=256, offsets=list(range(0, 257, 32)), clamp=15.,
                                      tm=tm, sb=types.SimpleNamespace(n=100), _to512=to512)
        comp = types.SimpleNamespace(order=['trustmark'], _tiled=tiled, frag={'trustmark': tm},
                                     sb=tiled.sb, SPEC={'trustmark': ('logit', 'raw_logits')})
        comp._frag_llr = types.MethodType(frag_llr, comp)
        package = types.ModuleType('src'); package.__path__ = []
        with patch.dict(sys.modules, {'src': package, 'src.soft_fusion': soft}):
            for size in ((512, 512), (621, 397)):
                pixels = np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(size[1], size[0], 3)
                image = PILImage.fromarray(pixels)
                views = list(UD._geometry_views(comp, image, dict(order=['trustmark'], fe_on=['tile'])))
                self.assertEqual(len(views), 81)
                for index in (0, 40, 80):
                    candidate = views[index]
                    source, ox, oy = candidate['tile']
                    calls.clear()
                    original = win(tiled, source, ox, oy, perm, mask)
                    replacement = UD._acceptance_llr(comp, candidate, 'trustmark', 'id')
                    self.assertEqual(calls[0], calls[1])
                    self.assertTrue(np.array_equal(np.sign(original), np.sign(replacement)))

    def test_derived_frontend_mapping_is_validated(self):
        cfg = dict(S=['VINE', 'TrustMark'], order=['TrustMark', 'VINE'],
                   s={'VINE': .6, 'TrustMark': 1.0}, fe_on=['scale'],
                   fe={'resync': False, 'scale': True, 'tile': False, 'angle': False})
        self.assertEqual(UD.budget_for_config(cfg)['max_tests'], 136)
        without_list = {k: value for k, value in cfg.items() if k != 'fe_on'}
        self.assertEqual(UD.normalize_config(without_list), UD.normalize_config(cfg))
        for bad_mapping in ({'tile': True}, {'scale': 1}, {'unknown': False}, [], None):
            with self.assertRaises(ValueError):
                UD.budget_for_config(dict(cfg, fe=bad_mapping))
        with self.assertRaises(ValueError):
            UD.budget_for_config(dict(cfg, resync=False))

    def test_tile_uses_raw_report_llrs_and_rejects_nonfinite_before_clipping(self):
        def scorer(name, image):
            if '|crop:' not in image.label:
                return vector(0)
            return vector(100, 100.0) if name == 'trustmark' else vector(0, 20.0)
        comp = Composite(order=('vine', 'trustmark'), stages=('tile',), scorer=scorer)
        result = UD.decode_detailed(comp, Image(), ('id', WORD), .01)
        self.assertTrue(result['detected'])
        self.assertEqual(result['accepted_kind'], 'tile')
        self.assertEqual(result['fused_ba'], 1.0)
        self.assertEqual(len(comp.reads), 4)  # two primary + one read per adopted-view fragment
        def nonfinite(name, image):
            return np.full(100, -np.inf) if '|crop:' in image.label else vector(0)
        comp = Composite(stages=('tile',), order=('trustmark',), scorer=nonfinite)
        with self.assertRaises(ValueError):
            UD.decode_detailed(comp, Image(), ('id', WORD), .01)

    def test_budget_all_frontends_and_multiple_stages(self):
        order = ['VINE', 'TrustMark', 'VideoSeal']
        expected = {(): 4, ('resync',): 19, ('scale',): 137,
                    ('tile',): 85, ('angle',): 199,
                    ('resync', 'scale', 'tile', 'angle'): 428}
        for stages, count in expected.items():
            self.assertEqual(UD.budget_for_config(dict(order=order, fe_on=list(stages)))['max_tests'], count)
        self.assertEqual(UD.budget_for_config(dict(order=['vine'], fe_on=[]))['max_tests'], 1)
        self.assertEqual(UD.budget_for_config(dict(order=['vine', 'trustmark'], fe_on=[]))['max_tests'], 3)

    def test_threshold_is_exact_minimal_integer_for_all_six_alphas(self):
        for stages in ([], ['resync'], ['scale'], ['tile'], ['angle'], ['resync', 'scale', 'tile', 'angle']):
            cfg = dict(order=['vine', 'trustmark', 'videoseal'], fe_on=stages)
            budget = UD.budget_for_config(cfg)['max_tests']
            for alpha in ALPHAS:
                count = UD.threshold_count(cfg, alpha)
                numerator, denominator = alpha.as_integer_ratio()
                tail = sum(math.comb(100, j) for j in range(count, 101))
                self.assertLessEqual(budget * tail * denominator, numerator * 2 ** 100)
                previous = sum(math.comb(100, j) for j in range(count - 1, 101))
                self.assertGreater(budget * previous * denominator, numerator * 2 ** 100)
                self.assertEqual(UD.threshold_for_config(cfg, alpha), count / 100)
        cfg = dict(order=['vine'], fe_on=[])
        self.assertEqual([UD.threshold_count(cfg, a) for a in ALPHAS], [57, 63, 69, 74, 80, 83])

    def test_subnormal_alpha_gets_impossible_threshold(self):
        cfg = dict(order=['vine'], fe_on=[])
        self.assertEqual(UD.threshold_count(cfg, 5e-324), 101)
        comp = Composite(scorer=lambda *_: vector(100))
        result = UD.decode_detailed(comp, Image(), ('id', WORD), 5e-324)
        self.assertFalse(result['detected'])
        self.assertFalse(result['threshold_attainable'])

    def test_strict_zero_ties_and_nonfinite_values(self):
        self.assertEqual(UD.strict_matches(np.zeros(100), WORD), 0)
        self.assertEqual(UD.strict_matches(-np.zeros(100), WORD), 0)
        self.assertEqual(UD.strict_matches(vector(74), WORD), 74)
        self.assertEqual(UD.strict_accuracy(vector(74), WORD), .74)
        self.assertEqual(UD.strict_matches(np.ones(100), np.ones(100)), 100)
        for bad in (math.inf, -math.inf, math.nan):
            values = np.ones(100); values[0] = bad
            with self.assertRaises(ValueError):
                UD.strict_matches(values, WORD)
        for values in (np.ones(99), ['1'] * 100, np.ones((10, 10)), np.ones(100, dtype=complex)):
            with self.assertRaises(ValueError):
                UD.strict_matches(values, WORD)

    def test_zero_tie_upper_tail_dominates_for_rademacher_fusion(self):
        # Explicit enumeration: strict matching never gives a zero sum a
        # free bit. Even with frequent ties its tail is <= Binomial(n,.5).
        n = 4
        match_counts = []
        for signs in itertools.product((-1, 1), repeat=2 * n):
            match_counts.append(sum(signs[j] + signs[n + j] < 0 for j in range(n)))
        for threshold in range(n + 1):
            probability = sum(k >= threshold for k in match_counts) / len(match_counts)
            upper = sum(math.comb(n, k) for k in range(threshold, n + 1)) / 2 ** n
            self.assertLessEqual(probability, upper)

    def test_no_identity_bypass_and_install_preserves_embedding(self):
        class Local(Composite):
            pass
        embedding = Local.embed
        UD.install(Local)
        UD.install(Local)
        self.assertIs(Local.embed, embedding)
        comp = Local()
        result = comp.decode(Image(), ('id', WORD), .01)
        self.assertEqual(result, (0.0, False))
        self.assertFalse(comp.geo_cascade(Image(), 'id', WORD))
        self.assertEqual(comp.embed(), 'unchanged embedding')

    def test_primary_fusion_can_be_the_only_acceptance(self):
        a, b = np.ones(100), np.ones(100)
        a[:20] = b[:20] = -1
        a[20:60], b[20:60] = -2, 1
        a[60:], b[60:] = 1, -2
        comp = Composite(('vine', 'trustmark'), scorer=lambda name, image: a if name == 'vine' else b)
        result = UD.decode_detailed(comp, Image(), ('id', WORD), 1e-6)
        self.assertTrue(result['detected'])
        self.assertEqual(result['accepted_path'], 'primary:fused')
        self.assertEqual(result['accepted_score'], 1)
        self.assertEqual(result['best_fragment_ba'], .6)
        self.assertEqual(result['attempted_tests'], 3)

    def test_primary_pass_does_not_initialize_geometry(self):
        comp = Composite(('vine', 'trustmark'), ('resync',), scorer=lambda *_: vector(100))
        result = UD.decode_detailed(comp, Image(), ('id', WORD), .01)
        self.assertTrue(result['detected'])
        self.assertFalse(result['used_geometry'])
        self.assertEqual(comp.geometry_actions, [])
        self.assertEqual(len(comp.reads), 2)
        self.assertEqual(result['max_tests'], 13)

    def test_resync_early_stop_and_report_reads_are_separate(self):
        comp = Composite(('trustmark', 'vine'), ('resync',),
                         scorer=lambda name, image: vector(95 if name == 'vine' and image.label == 'rectified' else 50))
        result = UD.decode_detailed(comp, Image(), ('id', WORD), 1e-6)
        self.assertTrue(result['detected'])
        self.assertEqual(result['accepted_path'], 'resync:rectified:vine')
        self.assertEqual(result['accepted_score'], .95)
        self.assertEqual(result['attempted_tests'], 4)  # three primary + one actual geo acceptance test
        self.assertEqual(len(comp.reads), 4)  # includes TM read for the accepted-view report
        self.assertEqual(result['max_tests'], 13)
        self.assertEqual(comp.geometry_actions, ['rectify'])

    def test_exact_maximum_geometric_attempt_counts(self):
        for order, stage, budget in ((('vine', 'trustmark'), 'resync', 13),
                                     (('vine',), 'scale', 134),
                                     (('trustmark',), 'tile', 82),
                                     (('vine', 'trustmark', 'videoseal'), 'angle', 199)):
            comp = Composite(order, (stage,))
            with angle_module(range(65)):
                result = UD.decode_detailed(comp, Image(), ('id', WORD), .01)
            self.assertFalse(result['detected'])
            self.assertEqual(result['attempted_tests'], budget)
            self.assertEqual(result['max_tests'], budget)

    def test_geometry_fusion_is_report_only(self):
        a, b = np.ones(100), np.ones(100)
        a[:20] = b[:20] = -1
        a[20:60], b[20:60] = -2, 1
        a[60:], b[60:] = 1, -2
        def score(name, image):
            return np.zeros(100) if image.label == 'att' else (a if name == 'vine' else b)
        comp = Composite(('vine', 'trustmark'), ('resync',), scorer=score)
        result = UD.decode_detailed(comp, Image(), ('id', WORD), 1e-6)
        self.assertEqual(UD.strict_accuracy(a + b, WORD), 1)
        self.assertFalse(result['detected'])
        self.assertEqual(result['accepted_score'], 0)  # failed result retains primary view

    def test_tile_only_trustmark_accepts_but_all_fragments_are_reported(self):
        def score(name, image):
            if image.label == 'att':
                return vector(50)
            return vector(100 if name == 'vine' else 90)
        comp = Composite(('vine', 'trustmark'), ('tile',), scorer=score)
        result = UD.decode_detailed(comp, Image(), ('id', WORD), 1e-6)
        self.assertTrue(result['detected'])
        self.assertTrue(result['accepted_path'].endswith(':trustmark'))
        self.assertEqual(result['accepted_score'], .9)
        self.assertEqual(result['best_fragment_ba'], 1)
        self.assertEqual(result['attempted_tests'], 4)
        comp = Composite(('vine', 'trustmark'), ('tile',),
                         scorer=lambda name, image: vector(100 if name == 'vine' and image.label != 'att' else 40))
        result = UD.decode_detailed(comp, Image(), ('id', WORD), 1e-6)
        self.assertFalse(result['detected'])
        self.assertFalse(any(name == 'vine' and label != 'att' for name, label in comp.reads))

    def test_shared_threshold_results_match_separate_early_stop_decodes(self):
        def score(name, image):
            if name != 'vine' or image.label == 'att':
                return vector(50)
            if image.label == 'rectified':
                return vector(70)
            if image.label.endswith('rotate:-3'):
                return vector(80)
            return vector(100)
        comp = Composite(('vine', 'trustmark'), ('resync',), scorer=score)
        batch = UD.decode_thresholds_detailed(comp, Image(), ('id', WORD), ALPHAS)
        shared_reads = len(comp.reads)
        separate_reads = 0
        for alpha in ALPHAS:
            single = Composite(('vine', 'trustmark'), ('resync',), scorer=score)
            result = UD.decode_detailed(single, Image(), ('id', WORD), alpha)
            separate_reads += len(single.reads)
            shared = batch['by_threshold'][batch['alpha_to_threshold'][repr(alpha)]]
            for key in ('detected', 'accepted_score', 'best_fragment_ba', 'per_fragment_ba',
                        'fused_ba', 'accepted_path', 'attempted_tests', 'tests'):
                self.assertEqual(shared[key], result[key], (alpha, key))
        self.assertLess(shared_reads, separate_reads)

    def test_trace_all_keeps_first_acceptance_but_records_complete_search(self):
        comp = Composite(('vine',), ('resync',), scorer=lambda *_: vector(100))
        result = UD.decode_detailed(comp, Image(), ('id', WORD), .01, trace_all=True)
        self.assertEqual(result['accepted_path'], 'primary:vine')
        self.assertEqual(result['attempted_tests'], 1)
        self.assertEqual(len(result['tests']), 1)
        self.assertEqual(len(result['all_tests']), 6)

    def test_decode_no_cascade_switch_is_respected(self):
        comp = Composite(('vine',), ('resync',), scorer=lambda name, image: vector(100 if image.label != 'att' else 0))
        comp.geo = False
        result = UD.decode_detailed(comp, Image(), ('id', WORD), .01)
        self.assertFalse(result['detected'])
        self.assertEqual(result['max_tests'], 1)
        self.assertEqual(comp.geometry_actions, [])

    def test_invalid_configuration_alpha_and_unbudgeted_geometry(self):
        for cfg in (dict(order=['unknown']), dict(order=[]), dict(order=['vine', 'vine']),
                    dict(order=['vine'], fe_on=['tile']), dict(order=['trustmark'], fe_on=['scale']),
                    dict(order=['vine'], fe_on=['unknown']), dict(order=['vine'], unexpected=True),
                    dict(order=['vine'], s={'vine': float('nan')})):
            with self.assertRaises(ValueError):
                UD.budget_for_config(cfg)
        for alpha in (0, 1, -1, float('nan'), float('inf'), True, '.01'):
            with self.assertRaises(ValueError):
                UD.threshold_count(dict(order=['vine']), alpha)
        comp = Composite(('vine',), ('scale',)); comp._vss = .001
        with self.assertRaisesRegex(ValueError, 'scale-search'):
            UD.decode_detailed(comp, Image(), ('id', WORD), .01)
        comp = Composite(('vine',), ('angle',))
        with angle_module(range(66)):
            with self.assertRaisesRegex(ValueError, 'angle candidates'):
                UD.decode_detailed(comp, Image(), ('id', WORD), .01)
        comp = Composite(('trustmark',), ('tile',)); comp._tiled.offsets = list(range(17))
        with self.assertRaisesRegex(ValueError, 'tile-search'):
            UD.decode_detailed(comp, Image(), ('id', WORD), .01)


if __name__ == '__main__':
    unittest.main()
