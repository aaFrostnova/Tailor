"""CPU regressions for the new selection protocol; no old result is mutated."""
import copy
import sys
import unittest
from pathlib import Path
U = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(U / 'code'), str(U / 'fixed/code')]
import numpy as np
import z3
import capacity_protocol as CP
import watermark_smt_v2 as W
from surrogate_model import PWL, Surrogate
from composite_prediction import composite_expression, NumericBackend, Z3Backend, predict_composite
from unified_detector import threshold_count, threshold_for_config


def constant(value):
    return PWL([0.1, 2.0], [value, value])


def surrogate():
    fs = ['VINE', 'TrustMark', 'VideoSeal']
    base = {(f, 'crop50'): constant(.8) for f in fs}
    delta = {(g, f, 'crop50'): constant(0) for g in fs for f in fs if g != f}
    fe = {f'base_fe_{stage}_{f}|crop50': constant(.98)
          for stage in ('resync', 'scale', 'angle', 'tile') for f in fs}
    fe.update({name: constant(0) for name in ('nested_penalty', 'penalty_fe_resync', 'penalty_fe_tile')})
    return Surrogate(fs, ['crop50'], {f: (.1, 2) for f in fs}, base, delta,
                     {f: constant(1) for f in fs}, {tuple(sorted((f,g))): constant(0) for i,f in enumerate(fs) for g in fs[i+1:]},
                     frontend=fe)


def cell(theta=70, score=.8, best=.6, detected=True):
    return dict(protocol=CP.CELL_PROTOCOL, n=1, order=['VINE'], by_threshold={str(theta):
        dict(threshold_count=theta, detected=[detected], used_geometry=[False],
             accepted_score=[score], best_fragment_ba=[best], fused_ba=[score], per_fragment_ba={'VINE':[best]})})


class ProtocolTests(unittest.TestCase):
    def test_exact_strength_cache_identity(self):
        import json
        x = .5000000004374595
        y = float(np.nextafter(x, 1.0))
        self.assertEqual(round(x, 3), round(y, 3))
        a = CP.exact_cfg_key(['VINE'], [], [x])
        b = CP.exact_cfg_key(['VINE'], [], [y])
        self.assertNotEqual(CP.exact_cfg_id(a), CP.exact_cfg_id(b))
        self.assertEqual(CP.exact_cfg_id(a), CP.exact_cfg_id(json.loads(json.dumps(a))))
        with self.assertRaises(ValueError):
            CP.exact_cfg_key(['VINE'], [], [float('nan')])

    def test_incomplete_threshold_coverage_remeasured(self):
        c = cell()
        self.assertTrue(CP.valid_cell(c, ['VINE'], 1))
        self.assertFalse(CP.valid_cell(c, ['VINE'], 1, cfg=dict(order=['VINE'],fe_on=[])))

    def test_no_legacy_cells(self):
        self.assertFalse(CP.valid_cell(dict(views=2, ba={'VINE':[1]}, idv=[True]), ['VINE'], 1))
        with self.assertRaises(ValueError):
            CP.combine(dict(views=2), ['VINE'], .7)

    def test_exact_bucket_and_decision(self):
        c = cell()
        self.assertTrue(CP.valid_cell(c, ['VINE'], 1))
        self.assertTrue(CP.combine(c, ['VINE'], .7)['ok'])
        with self.assertRaises(ValueError):
            CP.combine(c, ['VINE'], .71)
        c['by_threshold']['70']['detected'] = [False]
        self.assertFalse(CP.valid_cell(c, ['VINE'], 1))

    def test_capacity_separate_from_acceptance(self):
        # Fusion may accept while each individual fragment stays below tau.
        stats = CP.combine(cell(), ['VINE'], .7)
        self.assertEqual(stats['mean'], .8)
        self.assertEqual(stats['mean_best_fragment_ba'], .6)
        self.assertAlmostEqual(stats['capacity_bits_estimate'], CP.capacity_from_ba(.6))

    def test_judge_uses_full_config(self):
        cfg = dict(order=['VINE'], fe_on=['scale'], ms=10)
        request = dict(attacks=['crop50'], fpr=.1, min_psnr=20, max_ms=20)
        theta = threshold_count(dict(order=['VINE'], fe_on=['scale']), .1)
        c = cell(theta, .9, .9)
        result = CP.judge(request, cfg, {'crop50':dict(cell=c, psnr_db=30)})
        self.assertTrue(result['pass'])
        self.assertEqual(result['max_tests'], 134)
        self.assertEqual(result['threshold_count'], theta)
        self.assertGreater(theta, threshold_count(dict(order=['VINE'],fe_on=[]), .1))

    def test_legacy_rate_ignores_ids_and_keyed_view(self):
        sg = surrogate()
        sg._perimage['VINE|crop50'] = dict(xs=[.1,2], ba=[[.4],[.4]], ver=[[True],[True]])
        self.assertEqual(sg.rate_curve('VINE','crop50',.7).ys, [0,0])
        rec = dict(ba={'VINE':[.4]}, cas_ba={'VINE':[1]}, idv=[True], cas_ok=[True])
        sg._perimage['VINE|crop50']['raw_views'] = [copy.deepcopy(rec), copy.deepcopy(rec)]
        self.assertEqual(sg.rate_curve('VINE','crop50',.7).ys, [0,0])
        shifted = sg.with_composite_offsets(rate_offsets={('VINE','crop50'):-.1})
        self.assertAlmostEqual(shifted._perimage['VINE|crop50']['raw_views'][0]['ba']['VINE'][0], .3)
        self.assertEqual(shifted.rate_curve('VINE','crop50',.7).ys, [0,0])

    def test_scenario_preserves_original_alpha(self):
        alpha = 2**-37
        scen = CP.solver_scenario(dict(attacks=['crop50'],fpr=alpha,min_psnr=20,max_ms=100))
        self.assertEqual(scen['fpr'], alpha)
        with self.assertRaises(ValueError):
            W.presence_threshold(scen['min_ba'], 3)


class CompositeTests(unittest.TestCase):
    def test_only_selected_last_pair_suppresses_delta(self):
        sg = surrogate()
        sg._frontend['base_fe_resync_VINE|crop50|with_TrustMark'] = constant(.80)
        sg._frontend['base_fe_resync_VINE|crop50|with_VideoSeal'] = constant(.85)
        sg._delta[('TrustMark','VINE','crop50')] = constant(.10)
        sg._delta[('VideoSeal','VINE','crop50')] = constant(.05)
        s = {f:1 for f in sg.fragments}
        after = lambda f,g: sg.fragments.index(g) > sg.fragments.index(f)
        r = composite_expression(sg,'VINE','crop50',s,after,{'resync':True},NumericBackend())
        self.assertAlmostEqual(r['mean'], .75)
        self.assertEqual([x['pair_used'] for x in r['delta_terms']], [False,True])
        backend = Z3Backend()
        symbolic = composite_expression(sg,'VINE','crop50',s,after,{'resync':True},backend)
        solver = z3.Solver(); solver.add(*backend.constraints)
        self.assertEqual(solver.check(), z3.sat)
        self.assertAlmostEqual(float(solver.model().eval(symbolic['mean']).as_fraction()), .75)

    def test_impossible_threshold_not_clipped(self):
        cfg = dict(order=['VINE'],fe_on=[],s={'VINE':1})
        result = predict_composite(surrogate(), cfg, 'crop50', tau=1.01)
        self.assertEqual(result['mean_threshold'], 1.01)
        self.assertFalse(result['attack_pass'])

    def test_symbolic_threshold_all_cardinalities_and_stages(self):
        sg = surrogate()
        scen = CP.solver_scenario(dict(attacks=['crop50'],fpr=.01,min_psnr=0,max_ms=1e8), margin=.02)
        built = W.build(**scen, enable_order=True, continuous_strength=True, surrogate=sg, clean_floor={}, det_min=0)
        opt,use,rs,ns,*_ = built
        for stage in (None,'resync','scale','angle','tile'):
            host = 'TrustMark' if stage == 'tile' else 'VINE'
            names = [host] + [f for f in sg.fragments if f != host]
            for n in (1,2,3):
                cfg = dict(order=names[:n],fe_on=[] if stage is None else [stage])
                solver = z3.Solver(); solver.add(*opt.assertions())
                solver.add(*[v == (f in names[:n]) for f,v in use.items()])
                solver.add(*[v == (key == stage) for key,v in opt._fevars.items()])
                self.assertEqual(solver.check(), z3.sat, cfg)
                m = solver.model()
                self.assertEqual(float(m.eval(opt._threshold_expr).as_fraction()), threshold_for_config(cfg,.01))
                self.assertAlmostEqual(float(m.eval(opt._mean_target_expr).as_fraction()), threshold_for_config(cfg,.01)+.02)

    def test_rate_uses_geometry_threshold(self):
        sg = surrogate()
        plain = threshold_for_config(dict(order=['VINE'],fe_on=[]), .1)
        geo = threshold_for_config(dict(order=['VINE'],fe_on=['scale']), .1)
        mid = (plain+geo)/2
        sg._perimage['scale:VINE|crop50'] = dict(xs=[.1,2], ba=[[mid]*10,[mid]*10], ver=[[True]*10,[True]*10])
        scen = CP.solver_scenario(dict(attacks=['crop50'],fpr=.1,min_psnr=0,max_ms=1e8), margin=0)
        opt,use,rs,ns,*_ = W.build(**scen, enable_order=True, continuous_strength=True, surrogate=sg, clean_floor={}, rate_margin=0)
        opt.add(use['VINE'], z3.Not(use['TrustMark']), z3.Not(use['VideoSeal']), ns)
        self.assertEqual(opt.check(), z3.unsat)


if __name__ == '__main__':
    unittest.main()
