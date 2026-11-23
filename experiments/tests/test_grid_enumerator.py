"""C2 raw-mode search: exact on-knot values, flagged fallbacks, top-3, budget statuses, determinism, CLI."""
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grid import frozen  # noqa: E402
from grid import knots as K  # noqa: E402

AVAILABLE = frozen.available()
if AVAILABLE:
    from grid import grid_enumerator as G  # noqa: E402
    from grid.grid_enumerator import FRAGMENTS, GridSearch, margins, search  # noqa: E402
    from grid.tables import GridTables  # noqa: E402

TIMING = ('setup_s', 'search_s', 'total_s', 'first_feasible_s', 'over_budget')
# Request ids (field i) chosen to span the classes; 4050 has ranked candidates but no feasible one.
SAMPLE = (0, 366, 1100, 1832, 2500, 3000, 3500, 4000, 4392, 5400, 6300, 7000)
UNSAT_WITH_CANDIDATES = 4050


@unittest.skipUnless(AVAILABLE, 'relocated run root not present')
class RawSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = frozen.boot()
        cls.T = GridTables(mode='raw', ns=cls.ns)
        cls.sg = cls.T.sg
        cls.requests = {r['i']: r for r in json.loads(frozen.requests_path().read_text())}
        cls.results = {i: search(cls.T, cls.requests[i]['request']) for i in SAMPLE}

    def gs(self, request, stage='initial'):
        rate_margin, mean_margin = margins(self.ns, stage)
        return GridSearch(self.T, request, rate_margin=rate_margin, mean_margin=mean_margin)

    # ---- no interpolation ------------------------------------------------------------------------------------
    def test_every_returned_strength_is_a_measured_knot(self):
        n = 0
        for i, res in self.results.items():
            for c in res['candidates']:
                for f, x in c['s'].items():
                    self.assertIn(x, K.EXPECTED_KNOTS[f])
                    self.assertEqual(K.EXPECTED_KNOTS[f][c['knot_index'][f]], x)
                    self.assertGreaterEqual(x, res['strength_floor'][f])
                    n += 1
        self.assertGreater(n, 20)

    def test_ranked_space_holds_knot_indices_only(self):
        gs = self.gs(self.requests[4000]['request'])
        for fi, f in enumerate(FRAGMENTS):
            col = gs.c_kn[:, fi]
            sel = col >= 0
            self.assertTrue(np.all(col[sel] < len(K.EXPECTED_KNOTS[f])))
            self.assertTrue(np.all(self.T.kx[f][col[sel]] >= gs.lo[f]))

    # ---- exact recorded values ---------------------------------------------------------------------------------
    def direct(self, key, f, knot, tau):
        rec = self.sg._perimage[key]
        for x, row in zip(rec['xs'], K.record_rows(rec, f)):
            if abs(x - knot) <= 1e-9:
                row = np.asarray(row, float)
                return math.fsum(row) / len(row), float(np.mean(row >= tau))
        return None

    def test_on_knot_means_and_rates_are_the_raw_per_image_values(self):
        checked = 0
        for i, res in self.results.items():
            gs = self.gs(self.requests[i]['request'])
            for c in res['candidates']:
                ev = gs.evaluate(c['order'], c['fe_on'][0] if c['fe_on'] else None, c['s'])
                for a, by_f in ev['cells'].items():
                    for f, cell in by_f.items():
                        if cell['mean_source'].startswith('raw:'):
                            key = cell['mean_source'][4:]
                            mean, rate = self.direct(key, f, c['s'][f], c['live_threshold'])
                            if not cell['drops']:
                                self.assertEqual(cell['mean'], min(1.0, mean))
                            self.assertEqual(cell['level'], min(1.0, mean))
                            checked += 1
                        if cell['raw_rate'] is not None:
                            _, rate = self.direct(cell['rate_source'], f, c['s'][f], c['live_threshold'])
                            self.assertEqual(cell['raw_rate'], rate)
                            self.assertEqual(cell['rate_pass'], rate >= ev['rate_level'] - 1e-12)
        self.assertGreater(checked, 100)

    def test_raw_rate_is_not_isotonic_smoothed(self):
        # VINE alone at 0.5 under ctrlregen_s05, tau 0.57 (fpr 0.1, one fragment), level 0.90 (margin 00):
        # the raw fraction clears 0.90, the solver's isotonic fit of the same record does not.
        req = dict(attacks=['ctrlregen_s05'], fpr=0.1, min_psnr=0.0, max_ms=1e9)
        cell = self.gs(req, '00').evaluate(('VINE',), None, {'VINE': 0.5})['cells']['ctrlregen_s05']['VINE']
        self.assertEqual(self.direct('VINE|ctrlregen_s05', 'VINE', 0.5, 0.57)[1], cell['raw_rate'])
        self.assertGreaterEqual(cell['raw_rate'], 0.9)
        self.assertTrue(cell['rate_pass'])
        self.assertFalse(cell['rate_pass_solver'])
        # And the opposite: VideoSeal at 3.2 under crop50, tau 0.69, level 0.96: raw fails, smoothed passes.
        req = dict(attacks=['crop50'], fpr=1e-4, min_psnr=0.0, max_ms=1e9)
        cell = self.gs(req).evaluate(('VideoSeal',), None, {'VideoSeal': 3.2})['cells']['crop50']['VideoSeal']
        self.assertLess(cell['raw_rate'], 0.96)
        self.assertFalse(cell['rate_pass'])
        self.assertTrue(cell['rate_pass_solver'])

    def test_missing_records_and_knots_fall_back_to_fitted_curves_and_are_flagged(self):
        req = dict(attacks=['editing_ip2p_s20_v1', 'unmarker', 'jpeg25'], fpr=0.01, min_psnr=0.0, max_ms=1e9)
        gs = self.gs(req)
        ev = gs.evaluate(('VINE',), None, {'VINE': 0.3})                 # editing has no 0.3 knot
        cell = ev['cells']['editing_ip2p_s20_v1']['VINE']
        self.assertEqual(cell['mean_source'], 'fitted:base:VINE|editing_ip2p_s20_v1')
        self.assertEqual(cell['mean'], self.sg.base('VINE', 'editing_ip2p_s20_v1').eval(0.3))
        self.assertIsNone(cell['raw_rate'])
        self.assertIn('mean_fitted:base:VINE|editing_ip2p_s20_v1', ev['flags'])
        self.assertIn('rate_fitted:VINE|editing_ip2p_s20_v1', ev['flags'])
        self.assertEqual(ev['cells']['jpeg25']['VINE']['mean_source'], 'raw:VINE|jpeg25')
        ev = gs.evaluate(('VINE',), None, {'VINE': 1.0})                 # editing measured at 1.0
        self.assertEqual(ev['cells']['editing_ip2p_s20_v1']['VINE']['mean_source'], 'raw:VINE|editing_ip2p_s20_v1')
        self.assertNotIn('mean_fitted:base:VINE|editing_ip2p_s20_v1', ev['flags'])
        ev = gs.evaluate(('VINE',), None, {'VINE': 0.17})                # UnMarker lacks VINE 0.14 and 0.17
        self.assertIn('mean_fitted:base:VINE|unmarker', ev['flags'])
        self.assertIn('rate_fitted:VINE|unmarker', ev['flags'])
        # A decode-side replacement curve without per-image data: TrustMark under the VINE scale search.
        req = dict(attacks=['jpeg25', 'crop75'], fpr=0.01, min_psnr=0.0, max_ms=1e9)
        ev = self.gs(req).evaluate(('VINE', 'TrustMark'), 'scale', {'VINE': 0.5, 'TrustMark': 1.0})
        cell = ev['cells']['jpeg25']['TrustMark']
        self.assertEqual(cell['mean_source'], 'fitted:base_fe_scale_TrustMark|jpeg25')
        self.assertEqual(cell['rate_source'], 'TrustMark|jpeg25')        # scale does not re-embed TrustMark
        self.assertIn('mean_fitted:base_fe_scale_TrustMark|jpeg25', ev['flags'])
        self.assertEqual(ev['cells']['jpeg25']['VINE']['mean_source'], 'raw:scale:VINE|jpeg25')
        self.assertEqual(ev['label'], 'grid_lookup_surrogate')

    def test_composition_applies_delta_at_the_overwriter_knot(self):
        req = dict(attacks=['jpeg25'], fpr=0.01, min_psnr=0.0, max_ms=1e9)
        gs = self.gs(req)
        for order in (('VINE', 'TrustMark'), ('TrustMark', 'VINE')):
            ev = gs.evaluate(order, None, {'VINE': 0.4, 'TrustMark': 1.15})
            host, over = order
            cell = ev['cells']['jpeg25'][host]
            expected = self.sg.delta(over, host, 'jpeg25').eval(1.15 if over == 'TrustMark' else 0.4)
            self.assertEqual(cell['drops'], {over: expected} if expected else {})
            self.assertEqual(cell['mean'], cell['level'] - expected)
            self.assertEqual(ev['cells']['jpeg25'][over]['drops'], {})   # nothing is embedded after it
            self.assertEqual(ev['label'], 'grid_lookup_surrogate')

    # ---- top-3 and search order ----------------------------------------------------------------------------
    def test_top3_are_the_best_distinct_structures(self):
        for i in (0, 4000, 6300):
            gs = self.gs(self.requests[i]['request'])
            feasible = gs.evaluate_positions(0, gs.n_ranked)
            best = {}
            for pos in np.flatnonzero(feasible):
                best.setdefault(int(gs.c_sid[pos]), int(pos))
            expected = sorted(best.values())[:3]
            res = self.results[i]
            self.assertEqual([c['position'] for c in res['candidates']], expected)
            structs = [(tuple(c['order']), tuple(c['fe_on'])) for c in res['candidates']]
            self.assertEqual(len(set(structs)), len(structs))
            D = [c['D_pred'] for c in res['candidates']]
            self.assertEqual(D, sorted(D))
            self.assertTrue(np.all(np.diff(gs.c_D) >= 0))                  # visiting order is by predicted D

    def test_early_stop_and_full_exhaustion_return_the_same_top3(self):
        for i in (0, 4000, 6300):
            full = search(self.T, self.requests[i]['request'], stop_when_complete=False)
            self.assertTrue(full['exhausted'])
            self.assertEqual(full['examined'], full['n_ranked'])
            strip = lambda cs: [{k: v for k, v in c.items()} for c in cs]
            self.assertEqual(json.dumps(strip(full['candidates']), sort_keys=True),
                             json.dumps(strip(self.results[i]['candidates']), sort_keys=True))

    def test_deterministic(self):
        for i in (0, 3500):
            a = search(self.T, self.requests[i]['request'])
            b = search(self.T, self.requests[i]['request'])
            clean = lambda r: json.dumps({k: v for k, v in r.items() if k not in TIMING}, sort_keys=True)
            self.assertEqual(clean(a), clean(b))
            self.assertEqual(clean(a), clean(self.results[i]))

    def test_candidates_respect_every_rule(self):
        for i, res in self.results.items():
            req = self.requests[i]['request']
            for c in res['candidates']:
                self.assertTrue(c['feasible'])
                self.assertLessEqual(c['ms'], req['max_ms'])
                self.assertGreaterEqual(c['psnr_pred'], req['min_psnr'] - 1e-9)
                for a in req['attacks']:
                    self.assertTrue(c['coverage'][a])
                for f in c['order']:
                    self.assertTrue(c['useful'][f])
                if c['fe_on']:
                    self.assertIn(c['fe_on'][0], res['fe_responsible'])
                self.assertEqual(c['label'], 'grid_lookup_surrogate' if len(c['order']) > 1 else 'onknot_lookup')

    # ---- budget statuses ---------------------------------------------------------------------------------------
    def test_statuses_are_distinguishable(self):
        r0 = self.requests[0]['request']
        self.assertEqual(self.results[0]['status'], 'sat')
        zero = search(self.T, r0, budget_s=0.0)
        self.assertEqual((zero['status'], zero['examined'], zero['exhausted'], zero['complete']),
                         ('budget_exhausted', 0, False, False))
        self.assertEqual(zero['candidates'], [])
        unsat = self.results.get(UNSAT_WITH_CANDIDATES) or search(self.T, self.requests[UNSAT_WITH_CANDIDATES]['request'])
        self.assertEqual(unsat['status'], 'model_unsat')
        self.assertGreater(unsat['n_ranked'], 0)
        self.assertEqual(unsat['examined'], unsat['n_ranked'])
        self.assertTrue(unsat['exhausted'])
        empty = search(self.T, dict(r0, min_psnr=80.0))
        self.assertEqual((empty['status'], empty['n_ranked'], empty['exhausted']), ('model_unsat', 0, True))

    def test_budget_stops_mid_search_with_a_fake_clock(self):
        req = self.requests[UNSAT_WITH_CANDIDATES]['request']
        clock = itertools.count()
        with mock.patch.object(G.time, 'perf_counter', side_effect=lambda: float(next(clock))):
            res = search(self.T, req, budget_s=4.0)
        self.assertEqual(res['status'], 'budget_exhausted')
        self.assertFalse(res['exhausted'])
        self.assertGreater(res['examined'], 0)
        self.assertLess(res['examined'], res['n_ranked'])
        # The same clock on request 0, whose optimum lies in the first chunk: sat within the budget, not complete.
        self.assertLess(self.results[0]['candidates'][0]['position'], G.FIRST_CHUNK)
        clock = itertools.count()
        with mock.patch.object(G.time, 'perf_counter', side_effect=lambda: float(next(clock))):
            res = search(self.T, self.requests[0]['request'], budget_s=4.0, stop_when_complete=False)
        self.assertEqual(res['status'], 'sat')
        self.assertFalse(res['complete'])
        self.assertFalse(res['exhausted'])
        self.assertEqual(res['candidates'][0]['position'], self.results[0]['candidates'][0]['position'])

    # ---- raw distortion table ----------------------------------------------------------------------------------
    def test_raw_single_fragment_distortion_replaces_fitted_terms(self):
        stages = {'VINE': ('none', 'resync', 'scale'), 'TrustMark': ('none', 'resync', 'tile'),
                  'VideoSeal': ('none', 'resync')}
        table = {f: {st: {repr(x): {'mean_mse': 1.0 + 0.5 * j + 0.01 * i, 'mean_psnr': 0.0}
                          for i, x in enumerate(K.EXPECTED_KNOTS[f])} for j, st in enumerate(sts)}
                 for f, sts in stages.items()}
        doc = dict(schema='tailor_exp_raw_distortion_v1', distortion_domain='mse_255', table=table)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'raw.json'
            path.write_text(json.dumps(doc))
            TR = GridTables(mode='raw', raw_distortion=path, ns=self.ns)
            path.write_text(json.dumps({'VINE': {'none': {'0.31': 1.0}}}))
            with self.assertRaises(ValueError):
                GridTables(mode='raw', raw_distortion=path, ns=self.ns)
        req = dict(attacks=['jpeg25', 'crop75'], fpr=0.01, min_psnr=0.0, max_ms=1e9)
        rate_margin, mean_margin = margins(self.ns, 'initial')
        gs = GridSearch(TR, req, rate_margin=rate_margin, mean_margin=mean_margin)
        v = lambda f, st, x: table[f][st][repr(x)]['mean_mse']
        ev = gs.evaluate(('VINE',), 'scale', {'VINE': 0.5})
        self.assertEqual(ev['D_pred'], v('VINE', 'scale', 0.5))
        self.assertNotIn('distortion_fitted:VINE', ev['flags'])
        ev = gs.evaluate(('TrustMark', 'VINE'), None, {'VINE': 0.5, 'TrustMark': 1.0})
        self.assertAlmostEqual(ev['D_pred'], v('VINE', 'none', 0.5) + v('TrustMark', 'none', 1.0)
                               + self.sg.e('VINE', 'TrustMark').eval(1.5), places=12)
        # resync is charged once, on TrustMark when present, else on the first selected fragment.
        ev = gs.evaluate(('VideoSeal', 'VINE'), 'resync', {'VINE': 0.5, 'VideoSeal': 1.0})
        self.assertAlmostEqual(ev['D_pred'], v('VINE', 'resync', 0.5) + v('VideoSeal', 'none', 1.0)
                               + self.sg.e('VINE', 'VideoSeal').eval(1.5), places=12)
        # Without a table the fitted d plus the solver's stage penalty are used and flagged.
        ev = self.gs(req).evaluate(('VINE',), 'scale', {'VINE': 0.5})
        self.assertAlmostEqual(ev['D_pred'], self.sg.d('VINE').eval(0.5) +
                               self.sg.frontend('nested_penalty').eval(0.5), places=12)
        self.assertIn('distortion_fitted:VINE', ev['flags'])
        self.assertIn('stage_penalty_fitted:scale', ev['flags'])

    # ---- CLI -----------------------------------------------------------------------------------------------------
    def test_cli_writes_one_row_per_request_margin_and_budget(self):
        from grid import run_budgeted_search as R
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'c2.jsonl'
            R.main([str(out), '--indices', f'0,{UNSAT_WITH_CANDIDATES}', '--budgets', 'inf,0',
                    '--margins', 'initial,00'])
            rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(len(rows), 2 * 2 * 2)
        for row in rows:
            for k in ('spec_id', 'status', 'candidates', 'examined', 'total_s', 'first_feasible_s', 'exhausted',
                      'n_space', 'surrogate_sha256', 'rate_margin', 'mean_margin'):
                self.assertIn(k, row)
            self.assertEqual(row['n_space'], 58433)
            for c in row['candidates']:
                for k in ('order', 'fe_on', 's', 'psnr_pred', 'ms', 'max_tests', 'threshold_count', 'flags'):
                    self.assertIn(k, c)
                self.assertNotIn('cells', c)
        by = {(r['i'], r['margin_stage'], r['budget_s']): r for r in rows}
        self.assertEqual(by[(0, 'initial', None)]['status'], 'sat')
        self.assertEqual(by[(0, 'initial', 0.0)]['status'], 'budget_exhausted')
        self.assertEqual(by[(UNSAT_WITH_CANDIDATES, 'initial', None)]['status'], 'model_unsat')
        self.assertEqual(by[(0, '00', None)]['rate_margin'], 0.0)


if __name__ == '__main__':
    unittest.main()
