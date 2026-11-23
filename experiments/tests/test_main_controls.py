"""Main controls (A1 SingleOptimized, A3 AlwaysAllThree): restrictions, search flow, cache origin, bridging.

The modules import the relocated frozen runner, so these tests need the prepared run root; they are
skipped when it is absent.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RUN = Path(os.environ.get('TAILOR_RUNS', 'runs') + '/main_controls_20261109')
os.environ.setdefault('TAILOR_MAIN_CONTROLS_RUN', str(RUN))
HERE = Path(__file__).resolve().parents[1] / 'main_controls'
AVAILABLE = (Path(os.environ['TAILOR_MAIN_CONTROLS_RUN']) / 'root/fixed/code/campaign.py').exists()
if AVAILABLE:
    sys.path.insert(0, str(HERE))
    import common  # noqa: E402
    import mc_controller as D  # noqa: E402
    import bridge as B  # noqa: E402


CAND = dict(order=['VINE'], fe_on=[], s={'VINE': .5})


def cell(order, n, detected, score=0.9, theta='80'):
    return dict(protocol='unified_raw_ba_v1', n=n, order=list(order), by_threshold={theta: dict(
        threshold_count=int(theta), detected=list(detected), used_geometry=[False] * n,
        accepted_score=[score if d else 0.5 for d in detected], best_fragment_ba=[score if d else 0.5 for d in detected],
        fused_ba=[0.5] * n, per_fragment_ba={f: [score if d else 0.5 for d in detected] for f in order})})


@unittest.skipUnless(AVAILABLE, 'relocated run root not present')
class RestrictionTest(unittest.TestCase):
    def model(self, variant):
        import z3
        o = z3.Solver()
        u = {f: z3.Bool(f) for f in common.FRAGMENTS}
        o._svars = {f: z3.Real('s' + f) for f in u}
        o.add(z3.Or(*u.values()))
        common.add_restriction(variant, (o, u))
        return o, u

    def test_single_excludes_pairs_but_keeps_strength_free(self):
        import z3
        o, u = self.model('SingleOptimized')
        o.push()
        o.add(u['VINE'], u['TrustMark'])
        self.assertEqual(o.check(), z3.unsat)
        o.pop()
        o.add(u['VINE'], o._svars['VINE'] == z3.RealVal('.73'))
        self.assertEqual(o.check(), z3.sat)

    def test_all_three_forces_every_fragment_and_keeps_strength_free(self):
        import z3
        o, u = self.model('AlwaysAllThree')
        o.push()
        o.add(z3.Not(u['VideoSeal']))
        self.assertEqual(o.check(), z3.unsat)
        o.pop()
        o.add(o._svars['VINE'] == z3.RealVal('.41'), o._svars['TrustMark'] == z3.RealVal('1.7'))
        self.assertEqual(o.check(), z3.sat)
        m = o.model()
        self.assertTrue(all(z3.is_true(m.eval(v)) for v in u.values()))

    def test_unknown_variant_is_refused(self):
        with self.assertRaises(ValueError):
            self.model('UnitStrength')

    def test_only_a3_relaxes_the_passenger_rule(self):
        self.assertEqual(common.NO_PASSENGERS, {'SingleOptimized': True, 'AlwaysAllThree': False})

    def test_candidate_shape_is_checked(self):
        common.check_candidate('SingleOptimized', dict(order=['VINE']))
        common.check_candidate('AlwaysAllThree', dict(order=['VideoSeal', 'VINE', 'TrustMark']))
        with self.assertRaises(AssertionError):
            common.check_candidate('SingleOptimized', dict(order=['VINE', 'TrustMark']))
        with self.assertRaises(AssertionError):
            common.check_candidate('AlwaysAllThree', dict(order=['VINE', 'TrustMark']))

    def test_schedule_is_the_frozen_arm_06_ladder(self):
        self.assertEqual(common.STAGES, ('initial', '04', '02', '00'))
        self.assertEqual([common.margins('06', s) for s in common.STAGES],
                         [(.06, .02), (0.04, 0.02), (0.02, 0.01), (0.0, 0.0)])

    def test_frozen_modules_come_from_the_relocated_root(self):
        for m in (common.C, common.CP, common.MFP):
            self.assertEqual(Path(m.__file__).resolve().parent, common.ROOT / 'fixed/code')


@unittest.skipUnless(AVAILABLE, 'relocated run root not present')
class FlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.p = patch.object(D, 'output', lambda t: self.root / (t['stage'] + '.json'))
        self.p.start()
        self.q = patch.object(D, '_walk_step', lambda stage, c, v: dict(stage=stage, rank=c['rank'],
                                                                        passed=v['pass'], pending=v['pending']))
        self.q.start()
        self.t = dict(variant='SingleOptimized', arm='06', cls='C4', i=1, spec_id='x',
                      request={'attacks': ['noise'], 'fpr': 1e-9, 'min_psnr': 35., 'max_ms': 2000.})
        self.c = dict(rank=1, order=['VINE'], fe_on=[], s={'VINE': .5})

    def tearDown(self):
        self.p.stop()
        self.q.stop()
        self.temp.cleanup()

    def put(self, stage, cs):
        common.atomic(self.root / (stage + '.json'), dict(candidates=cs))

    def inspect(self, passed=False, pending=()):
        cpu, gpu = [], {}
        with patch.object(D, 'judge', return_value={'pass': passed, 'pending': list(pending), 'attacks': {}}):
            result = D.inspect_target(self.t, cpu, gpu)
        return result, cpu, gpu

    def test_missing_measurement_queues_gpu_and_never_refines(self):
        self.put('initial', [self.c])
        r, c, g = self.inspect(pending=['unmarker'])
        self.assertEqual(r['status'], 'pending_live')
        self.assertFalse(c)
        self.assertEqual(len(g), 1)

    def test_complete_failure_refines_once_then_backs_off_to_04(self):
        self.put('initial', [self.c])
        r, c, g = self.inspect()
        self.assertEqual(c[0]['stage'], 'initial_patch')
        self.put('initial_patch', [])
        r, c, g = self.inspect()
        self.assertEqual(r['status'], 'pending_fallback')
        self.assertEqual(c[0]['stage'], '04')

    def test_04_has_no_dependency_on_another_arm(self):
        self.put('initial', [])
        r, c, g = self.inspect()
        self.assertEqual([x['stage'] for x in c], ['04'])
        self.assertEqual(r['status'], 'pending_fallback')

    def test_pass_stops(self):
        self.put('initial', [self.c])
        r, c, g = self.inspect(passed=True)
        self.assertEqual(r['status'], 'accepted')
        self.assertFalse(c)
        self.assertFalse(g)

    def test_pending_correction_blocks_backoff(self):
        self.put('initial', [self.c])
        self.put('initial_patch', [dict(self.c, rank=4)])
        cpu, gpu = [], {}

        def judge(req, c):
            return {'pass': False, 'pending': ['noise'] if c['rank'] == 4 else [], 'attacks': {}}
        with patch.object(D, 'judge', side_effect=judge):
            r = D.inspect_target(self.t, cpu, gpu)
        self.assertEqual(r['status'], 'pending_live')
        self.assertFalse(cpu)

    def test_terminal_statuses(self):
        for stage in common.STAGES:
            self.put(stage, [])
        r, c, g = self.inspect()
        self.assertEqual(r['status'], 'model_unsat')
        for stage in common.STAGES:
            self.put(stage, [self.c])
            self.put(stage + '_patch', [])
        r, c, g = self.inspect()
        self.assertEqual(r['status'], 'live_exhausted')
        self.assertEqual([w['stage'] for w in r['walk']], list(common.STAGES))

    def test_backoff_keeps_the_request(self):
        self.put('initial', [self.c])
        self.put('initial_patch', [])
        r, c, g = self.inspect()
        self.assertEqual(c[0]['request'], self.t['request'])

    def test_corrections_and_backoff_run_before_new_initial_proposals(self):
        tasks = [dict(i=5, stage='initial', variant='SingleOptimized'), dict(i=9, stage='04_patch', variant='AlwaysAllThree'),
                 dict(i=1, stage='initial', variant='AlwaysAllThree'), dict(i=2, stage='02', variant='SingleOptimized')]
        self.assertEqual([t['stage'] for t in D.fair_cpu_tasks(tasks)], ['04_patch', '02', 'initial', 'initial'])

    def test_transient_nfs_errors_are_retried_and_others_are_not(self):
        import errno
        self.assertTrue(common.transient(OSError(errno.ESTALE, 'Stale file handle')))
        self.assertTrue(common.transient(OSError(errno.EIO, 'I/O error')))
        self.assertTrue(common.transient(json.JSONDecodeError('x', '', 0)))
        self.assertFalse(common.transient(OSError(errno.ENOENT, 'missing')))
        self.assertFalse(common.transient(KeyError('ms')))

    def test_controller_survives_a_stale_handle_during_a_sweep(self):
        import errno
        calls = []

        def sweep():
            calls.append(1)
            if len(calls) == 1:
                raise OSError(errno.ESTALE, 'Stale file handle')
            return dict(complete=True)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(D, 'RUN', Path(tmp)), patch.object(D, 'verify_sources'), \
                patch.object(D, 'sweep', side_effect=sweep), patch.object(D, 'sha', return_value='x'), \
                patch.object(D.time, 'sleep'):
            D.main()
        self.assertEqual(len(calls), 2)

    def test_every_queued_task_kind_can_be_ranked(self):
        # Regression: ranking read task['attacks'] for CPU proposal tasks, which have none, and every new CPU
        # worker died at start (2026-11-09 21:55 to 2026-11-10 00:30).
        import worker as Wk
        run = Path(os.environ['TAILOR_MAIN_CONTROLS_RUN'])
        q = run / 'queue_cpu.json'
        if q.exists() and json.loads(q.read_text()):
            task = json.loads(Path(json.loads(q.read_text())[0]).read_text())
            self.assertEqual(Wk.priority('cpu', task), 0)
        self.assertEqual(Wk.priority('cpu', dict(variant='AlwaysAllThree', stage='initial', request={})), 0)
        self.assertEqual(Wk.priority('bridge', dict(attacks=['unmarker'])), 0)

    def test_a100_workers_start_with_work_only_they_can_do(self):
        import worker as Wk
        with tempfile.TemporaryDirectory() as tmp:
            approval = Path(tmp) / 'light.json'
            approval.write_text(json.dumps(dict(complete=True, attacks={'blur': dict(approved=True)})))
            with patch.object(common, 'LIGHT_APPROVAL', approval), \
                    patch.object(common, 'LIGHT_CHECK', Path(tmp) / 'absent.json'):
                self.assertEqual(Wk.priority('gpu', dict(attacks=['blur', 'unmarker'])), 0)
                self.assertEqual(Wk.priority('gpu', dict(attacks=['blur', 'editing_ip2p_s20_v1'])), 1)
                self.assertEqual(Wk.priority('gpu', dict(attacks=['blur', 'crop_jpeg'])), 1)
                self.assertEqual(Wk.priority('gpu', dict(attacks=['blur'])), 3)
                self.assertEqual(Wk.priority('gpu_light', dict(attacks=['blur'])), 1)
                self.assertEqual(Wk.priority('gpu_light', dict(attacks=['blur', 'unmarker'])), 0)
                # once UnMarker is approved for L40S, A100 workers leave it after their own A100-only work
                approval.write_text(json.dumps(dict(complete=True, attacks={'blur': dict(approved=True),
                                                                            'unmarker': dict(approved=True)})))
                self.assertEqual(Wk.priority('gpu', dict(attacks=['blur', 'unmarker'])), 2)
                self.assertEqual(Wk.priority('gpu', dict(attacks=['blur', 'crop_jpeg'])), 1)

    def test_light_workers_run_on_l40s_and_take_only_approved_attacks(self):
        import worker as Wk
        cmd = D.submit_command('gpu_light', 3)
        self.assertIn('gpu', cmd)
        self.assertIn('--gres=gpu:l40s:1', cmd)
        self.assertEqual(cmd[-1], 'gpu_light')
        a100 = D.submit_command('gpu', 2)
        self.assertIn('--constraint=a100-80g', a100)
        self.assertIn('gpu', a100)
        self.assertIn('cpu', D.submit_command('cpu', 4))
        with tempfile.TemporaryDirectory() as tmp:
            approval = Path(tmp) / 'light.json'
            with patch.object(common, 'LIGHT_APPROVAL', approval), \
                    patch.object(common, 'LIGHT_CHECK', Path(tmp) / 'absent.json'), \
                    patch.object(Wk, 'load_cell', return_value=None):
                self.assertFalse(Wk.eligible('gpu_light', dict(candidate=CAND, attacks=['blur'])))      # no cross-GPU check yet
                approval.write_text(json.dumps(dict(complete=True, attacks={
                    'blur': dict(approved=True), 'regen': dict(approved=True), 'crop_jpeg': dict(approved=False),
                    'unmarker': dict(approved=True)})))
                self.assertTrue(Wk.eligible('gpu_light', dict(candidate=CAND, attacks=['blur', 'unmarker'])))
                self.assertFalse(Wk.eligible('gpu_light', dict(candidate=CAND, attacks=['editing_ip2p_s20_v1'])))
                self.assertFalse(Wk.eligible('gpu_light', dict(candidate=CAND, attacks=['blur', 'crop_jpeg'])))
                self.assertFalse(Wk.eligible('gpu_light', dict(candidate=CAND, attacks=['unmarker', 'ctrlregen_s03'])))
                self.assertTrue(Wk.eligible('gpu_light', dict(candidate=CAND, attacks=['blur', 'regen'])))
                self.assertTrue(Wk.eligible('gpu', dict(attacks=['blur', 'unmarker', 'crop_jpeg'])))
        self.assertEqual(D.DEFAULT_CAPACITY['gpu_light'], 0)

    def test_an_attack_measured_already_does_not_bind_a_light_worker(self):
        import worker as Wk
        with tempfile.TemporaryDirectory() as tmp:
            approval = Path(tmp) / 'light.json'
            approval.write_text(json.dumps(dict(complete=True, attacks={
                'blur': dict(approved=True), 'unmarker': dict(approved=True), 'crop_jpeg': dict(approved=False)})))
            task = dict(attacks=['blur', 'crop_jpeg', 'unmarker'])
            with patch.object(common, 'LIGHT_APPROVAL', approval), \
                    patch.object(common, 'LIGHT_CHECK', Path(tmp) / 'absent.json'):
                task = dict(task, candidate=dict(order=['VINE'], fe_on=[], s={'VINE': .5}))
                cached = lambda c, a, frozen=True: dict(cell={}) if a == 'crop_jpeg' else None
                with patch.object(Wk, 'load_cell', side_effect=cached):                # crop_jpeg measured on A100
                    self.assertTrue(Wk.eligible('gpu_light', task))
                with patch.object(Wk, 'load_cell', return_value=None):                 # crop_jpeg still missing
                    self.assertFalse(Wk.eligible('gpu_light', task))
                seen = []
                with patch.object(Wk, 'load_cell', side_effect=lambda c, a, frozen=True: seen.append(a)):
                    Wk.eligible('gpu_light', task)
                self.assertEqual(seen, ['crop_jpeg'])                                  # only blocked attacks looked up

    def test_a_check_run_allows_only_its_attacks_under_test(self):
        import worker as Wk
        with tempfile.TemporaryDirectory() as tmp:
            check = Path(tmp) / 'light_check.json'
            check.write_text(json.dumps(dict(attacks_under_test=['unmarker', 'ctrlregen_s03'])))
            with patch.object(common, 'LIGHT_APPROVAL', Path(tmp) / 'absent.json'), \
                    patch.object(common, 'LIGHT_CHECK', check), \
                    patch.object(Wk, 'load_cell', return_value=None):
                self.assertTrue(Wk.eligible('gpu_light', dict(candidate=CAND, attacks=['unmarker', 'ctrlregen_s03'])))
                self.assertFalse(Wk.eligible('gpu_light', dict(candidate=CAND, attacks=['unmarker', 'editing_ip2p_s20_v1'])))

    def test_gpu_task_refuses_an_unapproved_missing_attack_on_l40s(self):
        import gpu as G
        with tempfile.TemporaryDirectory() as tmp:
            root, run = Path(tmp) / 'root', Path(tmp) / 'run'
            (root / 'fixed/checkpoints').mkdir(parents=True)
            approval = Path(tmp) / 'light.json'
            approval.write_text(json.dumps(dict(complete=True, attacks={
                'blur': dict(approved=True), 'crop_jpeg': dict(approved=False)})))
            task = dict(candidate=dict(order=['VINE'], fe_on=[], s={'VINE': .5}), attacks=['blur', 'crop_jpeg'],
                        cls='C1')
            with patch.object(G, 'ROOT', root), patch.object(G, 'RUN', run), \
                    patch.object(G, 'device', return_value=('NVIDIA L40S', 'node027')), \
                    patch.object(common, 'LIGHT_APPROVAL', approval), \
                    patch.object(common, 'LIGHT_CHECK', Path(tmp) / 'absent.json'), \
                    patch.object(G, 'subprocess') as sp:
                with patch.object(G, 'load_cell', return_value=None):
                    with self.assertRaises(SystemExit) as cm:
                        G.run(task)
                    self.assertEqual(cm.exception.code, 75)
                sp.run.assert_not_called()

    def test_only_configurations_with_a_reusable_frozen_cell_wait_for_the_bridging_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen = Path(tmp)
            reusable = dict(candidate=dict(order=['VINE'], fe_on=[], s={'VINE': .5}), attacks=['blur', 'regen'])
            fresh = dict(candidate=dict(order=['VINE'], fe_on=[], s={'VINE': .6}), attacks=['blur'])
            key = common.cfgkey(reusable['candidate'])
            (frozen / common.cell_name(key, 'regen')).write_text('{}')
            with patch.object(D, 'FROZEN_CELLS', frozen):
                ready, deferred = D.split_until_bridged([reusable, fresh], decided=False)
                self.assertEqual((ready, deferred), ([fresh], [reusable]))
                ready, deferred = D.split_until_bridged([reusable, fresh], decided=True)
                self.assertEqual((ready, deferred), ([reusable, fresh], []))

    def test_gpu_workers_start_before_the_bridging_decision(self):
        status = dict(bridging_decided=False, pending_cpu=0, pending_gpu_configs=3)
        with patch.object(D, 'active_jobs', return_value=[]), patch.object(D, 'runnable_gpu_count', return_value=3), \
                patch.object(D, 'capacity', return_value=dict(cpu=0, gpu=8, gpu_light=0)), \
                patch.object(D, 'atomic'), patch.object(D.subprocess, 'check_output', return_value='123') as sub, \
                tempfile.TemporaryDirectory() as tmp, patch.object(D, 'RUN', Path(tmp)):
            D.ensure_workers({}, status)
        self.assertEqual(sub.call_count, 1)
        self.assertIn('--constraint=a100-80g', sub.call_args[0][0])


@unittest.skipUnless(AVAILABLE, 'relocated run root not present')
class CellOriginTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.cells, self.frozen, self.approval = base / 'cells', base / 'frozen', base / 'bridging/approval.json'
        self.cells.mkdir()
        self.frozen.mkdir()
        self.patches = [patch.object(common, 'CELLS', self.cells), patch.object(common, 'FROZEN_CELLS', self.frozen),
                        patch.object(common, 'APPROVAL', self.approval),
                        patch.object(common, 'group_path', lambda cls, key: base / 'none' / cls)]
        for p in self.patches:
            p.start()
        common._approval.update(mtime=None, attacks=frozenset())
        self.c = dict(order=['VINE'], fe_on=[], s={'VINE': .5})
        self.key = common.cfgkey(self.c)
        self.thr = str(common.CP.REQUEST_ALPHAS and 80)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        common._approval.update(mtime=None, attacks=frozenset())
        self.temp.cleanup()

    def write(self, folder, attack, psnr):
        rec = dict(cell=cell(['VINE'], 100, [True] * 95 + [False] * 5), psnr_db=psnr, key=list(map(list, self.key)))
        common.atomic(folder / common.cell_name(self.key, attack), rec)

    def load(self, attack, **kw):
        with patch.object(common.CP, 'valid_cell', lambda c, order, n, cfg=None, alphas=None: isinstance(c, dict)):
            return common.load_cell(self.c, attack, **kw)

    def test_frozen_cells_need_an_approval(self):
        self.write(self.frozen, 'blur', 40.0)
        self.assertIsNone(self.load('blur'))
        common.atomic(self.approval, dict(policy=common.POLICY, complete=True,
                                          attacks={'blur': dict(approved=True), 'regen': dict(approved=False)}))
        rec = self.load('blur')
        self.assertEqual(rec['origin'], 'frozen')
        self.assertIsNone(self.load('blur', frozen=False))
        self.write(self.frozen, 'regen', 40.0)
        self.assertIsNone(self.load('regen'))

    def test_run_cells_win_over_frozen_cells(self):
        common.atomic(self.approval, dict(policy=common.POLICY, complete=True, attacks={'blur': dict(approved=True)}))
        self.write(self.frozen, 'blur', 40.0)
        self.write(self.cells, 'blur', 41.0)
        rec = self.load('blur')
        self.assertEqual((rec['origin'], rec['psnr_db']), ('run', 41.0))

    def test_incomplete_approval_is_refused(self):
        common.atomic(self.approval, dict(policy=common.POLICY, complete=False, attacks={}))
        self.write(self.frozen, 'blur', 40.0)
        with self.assertRaises(ValueError):
            self.load('blur')


@unittest.skipUnless(AVAILABLE, 'relocated run root not present')
class BridgeTest(unittest.TestCase):
    def test_sample_is_stratified_and_deterministic(self):
        pop = {}
        for k in (1, 2, 3):
            for j in range(6):
                pop[f'{k}{j:063d}'] = set(B.MAIN_ATTACKS) if j < 2 else set(B.IN_PROCESS)
        frag = lambda c: int(c[0])
        sample = B.select_sample(pop, frag)
        self.assertEqual(sample, B.select_sample(dict(reversed(list(pop.items()))), frag))
        self.assertEqual([k for _, k, _ in sample], [1, 1, 1, 2, 2, 2, 3, 3, 3])
        full = [c for c, k, a in sample if len(a) == 20]
        self.assertEqual(len(full), 3)
        self.assertTrue(all(set(B.MAIN_ATTACKS) <= pop[c] for c in full))
        for k in (1, 2, 3):
            ids = sorted((c for c in pop if frag(c) == k), key=B.order_key)
            first_full = next(c for c in ids if len(pop[c]) == 20)
            self.assertIn((first_full, k, list(B.MAIN_ATTACKS)), sample)

    def test_compare_cells_reports_rate_mean_and_agreement(self):
        old = cell(['VINE'], 100, [True] * 95 + [False] * 5)
        new = cell(['VINE'], 100, [True] * 93 + [False] * 7)
        rows = B.compare_cells(old, new, ['80'])
        self.assertAlmostEqual(rows[0]['d_rate'], -0.02)
        self.assertAlmostEqual(rows[0]['agreement'], 0.98)
        self.assertIsNone(B.compare_cells(old, new, ['81']))

    def test_decision_rules(self):
        row = lambda dr, dm: dict(d_rate=dr, d_mean=dm, agreement=1.0)
        self.assertTrue(B.judge_attack('blur', [row(0.03, 0.01), row(-0.01, 0)])['approved'])
        self.assertFalse(B.judge_attack('blur', [row(0.04, 0.0)])['approved'])
        self.assertFalse(B.judge_attack('blur', [row(0.0, 0.011)])['approved'])
        self.assertTrue(B.judge_attack('regen', [row(0.08, 0.02), row(-0.06, -0.01)])['approved'])
        self.assertFalse(B.judge_attack('regen', [row(0.05, 0.0), row(0.05, 0.0)])['approved'])
        self.assertFalse(B.judge_attack('regen', [row(0.11, 0.0), row(-0.11, 0.0)])['approved'])
        self.assertFalse(B.judge_attack('regen', [])['approved'])
        self.assertEqual(B.STOCHASTIC | B.DETERMINISTIC, frozenset(B.MAIN_ATTACKS))
        self.assertEqual(len(B.MAIN_ATTACKS), 20)


@unittest.skipUnless(AVAILABLE, 'relocated run root not present')
class GpuPreflightTest(unittest.TestCase):
    """The frozen driver judges the plan row before measuring; every candidate shape must survive that."""

    def setUp(self):
        import gpu
        self.gpu = gpu
        self.c = dict(order=['TrustMark'], fe_on=['tile'], fe=['tile'], s={'TrustMark': 1.0})

    def test_candidate_without_latency_or_table_psnr_fails_on_cpu(self):
        for missing in ('ms', 'psnr_db'):
            c = dict(self.c, ms=0.0, psnr_db=40.0)
            c.pop(missing)
            with self.assertRaises(KeyError):
                self.gpu.preflight(dict(candidate=c, attacks=['blur', 'jpeg25']))

    def test_complete_candidate_passes_and_leaves_every_attack_pending(self):
        row = self.gpu.preflight(dict(candidate=dict(self.c, ms=0.0, psnr_db=40.0), attacks=['jpeg25', 'blur']))
        self.assertTrue(row['open'])

    def test_bridging_tasks_pass_the_preflight(self):
        selection = Path(os.environ['TAILOR_MAIN_CONTROLS_RUN']) / 'bridging/queue.json'
        if not selection.exists():
            self.skipTest('no bridging queue')
        for name in json.loads(selection.read_text()):
            task = json.loads(Path(name).read_text())
            if 'psnr_db' in task['candidate']:
                self.gpu.preflight(task)

    def test_image_set_override_only_with_a_file_and_only_for_the_driver_call(self):
        if str(common.C.CF) not in sys.path:
            sys.path.append(str(common.C.CF))      # campaign.boot puts it on the path in real runs
        import src.image_pool as pool
        original = pool.sample
        try:
            with tempfile.TemporaryDirectory() as tmp:
                self.assertIsNone(self.gpu.install_image_set(Path(tmp) / 'none.json'))
                self.assertIs(pool.sample, original)
                doc = dict(images_sha256='abc', images=[dict(path=f'/x/{i}.png') for i in range(3)])
                p = Path(tmp) / 'image_set.json'
                p.write_text(json.dumps(doc))
                self.assertEqual(self.gpu.install_image_set(p), 'abc')
                self.assertEqual(pool.sample(3, offset=300), ['/x/0.png', '/x/1.png', '/x/2.png'])
                with self.assertRaises(ValueError):
                    pool.sample(100, offset=300)
                with self.assertRaises(ValueError):
                    pool.sample(3, offset=0)
        finally:
            pool.sample = original

    def test_proposal_candidates_pass_the_preflight(self):
        root = Path(os.environ['TAILOR_MAIN_CONTROLS_RUN']) / 'proposals'
        seen = 0
        for variant in ('SingleOptimized', 'AlwaysAllThree'):
            for p in sorted((root / variant / '06').glob('*/initial.json'))[:20]:
                for c in json.loads(p.read_text())['candidates']:
                    self.gpu.preflight(dict(candidate=c, attacks=['blur']))
                    seen += 1
        if not seen:
            self.skipTest('no proposals yet')


if __name__ == '__main__':
    unittest.main()
