#!/usr/bin/env python3
"""CPU synthetic checks of per-request ID export; never opens campaign data."""
import collections
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    'unified_id_export_integration', HERE / 'export_unified_evaluation.py')
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class IdIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = EXPORT.support_module()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'operations').mkdir()
        put(self.root / 'fallback/pipeline_audit.json',
            dict(passed=True, arm_request_evaluations=20000))
        put(self.root / 'single_baselines/policy_freeze.json', {})
        put(self.root / 'operations/holdout_reuse_decision.json',
            dict(newly_unseen_holdout_claim=False,
                 holdout_reused_across_protocol_revisions=True))
        self.gate = dict(ready=True, pending=[], audits={'synthetic_complete': {}},
                         evidence={})

    def lane(self, module, root, is_main, inputs):
        """Complete key population, with deliberately different per-request data."""
        method = root.name
        fragment = module.SINGLE_FRAGMENTS.get(method, 'VINE')
        cfg = dict(S=[fragment], order=[fragment], s={fragment: 1.0}, fe_on=[])
        frozen, final = {}, {}
        baseline_offset = (0 if is_main else
                           next(i for i, (_, name) in enumerate(EXPORT.METHODS)
                                if name == method))
        for cls_index, cls in enumerate(module.CLASSES):
            for arm in module.ARMS:
                for index in range(2000):
                    key = (cls, arm, index)
                    # The original FPR is identical across methods and arms, but
                    # differs across classes to expose accidental mean/global use.
                    request = dict(attacks=['jpeg'], fpr=2.0 ** (-10 - cls_index),
                                   min_psnr_db=30.0, max_ms=1000.0)
                    selected = (dict(cfg=cfg, cfg_sha256='synthetic-' + fragment)
                                if index < 3 else None)
                    frozen[key] = dict(inputs=request, selected=selected,
                                       terminal_status=('live_pass' if selected else 'no_candidate'))
                    passed = index < 2
                    capacity = ((42.0 if arm == '04' else 32.0) + 2 * cls_index
                                if is_main else 38.0 + baseline_offset + 2 * cls_index)
                    if index == 1:
                        capacity = 5.0  # Passing evidence, but no positive ID length.
                    final[key] = dict(final_pass=passed if selected else None,
                                      capacity_bits_estimate=capacity,
                                      psnr_db=40.0, latency_ms=100.0)
        return ({'acceptance_protocol': EXPORT.PROTOCOL}, frozen,
                {'synthetic_identical_splits': True}, {'tasks': []}, final,
                {'status': 'complete'})

    def assemble(self):
        # Real summary, pairing, advice and table helpers run over all 70k rows.
        # Frozen/report validation has separate tests; this fixture supplies
        # synthetic, already-audited lane rows without touching GPU/cache files.
        with mock.patch.object(EXPORT, 'support_module', return_value=self.helper), \
             mock.patch.object(EXPORT, 'load_lane', side_effect=self.lane), \
             mock.patch.object(self.helper, 'pending_single'):
            return EXPORT.assemble(self.root, self.gate)

    def test_complete_population_keeps_per_request_values_and_statuses(self):
        files, provenance, records = self.assemble()
        self.assertEqual(set(files), {'tab/main.tex', 'tab/classes.tex', 'tab/main_quality.tex'})
        # The same audited metrics still generate the same three tables; ID
        # advice contributes only its separate report and provenance fields.
        expected_tables = {
            'tab/main.tex': self.helper.table_main(provenance['methods']),
            'tab/classes.tex': self.helper.table_classes(provenance['methods']),
            'tab/main_quality.tex': self.helper.table_quality(
                provenance['methods'], provenance['paired_comparisons']),
        }
        self.assertEqual(files, {name: EXPORT.revised_caption(text, EXPORT.CAPTIONS[name])
                                 for name, text in expected_tables.items()})
        self.assertEqual(len(records), 70000)
        counts = collections.Counter(row['method'] for row in records)
        self.assertEqual(counts, {name: 10000 for name in self.helper.METHODS})
        self.assertEqual(len({(r['method'], r['class'], r['request_id']) for r in records}), 70000)
        by_key = {(r['method'], r['class'], r['request_id']): r for r in records}
        expected = [
            ('main04', 'C1', 42.0, 2.0 ** -10, 32),
            ('main06', 'C1', 32.0, 2.0 ** -10, 22),
            ('main04', 'C2', 44.0, 2.0 ** -11, 33),
            ('H1_rule', 'C1', 38.0, 2.0 ** -10, 28),
            ('S_VideoSeal', 'C1', 42.0, 2.0 ** -10, 32),
        ]
        for method, cls, capacity, fpr, maximum in expected:
            with self.subTest(method=method, cls=cls):
                row = by_key[method, cls, 0]
                guidance = row['guidance']
                self.assertEqual(guidance['status'], 'available')
                self.assertEqual(guidance['capacity_bits_estimate'], capacity)
                self.assertEqual(row['request_fpr'], fpr)
                self.assertEqual(guidance['fpr'], fpr)
                self.assertEqual(guidance['id_bits_min'], 1)
                self.assertEqual(guidance['id_bits_max'], maximum)
                self.assertEqual(guidance['id_space_size_max'], str(2 ** maximum))
        for method in self.helper.METHODS:
            expected_arm = '06' if method == 'main06' else '04'
            self.assertEqual({r['arm'] for r in records if r['method'] == method}, {expected_arm})
            self.assertEqual(provenance['methods'][method]['denominator'], 10000)
            for index, status in ((1, 'empty'), (2, 'final_failed'), (3, 'not_deployed')):
                guidance = by_key[method, 'C1', index]['guidance']
                self.assertEqual(guidance['status'], status)
                for field in ('id_bits_min', 'id_bits_max',
                              'id_space_size_min', 'id_space_size_max'):
                    self.assertIsNone(guidance[field])
                if status != 'empty':
                    self.assertIsNone(guidance['capacity_bits_estimate'])
            statuses = collections.Counter(r['guidance']['status'] for r in records
                                           if r['method'] == method)
            self.assertEqual(statuses, {'available': 5, 'empty': 5,
                                        'final_failed': 5, 'not_deployed': 9985})
        for row in records:
            self.assertIsNone(row['guidance']['selected_id_bits'])
            self.assertEqual(row['guidance']['selection_by'], 'user')
            self.assertTrue(row['guidance']['model_guidance_only'])
            self.assertFalse(row['guidance']['range_deployment_validated'])
        self.assertFalse((self.root / 'operations/paper_staging').exists())

    def test_incomplete_audit_cannot_consume_or_publish_guidance(self):
        pending = dict(ready=False, pending=['synthetic_main_pipeline_audit.json'])
        with mock.patch.object(EXPORT, 'completion_gate', return_value=pending), \
             mock.patch.object(EXPORT, 'assemble') as assemble, \
             mock.patch.object(self.helper, 'write_id_guidance') as write:
            result = EXPORT.export_once(self.root)
        self.assertFalse(result['complete'])
        assemble.assert_not_called()
        write.assert_not_called()
        self.assertFalse((self.root / 'operations/paper_staging').exists())

    def test_tables_and_counts_match_exporter_before_id_wiring(self):
        previous_path = (HERE / 'revisions/id_guidance_20260915_before/'
                         'export_unified_evaluation.py')
        spec = importlib.util.spec_from_file_location('unified_export_before_id', previous_path)
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        lane_rows = []

        def observed_lane(*args):
            lane = self.lane(*args)
            # Probe both selected outcomes and non-deployments in both arms;
            # frozen choices and final verdicts must remain pure input records.
            for arm in self.helper.ARMS:
                for index in range(4):
                    key = ('C1', arm, index)
                    for rows in (lane[1], lane[4]):
                        lane_rows.append((rows[key], copy.deepcopy(rows[key])))
            return lane

        with mock.patch.object(previous, 'SUPPORT', EXPORT.SUPPORT), \
             mock.patch.object(previous, 'support_module', return_value=self.helper), \
             mock.patch.object(previous, 'load_lane', side_effect=observed_lane), \
             mock.patch.object(EXPORT, 'support_module', return_value=self.helper), \
             mock.patch.object(EXPORT, 'load_lane', side_effect=observed_lane), \
             mock.patch.object(self.helper, 'pending_single'):
            old_tables, old_provenance = previous.assemble(self.root, self.gate)
            new_tables, new_provenance, records = EXPORT.assemble(self.root, self.gate)
        self.assertEqual(new_tables, old_tables)
        self.assertEqual(new_provenance['methods'], old_provenance['methods'])
        self.assertEqual(new_provenance['paired_comparisons'], old_provenance['paired_comparisons'])
        self.assertEqual(new_provenance['dataset_equality'], old_provenance['dataset_equality'])
        self.assertEqual(len(records), 70000)
        for original, snapshot in lane_rows:
            self.assertEqual(original, snapshot)
        self.assertFalse((self.root / 'operations/paper_staging').exists())

    def test_staged_guidance_is_hash_bound_and_corrupt_reentry_is_rejected(self):
        assembled = self.assemble()
        before = {str(path): EXPORT.digest(path) for path in self.root.rglob('*.json')}

        def fresh_assembly(root, gate):
            # The exporter adds output hashes only to its private provenance copy.
            return assembled[0], copy.deepcopy(assembled[1]), assembled[2]

        with mock.patch.object(EXPORT, 'completion_gate', return_value=self.gate), \
             mock.patch.object(EXPORT, 'assemble', side_effect=fresh_assembly), \
             mock.patch.object(EXPORT, 'support_module', return_value=self.helper):
            first = EXPORT.export_once(self.root)
            second = EXPORT.export_once(self.root)
            self.assertEqual(first, second)
            snapshot = Path(first['snapshot'])
            guidance_path = snapshot / 'data/current_id_space_guidance.jsonl'
            provenance = EXPORT.read(snapshot / 'data/unified_evaluation_provenance.json')
            self.assertEqual(provenance['outputs']['data/current_id_space_guidance.jsonl'],
                             EXPORT.digest(guidance_path))
            manifest = provenance['id_space_guidance']
            self.assertEqual(manifest['records'], 70000)
            self.assertEqual(manifest['path'], 'data/current_id_space_guidance.jsonl')
            self.assertEqual(manifest['sha256'], EXPORT.digest(guidance_path))
            self.assertFalse(manifest['affects_solver_or_final_verdict'])
            self.assertEqual(first['id_space_guidance']['path'], str(guidance_path))
            self.assertEqual(first['id_space_guidance']['sha256'], manifest['sha256'])
            with guidance_path.open() as handle:
                self.assertEqual(sum(1 for _ in handle), 70000)
            self.assertTrue(first['paper_update_pending'])
            self.assertFalse(first['paper_sources_modified'])
            # Reentry must verify the JSONL itself, rather than trusting tables
            # or the previous latest.json success marker.
            guidance_path.write_text('{"corrupted":true}\n')
            with self.assertRaisesRegex(ValueError, 'frozen evidence changed'):
                EXPORT.export_once(self.root)
        self.assertEqual(before, {path: EXPORT.digest(path) for path in before})
        self.assertFalse(any((self.root / 'operations/paper_staging').glob('.building-*')))


if __name__ == '__main__':
    unittest.main()
