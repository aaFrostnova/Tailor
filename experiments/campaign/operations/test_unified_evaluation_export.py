"""CPU-only fail-closed staging checks; fixtures never touch campaign files."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('unified_export', HERE/'export_unified_evaluation.py')
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'operations').mkdir()

    def fixture(self):
        source = self.root/'fixture_source.txt'
        source.write_text('unchanged fixture')
        put(self.root/'source_hashes.json', {str(source): E.digest(source)})
        for location in [self.root] + [self.root/f/m for f, m in E.METHODS]:
            main = location == self.root
            report = location/('final_results.json' if main else 'final/report.json')
            put(report, {'fixture': True})
            audit = dict(passed=True, evidence={str(report): E.digest(report)})
            if main:
                audit.update(acceptance_protocol=E.PROTOCOL, original_requests=10000,
                    arm_request_evaluations=20000, independent_final_test=True,
                    actual_latency_measurements_complete=True, final_feedback_to_selection=False,
                    heldout_reused_across_protocol_revisions=True, newly_unseen_holdout_claim=False)
            else:
                audit.update(complete=True, terminal_requests=20000, no_final_feedback=True)
            put(location/'pipeline_audit.json', audit)
        for folder in ('baselines', 'single_baselines'):
            put(self.root/folder/'protocol_sources.json', {'hashes': {str(source): E.digest(source)}})
            methods = {}
            for f, method in E.METHODS:
                if f != folder:
                    continue
                p = self.root/f/method/'plans/proposals.json'
                put(p, {'fixture': True})
                methods[method] = dict(path=str(p), sha256=E.digest(p), rows=20000)
            put(self.root/folder/'policy_freeze.json', dict(source_hashes={str(source): E.digest(source)}, methods=methods))

    def test_pending_does_not_consume_report_or_create_staging(self):
        with mock.patch.object(E, 'assemble', side_effect=AssertionError('must not read reports')):
            status = E.export_once(self.root)
        self.assertFalse(status['complete'])
        self.assertEqual(len(status['pending']), 6)
        self.assertFalse((self.root/'operations/paper_staging').exists())

    def test_all_six_audits_are_required(self):
        self.fixture()
        self.assertTrue(E.completion_gate(self.root)['ready'])
        (self.root/'single_baselines/S_VINE/pipeline_audit.json').unlink()
        self.assertFalse(E.completion_gate(self.root)['ready'])

    def test_old_main_protocol_is_rejected(self):
        self.fixture()
        path = self.root/'pipeline_audit.json'
        a = E.read(path); a['acceptance_protocol'] = 'legacy'; put(path, a)
        with self.assertRaisesRegex(ValueError, 'another acceptance protocol'):
            E.completion_gate(self.root)

    def test_changed_report_or_source_is_rejected(self):
        self.fixture()
        (self.root/'baselines/H2_rule/final/report.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'frozen evidence changed'):
            E.completion_gate(self.root)
        self.fixture()
        (self.root/'fixture_source.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'frozen evidence changed'):
            E.completion_gate(self.root)

    def test_failed_lane_is_not_silently_pending(self):
        self.fixture()
        p = self.root/'single_baselines/S_VINE/pipeline_audit.json'
        a = E.read(p); a['passed'] = False; put(p, a)
        (self.root/'pipeline_audit.json').unlink()
        with self.assertRaisesRegex(ValueError, 'lane audit failed'):
            E.completion_gate(self.root)

    def test_publication_is_atomic_staged_and_idempotent(self):
        self.fixture()
        files = {'tab/main.tex': 'fixture table\n'}
        module = E.support_module()
        guidance = [dict(method='main04', arm='04', **{'class': 'C1'}, request_id=0,
            guidance=module.recommend_id_space(60, 2**-20))]
        def assembled(root, gate):
            return files, dict(script_sha256=E.digest(E.__file__), support={},
                completion_gate=gate, inputs={}, methods={'main04': dict(denominator=1, final_pass=1)}), guidance
        with mock.patch.object(E, 'assemble', side_effect=assembled):
            first = E.export_once(self.root)
            second = E.export_once(self.root)
        self.assertEqual(first, second)
        self.assertTrue(first['paper_update_pending'])
        self.assertFalse(first['paper_sources_modified'])
        self.assertEqual((Path(first['snapshot'])/'tab/main.tex').read_text(), files['tab/main.tex'])
        self.assertEqual(first['id_space_guidance']['records'], 1)
        self.assertTrue(Path(first['id_space_guidance']['path']).is_file())
        self.assertFalse(any((self.root/'operations/paper_staging').glob('.building-*')))

    def test_caption_discloses_reused_holdout_and_latency(self):
        m = E.support_module()
        counts = dict(name='main04', selected=10, selection_rate=.001, final_pass=9,
            final_rate=.0009, final_audit='complete')
        text = E.revised_caption(m.table_main({'main04': counts}), E.CAPTIONS['tab/main.tex'])
        self.assertIn('reused across protocol revisions', text)
        self.assertIn('cross-session component sum', E.CAPTIONS['tab/main_quality.tex'])
        self.assertNotIn('Current four-input request results', text)


if __name__ == '__main__':
    unittest.main()
