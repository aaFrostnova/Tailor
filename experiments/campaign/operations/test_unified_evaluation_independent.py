#!/usr/bin/env python3
"""Independent synthetic protocol-boundary checks; no campaign results are read."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('unified_export_independent', HERE / 'export_unified_evaluation.py')
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class ProtocolBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vendor = EXPORT.support_module()

    def lane(self, directory, *, main=False, deployed=True, row_tag=None,
             top_tag=None, frozen_tag=EXPORT.PROTOCOL):
        root = Path(directory)
        request = dict(attacks=['jpeg'], fpr=0.01, min_psnr_db=30, max_ms=1000)
        selected = {'cfg_sha256': 'synthetic-cfg'} if deployed else None
        frozen = dict(acceptance_protocol=frozen_tag, selection_sha256='synthetic-selection')
        rows = {('C1', '04', 0): dict(inputs=request, selected=selected)}
        row = dict(key='C1/04/0', inputs=request, final_feedback_used=False,
                   final_status='measured' if deployed else 'not_deployed',
                   final_pass=True if deployed else None)
        if deployed:
            row.update(cfg_sha256='synthetic-cfg', embedding_psnr_db={'mean': 40.0},
                       capacity_bits_estimate=20.0, latency_max_attack_mean_ms=100.0,
                       psnr_pass=True, latency_pass=True, empirical_detection_pass=True)
        if row_tag is not None:
            row['acceptance_protocol'] = row_tag
        report = dict(passed=True, complete=True, no_reselection=True,
                      selection_sha256='synthetic-selection', plan_sha256='synthetic-plan',
                      original_requests=10000, arm_request_evaluations=20000,
                      final_pass_requests=int(deployed), deployed_requests=int(deployed),
                      two_arms_are_paired_duplicates_for_rule=True, rows=[row])
        if top_tag is not None:
            report['acceptance_protocol'] = top_tag
        path = root / ('final_results.json' if main else 'final/report.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
        # Bypass population construction and audited hash fixtures only. The real
        # vendored streaming parser consumes the report and invokes the callback.
        with patch.object(self.vendor, 'load_frozen', return_value=(frozen, rows, {}, {'plan_sha256': 'synthetic-plan'})), \
             patch.object(self.vendor, 'audit_ready', return_value=(True, {})):
            return EXPORT.load_lane(self.vendor, root, main, self.vendor.Inputs())

    def test_measured_protocol_schema_is_enforced_at_the_correct_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Baseline reports carry the tag on each measured row, not at top level.
            result = self.lane(tmp, row_tag=EXPORT.PROTOCOL)
            self.assertTrue(result[4][('C1', '04', 0)]['final_pass'])
            for tag in (None, 'old_keyed_detector'):
                with self.subTest(baseline_row_tag=tag), self.assertRaisesRegex(ValueError, 'legacy baseline final'):
                    self.lane(tmp, row_tag=tag, top_tag=EXPORT.PROTOCOL)
            # Main reports instead pin the protocol at report top level.
            self.lane(tmp, main=True, top_tag=EXPORT.PROTOCOL)
            for tag in (None, 'old_keyed_detector'):
                with self.subTest(main_report_tag=tag), self.assertRaisesRegex(ValueError, 'legacy main final'):
                    self.lane(tmp, main=True, top_tag=tag, row_tag=EXPORT.PROTOCOL)

    def test_nondeployment_needs_no_row_tag_but_still_requires_new_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.lane(tmp, deployed=False)
            self.assertIsNone(result[4][('C1', '04', 0)]['final_pass'])
            for tag in (None, 'old_keyed_detector'):
                with self.subTest(frozen_tag=tag), self.assertRaisesRegex(ValueError, 'legacy frozen selection'):
                    self.lane(tmp, deployed=False, frozen_tag=tag)


if __name__ == '__main__':
    unittest.main()
