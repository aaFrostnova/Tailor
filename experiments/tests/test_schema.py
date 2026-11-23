"""Result schema: canonical status vocabulary and per-record validation."""
import copy
import unittest

from tailor_exp import protocol, schema

# Every raw status string found in the nine frozen per-request result files (census 2026-11-09).
FROZEN_RAW_STATUSES = ('accepted', 'fallback_live_exhausted', 'zero_margin_model_unsat', 'fixed_live_exhausted',
                       'live_exhausted', 'greedy_stalled', 'live_failed', 'model_unsat')


def accepted_record():
    order, geometry, alpha = ['videoseal', 'vine'], ['resync'], 1e-4
    t_c = protocol.max_tests(order, geometry)
    return dict(
        schema_version=schema.SCHEMA_VERSION, run_id='pr1_single_06', protocol_version=protocol.PROTOCOL,
        cohort_hash='a' * 64, spec_id='b' * 64, method='TAILOR-F', image_role='selection',
        key_seed='v5_key_encoder_master', decoder_hash='c' * 64, attack_param_id='d' * 64,
        margin_policy='m06_fallback_v2', latency_basis='modeled', status='accepted', raw_status='accepted',
        alpha=alpha, selected_configuration='e' * 64, strengths={'videoseal': 0.9, 'vine': 0.31},
        order=order, geometry=geometry, T_c=t_c, threshold=protocol.threshold_for_tests(t_c, alpha),
        predicted_distortion=41.2, actual_psnr=41.31, actual_ssim=0.9921, embed_ms=80.0, decode_ms=40.0,
        wall_clock_search_ms=12000.0, live_candidate_count=1, cache_provenance='live_topk/cells',
        readout_ref=dict(path='cells/e_jpeg25_pool300_n100.json', sha256='f' * 64))


class StatusVocabularyTest(unittest.TestCase):
    def test_early_stopping_statuses_are_distinguishable(self):
        terminal_failures = ['model_unsat', 'search_stalled', 'live_failed', 'live_exhausted', 'timeout',
                             'budget_exhausted']
        self.assertEqual(len(set(terminal_failures)), 6)
        for status in terminal_failures:
            self.assertTrue(schema.is_terminal(status), status)
            self.assertFalse(schema.is_satisfied(status), status)
        self.assertTrue(schema.is_satisfied('accepted'))

    def test_pending_states_are_not_terminal(self):
        for status in ('pending', 'missing_input', 'incompatible_checkpoint', 'audit_incomplete'):
            self.assertFalse(schema.is_terminal(status), status)
            self.assertFalse(schema.is_satisfied(status), status)

    def test_every_frozen_raw_status_has_a_canonical_status(self):
        for raw in FROZEN_RAW_STATUSES:
            self.assertIn(schema.canonical_status(raw), schema.STATUSES, raw)

    def test_frozen_raw_statuses_keep_unsat_stall_and_live_failure_apart(self):
        self.assertEqual(schema.canonical_status('zero_margin_model_unsat'), 'model_unsat')
        self.assertEqual(schema.canonical_status('greedy_stalled'), 'search_stalled')
        self.assertEqual(schema.canonical_status('live_failed'), 'live_failed')
        self.assertEqual(schema.canonical_status('fallback_live_exhausted'), 'live_exhausted')
        self.assertEqual(schema.canonical_status('fixed_live_exhausted'), 'live_exhausted')

    def test_unknown_raw_status_is_rejected(self):
        with self.assertRaises(schema.SchemaError):
            schema.canonical_status('passed')


class RecordValidationTest(unittest.TestCase):
    def test_valid_accepted_record_passes(self):
        schema.validate_record(accepted_record())

    def assertInvalid(self, **changes):
        record = accepted_record()
        for key, value in changes.items():
            if value is KeyError:
                del record[key]
            else:
                record[key] = value
        with self.assertRaises(schema.SchemaError):
            schema.validate_record(record)

    def test_missing_field_is_rejected(self):
        self.assertInvalid(cohort_hash=KeyError)

    def test_wrong_threshold_is_rejected(self):
        self.assertInvalid(threshold=accepted_record()['threshold'] - 1)

    def test_wrong_test_count_is_rejected(self):
        self.assertInvalid(T_c=3)

    def test_two_geometric_stages_are_rejected(self):
        record = accepted_record()
        record['geometry'] = ['resync', 'angle']
        record['T_c'] = protocol.max_tests(record['order'], record['geometry'])
        record['threshold'] = protocol.threshold_for_tests(record['T_c'], record['alpha'])
        with self.assertRaises(schema.SchemaError):
            schema.validate_record(record)

    def test_strengths_must_cover_exactly_the_selected_fragments(self):
        self.assertInvalid(strengths={'vine': 0.31})

    def test_accepted_record_needs_per_image_readouts(self):
        self.assertInvalid(readout_ref=None)

    def test_raw_status_must_map_to_status(self):
        self.assertInvalid(raw_status='greedy_stalled')

    def test_unknown_image_role_is_rejected(self):
        self.assertInvalid(image_role='final')

    def test_wrong_protocol_version_is_rejected(self):
        self.assertInvalid(protocol_version='legacy_identity_or')

    def test_pending_record_needs_no_configuration(self):
        record = accepted_record()
        record.update(status='pending', raw_status='pending', selected_configuration=None, strengths=None,
                      order=None, geometry=None, T_c=None, threshold=None, actual_psnr=None, actual_ssim=None,
                      readout_ref=None, live_candidate_count=0)
        schema.validate_record(record)

    def test_live_failure_needs_an_attempted_candidate(self):
        record = accepted_record()
        record.update(status='live_exhausted', raw_status='live_exhausted', live_candidate_count=0)
        with self.assertRaises(schema.SchemaError):
            schema.validate_record(record)

    def test_live_failure_threshold_is_recomputed(self):
        record = accepted_record()
        record.update(status='live_failed', raw_status='live_failed', threshold=record['threshold'] + 3)
        with self.assertRaises(schema.SchemaError):
            schema.validate_record(record)

    def test_configuration_is_forbidden_without_a_live_measurement(self):
        for status in ('model_unsat', 'search_stalled', 'timeout', 'pending'):
            with self.subTest(status=status):
                record = accepted_record()
                record.update(status=status, raw_status=status, live_candidate_count=0, readout_ref=None)
                with self.assertRaises(schema.SchemaError):
                    schema.validate_record(record)

    def test_live_candidates_need_readouts_whatever_the_status(self):
        record = accepted_record()
        record.update(status='timeout', raw_status='timeout', live_candidate_count=7, readout_ref=None,
                      selected_configuration=None, strengths=None, order=None, geometry=None, T_c=None,
                      threshold=None, actual_psnr=None, actual_ssim=None)
        with self.assertRaises(schema.SchemaError):
            schema.validate_record(record)

    def test_counts_and_thresholds_must_be_exact_integers(self):
        good = accepted_record()
        for key, value in (('T_c', True), ('T_c', float(good['T_c'])), ('threshold', float(good['threshold'])),
                           ('live_candidate_count', 1.0)):
            with self.subTest(key=key, value=value):
                self.assertInvalid(**{key: value})

    def test_numeric_fields_must_be_finite_numbers(self):
        for key in ('embed_ms', 'decode_ms', 'predicted_distortion', 'actual_ssim', 'wall_clock_search_ms'):
            with self.subTest(key=key):
                self.assertInvalid(**{key: 'fast'})
        self.assertInvalid(actual_psnr=float('nan'))

    def test_order_and_geometry_must_be_lists(self):
        self.assertInvalid(order={'videoseal': 1, 'vine': 2})
        self.assertInvalid(geometry='resync')

    def test_live_exhausted_without_a_final_configuration_is_valid(self):
        record = accepted_record()
        record.update(status='live_exhausted', raw_status='fallback_live_exhausted', live_candidate_count=3,
                      selected_configuration=None, strengths=None, order=None, geometry=None, T_c=None,
                      threshold=None, actual_psnr=None, actual_ssim=None)
        schema.validate_record(record)

    def test_validation_does_not_modify_the_record(self):
        record = accepted_record()
        before = copy.deepcopy(record)
        schema.validate_record(record)
        self.assertEqual(record, before)


if __name__ == '__main__':
    unittest.main()
