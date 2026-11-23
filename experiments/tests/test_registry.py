"""Run registry: combined tables may only draw on runs that share the frozen evaluation identity."""
import unittest

from tailor_exp import registry
from tests.test_schema import accepted_record


def tag(**changes):
    base = dict(protocol_version='unified_raw_ba_v1', cohort_hash='a' * 64, decoder_hash='c' * 64,
                image_role='selection', attack_param_id='d' * 64, margin_policy='m06_fallback_v2',
                latency_basis='modeled', key_seed='v5_key_encoder_master')
    base.update(changes)
    return base


def record_of(run_id, **changes):
    record = accepted_record()
    record.update(run_id=run_id, **changes)
    return record


class CohortProtocolIsolationTest(unittest.TestCase):
    def setUp(self):
        self.reg = registry.Registry()
        self.reg.register('tailor_f', tag())

    def test_identical_identities_combine(self):
        self.reg.register('single', tag())
        combined = self.reg.combine(['tailor_f', 'single'])
        self.assertEqual(combined['cohort_hash'], 'a' * 64)

    def test_any_identity_difference_is_refused(self):
        changes = dict(protocol_version='legacy_identity_or', cohort_hash='e' * 64, decoder_hash='f' * 64,
                       image_role='heldout', attack_param_id='0' * 64, latency_basis='measured',
                       key_seed='other_key', margin_policy='m04_fallback_v2')
        for field, value in changes.items():
            with self.subTest(field=field):
                reg = registry.Registry()
                reg.register('tailor_f', tag())
                reg.register('other', tag(**{field: value}))
                with self.assertRaises(registry.RegistryError):
                    reg.combine(['tailor_f', 'other'])

    def test_margin_policy_may_differ_only_when_declared(self):
        self.reg.register('tailor_f_04', tag(margin_policy='m04_fallback_v2'))
        combined = self.reg.combine(['tailor_f', 'tailor_f_04'], allow_differ=('margin_policy',))
        self.assertEqual(combined['margin_policy'], ('m04_fallback_v2', 'm06_fallback_v2'))

    def test_identity_fields_can_never_be_declared_to_differ(self):
        self.reg.register('other', tag(cohort_hash='e' * 64))
        for field in ('protocol_version', 'cohort_hash', 'decoder_hash', 'image_role', 'attack_param_id',
                      'key_seed', 'latency_basis'):
            with self.subTest(field=field):
                with self.assertRaises(registry.RegistryError):
                    self.reg.combine(['tailor_f', 'other'], allow_differ=(field,))

    def test_empty_combination_is_refused(self):
        with self.assertRaises(registry.RegistryError):
            self.reg.combine([])

    def test_reregistering_a_run_with_another_identity_is_refused(self):
        with self.assertRaises(registry.RegistryError):
            self.reg.register('tailor_f', tag(cohort_hash='e' * 64))

    def test_incomplete_tag_is_refused(self):
        bad = tag()
        del bad['key_seed']
        with self.assertRaises(registry.RegistryError):
            self.reg.register('incomplete', bad)

    def test_unknown_run_is_refused(self):
        with self.assertRaises(registry.RegistryError):
            self.reg.combine(['tailor_f', 'never_registered'])


class RecordIdentityTest(unittest.TestCase):
    def setUp(self):
        self.reg = registry.Registry()
        self.reg.register('tailor_f', tag())

    def test_schema_valid_records_with_the_run_identity_pass(self):
        self.assertEqual(self.reg.check_records('tailor_f', [record_of('tailor_f')]), 1)

    def test_record_with_another_key_seed_is_refused(self):
        with self.assertRaises(registry.RegistryError):
            self.reg.check_records('tailor_f', [record_of('tailor_f'), record_of('tailor_f', key_seed='other')])

    def test_record_of_another_run_is_refused(self):
        with self.assertRaises(registry.RegistryError):
            self.reg.check_records('tailor_f', [record_of('single')])

    def test_schema_invalid_record_is_refused(self):
        with self.assertRaises(registry.RegistryError):
            self.reg.check_records('tailor_f', [record_of('tailor_f', threshold=3)])


if __name__ == '__main__':
    unittest.main()
