"""CPU-only tests of the deployment interface. No model is loaded.

    python solver/test_deployment.py

Identities: released images carry dep:<name>, never the calibration namespace img_<n>. Keys: a
deployment key is at least 32 secret bytes and never the published evaluation key. Routing: the
identity check runs before the composite is touched. Evaluation: embed(pil, idx) still embeds under
img_<idx> with the same per-fragment calls, so every measured number is unaffected.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "solver"), os.path.join(ROOT, "measurement")]

import deployment as D  # noqa: E402


class Identities(unittest.TestCase):
    def test_deployment_identities_pass(self):
        for identity in ("dep:alice", "dep:7f3a-01", "dep:user/42"):
            self.assertEqual(D.check_identity(identity), identity)

    def test_the_calibration_namespace_is_refused(self):
        # img_00000..img_00099 are the live-calibration identities; evaluation also used img_10000+
        # (clean covers) and img_7000000+ (fresh-identity negatives). The whole prefix is reserved.
        for identity in ("img_00000", "img_00099", "img_10000", "img_7000123", "img_", "img_dep:alice"):
            with self.assertRaises(D.ReservedIdentityError):
                D.check_identity(identity)

    def test_other_forms_are_refused(self):
        for identity in ("alice", "dep:", "", "DEP:alice", "dep:a\nb"):
            with self.assertRaises(ValueError):
                D.check_identity(identity)
        with self.assertRaises(TypeError):
            D.check_identity(7)


class Keys(unittest.TestCase):
    def test_a_new_key_is_32_fresh_bytes(self):
        first, second = D.new_key(), D.new_key()
        self.assertEqual(len(bytes.fromhex(first)), 32)
        self.assertNotEqual(first, second)
        self.assertEqual(D.check_key(bytes.fromhex(first)), bytes.fromhex(first))

    def test_the_evaluation_key_and_short_keys_are_refused(self):
        with self.assertRaises(ValueError):
            D.check_key(D.EVALUATION_KEY)
        with self.assertRaises(ValueError):
            D.check_key(b"\x00" * 31)
        with self.assertRaises(TypeError):
            D.check_key("00" * 32)

    def test_load_key_from_the_environment_or_a_file(self):
        key = D.new_key()
        self.assertEqual(D.load_key({"TAILOR_MASTER_KEY": key}), bytes.fromhex(key))
        with tempfile.NamedTemporaryFile("w", suffix=".key", delete=False) as f:
            f.write(key + "\n")
        try:
            self.assertEqual(D.load_key({"TAILOR_MASTER_KEY_FILE": f.name}), bytes.fromhex(key))
        finally:
            os.unlink(f.name)
        with self.assertRaises(KeyError):
            D.load_key({})
        with self.assertRaises(KeyError):
            D.load_key({"TAILOR_MASTER_KEY": key, "TAILOR_MASTER_KEY_FILE": "unused"})
        with self.assertRaises(ValueError):
            D.load_key({"TAILOR_MASTER_KEY": "not hex"})
        with self.assertRaises(ValueError):
            D.load_key({"TAILOR_MASTER_KEY": D.EVALUATION_KEY.hex()})


class Routing(unittest.TestCase):
    def setUp(self):
        self.comp = mock.Mock()
        self.comp.secret_for_identity.side_effect = lambda identity: (identity, np.zeros(100, np.uint8))
        self.deployment = D.Deployment._around(self.comp, bytes.fromhex(D.new_key()))

    def test_a_reserved_identity_never_reaches_the_composite(self):
        with self.assertRaises(D.ReservedIdentityError):
            self.deployment.embed(object(), "img_00001")
        with self.assertRaises(D.ReservedIdentityError):
            self.deployment.verify(object(), "img_00001", 1e-6)
        self.comp.secret_for_identity.assert_not_called()
        self.comp._embed_secret.assert_not_called()

    def test_embed_uses_the_deployment_identity(self):
        image = object()
        self.deployment.embed(image, "dep:alice")
        (passed_image, (identity, _)), _ = self.comp._embed_secret.call_args
        self.assertIs(passed_image, image)
        self.assertEqual(identity, "dep:alice")


class EvaluationEmbedding(unittest.TestCase):
    """embed(pil, idx) with stub fragments: same identity, codeword, masks and calls as before."""

    def test_embed_is_unchanged(self):
        from PIL import Image
        import eval_matrix as EM
        from src.shortened_bch import ShortenedBCH
        from src.vine_crypto_wrapper import apply_crypto, derive_method_keyed_constants

        calls = []

        class Fragment:
            def __init__(self, name):
                self.name = name

            def get_perm_M(self, identity):
                calls.append(("keys", self.name, identity))
                return derive_method_keyed_constants(EM.KEY, identity, self.name, 100)

            def embed_with_target(self, image, target, strength):
                calls.append(("embed", self.name, tuple(int(b) for b in target), strength))
                return image

        comp = EM.OursComposite.__new__(EM.OursComposite)
        comp.sb, comp.order = ShortenedBCH(), ["videoseal", "trustmark", "vine"]
        comp.strength = {"videoseal": 0.9, "trustmark": 1.3, "vine": 0.7}
        comp.nested = comp.fe_tile = comp.resync = False
        comp.frag = {name: Fragment(name) for name in comp.order}

        image, (identity, word) = comp.embed(Image.new("RGB", (512, 512)), 7)
        self.assertEqual(identity, "img_00007")
        expected_word = comp.sb.encode(EM._id_payload("img_00007", comp.sb.data_bits))
        self.assertTrue(np.array_equal(word, expected_word))
        self.assertTrue(np.array_equal(comp.secret_for(7)[1], comp.secret_for_identity("img_00007")[1]))
        expected = []
        for name in comp.order:
            perm, mask = derive_method_keyed_constants(EM.KEY, "img_00007", name, 100)
            target = apply_crypto(expected_word, perm, mask)
            expected += [("keys", name, "img_00007"),
                         ("embed", name, tuple(int(b) for b in target), comp.strength[name])]
        self.assertEqual(calls, expected)
        self.assertEqual(image.size, (512, 512))


if __name__ == "__main__":
    unittest.main()
