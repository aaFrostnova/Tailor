"""Releasing images: deployment identities and the deployment key.

Characterization, live calibration and every evaluation in this repository embed under calibration
identities img_<n> (OursComposite.secret_for) and under the evaluation key KEY, which is public with
the code. Both are reserved for that purpose. The false-positive guarantee for a configuration that
live calibration selects adaptively (paper, Appendix B, condition (F)) holds for an identity whose
keyed permutation and mask calibration never evaluated, and its computational form assumes a secret
key. This module is the interface for embedding under a caller-chosen identity, and it enforces both
conditions: a released image carries an identity dep:<name>, never img_<n>, and the key is a secret
of at least 32 bytes, never KEY.

    python solver/deployment.py keygen > tailor.key        # 32 random bytes, written in hex
    export TAILOR_MASTER_KEY_FILE=tailor.key
    python solver/deployment.py embed  --config cfg.json --identity dep:alice in.png out.png
    python solver/deployment.py verify --config cfg.json --identity dep:alice --alpha 1e-6 out.png

`cfg.json` is the configuration the solver returned for the request (the dict OursComposite takes)
and `--alpha` is the request's false-positive budget. Verification runs the paper's acceptance rule,
measurement/unified_detector.py.
"""
import argparse
import hmac
import json
import os
import secrets
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(ROOT, "measurement"), os.path.join(ROOT, "solver"), ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DEPLOYMENT_PREFIX = "dep:"
CALIBRATION_PREFIX = "img_"                    # OursComposite.secret_for
MIN_KEY_BYTES = 32
EVALUATION_KEY = b"v5_key_encoder_master"      # eval_matrix.KEY, published with the code


class ReservedIdentityError(ValueError):
    """The identity is in the calibration namespace img_<n>."""


def check_identity(identity):
    """Return `identity` if a released image may carry it, else raise."""
    if not isinstance(identity, str):
        raise TypeError("an identity is a string")
    if identity.startswith(CALIBRATION_PREFIX):
        raise ReservedIdentityError(
            f"{identity!r} is in the calibration namespace img_<n>, which only calibration and evaluation use")
    if not identity.startswith(DEPLOYMENT_PREFIX) or len(identity) == len(DEPLOYMENT_PREFIX):
        raise ValueError(f"a deployment identity has the form {DEPLOYMENT_PREFIX}<name>, not {identity!r}")
    if not identity.isprintable():
        raise ValueError("an identity must be printable")
    return identity


def check_key(key):
    """Return `key` as bytes if it may serve as a deployment key, else raise."""
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("a key is bytes")
    key = bytes(key)
    if hmac.compare_digest(key, EVALUATION_KEY):
        raise ValueError("the evaluation key is public with this code and must not mark released images")
    if len(key) < MIN_KEY_BYTES:
        raise ValueError(f"a deployment key needs at least {MIN_KEY_BYTES} random bytes, not {len(key)}")
    return key


def new_key():
    """A fresh deployment key: 32 bytes from the operating system's generator, in hex."""
    return secrets.token_hex(MIN_KEY_BYTES)


def load_key(environ=None):
    """The deployment key, from TAILOR_MASTER_KEY (hex) or TAILOR_MASTER_KEY_FILE (a file holding hex)."""
    env = os.environ if environ is None else environ
    value, path = env.get("TAILOR_MASTER_KEY"), env.get("TAILOR_MASTER_KEY_FILE")
    if bool(value) == bool(path):
        raise KeyError("set exactly one of TAILOR_MASTER_KEY (hex) and TAILOR_MASTER_KEY_FILE")
    if path:
        with open(path) as f:
            value = f.read()
    try:
        key = bytes.fromhex(value.strip())
    except ValueError:
        raise ValueError("the deployment key must be written in hex") from None
    return check_key(key)


class Deployment:
    """One configuration returned by the solver, deployed under a secret key."""

    def __init__(self, config, key, dev="cuda", **composite_kwargs):
        from eval_matrix import KEY, OursComposite
        if KEY != EVALUATION_KEY:
            raise AssertionError("deployment.EVALUATION_KEY must equal eval_matrix.KEY")
        self.key = check_key(key)
        self.comp = OursComposite(dev, config=config, master_key=self.key, **composite_kwargs)

    @classmethod
    def _around(cls, composite, key):
        """A Deployment over an existing composite, for tests."""
        self = cls.__new__(cls)
        self.key, self.comp = check_key(key), composite
        return self

    def secret(self, identity):
        """(identity, codeword) of a deployment identity."""
        return self.comp.secret_for_identity(check_identity(identity))

    def embed(self, pil, identity):
        """The watermarked 512x512 image and its (identity, codeword)."""
        return self.comp._embed_secret(pil, self.secret(identity))

    def verify(self, pil, identity, alpha):
        """The acceptance record of measurement/unified_detector.py under `identity` at budget `alpha`."""
        secret = self.secret(identity)
        import unified_detector
        return unified_detector.decode_detailed(self.comp, pil, secret, alpha=alpha)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Embed and verify under deployment identities.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("keygen", help="print a new deployment key in hex")
    for name in ("embed", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, help="the configuration the solver returned (JSON)")
        command.add_argument("--identity", required=True, help="dep:<name>")
        command.add_argument("--device", default="cuda")
        if name == "embed":
            command.add_argument("source")
            command.add_argument("target")
        else:
            command.add_argument("--alpha", type=float, required=True, help="the request's false-positive budget")
            command.add_argument("image")
    args = parser.parse_args(argv)
    if args.command == "keygen":
        print(new_key())
        return 0
    try:                                          # refuse before loading a model
        check_identity(args.identity)
        key = load_key()
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(error.args[0] if error.args else str(error))
    with open(args.config) as f:
        config = json.load(f)
    from PIL import Image
    deployment = Deployment(config, key, dev=args.device)
    if args.command == "embed":
        image, _ = deployment.embed(Image.open(args.source).convert("RGB"), args.identity)
        image.save(args.target)
        return 0
    record = deployment.verify(Image.open(args.image).convert("RGB"), args.identity, args.alpha)
    print(json.dumps({k: record[k] for k in ("detected", "accepted_path", "bit_accuracy",
                                             "threshold_count", "max_tests", "alpha")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
