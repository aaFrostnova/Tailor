"""Payload encoding + BCH error correction for v2 fingerprint.

Maps a string image_id to a 64-bit payload, encodes via BCH(127, 64, t=10)
into a 127-bit codeword. Codeword bits get embedded as sign modulations
across fragment spatial regions (see bit_assignment.py).

The BCH layer tolerates up to 10 bit-flips before decode failure, providing
the redundancy needed for SD regeneration attacks where individual fragment
signs may flip.
"""

import hashlib
from typing import Tuple, Optional

import numpy as np

try:
    import bchlib  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "bchlib is required: pip install bchlib"
    ) from e


# BCH(127, 64, t=10) — codeword 127 bits, data 64 bits, corrects 10 errors.
# Other supported configs (set via BCHCodec(m, t)):
#   m=7, t=10:  n=127, k=64  → BCH(127, 64, t=10)  [default]
#   m=7, t=15:  n=127, k=36  → BCH(127, 36, t=15)
#   m=8, t=18:  n=255, k=131 → BCH(255, 131, t=18)
DEFAULT_BCH_M = 7
DEFAULT_BCH_T = 10
N_DATA_BITS = 64           # payload bit width
N_CODEWORD_BITS = 127      # n = 2^m - 1
N_ECC_BITS = 63            # codeword - data


def _bits_to_bytes(bits: np.ndarray, n_bytes: Optional[int] = None) -> bytes:
    """Pack a length-N {0,1} ndarray into a bytes string, MSB-first per byte.

    If n_bytes is provided, pad output to that many bytes (extra bytes are zero).
    """
    n = len(bits)
    needed = (n + 7) // 8
    if n_bytes is None:
        n_bytes = needed
    if n_bytes < needed:
        raise ValueError(f"n_bytes={n_bytes} too small for {n} bits")
    out = bytearray(n_bytes)
    for i in range(n):
        if int(bits[i]):
            out[i // 8] |= 1 << (7 - (i % 8))
    return bytes(out)


def _bytes_to_bits(data: bytes, n_bits: int) -> np.ndarray:
    """Unpack the first n_bits of a bytes string MSB-first into a uint8 ndarray."""
    out = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        byte = data[i // 8]
        out[i] = (byte >> (7 - (i % 8))) & 1
    return out


class BCHCodec:
    """BCH(n=127, k=64, t=10) encoder + decoder, bit-array I/O.

    Parameters mirror bchlib.BCH(t, m) defaults for our use case. The data is
    64 bits, the ECC is 63 bits, and the full codeword is 127 bits.
    """

    # Primitive polynomials for GF(2^m)
    _POLYS = {5: 37, 7: 137}

    def __init__(self, m: int = DEFAULT_BCH_M, t: int = DEFAULT_BCH_T):
        try:
            self.bch = bchlib.BCH(t=t, m=m)
        except TypeError:
            poly = self._POLYS.get(m, 137)
            self.bch = bchlib.BCH(poly, t)
        self.m = m
        self.t = t
        self.n = self.bch.n
        self.ecc_bits = self.bch.ecc_bits
        self.ecc_bytes = self.bch.ecc_bytes
        # Probe largest data_bytes for which encode+decode succeed
        # (bchlib's allowed data length depends on m, t internally).
        max_db = 1
        for nd in range(1, 64):
            try:
                _data = bytes(nd)
                _ecc = self.bch.encode(_data)
                _ = self.bch.decode(bytearray(_data), bytearray(_ecc))
                max_db = nd
            except Exception:
                break
        self.data_bytes = max_db
        self.data_bits = self.data_bytes * 8

    def encode(self, payload_bits: np.ndarray) -> np.ndarray:
        """Encode a length-`data_bits` {0,1} ndarray to a length-`n` codeword."""
        if len(payload_bits) != self.data_bits:
            raise ValueError(
                f"payload_bits must have length {self.data_bits}, got {len(payload_bits)}"
            )
        data_bytes_buf = _bits_to_bytes(
            payload_bits.astype(np.uint8), n_bytes=self.data_bytes,
        )
        ecc_buf = self.bch.encode(data_bytes_buf)
        ecc_bits = _bytes_to_bits(ecc_buf, self.ecc_bits)
        # Total logical codeword = data_bits || ecc_bits; pad/truncate to self.n
        full = np.concatenate([payload_bits.astype(np.uint8), ecc_bits]).astype(np.uint8)
        if len(full) >= self.n:
            return full[: self.n]
        # If somehow shorter (shouldn't happen), zero-pad
        out = np.zeros(self.n, dtype=np.uint8)
        out[: len(full)] = full
        return out

    def decode(
        self, codeword_bits: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int]:
        """Decode a length-127 codeword, correcting up to t bit flips.

        Returns:
            (recovered_data_bits, n_errors_corrected)
            or (None, -1) if uncorrectable.
        """
        if len(codeword_bits) != self.n:
            raise ValueError(
                f"codeword_bits must have length {self.n}, got {len(codeword_bits)}"
            )
        data_bits = codeword_bits[: self.data_bits].astype(np.uint8)
        ecc_bits = codeword_bits[self.data_bits :].astype(np.uint8)

        data_buf = bytearray(_bits_to_bytes(data_bits, n_bytes=self.data_bytes))
        ecc_buf = bytearray(_bits_to_bytes(ecc_bits, n_bytes=self.ecc_bytes))

        # bchlib API differs across versions:
        #   newer: decode() returns int, correct() applies in-place
        #   older: decode() returns (n_err, data, ecc) tuple
        decode_result = self.bch.decode(data_buf, ecc_buf)
        if isinstance(decode_result, tuple):
            n_err, data_buf, ecc_buf = decode_result
        else:
            n_err = decode_result
            self.bch.correct(data_buf, ecc_buf)

        corrected_bits = _bytes_to_bits(bytes(data_buf), self.data_bits)
        return corrected_bits, int(n_err)


# -------------------------------------------------------------- image_id ↔ bits

def image_id_to_payload(image_id: str, n_bits: int = N_DATA_BITS) -> np.ndarray:
    """Map a string image_id to a deterministic n_bits payload via SHA-256.

    Caller is responsible for storing the mapping image_id ↔ payload elsewhere.
    Two different image_id strings may collide if n_bits is short, but at 64
    bits the collision probability is negligible for any realistic registry.
    """
    digest = hashlib.sha256(image_id.encode("utf-8")).digest()
    return _bytes_to_bits(digest, n_bits)


def payload_to_hex(payload_bits: np.ndarray) -> str:
    """Convert a payload bit-array to a hex string for registry lookup."""
    return _bits_to_bytes(payload_bits.astype(np.uint8)).hex()


# ----------------------------------------------------------------- sign utility

def bits_to_signs(bits: np.ndarray) -> np.ndarray:
    """Map bit array {0, 1} to sign array {+1, -1}.

    Convention: bit 0 → +1, bit 1 → -1. The decoder reads the sign of a
    correlation and inverts this mapping.
    """
    return np.where(bits.astype(np.int8) == 0, 1, -1).astype(np.int8)


def signs_to_bits(signs: np.ndarray) -> np.ndarray:
    """Inverse of bits_to_signs. Treats non-negative as bit 0, negative as bit 1."""
    return np.where(signs >= 0, 0, 1).astype(np.uint8)
