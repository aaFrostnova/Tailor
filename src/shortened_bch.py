"""Shortened BCH(100, 37, t=10) over the existing BCH(127, 64, t=10) codec.

The bit-carrying watermark methods (VINE, DFT-Kred, Quant-QIM) all embed exactly
100 bits. To make every codeword bit covered by every method (so per-bit fusion
+ full BCH error correction work), we *shorten* BCH(127, 64) to a 100-bit
transmitted codeword by fixing the last 27 data positions to zero and not
transmitting them:

    full codeword (127) = [ data(64) | ecc(63) ]
                            data[0:37] transmitted; data[37:64] = 0 (shortened)
    transmitted (100)   = [ data[0:37] | ecc(63) ]

Shortened positions are known-zero, so reinserting them at decode adds no error;
errors-only bchlib still corrects up to t=10 flips among the 100 transmitted
bits. This keeps the BCH algorithm (no ECC swap) while making all three 100-bit
methods carry the SAME complete codeword.

The object exposes the same (`n`, `data_bits`, `encode`, `decode`) surface as
`BCHCodec`, so `src.soft_bch.chase_decode` / `decode_and_verify` work on it
unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.payload import BCHCodec


class ShortenedBCH:
    def __init__(self, codec: Optional[BCHCodec] = None, n_tx: int = 100):
        self.codec = codec or BCHCodec()
        self.n_full = self.codec.n              # 127
        self.k_full = self.codec.data_bits      # 64
        self.ecc_bits = self.codec.ecc_bits     # 63
        self.n = int(n_tx)                       # 100 transmitted bits
        self.data_bits = self.n - self.ecc_bits  # 37 payload bits
        if not (1 <= self.data_bits <= self.k_full):
            raise ValueError(
                f"n_tx={n_tx} implies data_bits={self.data_bits}, "
                f"must be in [1, {self.k_full}]"
            )
        # Positions inside the full 127-bit codeword.
        self.data_pos = np.arange(self.data_bits)                       # 0..36
        self.zero_pos = np.arange(self.data_bits, self.k_full)          # 37..63
        self.ecc_pos = np.arange(self.k_full, self.n_full)             # 64..126
        self.tx_pos = np.concatenate([self.data_pos, self.ecc_pos])    # length n
        assert len(self.tx_pos) == self.n

    def encode(self, payload_bits: np.ndarray) -> np.ndarray:
        """payload (data_bits) -> transmitted codeword (n)."""
        payload_bits = np.asarray(payload_bits, dtype=np.uint8)
        if len(payload_bits) != self.data_bits:
            raise ValueError(
                f"payload must have length {self.data_bits}, got {len(payload_bits)}"
            )
        data64 = np.zeros(self.k_full, dtype=np.uint8)
        data64[: self.data_bits] = payload_bits
        cw = self.codec.encode(data64)
        return cw[self.tx_pos].astype(np.uint8)

    def _assemble_full(self, tx_bits: np.ndarray) -> np.ndarray:
        cw = np.zeros(self.n_full, dtype=np.uint8)
        cw[self.tx_pos] = np.asarray(tx_bits, dtype=np.uint8)
        # zero_pos already 0 (known shortened bits)
        return cw

    def decode(self, tx_bits: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """transmitted codeword (n) -> (payload data_bits, n_err) or (None, -1)."""
        cw = self._assemble_full(tx_bits)
        data64, n_err = self.codec.decode(cw)
        if data64 is None or n_err < 0:
            return None, -1
        return np.asarray(data64[: self.data_bits], dtype=np.uint8), int(n_err)
