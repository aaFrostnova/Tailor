"""Chase-II soft-decision decoding over the existing BCH codec (Component A.3/C.1).

bchlib 2.1.3 exposes only errors-only decode (no native soft/erasure decode).
We wrap `BCHCodec` with Chase-II: flip the least-reliable positions through all
2^p patterns, run the hard BCH decoder on each, and keep the candidate codeword
that best agrees with the soft LLR vector (minimum correlation distance). This
exploits the soft per-bit signal produced by `src.soft_fusion` and extends the
effective correction radius beyond the raw t=10 for the easy-error case.

For erasures (Component C), `decode_with_erasures` forces a known-damaged set of
positions into the flip set (with LLR := 0), recovering most of the benefit of
true erasure decoding without changing the ECC. NOTE: this repairs the PAYLOAD;
it does not restore image content.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.payload import BCHCodec, image_id_to_payload


def hard_from_llr(llr: np.ndarray) -> np.ndarray:
    return (np.asarray(llr) > 0).astype(np.uint8)


def _try_decode(codec: BCHCodec, codeword_bits: np.ndarray):
    """Run the hard BCH decoder; return corrected data bits or None on failure."""
    try:
        data, n_err = codec.decode(codeword_bits.astype(np.uint8))
    except Exception:
        return None, -1
    if data is None or n_err < 0:
        return None, -1
    return np.asarray(data, dtype=np.uint8), int(n_err)


def chase_decode(
    llr: np.ndarray,
    codec: Optional[BCHCodec] = None,
    p: int = 8,
    forced_idx: Optional[Sequence[int]] = None,
    max_patterns: int = 1024,
) -> Optional[Dict]:
    """Chase-II decode of a length-n LLR vector.

    Args:
        llr: per-logical-bit LLR (LLR > 0 -> bit 1), length codec.n (127).
        p: number of least-reliable positions to enumerate (2^p patterns).
        forced_idx: positions that MUST be in the flip set (e.g. erasures).
        max_patterns: hard cap on decode attempts.

    Returns the best candidate dict {data, codeword, cost, n_err, n_flips} or
    None if no flip pattern decodes.
    """
    codec = codec or BCHCodec()
    llr = np.asarray(llr, dtype=np.float64).copy()
    n = codec.n
    if len(llr) != n:
        raise ValueError(f"llr must have length {n}, got {len(llr)}")

    forced = list(dict.fromkeys(int(i) for i in (forced_idx or [])))
    absl = np.abs(llr)
    # Least-reliable positions first; forced positions (often |llr|=0) sort to
    # the front naturally but we union them in explicitly to be safe.
    order = [int(i) for i in np.argsort(absl)]
    flip_pool: List[int] = list(dict.fromkeys(forced + order))[: max(p, len(forced))]

    hard = hard_from_llr(llr)
    best: Optional[Dict] = None
    n_tried = 0

    # Enumerate flip subsets of the pool, smallest subsets first.
    for k in range(len(flip_pool) + 1):
        for combo in itertools.combinations(flip_pool, k):
            if n_tried >= max_patterns:
                return best
            n_tried += 1
            cw = hard.copy()
            for idx in combo:
                cw[idx] ^= 1
            data, n_err = _try_decode(codec, cw)
            if data is None:
                continue
            cand = codec.encode(data)
            # Correlation distance: penalize disagreements weighted by reliability.
            cost = float(np.sum(absl * (cand != hard)))
            if best is None or cost < best["cost"]:
                best = {
                    "data": data,
                    "codeword": cand,
                    "cost": cost,
                    "n_err": n_err,
                    "n_flips": len(combo),
                }
    return best


def decode_with_erasures(
    llr: np.ndarray,
    erasure_idx: Sequence[int],
    codec: Optional[BCHCodec] = None,
    p: int = 8,
    max_erasures: int = 10,
) -> Optional[Dict]:
    """Erasure-assisted decode (Component C.1).

    Sets the LLR of known-damaged positions to 0 and forces them into the Chase
    flip set, so only the remaining hard errors must fall within t. Caps the
    erasure count to keep 2^p tractable.
    """
    codec = codec or BCHCodec()
    llr = np.asarray(llr, dtype=np.float64).copy()
    erasure_idx = list(dict.fromkeys(int(i) for i in erasure_idx))[:max_erasures]
    llr[erasure_idx] = 0.0
    p_eff = max(p, len(erasure_idx))
    return chase_decode(llr, codec=codec, p=p_eff, forced_idx=erasure_idx)


def decode_and_verify(
    llr: np.ndarray,
    image_id: str,
    codec: Optional[BCHCodec] = None,
    p: int = 8,
    erasure_idx: Optional[Sequence[int]] = None,
) -> Dict:
    """Full detection: Chase-decode the fused LLR and check payload identity.

    Detection is a true identity match against `image_id_to_payload(image_id)`,
    which is far stronger than a bit-accuracy threshold.
    """
    codec = codec or BCHCodec()
    if erasure_idx is not None and len(erasure_idx) > 0:
        best = decode_with_erasures(llr, erasure_idx, codec=codec, p=p)
    else:
        best = chase_decode(llr, codec=codec, p=p)

    expected = image_id_to_payload(image_id, n_bits=codec.data_bits)
    if best is None:
        return {
            "detected": False,
            "payload_bits": None,
            "expected_bits": expected,
            "n_err": -1,
            "data_bit_acc": 0.0,
        }
    data = best["data"]
    n = min(len(data), len(expected))
    acc = float(np.mean(data[:n] == expected[:n]))
    detected = bool(np.array_equal(data[:n], expected[:n]))
    return {
        "detected": detected,
        "payload_bits": data,
        "expected_bits": expected,
        "n_err": best["n_err"],
        "n_flips": best["n_flips"],
        "cost": best["cost"],
        "data_bit_acc": acc,
    }
