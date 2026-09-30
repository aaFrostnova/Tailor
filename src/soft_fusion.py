"""Soft per-bit LLR fusion across watermark methods (Component A).

The multi-method detector embeds the SAME 127-bit BCH codeword with each
method, under a per-method (perm sigma, mask M). Each decoder produces a soft
per-bit value in its own *target* (scrambled) domain. This module:

  1. converts each method's soft output to a log-likelihood ratio (LLR), where
     LLR > 0 means "logical codeword bit = 1";
  2. aligns it back to the shared logical codeword frame (the soft
     generalization of `undo_crypto`);
  3. fuses the aligned LLRs across methods by weighted summation
     (maximal-ratio combining), which is optimal when the methods are
     conditionally independent given the codeword.

Sign convention (must match src.payload and src.vine_crypto_wrapper):
  - A method's soft output is treated as evidence about the *target bit* at the
    encoder position r. For a probability p_r = P(target_r = 1), the target LLR
    is logit(p_r). For native logits, the value is already log P(1)/P(0).
  - `apply_crypto` sets target_r = cw_j  when M[r] = +1, and target_r = 1 - cw_j
    when M[r] = -1, where r = perm[j] and cw_j is the logical codeword bit.
    Hence LLR(cw_j) = M[r] * LLR(target_r).  Hard-thresholding the result
    reproduces `undo_crypto` exactly (see tests/test_soft_fusion.py).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


# ----------------------------------------------------------- soft -> LLR

def prob_to_llr(p: np.ndarray, eps: float = 1e-6, scale: float = 1.0) -> np.ndarray:
    """Sigmoid probabilities in [0,1] -> LLR = log(p/(1-p)).

    Used for VINE, whose ConvNeXt decoder ends in a sigmoid (probability that
    the target bit is 1). `scale` lets the caller calibrate this method's
    contribution relative to others.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return scale * np.log(p / (1.0 - p))


def logits_passthrough(logits: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Native logits (DFT-Kred / QIM learned heads) are already LLR-scaled.

    The convention there is `bit = 1 if logit > 0`, matching our LLR sign.
    """
    return scale * np.asarray(logits, dtype=np.float64)


def qim_margin_to_llr(mag: np.ndarray, delta: float, scale: float = 1.0) -> np.ndarray:
    """Analytical soft value for block-mean / DFT-magnitude QIM carriers.

    Bit-0 lattice = {k*delta}, bit-1 lattice = {k*delta + delta/2}. The signed
    margin (distance to the bit-0 lattice minus distance to the bit-1 lattice)
    is positive when the carrier sits closer to the bit-1 lattice, i.e. evidence
    for bit 1. This mirrors the distance-to-nearest-of-two-lattices decode rule
    used by the fragment wrappers when they expose raw logits.
    """
    mag = np.asarray(mag, dtype=np.float64)
    # nearest bit-0 lattice point: round(mag/delta)*delta
    d0 = np.abs(mag - np.round(mag / delta) * delta)
    # nearest bit-1 lattice point: (round(mag/delta - 0.5) + 0.5)*delta
    d1 = np.abs(mag - (np.round(mag / delta - 0.5) + 0.5) * delta)
    return scale * (d0 - d1)


def sum_redundant_carriers(per_carrier_llr: np.ndarray, carriers_per_bit: int) -> np.ndarray:
    """Combine K*M redundant carriers per bit by summing their LLRs.

    Redundant carriers are conditionally independent observations of the same
    bit, so summing log-likelihoods is the optimal combiner and strictly
    dominates the majority vote used in the prototype decoder.

    `per_carrier_llr` is laid out bit-major: [bit0_c0, bit0_c1, ..., bit1_c0, ...].
    """
    per_carrier_llr = np.asarray(per_carrier_llr, dtype=np.float64)
    n = per_carrier_llr.size // carriers_per_bit
    return per_carrier_llr[: n * carriers_per_bit].reshape(n, carriers_per_bit).sum(axis=1)


# ----------------------------------------------------------- align to codeword

def align_llr_to_codeword(
    llr_target: np.ndarray,
    perm: np.ndarray,
    M: np.ndarray,
    n_codeword: int = 127,
) -> np.ndarray:
    """Map a method's target-domain LLR to the shared logical codeword frame.

    Logical bit j is carried by target position r = perm[j], with a sign flip
    when M[r] = -1. Positions not covered by this method (j >= len(perm)) are
    left at 0 (no information). The hard threshold (out > 0) of the result
    equals `undo_crypto(prob, perm, M)` when llr_target = logit(prob).
    """
    perm = np.asarray(perm, dtype=np.int64)
    M = np.asarray(M, dtype=np.float64)
    llr_target = np.asarray(llr_target, dtype=np.float64)
    n = len(perm)
    out = np.zeros(n_codeword, dtype=np.float64)
    out[:n] = M[perm] * llr_target[perm]
    return out


def method_soft_to_codeword_llr(
    soft: np.ndarray,
    perm: np.ndarray,
    M: np.ndarray,
    kind: str = "prob",
    delta: Optional[float] = None,
    scale: float = 1.0,
    n_codeword: int = 127,
) -> np.ndarray:
    """Convenience: convert a method's raw soft output to aligned codeword LLR.

    kind: "prob" (sigmoid probs), "logit" (native logits), "qim" (needs delta).
    """
    if kind == "prob":
        llr_t = prob_to_llr(soft, scale=scale)
    elif kind == "logit":
        llr_t = logits_passthrough(soft, scale=scale)
    elif kind == "qim":
        if delta is None:
            raise ValueError("delta required for kind='qim'")
        llr_t = qim_margin_to_llr(soft, delta, scale=scale)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return align_llr_to_codeword(llr_t, perm, M, n_codeword=n_codeword)


# ----------------------------------------------------------- fuse

def fuse_llrs(
    aligned: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    n_codeword: int = 127,
) -> np.ndarray:
    """Maximal-ratio combining: L_fused[j] = sum_m w_m * L_{m,j}.

    Each value in `aligned` is a length-n_codeword LLR array already mapped to
    the logical frame. `weights` defaults to 1.0 per method.
    """
    L = np.zeros(n_codeword, dtype=np.float64)
    for name, a in aligned.items():
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        a = np.asarray(a, dtype=np.float64)
        L[: len(a)] += w * a[:n_codeword]
    return L


def llr_to_bits(llr: np.ndarray) -> np.ndarray:
    """Hard decision: logical bit = 1 where LLR > 0, else 0."""
    return (np.asarray(llr) > 0).astype(np.uint8)


def erasure_positions(llr: np.ndarray, abs_thresh: float) -> np.ndarray:
    """Indices whose fused |LLR| is below a confidence threshold (erasures)."""
    return np.where(np.abs(np.asarray(llr)) < abs_thresh)[0]
