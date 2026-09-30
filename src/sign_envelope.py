"""Keyed sign envelope for v3 hybrid (learnable carrier + crypto envelope).

For each image_id we derive a permutation σ and a ±1 mask M from
(master_key, image_id) via HKDF-SHA512. The 127-bit BCH codeword `c` is
then mapped to a (3, H, W) sign pattern that flips the sign of the
shared template `T` in 127 disjoint spatial regions:

    embedded = clip(x + jnd(x) ⊙ T ⊙ S, 0, 1)

Decode:
    residual = suspect − orig
    For each region r:
        ip_r = <residual_r, T_r>          (inner product per region)
        pred_sign_r = sign(ip_r)
        recovered_bit_r = (1 − pred_sign_r · M[r]) / 2

The 127 bits are then BCH-decoded into a 64-bit payload that should match
image_id_to_payload(image_id).

Security rests on the keyed M and σ — an adversary without the master_key
cannot predict which region carries which bit, nor whether it should be +1
or −1, even if the template T is public.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import List, Tuple

import numpy as np


# ----------------------------------------------------------- HKDF -> uint64 stream

def _hkdf_uint64_stream(key: bytes, salt: bytes, n: int) -> np.ndarray:
    """Generate n uint64 deterministic values via HMAC-SHA512 with counter."""
    out = bytearray()
    counter = 0
    while len(out) < n * 8:
        chunk = hmac.new(key, salt + counter.to_bytes(4, "big"), hashlib.sha512).digest()
        out.extend(chunk)
        counter += 1
    return np.frombuffer(bytes(out[: n * 8]), dtype=np.uint64).copy()


def derive_keyed_constants(
    master_key: bytes, image_id: str, n_bits: int = 127,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive (permutation σ ∈ S_n_bits, ±1 mask M ∈ {±1}^n_bits)."""
    salt = b"v3_envelope/" + image_id.encode("utf-8")
    # σ from HMAC counter stream
    perm_stream = _hkdf_uint64_stream(master_key, salt + b"/perm", n_bits)
    perm = np.arange(n_bits, dtype=np.int64)
    for i in range(n_bits - 1, 0, -1):
        j = int(perm_stream[i] % (i + 1))
        perm[i], perm[j] = perm[j], perm[i]

    # ±1 mask M
    mask_stream = _hkdf_uint64_stream(master_key, salt + b"/mask", n_bits)
    M = np.where(mask_stream % 2 == 0, 1, -1).astype(np.int8)

    return perm, M


# --------------------------------------------------------------------- regions

def make_region_masks(
    shape: Tuple[int, int, int],
    n_bits: int = 127,
) -> Tuple[List[np.ndarray], int, int]:
    """Tile image space into n_bits roughly-equal rectangular regions.

    Returns (masks, grid_rows, grid_cols). `masks[i]` is a bool array of `shape`.
    """
    C, H, W = shape
    # Pick a grid close to square
    grid_n = int(np.ceil(np.sqrt(n_bits)))     # 12 for n_bits=127
    cell_h = H // grid_n
    cell_w = W // grid_n

    masks: List[np.ndarray] = []
    for i in range(n_bits):
        r, c = i // grid_n, i % grid_n
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < grid_n - 1 else H
        x0 = c * cell_w
        x1 = (c + 1) * cell_w if c < grid_n - 1 else W
        m = np.zeros(shape, dtype=bool)
        m[:, y0:y1, x0:x1] = True
        masks.append(m)
    return masks, grid_n, grid_n


def build_sign_pattern(
    codeword: np.ndarray,
    perm: np.ndarray,
    mask: np.ndarray,
    region_masks: List[np.ndarray],
    shape: Tuple[int, int, int],
) -> np.ndarray:
    """Construct a (C, H, W) ±1 sign tensor.

    The j-th codeword bit `codeword[j]` is mapped to region `perm[j]` and
    the embedded sign is `M[perm[j]] · (1 − 2·codeword[j])`. This couples
    every codeword bit to one region.
    """
    if len(codeword) != len(perm):
        raise ValueError(f"len(codeword)={len(codeword)} != len(perm)={len(perm)}")
    n_bits = len(codeword)
    sign_map = np.ones(shape, dtype=np.float32)
    for j in range(n_bits):
        region_idx = int(perm[j])
        bit = int(codeword[j])
        sign = float(mask[region_idx]) * (1.0 if bit == 0 else -1.0)
        sign_map[region_masks[region_idx]] = sign
    return sign_map


# ----------------------------------------------------------------- recover bits

def recover_codeword_from_residual(
    residual: np.ndarray,
    template_T: np.ndarray,
    perm: np.ndarray,
    mask: np.ndarray,
    region_masks: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Statistical decode of 127 codeword bits.

    For each region: ip_r = <residual_r, T_r>; pred_sign_r = sign(ip_r);
    recovered bit at perm⁻¹(r) is (1 − pred_sign_r · M[r]) / 2.
    """
    n_bits = len(perm)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n_bits)

    region_signs = np.zeros(n_bits, dtype=np.int8)
    region_scores = np.zeros(n_bits, dtype=np.float64)
    for r in range(n_bits):
        rmask = region_masks[r]
        ip = float(np.sum(residual[rmask] * template_T[rmask]))
        region_scores[r] = ip
        region_signs[r] = 1 if ip >= 0 else -1

    # Recover bits via inverse permutation
    bits = np.zeros(n_bits, dtype=np.uint8)
    scores = np.zeros(n_bits, dtype=np.float64)
    for j in range(n_bits):
        r = int(perm[j])
        decoded_sign = int(region_signs[r]) * int(mask[r])
        bits[j] = 0 if decoded_sign > 0 else 1
        scores[j] = abs(region_scores[r])
    return bits, scores
