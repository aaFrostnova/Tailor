"""Zero-training crypto wrapper for VINE pretrained model.

Wraps VINE-R-Enc/Dec with per-image (σ, M) permutation/mask.
The neural network is unchanged — crypto is purely pre/post processing.

Usage:
    wrapper = VineCryptoWrapper(master_key=b"my_secret_key")
    pil_wm = wrapper.embed(pil_image, image_id="img_001")
    result = wrapper.detect(pil_suspect, image_id="img_001")
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.paths import VINE_HF, VINE_REPO, require


def _on_path():
    """Put VINE's own source tree on the path. Only the two loaders below need it, so the
    crypto helpers in this module stay importable without VINE installed."""
    require(VINE_REPO, "VINE", "python tools/fetch_models.py --only vine", "TAILOR_VINE_REPO")
    for p in (VINE_REPO, os.path.join(VINE_REPO, "vine", "src")):
        if p not in sys.path:
            sys.path.insert(0, p)

from src.sign_envelope import _hkdf_uint64_stream
from src.payload import BCHCodec, image_id_to_payload


def derive_method_keyed_constants(
    master_key: bytes, image_id: str, method_name: str, n_bits: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive (σ, M) with method-specific salt for independence between methods."""
    salt = f"multi_method/{method_name}/{image_id}".encode("utf-8")
    perm_stream = _hkdf_uint64_stream(master_key, salt + b"/perm", n_bits)
    perm = np.arange(n_bits, dtype=np.int64)
    for i in range(n_bits - 1, 0, -1):
        j = int(perm_stream[i] % (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    mask_stream = _hkdf_uint64_stream(master_key, salt + b"/mask", n_bits)
    M = np.where(mask_stream % 2 == 0, 1, -1).astype(np.int8)
    return perm, M


def apply_crypto(codeword: np.ndarray, perm: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply crypto transform: codeword → target bits for encoder."""
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    bit_at_pos = codeword[inv_perm]
    signs = M.astype(np.float32) * (1.0 - 2.0 * bit_at_pos.astype(np.float32))
    target_bits = ((1 - signs) / 2).astype(np.float32)
    return target_bits


def undo_crypto(decoded_bits: np.ndarray, perm: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Undo crypto transform: decoder output → recovered codeword."""
    n = len(perm)
    recovered = np.zeros(n, dtype=np.uint8)
    for j in range(n):
        r = int(perm[j])
        decoded_sign = 1 if decoded_bits[r] < 0.5 else -1
        original_sign = decoded_sign * int(M[r])
        recovered[j] = 0 if original_sign > 0 else 1
    return recovered


class VineCryptoWrapper:
    """Crypto wrapper around VINE pretrained encoder/decoder."""

    def __init__(self, master_key: bytes = b"v5_key_encoder_master",
                 method_name: str = "vine", n_bits: int = 100, device: str = "cuda",
                 detection_threshold: float = 0.75, variant: str = "R"):
        self.master_key = master_key
        self.method_name = method_name
        self.n_bits = n_bits
        self.device = device
        self.detection_threshold = detection_threshold
        self.variant = variant          # "R" (regen-robust, default) or "B" (higher-PSNR, mild threats)
        self.codec = BCHCodec()
        self._enc = None
        self._dec = None

    def _load_encoder(self):
        if self._enc is None:
            _on_path()
            from vine_turbo import VINE_Turbo
            self._enc = VINE_Turbo.from_pretrained(VINE_HF.format(variant=self.variant, part="Enc"))
            self._enc.to(self.device).eval()
        return self._enc

    def _load_decoder(self):
        if self._dec is None:
            _on_path()
            from stega_encoder_decoder import CustomConvNeXt
            self._dec = CustomConvNeXt.from_pretrained(VINE_HF.format(variant=self.variant, part="Dec"))
            self._dec.to(self.device).eval()
        return self._dec

    def _get_target_bits(self, image_id: str) -> torch.Tensor:
        payload = image_id_to_payload(image_id, n_bits=self.codec.data_bits)
        codeword = self.codec.encode(payload)
        cw_for_vine = codeword[:self.n_bits]
        perm, M = derive_method_keyed_constants(
            self.master_key, image_id, self.method_name, self.n_bits,
        )
        target = apply_crypto(cw_for_vine, perm, M)
        return torch.tensor(target, dtype=torch.float32)

    def get_perm_M(self, image_id: str):
        """Return this method's (perm, M) for aligning soft bits to the codeword."""
        return derive_method_keyed_constants(
            self.master_key, image_id, self.method_name, self.n_bits,
        )

    def embed_with_target(self, pil: Image.Image, target_bits: np.ndarray,
                           strength: float = 1.0) -> Image.Image:
        """Embed an explicit length-n_bits target (already crypto-applied).

        Used by the fused detector so all methods carry the SAME codeword.
        `strength` scales the encoder residual (1.0 = today's default, byte-identical).
        """
        from torchvision import transforms
        enc = self._load_encoder()
        target_bits = torch.tensor(
            np.asarray(target_bits, dtype=np.float32)
        ).unsqueeze(0).to(self.device)

        size = pil.size
        t256 = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        t_back = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
        resized = t256(pil).unsqueeze(0).to(self.device) * 2 - 1
        orig = transforms.ToTensor()(pil).unsqueeze(0).to(self.device) * 2 - 1
        with torch.no_grad():
            enc_256 = enc(resized, target_bits)
        residual = t_back(enc_256 - resized)
        encoded = torch.clamp((strength * residual + orig) * 0.5 + 0.5, 0, 1)
        return transforms.ToPILImage()(encoded[0].cpu())

    def raw_probs(self, pil: Image.Image) -> np.ndarray:
        """Raw per-bit sigmoid probabilities from the decoder (no crypto undo).

        These are the target-domain probabilities P(target_bit = 1), consumed by
        src.soft_fusion.prob_to_llr + align_llr_to_codeword for fusion.
        """
        from torchvision import transforms
        dec = self._load_decoder()
        t256 = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        img = t256(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = dec(img)[0].cpu().numpy()
        return probs

    def embed(self, pil: Image.Image, image_id: str) -> Image.Image:
        from torchvision import transforms
        enc = self._load_encoder()
        target_bits = self._get_target_bits(image_id).unsqueeze(0).to(self.device)

        size = pil.size
        t256 = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        t_back = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)

        resized = t256(pil).unsqueeze(0).to(self.device) * 2 - 1
        orig = transforms.ToTensor()(pil).unsqueeze(0).to(self.device) * 2 - 1

        with torch.no_grad():
            enc_256 = enc(resized, target_bits)
        residual = t_back(enc_256 - resized)
        encoded = torch.clamp((residual + orig) * 0.5 + 0.5, 0, 1)
        return transforms.ToPILImage()(encoded[0].cpu())

    def detect(self, pil: Image.Image, image_id: str) -> dict:
        from torchvision import transforms
        dec = self._load_decoder()

        t256 = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        img = t256(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = dec(img)[0].cpu().numpy()

        perm, M = derive_method_keyed_constants(
            self.master_key, image_id, self.method_name, self.n_bits,
        )
        recovered_cw = undo_crypto(logits, perm, M)

        payload = image_id_to_payload(image_id, n_bits=self.codec.data_bits)
        expected_cw = self.codec.encode(payload)[:self.n_bits]

        bit_acc = float(np.mean(recovered_cw == expected_cw))

        # Detection by bit_accuracy threshold (not BCH decode, because
        # we only embed n_bits of the full 127-bit codeword, so BCH
        # always fails on the missing ECC bits).
        detected = bit_acc >= self.detection_threshold

        return {
            "detected": bool(detected),
            "bit_accuracy": bit_acc,
            "method": self.method_name,
            "probs": logits.tolist(),       # raw P(target_bit=1), for soft fusion
            "perm": perm.tolist(),
            "M": M.tolist(),
        }
