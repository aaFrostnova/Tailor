"""TrustMark as a FUSIBLE codeword fragment (not an independent OR corroborator).

Empirically (scripts/defense/fuse_trustmark_test.py) fusing TrustMark's raw 100-bit decoder
output into the shared-codeword fusion: neutral on regen/rinse/signal/VAE (+/-0.003), and a
LARGE gain on geometric attacks it survives -- hflip fused bit-acc 0.515->0.885, crop75
0.503->0.829 -- because TrustMark's CNN decode is flip/center-crop robust and the per-image
crypto (perm, M) realigns its bits to the codeword. So it rescues geometric detection AND
cryptographic identification that the keyed transform/latent fragments cannot.

Uses use_ECC=False, secret_len=100: the decoder net natively emits 100 bits; the soft per-bit
signal is decoder.decoder(stego) (sign = bit), thresholded at >0 internally. We expose those
raw logits so soft_fusion can MRC-combine them with VINE/DFT/QIM.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.payload import BCHCodec, image_id_to_payload
from src.vine_crypto_wrapper import (
    apply_crypto, derive_method_keyed_constants, undo_crypto,
)


class TrustMarkFragment:
    """TrustMark carrying the shared shortened-BCH codeword, keyed per-image by (perm, M).
    Mirrors VineCryptoWrapper so it drops into the fused detector and the overwrite matrix."""

    def __init__(self, master_key: bytes = b"v5_key_encoder_master",
                 method_name: str = "trustmark", n_bits: int = 100,
                 model_type: str = "B", device: str = "cuda",
                 detection_threshold: float = 0.75):
        from trustmark import TrustMark
        self.tm = TrustMark(verbose=False, model_type=model_type, use_ECC=False, secret_len=n_bits)
        self.master_key = master_key
        self.method_name = method_name
        self.n_bits = n_bits
        self.detection_threshold = detection_threshold
        self.codec = BCHCodec()

    def get_perm_M(self, image_id: str):
        return derive_method_keyed_constants(self.master_key, image_id, self.method_name, self.n_bits)

    def _codeword_target(self, image_id: str) -> np.ndarray:
        cw = self.codec.encode(image_id_to_payload(image_id, n_bits=self.codec.data_bits))[:self.n_bits]
        perm, M = self.get_perm_M(image_id)
        return apply_crypto(cw, perm, M)

    def embed_with_target(self, pil: Image.Image, target_bits: np.ndarray,
                           strength: float = 1.0) -> Image.Image:
        s = "".join(str(int(b)) for b in np.asarray(target_bits, np.uint8)[:self.n_bits])
        return self.tm.encode(pil.convert("RGB"), s, MODE="binary", WM_STRENGTH=strength)

    def embed(self, pil: Image.Image, image_id: str) -> Image.Image:
        return self.embed_with_target(pil, self._codeword_target(image_id))

    def raw_logits(self, pil: Image.Image) -> np.ndarray:
        """Continuous per-bit decoder output (sign = bit, |.| = confidence) -> soft-fusion ready."""
        r = self.tm.model_resolution_dec
        img = pil.convert("RGB").resize((r, r), Image.BILINEAR)
        x = transforms.ToTensor()(img).unsqueeze(0).to(self.tm.decoder.device) * 2.0 - 1.0
        with torch.no_grad():
            logits = self.tm.decoder.decoder(x).cpu().numpy().reshape(-1)[:self.n_bits]
        return logits.astype(np.float64)

    def detect(self, pil: Image.Image, image_id: str) -> dict:
        logits = self.raw_logits(pil)
        hard = (logits > 0).astype(np.float64)
        perm, M = self.get_perm_M(image_id)
        recovered = undo_crypto(hard, perm, M)
        expected = self.codec.encode(image_id_to_payload(image_id, n_bits=self.codec.data_bits))[:self.n_bits]
        bit_acc = float(np.mean(recovered == expected))
        return {"detected": bool(bit_acc >= self.detection_threshold), "bit_accuracy": bit_acc,
                "method": self.method_name}
