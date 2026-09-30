"""Crypto wrapper around MaskWM (Mask Image Watermarking, NeurIPS 2025, arXiv:2504.12739).

MaskWM-D is a post-hoc, pixel-space JND-modulated additive watermark with a U2Net
mask-predictor for reference-free localization. We wrap the D_128bits variant so it
carries the SAME shortened-BCH codeword as VINE/DFT/QIM (first ``n_bits`` of the 128-bit
message), keyed per image by (perm sigma, sign-mask M) -- exactly like VineCryptoWrapper.

Role in the composite: an INDEPENDENT geometric/localization OR-corroborator (TrustMark
sibling). MaskWM reports geometric bit-acc 0.9998 vs TrustMark 0.7868, and it is
reference-free -- so it removes TrustMark's single-point-of-failure on crop/resize.
It does NOT address the regeneration gap (never tested vs diffusion img2img; its only
generative-adjacent test is a VAE round-trip, which our stack already survives at ~1.0).

The decoder returns UNCALIBRATED per-bit scores (raw nn.Linear output, thresholded at
0.5, not zero-centered). ``raw_scores`` exposes them; for soft-LLR fusion they must be
mapped LLR ~= k*(score-0.5) and calibrated (temperature/Platt) first -- hence v1 use is
the independent bit-accuracy OR tier, not a fused LLR member.

P0 NOTE (from the integration review): Decoder.forward self-gates -- it multiplies the
image by (its own mask_pred > 0.5) before bit extraction. After a hostile attack a
mis-firing mask-predictor can zero the image and collapse the bits. For a GLOBAL
watermark we pass an all-ones mask (``use_self_mask=False``, the default here) to
decouple bit-decode from mask-prediction. Set ``use_self_mask=True`` to reproduce the
paper's decode path.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Reuse the exact crypto + payload machinery the other fragments use.
from src.payload import BCHCodec, image_id_to_payload
from src.vine_crypto_wrapper import (
    apply_crypto,
    derive_method_keyed_constants,
    undo_crypto,
)

_MASKWM_REPO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "external", "MaskWM")


class MaskWMWrapper:
    """Crypto wrapper around the MaskWM-D pretrained encoder/decoder (D_128bits)."""

    def __init__(self,
                 ckpt_path: str,
                 master_key: bytes = b"v5_key_encoder_master",
                 method_name: str = "maskwm",
                 n_bits: int = 100,
                 msg_len: int = 128,
                 jnd_factor: float = 1.3,
                 device: str = "cuda",
                 detection_threshold: float = 0.75,
                 use_self_mask: bool = False,
                 repo_path: str = _MASKWM_REPO,
                 config_name: str = "D_128bits"):
        assert n_bits <= msg_len, f"n_bits {n_bits} must fit in msg_len {msg_len}"
        self.ckpt_path = ckpt_path
        self.master_key = master_key
        self.method_name = method_name
        self.n_bits = n_bits
        self.msg_len = msg_len
        self.jnd_factor = jnd_factor
        self.device = device
        self.detection_threshold = detection_threshold
        self.use_self_mask = use_self_mask
        self.repo_path = repo_path
        self.config_name = config_name
        self.codec = BCHCodec()
        self._model = None

    # ---- model loading -------------------------------------------------
    def _load(self):
        if self._model is None:
            if self.repo_path not in sys.path:
                sys.path.insert(0, self.repo_path)
            from omegaconf import OmegaConf
            from models.Mask_Model import WatermarkModel
            cfg = OmegaConf.load(
                os.path.join(self.repo_path, "configs", "model", f"{self.config_name}.yaml")
            )
            assert cfg["wm_enc_config"]["message_length"] == self.msg_len, (
                f"config message_length {cfg['wm_enc_config']['message_length']} != msg_len {self.msg_len}"
            )
            model = WatermarkModel(**cfg)
            if self.ckpt_path is not None:
                state = torch.load(self.ckpt_path, map_location="cpu")
                model.load_state_dict(state, strict=True)
            self._model = model.to(self.device).eval()
        return self._model

    def load_random_for_smoketest(self):
        """Construct with random weights (no checkpoint) to validate the pipeline."""
        self.ckpt_path = None
        return self._load()

    # ---- crypto / payload (mirrors VineCryptoWrapper) ------------------
    def _codeword_target(self, image_id: str) -> np.ndarray:
        """First n_bits of the BCH codeword, crypto-keyed for this method."""
        payload = image_id_to_payload(image_id, n_bits=self.codec.data_bits)
        codeword = self.codec.encode(payload)[:self.n_bits]
        perm, M = self.get_perm_M(image_id)
        return apply_crypto(codeword, perm, M)  # float {0,1}, length n_bits

    def get_perm_M(self, image_id: str):
        return derive_method_keyed_constants(
            self.master_key, image_id, self.method_name, self.n_bits,
        )

    def _build_message(self, target_bits: np.ndarray) -> torch.Tensor:
        """Pad the n_bits crypto target to the full msg_len with zeros."""
        msg = np.zeros(self.msg_len, dtype=np.float32)
        msg[:self.n_bits] = np.asarray(target_bits, dtype=np.float32)
        return torch.tensor(msg, dtype=torch.float32).unsqueeze(0).to(self.device)

    # ---- image <-> tensor (MaskWM uses [-1,1], net runs at 256) --------
    def _to_tensor(self, pil: Image.Image) -> torch.Tensor:
        arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0  # [H,W,3] in [0,1]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return t * 2.0 - 1.0  # [-1,1]

    def _to_pil(self, t: torch.Tensor) -> Image.Image:
        arr = ((t[0].clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))

    # ---- embed ---------------------------------------------------------
    def embed_with_target(self, pil: Image.Image, target_bits: np.ndarray) -> Image.Image:
        model = self._load()
        message = self._build_message(target_bits)
        img = self._to_tensor(pil)
        H, W = img.shape[-2:]
        img256 = F.interpolate(img, size=[256, 256], mode="bilinear", align_corners=False)
        with torch.no_grad():
            wm256 = model.encoder(img256, message, use_jnd=True, jnd_factor=self.jnd_factor, blue=True)
        residual = F.interpolate(wm256 - img256, size=[H, W], mode="bilinear", align_corners=False)
        wm = (residual + img).clamp_(-1, 1)
        return self._to_pil(wm)

    def embed(self, pil: Image.Image, image_id: str) -> Image.Image:
        return self.embed_with_target(pil, self._codeword_target(image_id))

    # ---- decode --------------------------------------------------------
    def _decode_scores(self, pil: Image.Image):
        """Return (raw per-bit scores length msg_len, mask_pred tensor)."""
        model = self._load()
        img = self._to_tensor(pil)
        img256 = F.interpolate(img, size=[256, 256], mode="bilinear", align_corners=False)
        with torch.no_grad():
            if self.use_self_mask:
                scores, mask_pred = model.decoder(img256)
            else:  # P0: all-ones mask -> bypass U2Net self-gating for global watermark
                ones = torch.ones(img256.shape[0], 1, 256, 256, device=self.device)
                scores, mask_pred = model.decoder(img256, mask=ones)
        return scores[0].cpu().numpy(), mask_pred

    def raw_scores(self, pil: Image.Image) -> np.ndarray:
        """Uncalibrated per-bit scores for the n_bits codeword positions (~near {0,1})."""
        return self._decode_scores(pil)[0][:self.n_bits]

    def localization_map(self, pil: Image.Image) -> np.ndarray:
        """U2Net per-pixel watermark-presence map in [0,1] (256x256)."""
        mask_pred = self._decode_scores(pil)[1]
        return mask_pred[0, 0].cpu().numpy()

    def detect(self, pil: Image.Image, image_id: str) -> dict:
        scores = self.raw_scores(pil)  # length n_bits, threshold at 0.5
        perm, M = self.get_perm_M(image_id)
        recovered_cw = undo_crypto(scores, perm, M)
        payload = image_id_to_payload(image_id, n_bits=self.codec.data_bits)
        expected_cw = self.codec.encode(payload)[:self.n_bits]
        bit_acc = float(np.mean(recovered_cw == expected_cw))
        return {
            "detected": bool(bit_acc >= self.detection_threshold),
            "bit_accuracy": bit_acc,
            "method": self.method_name,
            "scores": scores.tolist(),  # uncalibrated; map LLR~=k*(s-0.5) before fusion
            "perm": perm.tolist(),
            "M": M.tolist(),
        }
