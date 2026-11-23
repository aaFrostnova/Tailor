"""Baseline watermark method wrappers (reliable tier).

Each wrapper exposes:
  .name         str
  .n_bits       int   (native payload capacity)
  .embed(pil, bits)   -> watermarked PIL (RGB)
  .decode(pil)        -> np.ndarray of recovered bits (len n_bits)

Imports are lazy/guarded so a missing package does not break the others.
APIs are verified by smoke tests before the full run; adjust here if a smoke
test reveals a different signature.
"""
from __future__ import annotations
import os

import numpy as np
from PIL import Image


# ----------------------------------------------------------- invisible-watermark
class InvisibleWMMethod:
    """dwtDct / dwtDctSvd / rivaGan from the invisible-watermark package."""

    def __init__(self, method: str):
        self.method = method
        self.name = {"dwtDct": "DwtDct", "dwtDctSvd": "DwtDctSvd", "rivaGan": "RivaGAN"}[method]
        self.n_bits = 32  # reliable capacity for all three
        import cv2  # noqa
        from imwatermark import WatermarkEncoder, WatermarkDecoder
        self._cv2 = cv2
        self._Enc = WatermarkEncoder
        self._Dec = WatermarkDecoder
        if method == "rivaGan":
            WatermarkEncoder.loadModel()
            WatermarkDecoder.loadModel()

    def embed(self, pil, bits):
        bgr = self._cv2.cvtColor(np.asarray(pil.convert("RGB")), self._cv2.COLOR_RGB2BGR)
        enc = self._Enc()
        enc.set_watermark("bits", list(int(b) for b in bits))
        wm_bgr = enc.encode(bgr, self.method)
        rgb = self._cv2.cvtColor(wm_bgr, self._cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def decode(self, pil):
        bgr = self._cv2.cvtColor(np.asarray(pil.convert("RGB")), self._cv2.COLOR_RGB2BGR)
        dec = self._Dec("bits", self.n_bits)
        rec = dec.decode(bgr, self.method)
        return np.asarray(rec, dtype=np.uint8)[: self.n_bits]


# ----------------------------------------------------------- TrustMark
class TrustMarkMethod:
    def __init__(self, model_type="Q"):
        from trustmark import TrustMark
        self.name = f"TrustMark-{model_type}"
        self.tm = TrustMark(verbose=False, model_type=model_type)
        # capacity queried at runtime in smoke; default schema ~100 data bits
        try:
            self.n_bits = int(self.tm.schemaCapacity())
        except Exception:
            self.n_bits = 100

    def embed(self, pil, bits):
        s = "".join(str(int(b)) for b in bits)
        return self.tm.encode(pil.convert("RGB"), s, MODE="binary")

    def decode(self, pil):
        out = self.tm.decode(pil.convert("RGB"), MODE="binary")
        s = out[0] if isinstance(out, (tuple, list)) else out
        if not s:
            return np.zeros(self.n_bits, dtype=np.uint8)
        bits = np.array([int(c) for c in s if c in "01"], dtype=np.uint8)
        if len(bits) < self.n_bits:
            bits = np.concatenate([bits, np.zeros(self.n_bits - len(bits), np.uint8)])
        return bits[: self.n_bits]


# ----------------------------------------------------------- VINE (raw, no crypto)
class VineMethod:
    """Raw VINE-R or VINE-B: SD-Turbo encoder + ConvNeXt decoder, 100-bit, ref-free."""

    def __init__(self, variant="R", device="cuda", resolution=256):
        import os
        import sys
        from src.paths import VINE_REPO
        sys.path.insert(0, VINE_REPO)
        sys.path.insert(0, os.path.join(VINE_REPO, "vine", "src"))
        self.name = f"VINE-{variant}"
        self.n_bits = 100
        self.device = device
        self.resolution = resolution
        self._variant = variant
        self._enc = None
        self._dec = None

    def _load(self):
        if self._enc is None:
            from vine_turbo import VINE_Turbo
            from stega_encoder_decoder import CustomConvNeXt
            tag = "VINE-R" if self._variant == "R" else "VINE-B"
            self._enc = VINE_Turbo.from_pretrained(f"Shilin-LU/{tag}-Enc").to(self.device).eval()
            self._dec = CustomConvNeXt.from_pretrained(f"Shilin-LU/{tag}-Dec").to(self.device).eval()

    def embed(self, pil, bits):
        import torch
        from torchvision import transforms
        self._load()
        size = pil.size
        t = transforms.Compose([transforms.Resize(self.resolution, antialias=True),
                                transforms.CenterCrop(self.resolution), transforms.ToTensor()])
        x = (t(pil.convert("RGB")).unsqueeze(0).to(self.device) * 2 - 1)
        secret = torch.tensor(np.asarray(bits, np.float32)[None], device=self.device)
        with torch.no_grad():
            enc = self._enc(x, secret=secret)
        resid = transforms.Resize(size[::-1], antialias=True)(enc - x)
        orig = transforms.ToTensor()(pil.convert("RGB")).unsqueeze(0).to(self.device) * 2 - 1
        out = torch.clamp((resid + orig) * 0.5 + 0.5, 0, 1)
        return transforms.ToPILImage()(out[0].cpu())

    def decode(self, pil):
        import torch
        from torchvision import transforms
        self._load()
        t = transforms.Compose([transforms.Resize(self.resolution, antialias=True),
                                transforms.CenterCrop(self.resolution), transforms.ToTensor()])
        x = t(pil.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self._dec(x)[0].cpu().numpy()   # CustomConvNeXt outputs sigmoid probs
        return (probs > 0.5).astype(np.uint8)[: self.n_bits]


def build_methods(names, device="cuda"):
    """Instantiate requested methods; skip (with message) any that fail to load."""
    reg = {}
    specs = {
        "dwtDct": lambda: InvisibleWMMethod("dwtDct"),
        "dwtDctSvd": lambda: InvisibleWMMethod("dwtDctSvd"),
        "rivaGan": lambda: InvisibleWMMethod("rivaGan"),
        "trustmark": lambda: TrustMarkMethod("Q"),
        "trustmark_b": lambda: TrustMarkMethod("B"),
        "vine_r": lambda: VineMethod("R", device),
        "vine_b": lambda: VineMethod("B", device),
    }
    for n in names:
        try:
            reg[n] = specs[n]()
            print(f"[methods] loaded {n} -> {reg[n].name} (n_bits={reg[n].n_bits})", flush=True)
        except Exception as e:
            print(f"[methods] SKIP {n}: {type(e).__name__}: {e}", flush=True)
    return reg
