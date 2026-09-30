"""VideoSeal as a fusible codeword fragment -- the GEOMETRY fragment (drop-in replacement
for TrustMark). Meta VideoSeal (github.com/facebookresearch/videoseal) is a pixel-space
neural watermark that is dramatically more rotation/translation robust than TrustMark
(measured: VS-only vs TM rot10/rot90 0.12/0.06 -> 1.00/1.00 with raw bit-acc ~0.99;
border0.2 0.00 -> 0.88; FPR-verified 0). Like all pixel marks it is regen-dead, so VINE
remains the regeneration fragment. Mirrors TrustMarkFragment / VineCryptoWrapper so it
drops straight into the soft-fusion composite and TiledTrustMark.

Setup: needs external/videoseal on path + `pip install av`; the model card resolves config
paths relative to the repo root, so load() is wrapped in a temporary chdir.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
from PIL import Image

from src.payload import BCHCodec, image_id_to_payload
from src.vine_crypto_wrapper import apply_crypto, derive_method_keyed_constants, undo_crypto

_VSEAL_REPO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "external", "videoseal")


class VideoSealFragment:
    def __init__(self, master_key: bytes = b"v5_key_encoder_master", method_name: str = "videoseal",
                 n_bits: int = 100, card: str = "videoseal", device: str = "cuda",
                 detection_threshold: float = 0.75):
        if _VSEAL_REPO not in sys.path:
            sys.path.insert(0, _VSEAL_REPO)
        import videoseal
        cwd = os.getcwd()
        try:                                   # card configs are repo-root-relative
            os.chdir(_VSEAL_REPO)
            model = videoseal.load(card)
        finally:
            os.chdir(cwd)
        self.model = model.to(device).eval()
        self.device = device
        self.master_key = master_key
        self.method_name = method_name
        self.n_bits = n_bits
        self.msg_len = int(self.model.get_random_msg().shape[-1])
        self.detection_threshold = detection_threshold
        self.codec = BCHCodec()

    # ---- crypto / payload (mirrors TrustMarkFragment) ----
    def get_perm_M(self, image_id: str):
        return derive_method_keyed_constants(self.master_key, image_id, self.method_name, self.n_bits)

    def _codeword_target(self, image_id: str) -> np.ndarray:
        cw = self.codec.encode(image_id_to_payload(image_id, n_bits=self.codec.data_bits))[:self.n_bits]
        perm, M = self.get_perm_M(image_id)
        return apply_crypto(cw, perm, M)

    # ---- image <-> tensor (VideoSeal: [B,3,H,W] in [0,1], 512) ----
    def _to_t(self, pil: Image.Image) -> torch.Tensor:
        a = np.asarray(pil.convert("RGB").resize((512, 512)), np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _to_pil(self, t: torch.Tensor) -> Image.Image:
        a = (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(a)

    # ---- embed ----
    def embed_with_target(self, pil: Image.Image, target_bits: np.ndarray,
                           strength: float = 1.0) -> Image.Image:
        msg = torch.zeros(1, self.msg_len, device=self.device)
        msg[0, :self.n_bits] = torch.tensor(np.asarray(target_bits, np.float32)[:self.n_bits], device=self.device)
        # NOTE: the top-level model has no `scaling_w` of its own -- Wam/Videoseal only
        # forward the ctor arg into `self.blender = Blender(scaling_i, scaling_w, ...)`
        # (external/videoseal/videoseal/models/wam.py:61); the value actually consulted at
        # inference is `self.blender.scaling_w` (blender.py:68, every official eval script
        # sets `model.blender.scaling_w`). getattr(self.model, "scaling_w", None) is always
        # None, so scale the blender directly; fall back to a top-level attribute if some
        # future card variant exposes one there instead.
        blender = getattr(self.model, "blender", None)
        if blender is not None and hasattr(blender, "scaling_w"):
            holder = blender
        elif hasattr(self.model, "scaling_w"):
            holder = self.model
        else:
            holder = None
        old = getattr(holder, "scaling_w", None) if holder is not None else None
        if old is not None:
            holder.scaling_w = float(old) * float(strength)
        try:
            with torch.no_grad():
                out = self.model.embed(self._to_t(pil), msgs=msg, is_video=False)
        finally:
            if old is not None:
                holder.scaling_w = old
        return self._to_pil(out["imgs_w"])

    def embed(self, pil: Image.Image, image_id: str) -> Image.Image:
        return self.embed_with_target(pil, self._codeword_target(image_id))

    # ---- decode ----
    def raw_logits(self, pil: Image.Image) -> np.ndarray:
        """Per-bit decoder logits for the first n_bits (sign = bit, |.| = confidence)."""
        with torch.no_grad():
            preds = self.model.detect(self._to_t(pil), is_video=False)["preds"]   # F x (1+K)
        return preds[:, 1:][0, :self.n_bits].cpu().numpy().astype(np.float64)

    def detect(self, pil: Image.Image, image_id: str) -> dict:
        logits = self.raw_logits(pil)
        hard = (logits > 0).astype(np.float64)
        perm, M = self.get_perm_M(image_id)
        recovered = undo_crypto(hard, perm, M)
        expected = self.codec.encode(image_id_to_payload(image_id, n_bits=self.codec.data_bits))[:self.n_bits]
        ba = float(np.mean(recovered == expected))
        return {"detected": bool(ba >= self.detection_threshold), "bit_accuracy": ba, "method": self.method_name}


if __name__ == "__main__":
    import glob
    REPO = os.path.dirname(os.path.dirname(__file__)); sys.path.insert(0, REPO)
    from src.shortened_bch import ShortenedBCH
    sb = ShortenedBCH()
    f = VideoSealFragment(n_bits=sb.n)
    print(f"loaded VideoSealFragment: msg_len={f.msg_len} n_bits={f.n_bits}")
    def rot(im, d): return im.rotate(d, resample=Image.BILINEAR)
    imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
            sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:184]
    clean, r90, fp = [], [], []
    for j, p in enumerate(imgs):
        iid = f"sw_{180+j:05d}"; orig = Image.open(p).convert("RGB").resize((512, 512))
        cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); perm, M = f.get_perm_M(iid)
        tgt = apply_crypto(cw, perm, M)
        wm = f.embed_with_target(orig, tgt)
        clean.append(float(np.mean(undo_crypto((f.raw_logits(wm) > 0).astype(np.uint8), perm, M) == cw)))
        r90.append(float(np.mean(undo_crypto((f.raw_logits(rot(wm, 90)) > 0).astype(np.uint8), perm, M) == cw)))
        nid = f"neg_{j:05d}"; pn, Mn = f.get_perm_M(nid); cwn = sb.encode(image_id_to_payload(nid, n_bits=sb.data_bits))
        fp.append(float(np.mean(undo_crypto((f.raw_logits(orig) > 0).astype(np.uint8), pn, Mn) == cwn)))  # unwatermarked
    print(f"clean bit-acc {np.mean(clean):.3f} (want ~1.0)  rot90 bit-acc {np.mean(r90):.3f} (want ~1.0)  unwm bit-acc {np.mean(fp):.3f} (want ~0.5)")
    print("VSFRAG_SELFTEST_DONE")
