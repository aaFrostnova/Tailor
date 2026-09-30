"""Tiled-TrustMark: spatial-redundancy + self-sync enhancement borrowed from the
block-DWT method (Bulychev et al., WACV 2026) to close the composite's TRANSLATION-crop
gap at zero fidelity / zero FPR cost.

Why: in the VINE+TrustMark composite, VINE dies on ALL crops (latent shift), so crop
robustness rests entirely on TrustMark. Plain (global) TrustMark survives moderate crops
via its training aug but breaks under translation/border crops >=~20% (measured: border0.2
-> composite 0.00). Embedding the SAME crypto codeword redundantly in a g x g tile grid,
plus a sliding-window crypto-verify decode, makes ANY surviving aligned tile recover the
full codeword -- exactly the block-DWT "any block detects" property.

Detection cascade (all paths FPR-safe; crop path uses 37-bit exact crypto match):
  1. fused = VINE_global_LLR + sum(canonical-tile TM LLRs)   (FIXED tile positions; no
     max-of-many selection -> no FPR inflation). crypto-verify(fused) OR fused ba>=tau.
  2. fallback (only if 1 fails): sliding-window crypto-verify -- ANY window whose TM
     decode Chase-corrects to the exact 37-bit payload => detected.

Validated (scripts/defense/tiled_tm_v3.py, n=16): border0.1/0.2/0.3 0.69/0.00/0.00 ->
1.00/1.00/1.00; clean/jpeg/regen unchanged; PSNR cost -0.1dB; FPR 0/16 (== plain).
Zoom crops (cresize/corner 2x) stay ~0 -- an information-theoretic content-loss limit,
not addressable by tiling (the source paper itself gets only 76-91% at crop-50%).
"""
from __future__ import annotations
import numpy as np
from PIL import Image

from src.soft_fusion import method_soft_to_codeword_llr
from src.soft_bch import decode_and_verify


class TiledTrustMark:
    def __init__(self, tm, sb, G: int = 2, step: int = 32, clamp: float = 15.0):
        self.tm, self.sb, self.G, self.step, self.clamp = tm, sb, G, step, clamp
        self.cell = 512 // G
        self.offsets = list(range(0, 512 - self.cell + 1, step))           # sliding-window grid
        self.canon = [(ox, oy) for oy in range(0, 512, self.cell) for ox in range(0, 512, self.cell)]

    @staticmethod
    def _to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im

    # ---- embed: same crypto target into every g x g cell ----
    def embed(self, base_pil: Image.Image, target_bits: np.ndarray, strength: float = 1.0) -> Image.Image:
        out = np.asarray(self._to512(base_pil), np.float64).copy()
        c = self.cell
        for gy in range(self.G):
            for gx in range(self.G):
                cell = Image.fromarray(out[gy*c:(gy+1)*c, gx*c:(gx+1)*c].astype(np.uint8))
                wm = self.tm.embed_with_target(cell, target_bits, strength=strength)
                wm = wm.resize((c, c)) if wm.size != (c, c) else wm
                out[gy*c:(gy+1)*c, gx*c:(gx+1)*c] = np.asarray(wm, np.float64)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def _win_llr(self, att512, ox, oy, perm, M):
        win = att512.crop((ox, oy, ox + self.cell, oy + self.cell))
        return np.clip(method_soft_to_codeword_llr(self.tm.raw_logits(win), perm, M, kind="logit",
                                                   n_codeword=self.sb.n), -self.clamp, self.clamp)

    def canonical_llr(self, att_pil, perm, M) -> np.ndarray:
        """Sum of TM LLRs at the FIXED tile positions (the FPR-safe fused-path TM term)."""
        att = self._to512(att_pil); s = np.zeros(self.sb.n)
        for ox, oy in self.canon:
            s += self._win_llr(att, ox, oy, perm, M)
        return s

    def crop_recover(self, att_pil, image_id, perm, M, return_view: bool = False):
        """Crop-recovery fallback: True iff ANY sliding window crypto-verifies (37-bit exact).

        With `return_view` the verified window is handed back as well, so the caller can report the
        accuracy of the view it actually accepted rather than that of the un-shifted frame."""
        att = self._to512(att_pil)
        for oy in self.offsets:
            for ox in self.offsets:
                llr = self._win_llr(att, ox, oy, perm, M)
                if bool(decode_and_verify(llr, image_id, codec=self.sb)["detected"]):
                    if return_view:
                        return True, att.crop((ox, oy, ox + self.cell, oy + self.cell))
                    return True
        return (False, None) if return_view else False

    def detect(self, att_pil, image_id, perm, M, vine_llr_global: np.ndarray, tau: float) -> bool:
        """Full cascade. vine_llr_global = aligned VINE codeword LLR on the (attacked) image."""
        fused = np.asarray(vine_llr_global, np.float64) + self.canonical_llr(att_pil, perm, M)
        if bool(decode_and_verify(fused, image_id, codec=self.sb)["detected"]):
            return True
        if float(np.mean((fused > 0).astype(np.uint8) == self._expected(image_id))) >= tau:
            return True
        return self.crop_recover(att_pil, image_id, perm, M)   # slow path only on failure

    def _expected(self, image_id):
        from src.payload import image_id_to_payload
        return self.sb.encode(image_id_to_payload(image_id, n_bits=self.sb.data_bits)).astype(np.uint8)


if __name__ == "__main__":
    import os, sys, glob
    REPO = os.path.dirname(os.path.dirname(__file__)); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
    from src.shortened_bch import ShortenedBCH
    from src.payload import image_id_to_payload
    from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
    from src.trustmark_fragment import TrustMarkFragment
    from scipy.stats import binom
    KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70
    sb = ShortenedBCH(); tau = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
    tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
    tt = TiledTrustMark(tm, sb)
    def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
    def sr(c, w, a): C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64); return Image.fromarray(np.clip(C + a*(W-C), 0, 255).astype(np.uint8))
    def border(im, p): d = int(round(512*p)); reg = im.crop((d, d, 512, 512)); cv = Image.new("RGB", (512, 512)); cv.paste(reg, (0, 0)); return cv
    def vllr(att, pv, Mv): return np.clip(method_soft_to_codeword_llr(vine.raw_probs(to512(att)), pv, Mv, kind="prob", n_codeword=sb.n), -15, 15)
    imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) + sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:188]
    clean_ok = border_ok = fp = 0
    for j, fp_ in enumerate(imgs):
        iid = f"sw_{180+j:05d}"; orig = to512(Image.open(fp_).convert("RGB"))
        cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)); pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
        v = sr(orig, to512(vine.embed_with_target(orig, apply_crypto(cw, pv, Mv))), ALPHA)
        vt = sr(v, tt.embed(v, apply_crypto(cw, pt, Mt)), ALPHA)
        clean_ok += tt.detect(vt, iid, pt, Mt, vllr(vt, pv, Mv), tau)
        b = border(vt, 0.2); border_ok += tt.detect(b, iid, pt, Mt, vllr(b, pv, Mv), tau)
        nid = f"neg_{j:05d}"; pvn, Mvn = vine.get_perm_M(nid); ptn, Mtn = tm.get_perm_M(nid)
        fp += tt.detect(orig, nid, ptn, Mtn, vllr(orig, pvn, Mvn), tau)        # unwatermarked
    print(f"[tiled-TM selftest n={len(imgs)}] clean detect {clean_ok}/{len(imgs)}  border0.2 detect {border_ok}/{len(imgs)}  FPR {fp}/{len(imgs)}")
    print("TILED_TM_SELFTEST_DONE")
