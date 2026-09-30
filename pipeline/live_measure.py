"""LIVE MEASUREMENT EXECUTOR — runs a real embed -> attack -> decode for an ARBITRARY solver config.

This is what turns a LIVE-VERIFY verdict (strength bracketing, unvalidated resolution, or a CEGAR
contradiction) into an actual number, and it is the same executor the external audit uses.

  measure(config, attack, images) -> {bit_acc, per_fragment, psnr, n, source:"live"}
    config = (frags, resync, nested, alpha, strengths)  -- exactly what the solver returns
    bit_acc uses BEST-PATH semantics (max over chosen fragments), matching the solver's Or(...) coverage.

HONEST BOUNDARY: attacks are split by what can run in-process.
  LIVE_OK   : signal / geometry / VAE  -> measurable synchronously here.
  BATCH_ONLY: regen, rinse, ctrlregen*, unmarker -> need the ctrlregen / unmarker conda envs and a GPU
              job, so they return None and the caller must fall back to the table (flagged, never faked).
"""
import sys, os, io, json, time
import numpy as np
from PIL import Image, ImageFilter
CF = "/data/tailor/project"
SC = "/data/tailor/workspace/wm_dataset10k"
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense"):
    sys.path.insert(0, p)

BATCH_ONLY = {"regen", "rinse", "ctrlregen", "ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07", "unmarker"}

# FRONT-END-DEPENDENT attacks: the solver models these as defended by a front-end (resync un-warps
# rotation; the nested ring + fine scale search recovers crop / UnMarker). This executor performs the
# EMBED side of the front-ends but decodes BARE -- it does not run the geo-cascade (SyncSeal un-warp,
# rotation refine, VINE scale search). Measuring these bare therefore reports the UN-recovered number,
# which would silently "verify" a config the solver accepted for a different reason. They are refused
# here (returned as unmeasurable) until the geo-cascade decode is wired in.
FE_DEPENDENT = {"rot9": "resync", "rot30": "resync",
                "crop90": "nested", "crop75": "nested|resync", "crop50": "nested"}

def _jpeg(p, q):
    b = io.BytesIO(); p.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def _crop(p, keep):
    w, h = p.size; s = keep ** 0.5
    cw, ch = int(w * s), int(h * s); x, y = (w - cw) // 2, (h - ch) // 2
    return p.crop((x, y, x + cw, y + ch)).resize((w, h), Image.BICUBIC)
def _rot(p, deg):
    w, h = p.size
    return p.rotate(deg, resample=Image.BICUBIC, expand=False).resize((w, h), Image.BICUBIC)
def _noise(p, sd, seed=0):
    a = np.asarray(p, np.float32) + np.random.RandomState(seed).normal(0, sd, np.asarray(p).shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def _bright(p, k):
    return Image.fromarray(np.clip(np.asarray(p, np.float32) * k, 0, 255).astype(np.uint8))
def _contrast(p, k):
    a = np.asarray(p, np.float32); m = a.mean()
    return Image.fromarray(np.clip((a - m) * k + m, 0, 255).astype(np.uint8))
ATTACKS = {
    "clean":    lambda p: p,
    "jpeg25":   lambda p: _jpeg(p, 25),   "jpeg50": lambda p: _jpeg(p, 50),
    "blur":     lambda p: p.filter(ImageFilter.GaussianBlur(2.0)),
    "noise":    lambda p: _noise(p, 12.75),
    "bright":   lambda p: _bright(p, 0.5), "contrast": lambda p: _contrast(p, 0.5),
    "crop90":   lambda p: _crop(p, 0.90), "crop75": lambda p: _crop(p, 0.75), "crop50": lambda p: _crop(p, 0.50),
    "rot9":     lambda p: _rot(p, 9),     "rot30":  lambda p: _rot(p, 30),
}

class LiveMeasurer:
    """Loads the fragments once; measures any (config, attack) on real images."""
    DEFAULT_S = {"VINE": 1.0, "TrustMark": 1.0, "VideoSeal": 1.0}

    def __init__(self, device="cuda", n_images=24, cache_path=f"{SC}/live_cache.json"):
        from src.vine_crypto_wrapper import VineCryptoWrapper
        from src.trustmark_fragment import TrustMarkFragment
        from src.videoseal_fragment import VideoSealFragment
        from src.shortened_bch import ShortenedBCH
        from composite_external_eval import scale_resid, nested_vine_embed
        import glob
        self.scale_resid, self.nested_vine_embed = scale_resid, nested_vine_embed
        self.sb = ShortenedBCH(); self.NB = self.sb.n
        K = b"v5_key_encoder_master"
        self.FR = {"VINE": VineCryptoWrapper(K, "vine", self.NB, device, variant="R"),
                   "TrustMark": TrustMarkFragment(K, "trustmark", self.NB, model_type="B", device=device),
                   "VideoSeal": VideoSealFragment(K, "videoseal", self.NB, device=device)}
        self.imgs = sorted(glob.glob(f"{SC}/pool/*/img/*.png"))[:n_images]
        self.n = len(self.imgs)
        self.cache_path = cache_path
        self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        self.vae = {}

    def _hard(self, fn, pil):
        f = self.FR[fn]
        v = np.asarray(f.raw_probs(pil) if fn == "VINE" else f.raw_logits(pil)).ravel()[:self.NB]
        return (v > (0.5 if fn == "VINE" else 0.0)).astype(np.uint8)

    def _vae(self, pil, which):
        import torch
        from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor
        if which not in self.vae:
            m = (bmshj2018_hyperprior if which == "B" else cheng2020_anchor)(quality=3, pretrained=True)
            self.vae[which] = m.eval().to("cuda")
        x = torch.from_numpy(np.asarray(pil, np.float32).transpose(2, 0, 1)[None] / 255.).to("cuda")
        with torch.no_grad(): y = self.vae[which](x)["x_hat"].clamp(0, 1)
        return Image.fromarray((y[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

    def _embed(self, cover, frags, nested, strengths, order):
        """Embed the exact config through each fragment's NATIVE strength knob, in `order`."""
        img = cover; tgts = {}; strengths = strengths or {}
        for f in (order or list(frags)):
            if f not in frags: continue
            t = np.random.RandomState(abs(hash(f)) % 2**31).randint(0, 2, self.NB).astype(np.uint8)
            tgts[f] = t
            s = float(strengths.get(f, self.DEFAULT_S.get(f, 1.0)))
            if f == "VINE" and nested:
                img = self.nested_vine_embed(self.FR["VINE"], img, t, scales=(1.0, 0.75, 0.5), strength=s)
            else:
                img = self.FR[f].embed_with_target(img, t, strength=s)
            if img.size != cover.size: img = img.resize(cover.size)
        return img, tgts

    def measure(self, frags, attack, resync=False, nested=False, alpha=0.7, strengths=None, order=None):
        if attack in BATCH_ONLY:
            return None                                   # needs a cross-env GPU job; caller falls back
        if attack in FE_DEPENDENT and (resync or nested):
            return None                                   # bare decode would understate a front-end config
        # unified per-fragment strengths map: alpha is folded in as VINE's native strength only when
        # the caller hasn't already given VINE its own entry (back-compat for alpha-only callers).
        # Computed BEFORE the cache key so the key reflects exactly what _embed will receive.
        strengths = {**({'VINE': alpha} if 'VINE' not in (strengths or {}) else {}), **(strengths or {})}
        sstr = "_".join(f"{k}:{round(float(v), 4)}" for k, v in sorted(strengths.items()))
        ostr = ">".join(order or sorted(frags))
        # "v2|" version tag: pre-change cache entries were computed under the old post-hoc scale_resid
        # _embed (VINE-uses-alpha, TrustMark/VideoSeal hardcoded 0.70) and are no longer valid; the tag
        # plus the strengths/order hash guarantees stale v1 entries can never collide with a v2 lookup.
        key = f"v2|{'+'.join(sorted(frags))}|{attack}|r{int(resync)}|n{int(nested)}|s{sstr}|o{ostr}|{self.n}"
        if key in self.cache: return self.cache[key]
        af = ATTACKS.get(attack) or (lambda p: self._vae(p, attack[-1])) if attack in ("vaeB", "vaeC") else ATTACKS.get(attack)
        if af is None: return None
        per = {f: [] for f in frags}; ps = []; t0 = time.time()
        for fp in self.imgs:
            cover = Image.open(fp).convert("RGB").resize((512, 512))
            emb, tgts = self._embed(cover, frags, nested, strengths, order)
            mse = np.mean((np.asarray(cover, np.float32)/255 - np.asarray(emb, np.float32)/255) ** 2)
            ps.append(99.0 if mse < 1e-12 else float(10*np.log10(1.0/mse)))
            att = af(emb)
            for f in frags: per[f].append(float(np.mean(self._hard(f, att) == tgts[f])))
        res = {"bit_acc": max(float(np.mean(v)) for v in per.values()),      # best-path, matches the solver
               "per_fragment": {f: round(float(np.mean(v)), 4) for f, v in per.items()},
               "psnr": round(float(np.mean(ps)), 2), "n": self.n,
               "ms_per_image": round((time.time()-t0)*1000/max(1,self.n), 1), "source": "live"}
        self.cache[key] = res; json.dump(self.cache, open(self.cache_path, "w"), indent=1)
        return res
