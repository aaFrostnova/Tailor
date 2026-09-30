"""Measure the two solver-output dimensions that were formalized but unmeasured:
 (1) STRENGTH sweep for TrustMark and VideoSeal (VINE already has an alpha sweep): residual scale
     s in {0.3..1.0} -> PSNR + bit-acc under clean/jpeg/blur. Gives the TM/VS strength grids.
 (2) ORDER sweep: embed the 3-fragment composite in each of the 6 permutations at deployment strengths
     (VINE 1.0, TM/VS 0.70) and decode each fragment on the clean composite -> per-fragment bit-acc +
     composite PSNR. Shows the embed order changes the realized bit-acc (cross-fragment overwrite).
Uses each fragment's embed_with_target + raw_logits/raw_probs (crypto is a bijection, irrelevant to ba).
"""
import sys, os, io, json, itertools, glob
import numpy as np
from PIL import Image, ImageFilter
CF = "/data/tailor/project"
SC = "/data/tailor/workspace/wm_dataset10k"
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense"):
    sys.path.insert(0, p)
from composite_external_eval import scale_resid
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH

dev = "cuda"; KEY = b"v5_key_encoder_master"
sb = ShortenedBCH(); NB = sb.n
print("loading fragments...", flush=True)
vine = VineCryptoWrapper(KEY, "vine", NB, dev, variant="R")
tm = TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev)
vs = VideoSealFragment(KEY, "videoseal", NB, device=dev)
FR = {"VINE": vine, "TrustMark": tm, "VideoSeal": vs}
STR = {"VINE": 1.0, "TrustMark": 0.70, "VideoSeal": 0.70}   # deployment strengths (for the order sweep)
print("loaded.", flush=True)

def rng_bits(seed):
    return np.random.RandomState(seed).randint(0, 2, NB).astype(np.uint8)
def hard(fn, pil):
    f = FR[fn]
    if fn == "VINE":
        return (np.asarray(f.raw_probs(pil)).ravel()[:NB] > 0.5).astype(np.uint8)
    return (np.asarray(f.raw_logits(pil)).ravel()[:NB] > 0).astype(np.uint8)
def psnr(a, b):
    m = np.mean((np.asarray(a, np.float32)/255 - np.asarray(b, np.float32)/255)**2)
    return 99.0 if m < 1e-12 else float(10*np.log10(1.0/m))
def _jpeg(pil, q=50):
    buf = io.BytesIO(); pil.save(buf, "JPEG", quality=q); buf.seek(0); return Image.open(buf).convert("RGB")
ATT = {"clean": lambda x: x, "jpeg50": _jpeg, "blur": lambda x: x.filter(ImageFilter.GaussianBlur(2))}

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
files = sorted(glob.glob(f"{SC}/pool/*/img/*.png"))[:N]
print(f"n={len(files)} images", flush=True)
covers = [Image.open(fp).convert("RGB").resize((512, 512)) for fp in files]

# ---------- (1) STRENGTH SWEEP: TrustMark, VideoSeal ----------
STRENGTHS = [0.3, 0.5, 0.7, 0.9, 1.0]
strength = {}
for fn in ("TrustMark", "VideoSeal"):
    for s in STRENGTHS:
        ps, ba = [], {a: [] for a in ATT}
        for i, cover in enumerate(covers):
            tgt = rng_bits(i)
            wm = FR[fn].embed_with_target(cover, tgt)
            emb = scale_resid(cover, wm, s)
            if emb.size != (512, 512): emb = emb.resize((512, 512))
            ps.append(psnr(cover, emb))
            for a, af in ATT.items():
                ba[a].append(float(np.mean(hard(fn, af(emb)) == tgt)))
        strength[f"{fn}@{s}"] = {"psnr": round(float(np.mean(ps)), 2), **{a: round(float(np.mean(ba[a])), 3) for a in ATT}}
        print(f"  strength {fn}@{s}: PSNR {strength[f'{fn}@{s}']['psnr']}  clean {strength[f'{fn}@{s}']['clean']}  jpeg50 {strength[f'{fn}@{s}']['jpeg50']}  blur {strength[f'{fn}@{s}']['blur']}", flush=True)

# ---------- (2) ORDER SWEEP: 3-frag permutations, clean decode ----------
order = {}
for perm in itertools.permutations(["VINE", "TrustMark", "VideoSeal"]):
    ps, ba = [], {f: [] for f in perm}
    for i, cover in enumerate(covers):
        tgts = {f: rng_bits(i * 7 + j) for j, f in enumerate(perm)}
        img = cover
        for f in perm:
            wm = FR[f].embed_with_target(img, tgts[f])
            img = scale_resid(img, wm, STR[f])
            if img.size != (512, 512): img = img.resize((512, 512))
        ps.append(psnr(cover, img))
        for f in perm:
            ba[f].append(float(np.mean(hard(f, img) == tgts[f])))
    key = " -> ".join(perm)
    order[key] = {"psnr": round(float(np.mean(ps)), 2), **{f: round(float(np.mean(ba[f])), 3) for f in perm}}
    print(f"  order {key}: PSNR {order[key]['psnr']}  VINE {order[key]['VINE']}  TM {order[key]['TrustMark']}  VS {order[key]['VideoSeal']}", flush=True)

json.dump({"n": N, "strength_sweep": strength, "order_sweep": order,
           "note": "strength = residual scale (scale_resid); order at deployment strengths VINE 1.0 / TM 0.70 / VS 0.70"},
          open(f"{SC}/order_strength.json", "w"), indent=2)
print("wrote", f"{SC}/order_strength.json"); print("ORDER_STRENGTH_DONE")
