"""How much does a fragment's robustness move when the IMAGES change?

The patch rule shrinks a live measurement toward the table by prior^2/(prior^2+se^2), where `prior` is
how far one image population's behaviour can sit from another's. That number cannot be guessed: too
small and real distribution shifts are dismissed as noise, too large and sampling noise is written into
the model as if it were a shift. It is measurable -- the pool is built from five distinct sources, so
the spread of per-source means IS the quantity, with the within-source standard error subtracted off so
that sampling noise is not counted twice.
"""
import sys, glob, json, os
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT as CF, WORKSPACE as SC, WMATTACKER_REPO as WMA
for p in (CF, f"{CF}/solver", WMA):
    sys.path.insert(0, p)
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH
from src.attacks import attack_pil_any as attack_pil

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
sb = ShortenedBCH(); NB = sb.n; KEY = b"v5_key_encoder_master"; dev = "cuda"
FR = {"VINE": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
      "TrustMark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
      "VideoSeal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}
MID = {"VINE": 0.6, "TrustMark": 1.0, "VideoSeal": 1.0}
ATT = ["jpeg25", "crop75", "rot9", "regen"]
SRC = sorted({p.split("/pool/")[1].split("/")[0] for p in glob.glob(f"{SC}/pool/*/img")})
print(f"sources: {SRC}   N={N} per source", flush=True)

def hard(fn, pil):
    f = FR[fn]; v = f.raw_probs(pil) if fn == "VINE" else f.raw_logits(pil)
    v = np.asarray(v).ravel()[:NB]
    return (v > (0.5 if fn == "VINE" else 0.0)).astype(np.uint8)

rows = []
for f in FR:
    for a in ATT:
        per_src = {}
        for sname in SRC:
            files = sorted(glob.glob(f"{SC}/pool/{sname}/img/*.png"))[:N]
            if len(files) < 5: continue
            bas = []
            for i, fp in enumerate(files):
                cov = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC)
                t = np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
                emb = FR[f].embed_with_target(cov, t, strength=MID[f])
                if emb.size != (512, 512): emb = emb.resize((512, 512))
                v = attack_pil(a, emb, dev=dev)
                if v.size != (512, 512): v = v.resize((512, 512))
                bas.append(float(np.mean(hard(f, v) == t)))
            per_src[sname] = (float(np.mean(bas)), float(np.std(bas, ddof=1)), len(bas))
        if len(per_src) < 2: continue
        means = np.array([v[0] for v in per_src.values()])
        ses = np.array([v[1] / np.sqrt(v[2]) for v in per_src.values()])
        observed = float(np.std(means, ddof=1))
        within = float(np.sqrt(np.mean(ses ** 2)))          # sampling noise inside each source
        between = float(np.sqrt(max(observed ** 2 - within ** 2, 0.0)))
        rows.append({"frag": f, "attack": a, "per_source": per_src,
                     "sd_of_means": observed, "within_se": within, "between_sd": between})
        print(f"  {f:10s} {a:9s} means " + " ".join(f"{v[0]:.3f}" for v in per_src.values())
              + f"   sd(means)={observed:.4f}  within={within:.4f}  BETWEEN={between:.4f}", flush=True)
b = np.array([r["between_sd"] for r in rows])
print(f"\n=== between-source sd over {len(rows)} cells ===")
print(f"  median {np.median(b):.4f}   mean {b.mean():.4f}   p90 {np.percentile(b,90):.4f}   max {b.max():.4f}")
print(f"  -> prior_sd for the patch rule: {np.median(b):.3f} (median) / {np.percentile(b,90):.3f} (p90, conservative)")
json.dump({"rows": rows, "n_per_source": N,
           "prior_sd_median": float(np.median(b)), "prior_sd_p90": float(np.percentile(b, 90))},
          open(f"{SC}/prior_width.json", "w"), indent=2)
print("PRIOR_WIDTH_DONE")
