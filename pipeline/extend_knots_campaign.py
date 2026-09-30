"""Range extension: the three fragments were swept over three different distortion intervals (VINE 34.8
to 47.3 dB, TrustMark 37.1 to 47.5, VideoSeal 41.6 to 50.6), so a comparison between them is only
measured on 41.6 to 47.3 dB. 27.1% of the solved requests sit above 47.3 dB, where only VideoSeal has
data, and VideoSeal's curves are still climbing at its 41.6 dB edge (vaeC slope +0.19), where it was
called infeasible for want of measurement. This sweeps EXTRA knots so all three cover about 35 to 50 dB:

    VideoSeal  1.75 2.0 2.25 2.5 2.8 3.2   (40.3 down to 35.4 dB, from the fitted d(s))
    VINE       0.14 0.17                    (49.6, 48.5 dB; 0.14 is its measured presence floor)
    TrustMark  1.8 2.0                      (36.1, 35.2 dB)

Same measurement as perimage_campaign.py's inprocess mode (composite crypto codeword, per-image ba and
verify), on the ext9 slice, for the 14 in-process columns plus regen and rinse2x, with the per-image
MSE so the distortion curve d(f) extends with the same knots.
Usage: python extend_knots_campaign.py <frag> <strength>
"""
import sys, os, json, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT as CF, WORKSPACE as SC, WMATTACKER_REPO as WMA
for p in (CF, f"{CF}/solver", WMA, SC): sys.path.insert(0, p)
if len(sys.argv) < 3:
    sys.exit("usage: extend_knots_campaign.py {VINE|TrustMark|VideoSeal} <strength>")
F = sys.argv[1]; s = float(sys.argv[2]); sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
from eval_matrix import OursComposite
from src.attacks import attack_pil_any
from src.image_pool import sample as _pool_sample, composition_of
from src.soft_bch import decode_and_verify
FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}
NEW_KNOTS = {"VideoSeal": [1.75, 2.0, 2.25, 2.5, 2.8, 3.2], "VINE": [0.14, 0.17], "TrustMark": [1.8, 2.0]}
assert s in NEW_KNOTS[F], (F, s)
COLS = ["jpeg25", "blur", "noise", "bright", "contrast", "crop75", "crop50", "rot9", "rs256", "hflip",
        "crop_jpeg", "border20", "vaeB", "vaeC", "regen", "rinse2x"]
dev = "cuda"
OUT = f"{SC}/perimage_ext/inprocess"; os.makedirs(OUT, exist_ok=True)
files = _pool_sample(100, offset=0)
covers = [Image.open(f).convert("RGB").resize((512, 512), Image.BICUBIC) for f in files]
comp = OursComposite(dev, tm_variant="B", vine_variant="R",
                     config={"frags": [FKEY[F]], "order": [FKEY[F]], "strengths": {},
                             **W.frontend_config({"resync": False, "scale": False, "angle": False, "tile": False})})
comp.strength = dict(comp.DEFAULT_STRENGTH); comp.strength[FKEY[F]] = float(s)
def r512(img): return img if img.size == (512, 512) else img.resize((512, 512))
def mse(a, b): return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
print(f"[extend] {F} @ {s}: {len(covers)} images {composition_of(files)}, {len(COLS)} columns", flush=True)
ba = {a: [] for a in COLS}; ver = {a: [] for a in COLS}; clean_ba, dist = [], []
t0 = time.time()
for i, cov in enumerate(covers):
    emb, sec = comp.embed(cov, i); iid, tx = sec; emb = r512(emb)
    dist.append(mse(cov, emb))
    L0 = comp._frag_llr(FKEY[F], emb, iid); clean_ba.append(float(np.mean((L0 > 0).astype(np.uint8) == tx)))
    for a in COLS:
        v = r512(attack_pil_any(a, emb, dev=dev))
        L = comp._frag_llr(FKEY[F], v, iid)
        ba[a].append(float(np.mean((L > 0).astype(np.uint8) == tx)))
        ver[a].append(bool(decode_and_verify(L, iid, codec=comp.sb)["detected"]))
    if (i + 1) % 10 == 0: print(f"  {i+1}/{len(covers)} [{time.time()-t0:.0f}s]", flush=True)
D = float(np.mean(dist))
json.dump({"frag": F, "strength": s, "n": len(covers), "images": "pool offset 0 (ext9 slice)", "embed": "composite crypto codeword",
           "clean_ba": clean_ba, "mse": dist, "mse_mean": D, "psnr_db": 10 * np.log10(255.0 ** 2 / max(D, 1e-9)),
           "ba": ba, "ver": ver, "means": {a: round(float(np.mean(ba[a])), 4) for a in COLS}},
          open(f"{OUT}/{F}_{s}.json", "w"))
print(f"  {F}@{s}: PSNR {10*np.log10(255.0**2/max(D,1e-9)):.2f} dB, clean ba {np.mean(clean_ba):.3f}; "
      + " ".join(f"{a}={np.mean(ba[a]):.2f}" for a in COLS), flush=True)
print("EXTEND_DONE", flush=True)
