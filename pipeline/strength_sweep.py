"""Measure per-fragment detection across ATTACK STRENGTH sweeps (jpeg-Q, blur-sigma, noise-sigma,
crop-frac). Used to build the 3D PSNR-cost surface: at each (attack, strength) the frontier picks
the lightest (highest-PSNR) config that still defends -> stronger attack -> heavier config -> lower PSNR."""
import sys, os, glob, io, json
import numpy as np
from PIL import Image, ImageFilter
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT as REPO, WORKSPACE as SC
sys.path.insert(0, f"{REPO}/solver")
from wbench.methods import VineMethod, TrustMarkMethod

dev = "cuda"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
print("loading fragments...", flush=True)
FR = {"VINE-B": VineMethod("B", dev), "VINE-R": VineMethod("R", dev), "TrustMark": TrustMarkMethod("B")}
TAU = {k: float(binom.ppf(0.99, m.n_bits, 0.5) + 1) / m.n_bits for k, m in FR.items()}
print("loaded.", flush=True)

def a_jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=int(q)); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
def a_blur(im, s): return im.filter(ImageFilter.GaussianBlur(float(s)))
def a_noise(im, s):
    a = np.asarray(im, np.float64) + np.random.RandomState(0).normal(0, float(s)*255, (512, 512, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def a_crop(im, f):
    s = int(512*float(f)); o = (512-s)//2; return im.crop((o, o, o+s, o+s)).resize((512, 512))
SWEEPS = {"jpeg": (a_jpeg, [90, 70, 50, 30, 10]), "blur": (a_blur, [1, 2, 3, 4, 5]),
          "noise": (a_noise, [0.02, 0.05, 0.10, 0.15, 0.20]), "crop": (a_crop, [0.9, 0.75, 0.6, 0.5, 0.4])}

files = sorted(glob.glob(f"{SC}/pool/*/img/*.png"))[:N]
covers = [Image.open(f).convert("RGB").resize((512, 512)) for f in files]
det = {k: {a: {} for a in SWEEPS} for k in FR}
for fk, m in FR.items():
    bits = [np.random.RandomState(1000+i).randint(0, 2, m.n_bits).astype(np.uint8) for i in range(N)]
    embs = [m.embed(c, b) for c, b in zip(covers, bits)]
    embs = [e.resize((512, 512)) if e.size != (512, 512) else e for e in embs]
    for aname, (afn, levels) in SWEEPS.items():
        for lv in levels:
            hits = []
            for e, b in zip(embs, bits):
                at = afn(e, lv)
                if at.size != (512, 512): at = at.resize((512, 512))
                rec = np.asarray(m.decode(at)).astype(np.uint8); n = min(len(rec), len(b))
                hits.append(float(np.mean(rec[:n] == b[:n]) >= TAU[fk]))
            det[fk][aname][str(lv)] = float(np.mean(hits))
    print(f"  {fk} swept", flush=True)
json.dump({"det": det, "sweeps": {k: v[1] for k, v in SWEEPS.items()}}, open(f"{SC}/strength_sweep.json", "w"), indent=2)
print("=== per-fragment detection across strength ===")
for aname in SWEEPS:
    print(f"\n{aname}  levels={SWEEPS[aname][1]}")
    for fk in FR: print(f"  {fk:10s} " + " ".join(f"{det[fk][aname][str(lv)]:.2f}" for lv in SWEEPS[aname][1]))
print("STRENGTH_SWEEP_DONE"); os._exit(0)
