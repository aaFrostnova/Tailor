"""Where does each fragment start to work with NO attack?  (item 3 of the follow-up plan)

For each fragment, a fine strength sweep from well below the current range floor to its top, on N clean
cross-source images, recording per strength: mean bit accuracy and its spread, the fraction of images at
presence level (ba >= 0.63, the 1% zero-bit threshold) and at identity level (ba >= 0.90, the BCH(100,37)
limit), and the embed PSNR. The smallest strength whose identity-level fraction reaches 0.99 is the
candidate new range floor: knots below it measure a mark that cannot even be read unattacked.
Usage: python measure_clean_minimum.py [N=100]   -> clean_minimum_strength.json
"""
import sys, os, json, time
import numpy as np
from PIL import Image
CF = "/data/tailor/project"
SC = "/data/tailor/workspace/wm_dataset10k"
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense"):
    sys.path.insert(0, p)
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH
from src.image_pool import sample as _pool_sample, composition_of
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = os.environ.get("OUT_NAME", "clean_minimum_strength.json")
sb = ShortenedBCH(); NB = sb.n; KEY = b"v5_key_encoder_master"; dev = "cuda"
FR = {"VINE": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
      "TrustMark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
      "VideoSeal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}
# fine below the current floor, the existing knots above it
GRID = {"VINE":      [round(x, 3) for x in np.arange(0.02, 0.20, 0.02)] + [0.2, 0.3, 0.4, 0.6, 0.8, 1.0],
        "TrustMark": [round(x, 3) for x in np.arange(0.05, 0.40, 0.05)] + [0.4, 0.55, 0.7, 1.0, 1.3, 1.6],
        "VideoSeal": [round(x, 3) for x in np.arange(0.05, 0.50, 0.05)] + [0.5, 0.625, 0.75, 1.0, 1.25, 1.5]}
files = _pool_sample(N, offset=0)                # the fitting slice: the same images the table was fit on
print(f"N={N} sources={composition_of(files)} strengths={ {f: len(g) for f, g in GRID.items()} }", flush=True)
def tb(i): return np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
def r512(im): return im if im.size == (512, 512) else im.resize((512, 512))
def hard(fn, pil):
    f = FR[fn]; v = f.raw_probs(pil) if fn == "VINE" else f.raw_logits(pil)
    v = np.asarray(v).ravel()[:NB]
    return (v > (0.5 if fn == "VINE" else 0.0)).astype(np.uint8)
def psnr(a, b):
    m = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)); return 10 * np.log10(255.0 ** 2 / max(m, 1e-9))
obs = {f: {s: {"ba": [], "psnr": []} for s in GRID[f]} for f in FR}
t0 = time.time()
for i, fp in enumerate(files):
    cov = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC); t = tb(i)
    for f in FR:
        for s in GRID[f]:
            emb = r512(FR[f].embed_with_target(cov, t, strength=s))
            obs[f][s]["ba"].append(float(np.mean(hard(f, emb) == t))); obs[f][s]["psnr"].append(psnr(cov, emb))
    if (i + 1) % 10 == 0: print(f"  ...{i+1}/{N} [{time.time()-t0:.0f}s]", flush=True)
out = {"n": N, "pool": composition_of(files), "presence_tau": 0.63, "identity_beta": 0.90, "rows": {}, "floor": {}}
for f in FR:
    rows = []
    for s in GRID[f]:
        ba = np.array(obs[f][s]["ba"])
        rows.append({"s": s, "ba_mean": round(float(ba.mean()), 4), "ba_sd": round(float(ba.std(ddof=1)), 4),
                     "frac_presence": round(float(np.mean(ba >= 0.63)), 3), "frac_identity": round(float(np.mean(ba >= 0.90)), 3),
                     "psnr": round(float(np.mean(obs[f][s]["psnr"])), 2)})
    out["rows"][f] = rows
    fl = {lvl: next((r["s"] for r in rows if r[k] >= 0.99), None) for lvl, k in (("presence", "frac_presence"), ("identity", "frac_identity"))}
    out["floor"][f] = fl
    print(f"\n{f}: current floor {min(GRID[f]) if f=='VINE' else ''}", flush=True)
    print(f"  {'s':>6s} {'ba':>7s} {'sd':>6s} {'>=.63':>6s} {'>=.90':>6s} {'PSNR':>6s}", flush=True)
    for r in rows: print(f"  {r['s']:6.3f} {r['ba_mean']:7.4f} {r['ba_sd']:6.3f} {r['frac_presence']:6.2f} {r['frac_identity']:6.2f} {r['psnr']:6.2f}", flush=True)
    print(f"  -> smallest strength with >=99% of images at presence level: {fl['presence']}, at identity level: {fl['identity']}", flush=True)
json.dump(out, open(f"{SC}/{OUT}", "w"), indent=2); print(f"\nwrote {SC}/{OUT}"); print("CLEAN_MINIMUM_DONE")
