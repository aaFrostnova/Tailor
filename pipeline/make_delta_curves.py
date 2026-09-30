"""Interference as a function of the overwriting fragment's strength (follow-up item 4).

The table's delta_{g->f}(a) is one number, measured with both fragments at their mid strengths, and the
solver treats it as constant over the overwriting fragment's whole range. This measures it as a curve:
f embedded at its mid strength, then g embedded over it at each of g's nine knots; f decoded after the
attack; delta(s_g) = ba_f(alone at mid, same images) - ba_f(with g at s_g), clipped at zero. Only the two
ordered pairs the solver actually stacks (VINE over VideoSeal, VideoSeal over VINE) and the columns that
bind.  Usage: python make_delta_curves.py N CELL NCELLS   (one (pair, attack) cell per task)
"""
import sys, os, json, time, subprocess
import numpy as np
from PIL import Image
CF = "/data/tailor/project"
SC = "/data/tailor/workspace/wm_dataset10k"
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker"):
    sys.path.insert(0, p)
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH
from src.attacks import attack_pil_any
from src.image_pool import sample as _pool_sample, composition_of
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CELL = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NCELLS = int(sys.argv[3]) if len(sys.argv) > 3 else 22
OUT_TAG = os.environ.get("OUT_TAG", "delta_curves")
GRID = {f: [round(float(v), 4) for v in np.linspace(lo, hi, 9)]
        for f, (lo, hi) in {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6), "VideoSeal": (0.5, 1.5)}.items()}
MID = {f: GRID[f][len(GRID[f]) // 2] for f in GRID}       # the host's reference strength stays the original mid
# DELTA_GRID_JSON='{"VideoSeal": [1.75, 2.0]}' sweeps the OVERWRITING fragment over extra knots only (the
# range extension); the host f keeps its original mid strength so the new points join the old curve.
if os.environ.get("DELTA_GRID_JSON"):
    for _f, _vs in json.loads(os.environ["DELTA_GRID_JSON"]).items():
        GRID[_f] = [float(v) for v in _vs]
PAIRS = [("VINE", "VideoSeal"), ("VideoSeal", "VINE")]                  # (g overwrites f)
# DELTA_CELLS_FILE=<json list of [g, f, attack]> names the cells explicitly (the table-completion
# campaign sweeps the four remaining ordered pairs on 16 columns and VINE<->VideoSeal on the 6 columns
# the first campaign skipped); with it, PAIRS/COLS below are not used. The host's mid strength is
# unchanged, so these curves are directly comparable with the 22 measured earlier.
COLS = ["crop75", "crop50", "rot9", "hflip", "crop_jpeg", "jpeg25", "vaeB", "vaeC", "regen", "rinse2x", "rinse4x"]
CELLS = [(g, f, a) for (g, f) in PAIRS for a in COLS]
if os.environ.get("DELTA_CELLS_FILE"):
    CELLS = [tuple(c) for c in json.load(open(os.environ["DELTA_CELLS_FILE"]))]
assert len(CELLS) == NCELLS, (len(CELLS), NCELLS)
g, f, attack = CELLS[CELL]
print(f"N={N} cell {CELL}/{NCELLS}: delta_{g}->{f} under {attack}", flush=True)
sb = ShortenedBCH(); NB = sb.n; KEY = b"v5_key_encoder_master"; dev = "cuda"
FR = {"VINE": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
      "TrustMark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
      "VideoSeal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}
def tb(i): return np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
def r512(im): return im if im.size == (512, 512) else im.resize((512, 512))
def hard(fn, pil):
    fr = FR[fn]; v = fr.raw_probs(pil) if fn == "VINE" else fr.raw_logits(pil)
    v = np.asarray(v).ravel()[:NB]
    return (v > (0.5 if fn == "VINE" else 0.0)).astype(np.uint8)
files = _pool_sample(N, offset=0)
print("sources:", composition_of(files), flush=True)
alone, with_g = [], {s: [] for s in GRID[g]}
t0 = time.time()
for i, fp in enumerate(files):
    cover = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC); t = tb(i)
    base = r512(FR[f].embed_with_target(cover, t, strength=MID[f]))
    alone.append(float(np.mean(hard(f, r512(attack_pil_any(attack, base, dev=dev))) == t)))
    for s in GRID[g]:
        comp = r512(FR[g].embed_with_target(base, t, strength=s))
        with_g[s].append(float(np.mean(hard(f, r512(attack_pil_any(attack, comp, dev=dev))) == t)))
    if (i + 1) % max(1, N // 10) == 0: print(f"  ...{i+1}/{N} [{time.time()-t0:.0f}s]", flush=True)
a0 = float(np.mean(alone)); ys = [round(max(0.0, a0 - float(np.mean(with_g[s]))), 4) for s in GRID[g]]
print(f"  {f} alone at {MID[f]}: {a0:.4f};  delta_{g}->{f}(s_{g}) = " + ", ".join(f"{s}:{y:.3f}" for s, y in zip(GRID[g], ys)), flush=True)
try: git = subprocess.run(["git", "-C", CF, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
except Exception: git = "unknown"
out = f"{SC}/surrogate_{OUT_TAG}_{CELL}of{NCELLS}.json"
json.dump({"delta": {f"{g}|{f}|{attack}": {"xs": GRID[g], "ys": ys}}, "base": {},
           "reference": {f"{f}@{MID[f]}|{attack}|alone": round(a0, 4), f"{f}@{MID[f]}|{attack}|with_{g}": {str(s): round(float(np.mean(v)), 4) for s, v in with_g.items()}},
           "source": {"n": N, "pool": composition_of(files), "git": git, "measured_at": os.environ.get("TS", "unknown"),
                      "cell": [g, f, attack], "host_strength": MID[f], "note": "f at mid, g swept over its nine knots after f; delta = alone - with"}},
          open(out, "w"), indent=1)
print("wrote", out, flush=True); print("DELTA_CURVES_DONE", flush=True)
