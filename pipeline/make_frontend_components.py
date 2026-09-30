"""Measure each geometric front-end SEPARATELY, so the solver can choose among them.

The deployed cascade has four stages -- SyncSeal rectify, residual tilt, VINE ring scale search,
blind angle probe -- and they used to run as one bundle behind a single `geo` flag. The surrogate
inherited that shape: it held one curve for "resync" (measured on rot9 only) and one for "nested"
(crop75/crop50 only), and the solver filled the rest in with hardcoded `if attack in (...)` rules.
Three geometric columns -- rs256, hflip, crop_jpeg -- had no front-end entry at all, so a request
naming them gave the solver nothing to decide with.

This measures every stage on every geometric attack, each against a control decoded in the SAME run
with the cascade suppressed. What is stored is therefore the stage's own contribution, and a stage
that does nothing (or hurts) on a column is recorded as such rather than assumed away.

Sharded by (component, attack): pass SHARD NSHARD to take a slice of the 18 cells.
"""
import sys, os, json, time, subprocess
import numpy as np
from PIL import Image
CF = "/data/tailor/project"
SC = "/data/tailor/workspace/wm_dataset10k"
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense"):
    sys.path.insert(0, p)
from src.shortened_bch import ShortenedBCH
from src.image_pool import sample as _pool_sample, composition_of
from eval_matrix import GEO, OursComposite            # the SAME operators AND the SAME decoder

N      = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SHARD  = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NSHARD = int(sys.argv[3]) if len(sys.argv) > 3 else 1

sb = ShortenedBCH(); NB = sb.n; dev = "cuda"
GRID = {f: [round(float(v), 4) for v in np.linspace(lo, hi, 9)]
        for f, (lo, hi) in {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6),
                            "VideoSeal": (0.5, 1.5)}.items()}
# FE_GRID_JSON='{"VideoSeal": [1.75, 2.0]}' measures the stages at EXTRA knots only, for the range
# extension (the three fragments were swept over three different distortion intervals; the extension
# aligns them, and a stage's replacement curve has to exist at the new knots or the solver would read
# the clamped end value there). Fragments absent from the override are skipped entirely.
if os.environ.get("FE_GRID_JSON"):
    import json as _json
    GRID = {f: [float(v) for v in vs] for f, vs in _json.loads(os.environ["FE_GRID_JSON"]).items()}
FRAGS = [f for f in ["VINE", "TrustMark", "VideoSeal"] if f in GRID]
FKEY  = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}

# One entry per separately selectable front-end. `scale` carries the nested ring because the ring is
# what the scale search searches over; that ring's PSNR cost is measured as nested_penalty.
COMPONENTS = {
    "resync": {"resync": True,  "nested": False, "scale_search": False, "angle_sweep": False},
    "scale":  {"resync": False, "nested": True,  "scale_search": True,  "angle_sweep": False},
    "angle":  {"resync": False, "nested": False, "scale_search": False, "angle_sweep": True},
    # Spatial redundancy: the same codeword in every cell of a 2x2 grid, recovered by a sliding
    # window. It was built and validated but never became a decision -- it is not in the deployed
    # cascade, and the column it wins (translation crop) was not in the attack suite either.
    "tile":   {"resync": False, "nested": False, "scale_search": False, "angle_sweep": False,
               "tile": True},
}
ATTACKS = ["crop75", "crop50", "rot9", "rs256", "hflip", "crop_jpeg", "border20"]
# The three embed-changing stages change what the fragment looks like on EVERY column, not only the
# geometric ones: the solver reads a stage's replacement curve wherever one exists and falls back to
# the plain curve elsewhere, which credited an unmeasured embed (tiled TrustMark under VAE read the
# plain TrustMark curve and certified false SATs). FE_ATTACKS / FE_COMPONENTS select a different
# (stage, attack) grid; FE_OUT_TAG keeps the files apart; FE_PENALTY=0 skips the already-measured
# embed-side cost.
if os.environ.get("FE_ATTACKS"):    ATTACKS = os.environ["FE_ATTACKS"].split(",")
if os.environ.get("FE_COMPONENTS"): COMPONENTS = {k: COMPONENTS[k] for k in os.environ["FE_COMPONENTS"].split(",")}
OUT_TAG = os.environ.get("FE_OUT_TAG", "")
from src.attacks import attack_pil_any as _attack_any

CELLS = [(c, a) for c in COMPONENTS for a in ATTACKS]
MINE  = [c for k, c in enumerate(CELLS) if k % NSHARD == SHARD]
print(f"N={N} shard {SHARD}/{NSHARD}: {len(MINE)} of {len(CELLS)} cells -> {MINE}", flush=True)

_CACHE = {}
def _composite(frag_key, strength, flags):
    """The deployed composite with exactly one fragment and exactly one front-end enabled."""
    key = (frag_key, tuple(sorted(flags.items())))
    if key not in _CACHE:
        cfg = dict(flags); cfg.update({"frags": [frag_key], "order": [frag_key],
                                       "strengths": {frag_key: 1.0}})
        if len(_CACHE) >= 3:                      # single live composite per fragment set; models are large
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = OursComposite(dev, tm_variant="B", vine_variant="R", config=cfg)
    c = _CACHE[key]
    c.strength = dict(c.DEFAULT_STRENGTH); c.strength[frag_key] = float(strength)
    return c

def r512(im): return im if im.size == (512, 512) else im.resize((512, 512))

files = _pool_sample(N, offset=0)                 # all five sources, in the evaluation set's proportions
assert len(files) == N, f"pool sample short: {len(files)}"
print("sources:", composition_of(files), flush=True)

# Both sides are recorded in BIT ACCURACY, the unit the base curves are in and the unit the
# coverage clause compares against beta. `decode` reports the accuracy of the view the cascade
# accepted, so a stage that rescues an image raises this number rather than leaving it at the
# un-rectified value. Detection is kept alongside for the report, not for the solver.
on   = {(c, a, f, s): [] for (c, a) in MINE for f in FRAGS for s in GRID[f]}
off  = {(c, a, f, s): [] for (c, a) in MINE for f in FRAGS for s in GRID[f]}
don  = {(c, a, f, s): [] for (c, a) in MINE for f in FRAGS for s in GRID[f]}
doff = {(c, a, f, s): [] for (c, a) in MINE for f in FRAGS for s in GRID[f]}
# Wall time the stage actually costs. The solver used to charge 300 ms for resync and 1600 ms for
# nested, which were the BUNDLE's numbers -- with the stages separated each has to carry its own.
lat  = {(c, a): [] for (c, a) in MINE}
latc = {(c, a): [] for (c, a) in MINE}
# Fidelity cost of the EMBED-side front-ends. Only the nested ring was ever charged for one: the
# SyncSeal mark and the tiled ring also change the embedded image, so without this the solver can
# switch them on for free and will, since they can only raise coverage.
PEN_FRAG = {"resync": "trustmark", "scale": "vine", "angle": None, "tile": "trustmark"}
pen = {(c, f, s): [] for (c, a) in MINE for f in FRAGS for s in GRID[f]}

t0 = time.time()
for i, fp in enumerate(files):
    cover = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC)
    for (cname, aname) in MINE:
        flags = COMPONENTS[cname]
        for f in FRAGS:
            comp = _composite(FKEY[f], GRID[f][0], flags)
            for s in GRID[f]:
                comp.strength[FKEY[f]] = float(s)
                emb, sec = comp.embed(cover, i)
                att = r512(GEO[aname](emb)) if aname in GEO else r512(_attack_any(aname, emb, dev=dev))
                _t = time.time()
                ba_off, det_off = comp.decode_no_cascade(att, sec)  # same decoder, cascade suppressed
                _t_off = time.time() - _t
                _t = time.time()
                ba_on,  det_on  = comp.decode(att, sec)             # deployed path, this stage allowed
                lat[(cname, aname)].append((time.time() - _t) * 1000.0)
                latc[(cname, aname)].append(_t_off * 1000.0)
                off[(cname, aname, f, s)].append(float(ba_off))
                on[(cname, aname, f, s)].append(float(ba_on))
                doff[(cname, aname, f, s)].append(float(det_off))
                don[(cname, aname, f, s)].append(float(det_on))
                if os.environ.get("FE_PENALTY", "1") == "1" and PEN_FRAG.get(cname) == FKEY[f] and aname == ATTACKS[0]:
                    # measured once per (component, fragment, strength): the same embed with the
                    # front-end and without it, on the same cover
                    plain = _composite(FKEY[f], s, {"resync": False, "nested": False,
                                                    "scale_search": False, "angle_sweep": False,
                                                    "tile": False})
                    plain.strength[FKEY[f]] = float(s)
                    pemb, _ = plain.embed(cover, i)
                    A = np.asarray(r512(emb), np.float64); B = np.asarray(r512(pemb), np.float64)
                    C = np.asarray(cover, np.float64)
                    pen[(cname, f, s)].append(float(np.mean((A - C) ** 2) - np.mean((B - C) ** 2)))
    if (i + 1) % 5 == 0: print(f"  ...{i+1}/{N} [{time.time()-t0:.0f}s]", flush=True)

mean = lambda L: float(np.mean(L)) if L else float("nan")
fe = {}; LAT = {}
for (cname, aname) in MINE:
    for f in FRAGS:
        ys_on  = [round(mean(on[(cname, aname, f, s)]), 4) for s in GRID[f]]
        ys_off = [round(mean(off[(cname, aname, f, s)]), 4) for s in GRID[f]]
        fe[f"base_fe_{cname}_{f}|{aname}"] = {"xs": GRID[f], "ys": ys_on}
        fe[f"ctrl_fe_{cname}_{f}|{aname}"] = {"xs": GRID[f], "ys": ys_off}
        fe[f"det_fe_{cname}_{f}|{aname}"]  = {"xs": GRID[f],
            "ys": [round(mean(don[(cname, aname, f, s)]), 4) for s in GRID[f]]}
        fe[f"detctrl_fe_{cname}_{f}|{aname}"] = {"xs": GRID[f],
            "ys": [round(mean(doff[(cname, aname, f, s)]), 4) for s in GRID[f]]}
        d = [a - b for a, b in zip(ys_on, ys_off)]
        print(f"[{cname:6s} {aname:9s}] {f:10s} ba effect: "
              + ", ".join(f"{s}:{v:+.3f}" for s, v in zip(GRID[f], d)), flush=True)

for (cname, aname) in MINE:
    f = PEN_FRAG.get(cname)
    if f and aname == ATTACKS[0]:
        F = {v: k for k, v in FKEY.items()}[f]
        if not all(pen[(cname, F, s)] for s in GRID[F]):
            # FE_PENALTY=0 (or a cell that never reached the first attack) leaves these empty; a NaN
            # curve written here overrode the measured cost when the canonical table merged it.
            print(f"[{cname}] embed-side cost not measured in this run; no penalty curve written", flush=True)
            continue
        ys = [round(mean(pen[(cname, F, s)]), 4) for s in GRID[F]]
        fe[f"penalty_fe_{cname}"] = {"xs": GRID[F], "ys": ys}
        print(f"[{cname:6s}] fidelity cost over plain {F}: "
              + ", ".join(f"{s}:{v:+.3f}" for s, v in zip(GRID[F], ys)) + " (MSE)", flush=True)

for (cname, aname) in MINE:
    # The stage's own cost is what the deployed decode takes with it minus the same decode without
    # it, so model load and the primary decode are not charged to the front-end.
    LAT[f"latency_ms_{cname}|{aname}"] = round(mean(lat[(cname, aname)]) - mean(latc[(cname, aname)]), 2)
    print(f"[{cname:6s} {aname:9s}] stage latency: "
          f"{mean(lat[(cname,aname)]) - mean(latc[(cname,aname)]):.1f} ms "
          f"(with {mean(lat[(cname,aname)]):.1f}, without {mean(latc[(cname,aname)]):.1f})", flush=True)

try:
    git_rev = subprocess.run(["git", "-C", CF, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    git_rev = "unknown"
out = f"{SC}/surrogate_fe_components_{OUT_TAG + '_' if OUT_TAG else ''}{SHARD}of{NSHARD}.json"
json.dump({"frontend": fe, "latency": LAT, "base": {}, "delta": {},
           "source": {"n": N, "pool": "cross-source", "slice": [0, N], "git": git_rev,
                      "measured_at": os.environ.get("TS", "unknown"),
                      "cells": MINE, "shard": [SHARD, NSHARD],
                      "note": "each front-end stage measured alone against a same-run control "
                              "decoded with the cascade suppressed; attacks " + ("from eval_matrix.GEO" if not os.environ.get("FE_ATTACKS") else "FE_ATTACKS=" + os.environ["FE_ATTACKS"])}},
          open(out, "w"), indent=1)
print("wrote", out, flush=True)
print("FE_COMPONENTS_DONE", flush=True)
