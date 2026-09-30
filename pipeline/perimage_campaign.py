"""PER-IMAGE bit accuracy (and, where a crypto codeword was embedded, the keyed verification flag) for
every base cell of the surrogate table and the ring stage's cells on the adversarial columns.

Why: the table stores E[ba] per (fragment, attack, strength). The deployment accepts per image against
a threshold tau that depends on the request's false-positive budget and fragment count, so the quantity
a request needs is P(ba >= tau), which E[ba] cannot give when the per-image distribution is bimodal
(UnMarker under the ring at s=0.3: mean 0.694, half the images at 0.99 and half at chance). Per-image
values make P(ba >= tau) exact for ANY tau at build time, and E[ba] is still their mean.

Modes (one cell family each; the sbatch array fans them out):
  inprocess <frag> <knot>   embed the ext9 slice (pool offset 0, N=100) with ONE fragment at that knot
                            through the deployed composite (crypto codeword), apply the 14 in-process
                            attacks, record ba and verify per image.       -> perimage/inprocess/<F>_<s>.json
  ctrlregen <s03|s05|s07>   decode-only: regen_overlay_ext_n100_ctrlregen_<step>/attacked/solo_*.png,
                            random-bit embeds read with the fragment's own decoder vs RandomState(i).
  unmarker                  decode-only: unmarker9/att_<frag>_<s>/i*.png, same random-bit convention.
  ring                      decode-only: unmarker9_ring/knot_*/att_*/i*.png, composite embeds with the
                            scale stage on; best-path over the cascade's accepted view, ba + verify + det.
  mildregen                 reformat mildregen9_n100/shard*of04.json (regen, rinse2x; per-image lists
                            already stored there).
"""
import sys, os, json, glob, time
import numpy as np
from PIL import Image
CF = "/data/tailor/project"
SC = "/data/tailor/workspace/wm_dataset10k"
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
MODE = sys.argv[1]; ARGS = sys.argv[2:]; sys.argv = [sys.argv[0]]
OUT = f"{SC}/perimage"; os.makedirs(OUT, exist_ok=True)
KEY = b"v5_key_encoder_master"; NB = 100
FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}
UNKEY = {v: k for k, v in FKEY.items()}
GRID = {"VINE": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "TrustMark": [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6],
        "VideoSeal": [0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]}
INPROCESS = ["jpeg25", "blur", "noise", "bright", "contrast", "crop75", "crop50", "rot9", "rs256", "hflip",
             "crop_jpeg", "border20", "vaeB", "vaeC"]
dev = "cuda"


def r512(img): return img if img.size == (512, 512) else img.resize((512, 512))


def _deployed_read(comp, order, img, iid, tx):
    """Per-fragment bit accuracy and keyed verification on the view the DEPLOYED decoder accepts.

    eval_matrix.OursComposite.decode accepts on (any fragment verifies) or (the fused codeword
    verifies) or (the zero-bit presence test fires), and only then, on a miss, runs the geometric
    cascade and reports the rescued view. Firing the cascade on a verification miss alone would read a
    rescued view where the deployment had already accepted the primary one."""
    from eval_matrix import presence_detected
    from src.soft_fusion import fuse_llrs, llr_to_bits
    from src.soft_bch import decode_and_verify
    def _read(v):
        L = {f: comp._frag_llr(FKEY[f], v, iid) for f in order}
        ba = {f: float(np.mean((L[f] > 0).astype(np.uint8) == tx)) for f in order}
        ver = {f: bool(decode_and_verify(L[f], iid, codec=comp.sb)["detected"]) for f in order}
        fused = fuse_llrs({FKEY[f]: L[f] for f in order}, weights=None, n_codeword=comp.sb.n)
        fba = float(np.mean(llr_to_bits(fused) == tx))
        fver = bool(decode_and_verify(fused, iid, codec=comp.sb)["detected"])
        det = fver or any(ver.values()) or presence_detected(ba, fba, len(order))
        return ba, ver, det
    ba, ver, det = _read(img)
    if comp.geo and not det:
        ok, vw = comp.geo_cascade(img, iid, tx, return_view=True)
        if ok and vw is not None:
            ba2, ver2, _ = _read(vw)
            ba = ba2; ver = {f: ver[f] or ver2[f] for f in order}; det = True
    return ba, ver, float(det)



def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w")); print(f"  wrote {os.path.relpath(path, SC)}", flush=True)


# ----------------------------------------------------------------------------------------------
if MODE == "inprocess":
    import watermark_smt_v2 as W
    from eval_matrix import OursComposite
    from src.attacks import attack_pil_any
    from src.image_pool import sample as _pool_sample, composition_of
    from src.soft_bch import decode_and_verify
    F = ARGS[0]; K = int(ARGS[1]); s = GRID[F][K]
    files = _pool_sample(100, offset=0)                     # the ext9 slice: same images as the base table
    covers = [Image.open(f).convert("RGB").resize((512, 512), Image.BICUBIC) for f in files]
    comp = OursComposite(dev, tm_variant="B", vine_variant="R",
                         config={"frags": [FKEY[F]], "order": [FKEY[F]], "strengths": {},
                                 **W.frontend_config({"resync": False, "scale": False, "angle": False, "tile": False})})
    comp.strength = dict(comp.DEFAULT_STRENGTH); comp.strength[FKEY[F]] = float(s)
    print(f"[inprocess] {F} @ {s}: {len(covers)} images {composition_of(files)}, {len(INPROCESS)} attacks", flush=True)
    ba = {a: [] for a in INPROCESS}; ver = {a: [] for a in INPROCESS}; clean_ba = []
    t0 = time.time()
    for i, cov in enumerate(covers):
        emb, sec = comp.embed(cov, i); iid, tx = sec
        L0 = comp._frag_llr(FKEY[F], emb, iid); clean_ba.append(float(np.mean((L0 > 0).astype(np.uint8) == tx)))
        for a in INPROCESS:
            v = r512(attack_pil_any(a, emb, dev=dev))
            L = comp._frag_llr(FKEY[F], v, iid)
            ba[a].append(float(np.mean((L > 0).astype(np.uint8) == tx)))
            ver[a].append(bool(decode_and_verify(L, iid, codec=comp.sb)["detected"]))
        if (i + 1) % 20 == 0: print(f"  {i+1}/{len(covers)} [{time.time()-t0:.0f}s]", flush=True)
    save(f"{OUT}/inprocess/{F}_{s}.json",
         {"frag": F, "strength": s, "n": len(covers), "images": "pool offset 0 (ext9 slice)", "embed": "composite crypto codeword",
          "clean_ba": clean_ba, "ba": ba, "ver": ver,
          "means": {a: round(float(np.mean(ba[a])), 4) for a in INPROCESS}})
    print("INPROCESS_DONE", flush=True)

# ----------------------------------------------------------------------------------------------
elif MODE in ("ctrlregen", "unmarker"):
    from src.vine_crypto_wrapper import VineCryptoWrapper
    from src.trustmark_fragment import TrustMarkFragment
    from src.videoseal_fragment import VideoSealFragment
    FR = {"VINE": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
          "TrustMark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
          "VideoSeal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}
    def hard(F, pil):
        f = FR[F]; v = f.raw_probs(pil) if F == "VINE" else f.raw_logits(pil)
        v = np.asarray(v).ravel()[:NB]; return (v > (0.5 if F == "VINE" else 0.0)).astype(np.uint8)
    def tbits(i): return np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
    if MODE == "ctrlregen":
        step = ARGS[0]; col = f"ctrlregen_{step}"
        d = f"{SC}/regen_overlay_ext_n100_ctrlregen_{step}/attacked"
        CODE = {"V": "VINE", "T": "TrustMark", "S": "VideoSeal"}
        cells = {}
        files = sorted(glob.glob(f"{d}/solo_*.png")); t0 = time.time()
        print(f"[ctrlregen {step}] {len(files)} retained solo images in {d}", flush=True)
        for k, fp in enumerate(files):
            _, c, s_str, i_str = os.path.basename(fp)[:-4].split("_")
            F = CODE[c]; s = min(GRID[F], key=lambda g: abs(g - float(s_str))); i = int(i_str)
            img = r512(Image.open(fp).convert("RGB"))
            cells.setdefault((F, s), []).append((i, float(np.mean(hard(F, img) == tbits(i)))))
            if (k + 1) % 500 == 0: print(f"  {k+1}/{len(files)} [{time.time()-t0:.0f}s]", flush=True)
        for (F, s), v in cells.items():
            v.sort(); save(f"{OUT}/{col}/{F}_{s}.json",
                           {"frag": F, "strength": s, "attack": col, "n": len(v), "embed": "random bits (regen_overlay_ext)",
                            "idx": [i for i, _ in v], "ba": [b for _, b in v], "ver": None, "mean": round(float(np.mean([b for _, b in v])), 4)})
    else:
        col = "unmarker"; t0 = time.time()
        for d in sorted(glob.glob(f"{SC}/unmarker9/att_*_*")):
            bn = os.path.basename(d)[4:]; m, s_str = bn.rsplit("_", 1); F = UNKEY[m]; s = float(s_str)
            files = sorted(glob.glob(f"{d}/i*.png")); v = []
            for fp in files:
                i = int(os.path.basename(fp)[1:6]); img = r512(Image.open(fp).convert("RGB"))
                v.append((i, float(np.mean(hard(F, img) == tbits(i)))))
            save(f"{OUT}/{col}/{F}_{s}.json",
                 {"frag": F, "strength": s, "attack": col, "n": len(v), "embed": "random bits (unmarker9)",
                  "idx": [i for i, _ in v], "ba": [b for _, b in v], "ver": None, "mean": round(float(np.mean([b for _, b in v])), 4)})
            print(f"  {F}@{s} n={len(v)} mean {np.mean([b for _, b in v]):.3f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"{MODE.upper()}_DONE", flush=True)

# ----------------------------------------------------------------------------------------------
elif MODE == "ring":
    import watermark_smt_v2 as W
    from eval_matrix import OursComposite
    from src.soft_bch import decode_and_verify
    COLS = {"unmarker": "att_unmarker", "ctrlregen_s03": "att_ctrlregen_s03", "ctrlregen_s05": "att_ctrlregen_s05", "ctrlregen_s07": "att_ctrlregen_s07"}
    for kd in sorted(glob.glob(f"{SC}/unmarker9_ring/knot_*"), key=lambda p: float(p.split("knot_")[1])):
        s = float(kd.split("knot_")[1])
        comp = OursComposite(dev, tm_variant="B", vine_variant="R", config={"frags": ["vine"], "order": ["vine"], "strengths": {},
                             **W.frontend_config({"resync": False, "scale": True, "angle": False, "tile": False})})
        comp.strength = dict(comp.DEFAULT_STRENGTH); comp.strength["vine"] = s
        for col, sub in COLS.items():
            files = sorted(glob.glob(f"{kd}/{sub}/i*.png"))
            if not files: continue
            idx, ba, ver, det = [], [], [], []
            for fp in files:
                i = int(os.path.basename(fp)[1:6]); iid, tx = comp.secret_for(i)
                img = r512(Image.open(fp).convert("RGB"))
                _, d_ = comp.decode(img, (iid, tx))
                L = comp._frag_llr("vine", img, iid); b = float(np.mean((L > 0).astype(np.uint8) == tx))
                if b < 0.9 and comp.geo:                                   # best-path: the cascade's accepted view
                    ok, vw = comp.geo_cascade(img, iid, tx, return_view=True)
                    if ok and vw is not None:
                        L2 = comp._frag_llr("vine", vw, iid); b2 = float(np.mean((L2 > 0).astype(np.uint8) == tx))
                        if b2 > b: b, L = b2, L2
                idx.append(i); ba.append(b); ver.append(bool(decode_and_verify(L, iid, codec=comp.sb)["detected"])); det.append(float(d_))
            save(f"{OUT}/ring/VINE_{col}_{s}.json",
                 {"frag": "VINE", "stage": "scale", "strength": s, "attack": col, "n": len(ba), "embed": "composite crypto codeword, nested ring",
                  "idx": idx, "ba": ba, "ver": ver, "det": det, "mean": round(float(np.mean(ba)), 4), "det_rate": round(float(np.mean(det)), 4)})
            print(f"  ring VINE@{s} {col:14s} n={len(ba)} mean {np.mean(ba):.3f} verify {np.mean(ver):.2f} det {np.mean(det):.2f}", flush=True)
    print("RING_DONE", flush=True)

# ----------------------------------------------------------------------------------------------
elif MODE == "mildregen":
    acc = {}
    for p in sorted(glob.glob(f"{SC}/mildregen9_n100/shard*of04.json")):
        d = json.load(open(p))
        for k, v in d["solo"].items():
            a, F, s = k.split("|"); acc.setdefault((a, F, float(s)), []).extend([float(x) for x in v])
    for (a, F, s), v in sorted(acc.items()):
        if a not in ("regen", "rinse2x"): continue
        save(f"{OUT}/{a}/{F}_{s}.json",
             {"frag": F, "strength": s, "attack": a, "n": len(v), "embed": "random bits (mildregen9_n100)",
              "ba": v, "ver": None, "mean": round(float(np.mean(v)), 4)})
    print("MILDREGEN_DONE", flush=True)
elif MODE != "stage":
    raise SystemExit(f"unknown mode {MODE}")

# ----------------------------------------------------------------------------------------------
# stage <component> <frag> <strength>: per-image values for a FRONT-END cell. The composite is built
# with that one stage on; every attacked image is read on the view the deployed decoder accepts (the
# primary view, or the cascade's view when the keyed verification missed on the primary), so the
# recorded ba/verify are what the deployment delivers with the stage on. In-process columns only: the
# diffusion columns under a stage keep their det_fe_* curves (recorded at the decoder's 1% budget).
if MODE == "stage":
    import watermark_smt_v2 as W
    from eval_matrix import OursComposite
    from src.attacks import attack_pil_any
    from src.image_pool import sample as _pool_sample, composition_of
    from src.soft_bch import decode_and_verify
    COMPONENTS = {"resync": {"resync": True, "nested": False, "scale_search": False, "angle_sweep": False},
                  "scale":  {"resync": False, "nested": True, "scale_search": True, "angle_sweep": False},
                  "angle":  {"resync": False, "nested": False, "scale_search": False, "angle_sweep": True},
                  "tile":   {"resync": False, "nested": False, "scale_search": False, "angle_sweep": False, "tile": True}}
    C, F, s = ARGS[0], ARGS[1], float(ARGS[2])
    cols = ["crop75", "crop50", "rot9", "rs256", "hflip", "crop_jpeg", "border20"] if C == "angle" else INPROCESS
    # STAGE_COLS=regen,rinse2x measures the stage on other in-process columns; it then MUST write to its
    # own directory (STAGE_OUT), because the default one holds the 14-column cell of the same
    # (stage, fragment, knot) under the same file name. STAGE_N shrinks the image count for a smoke run.
    out_dir = os.environ.get("STAGE_OUT") or f"{OUT}/stage_{C}"
    if os.environ.get("STAGE_COLS"):
        cols = os.environ["STAGE_COLS"].split(",")
        assert os.environ.get("STAGE_OUT"), "STAGE_COLS needs STAGE_OUT (the default directory holds the 14-column cells)"
    files = _pool_sample(int(os.environ.get("STAGE_N", "100")), offset=0)
    covers = [Image.open(f).convert("RGB").resize((512, 512), Image.BICUBIC) for f in files]
    comp = OursComposite(dev, tm_variant="B", vine_variant="R",
                         config={"frags": [FKEY[F]], "order": [FKEY[F]], "strengths": {}, **COMPONENTS[C]})
    comp.strength = dict(comp.DEFAULT_STRENGTH); comp.strength[FKEY[F]] = float(s)
    print(f"[stage {C}] {F} @ {s}: {len(covers)} images {composition_of(files)}, {len(cols)} columns", flush=True)
    ba = {a: [] for a in cols}; ver = {a: [] for a in cols}; det = {a: [] for a in cols}; t0 = time.time()
    for i, cov in enumerate(covers):
        emb, sec = comp.embed(cov, i); iid, tx = sec; emb = r512(emb)
        for a in cols:
            v = r512(attack_pil_any(a, emb, dev=dev))
            b_, vr_, d_ = _deployed_read(comp, [F], v, iid, tx)
            ba[a].append(b_[F]); ver[a].append(vr_[F]); det[a].append(d_)
        if (i + 1) % 20 == 0: print(f"  {i+1}/{len(covers)} [{time.time()-t0:.0f}s]", flush=True)
    save(f"{out_dir}/{F}_{s}.json",
         {"frag": F, "stage": C, "strength": s, "n": len(covers), "images": "pool offset 0 (ext9 slice)",
          "embed": f"composite crypto codeword, stage {C} on", "ba": ba, "ver": ver, "det": det,
          "means": {a: round(float(np.mean(ba[a])), 4) for a in cols}, "det_rate": {a: round(float(np.mean(det[a])), 4) for a in cols}})
    print("STAGE_DONE", flush=True)
