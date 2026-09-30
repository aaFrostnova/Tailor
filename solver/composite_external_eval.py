"""Cross-env evaluation of the composite vs an EXTERNAL attacker (CtrlRegen+/UnMarker).

Advanced attacks live in their own conda envs (incompatible pins), so we can't import them
into the fingerprint interpreter. Pattern (3 processes):
  1. (fingerprint) --mode embed  : embed composite on N images -> embed_dir/ + meta.json
  2. (attacker env) batch attack : embed_dir/ -> attacked_dir/        [separate script/env]
  3. (fingerprint) --mode decode : decode attacked_dir/ with meta.json -> detection report

Detection = fused-BCH-verify OR fused-zerobit(>=tau) OR TrustMark, matching
benchmark_composite_defense.py. Supports --fragments for ablations.
"""
import argparse, glob, json, os, sys
import numpy as np, torch
from PIL import Image
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import MASKWM_CKPT, POOL, PROJECT as REPO
sys.path.insert(0, os.path.join(REPO, "solver"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
# MaskWM is an external baseline, not part of the deployed cascade, which imports this module only
# for scale_resid / nested_vine_embed / rot; it is therefore imported where it is used.
def _maskwm():
    from src.maskwm_wrapper import MaskWMWrapper
    return MaskWMWrapper
from src.syncseal_frontend import load_sync, sync_embed, sync_rectify, DEFAULT_JIT
from src.payload import image_id_to_payload

KEY = b"v5_key_encoder_master"
SPEC = {"vine": ("prob", "raw_probs", "target"),
        "trustmark": ("logit", "raw_logits", "target"),
        "videoseal": ("logit", "raw_logits", "target"),   # geometry fragment (rotation/crop-robust; replaces trustmark)
        "maskwm": ("prob", "raw_scores", "target")}       # MaskWM-D baseline (mask-adaptive pixel wm)


def build(fragments, dev, sb, tm_variant="B"):
    b = {
        "vine":      lambda: VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev),
        "trustmark": lambda: TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type=tm_variant, device=dev),
        "videoseal": lambda: VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=sb.n, device=dev),
        "maskwm":    lambda: _maskwm()(ckpt_path=MASKWM_CKPT,
                                           master_key=KEY, method_name="maskwm", n_bits=sb.n, device=dev),
    }
    return {n: b[n]() for n in fragments}


class PlainTrustMark:
    """TrustMark carrying its own bits rather than the shared codeword: the standalone tier
    the composite is OR-ed against, so that tier is a plain method and not a second view of
    the fused one. It is TrustMarkFragment with the per-image permutation left off, which is
    all that separates a fragment from the method it is built on."""

    def __init__(self, variant, n_bits, device):
        self.frag = TrustMarkFragment(method_name="trustmark_or", n_bits=n_bits,
                                      model_type=variant, device=device)
        self.n_bits = n_bits

    def embed(self, pil, bits):
        return self.frag.embed_with_target(pil, bits)

    def decode(self, pil):
        return (self.frag.raw_logits(pil) > 0).astype(np.uint8)


def tmbits_for(image_id, n):
    return np.random.RandomState(hash(image_id) % (2**31)).randint(0, 2, n).astype(np.uint8)


def scale_resid(cover, wm, alpha):
    """Scale one fragment's residual by alpha (alpha=1.0 = full strength, no-op).

    The next fragment embeds on the scaled result, so chaining this per fragment
    reproduces strength_tradeoff_eval.py exactly. Both images are aligned to 512.
    """
    if cover.size != (512, 512): cover = cover.resize((512, 512))
    if wm.size != (512, 512): wm = wm.resize((512, 512))
    if alpha == 1.0: return wm
    c = np.asarray(cover, np.float64); w = np.asarray(wm, np.float64)
    return Image.fromarray(np.clip(c + alpha * (w - c), 0, 255).astype(np.uint8))


def rot(img, deg):
    """Reflect-pad -> rotate (BICUBIC) -> center-crop 512. Rotation-resync operator: rotating an
    attacked image by a candidate -deg re-aligns the alignment-locked fragments (VINE/TrustMark)."""
    a = np.asarray(img); pad = 256
    big = Image.fromarray(np.pad(a, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    bw, bh = big.size; l = (bw - 512) // 2
    return big.crop((l, l, l + 512, l + 512))


def nested_vine_embed(m, cover, target, scales=(1.0, 0.75, 0.5), strength=1.0):
    """Embed the SAME crypto target as concentric VINE layers at each scale (full strength per layer).
    VINE's signal is a border ring, so a layer at scale K puts its ring at radius K -> nested rings that
    each survive a different crop depth. This is what the decode-side vine_scale search (B2) recovers.

    `strength` threads through to each layer's native embed_with_target knob (default 1.0 = prior
    behavior, unchanged for existing callers that don't pass it)."""
    if cover.size != (512, 512): cover = cover.resize((512, 512))
    x = cover
    for K in scales:
        if K >= 0.999:
            x = m.embed_with_target(x, target, strength=strength)
            if x.size != (512, 512): x = x.resize((512, 512))
        else:
            s = int(512 * K); o = (512 - s) // 2
            wm = m.embed_with_target(x.crop((o, o, o + s, o + s)), target, strength=strength)
            out = x.copy(); out.paste(wm.resize((s, s)), (o, o)); x = out
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["embed", "decode"], required=True)
    ap.add_argument("--fragments", nargs="+", default=["vine", "dft", "qim"])
    ap.add_argument("--n_images", type=int, default=50)
    ap.add_argument("--start_idx", type=int, default=4000)
    ap.add_argument("--image_dir", default=os.path.join(POOL, "A", "img"),
                    help="covers to embed; the default is split A of the measurement pool")
    ap.add_argument("--image_glob", default="*.png")
    ap.add_argument("--embed_dir", required=True, help="where embedded PNGs + meta.json live")
    ap.add_argument("--attacked_dir", help="(decode) dir of attacked PNGs; default=embed_dir")
    ap.add_argument("--attack_name", default="external")
    ap.add_argument("--tm_variant", default="B", help="TrustMark model variant (B strongest)")
    ap.add_argument("--strength", type=float, default=0.70,
                    help="per-fragment residual scale (1.0=full). 0.70 default = +2.7dB PSNR free, "
                         "detection still 1.0 incl regen (validated for vine+trustmark, n=16). "
                         "0.50 = +5.5dB with minor regen dip. See strength_tradeoff_eval.py.")
    ap.add_argument("--output", default="results/defense/composite_external.json")
    ap.add_argument("--no_bestpath", action="store_true", help="disable best-of-paths detection (per-fragment crypto-verify OR fused)")
    ap.add_argument("--resync", action="store_true", help="rotation-resync fallback: when @0deg detection fails, "
                    "search candidate angles and accept iff ANY fragment crypto-verifies (2^-37 zero false-accept, "
                    "self-validating). Recovers rot>=20deg (0%%->100%%). Only runs when the primary paths miss.")
    ap.add_argument("--resync_range", type=float, default=180.0, help="rotation search half-range in degrees (+-range). "
                    "MEASURED: the old default of 30 was a CONFIG artifact, not a limit -- rot45/60/75/120/150 all sat at "
                    "0.00-0.10 purely because the true angle was outside the grid. Widening to +-180 lifts every one of "
                    "them to 0.95 (rot90/180 partly work even at +-30 because they are exact axis-aligned transforms). "
                    "Cost is only more crypto-verify trials: union-bound FPR stays <= 2^-30.")
    ap.add_argument("--resync_step", type=float, default=3.0, help="rotation search step in degrees")
    ap.add_argument("--vine_scales", type=str, default="1.0,0.75,0.5", help="comma-sep concentric VINE ring scales "
                    "(outer->inner). Default 3 rings. Drop the outer ring with '0.75,0.5' (+~2.4dB PSNR, geometry "
                    "unchanged; regeneration margin is the axis to watch since the K=1.0 ring is the strongest "
                    "regen carrier).")
    ap.add_argument("--single_vine", dest="vine_nested", action="store_false", default=True,
                    help="disable nested-ring VINE embedding (default ON): embed VINE once full-frame instead of "
                    "as concentric layers {1,.75,.5}. Nested is the deployed default — it is what gives the "
                    "decode-side vine_scale search something to find under crop/zoom (single VINE dies on any crop "
                    "because the crop removes its border ring).")
    ap.add_argument("--vine_scale_step", type=float, default=0.005, help="blind SCALE-search step for the nested-VINE "
                    "layers (fraction of frame side). CALIBRATED, do not coarsen: VINE's scale capture width is <1%%, so "
                    "the previous 0.02-0.03 grid MISSED the layers entirely -- e.g. a 0.9 crop puts the K=0.5 layer at "
                    "0.5565 and the 0.03 grid's nearest point (0.55, 1.2%% off) decodes at ba 0.43 while the exact scale "
                    "decodes at ba 0.98. With this step, full UnMarker goes 0.16 -> 1.00 on VINE alone. Union-bound FPR "
                    "with 133 scale views + the rotation views stays <= 2^-29.6.")
    ap.add_argument("--syncseal", action="store_true", help="SyncSeal learned geometric front-end (arXiv 2509.15208): "
                    "embed mode adds a sync watermark (+43.5dB, no poison); decode mode, on primary miss, detects the 4 "
                    "corners + perspective-unwarps then decodes+crypto-verifies the rectified frame. Recovers rotation AND "
                    "crop-zoom/scale/perspective in ONE pass (crop75 0.13->0.96). crypto-verify stays the accept gate -> FPR unchanged.")
    ap.add_argument("--syncseal_jit", default=DEFAULT_JIT, help="path to syncmodel.jit.pt")
    ap.add_argument("--geo_cascade", action="store_true", help="UNIFIED geometric cascade (combines SyncSeal + "
                    "crypto-verify search, cheap->expensive, every stage crypto-verify-gated): A) SyncSeal one-shot "
                    "unwarp (all geometry, all fragments); B) local +-3/6deg crypto-verify refine around the rectified "
                    "frame (TM carrier, pushes rotation ->1.0); C) blind coarse-to-fine TM-guided angle search on the "
                    "original (regen fallback when the sync mark is dead). Fires only on primary miss.")
    ap.add_argument("--blind_coarse", type=float, default=10.0, help="coarse step (deg) for cascade stage C")
    ap.add_argument("--blind_gate", type=float, default=0.60, help="cascade stage-C early-abort: if the best coarse TM "
                    "bit-acc stays below this, the loss is INFORMATION (crop/regen) not misalignment -> skip the fine "
                    "search (no angle can cross the crypto floor). Separates recoverable rotation (coarse best >>gate) "
                    "from info-loss (coarse best ~0.50).")
    args = ap.parse_args()
    dev = "cuda"
    sb = ShortenedBCH(); tau = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
    frag = build(args.fragments, dev, sb, args.tm_variant)
    use_bestpath = not args.no_bestpath
    sync = load_sync(args.syncseal_jit, dev) if (args.syncseal or args.geo_cascade) else None
    # add the standalone TrustMark OR-tier only when neither pixel geometry fragment is fused
    use_or_tm = "trustmark" not in args.fragments and "videoseal" not in args.fragments
    if use_or_tm:
        tm = PlainTrustMark(args.tm_variant, sb.n, dev)

    if args.mode == "embed":
        os.makedirs(args.embed_dir, exist_ok=True)
        imgs = sorted(glob.glob(os.path.join(args.image_dir, args.image_glob)))[args.start_idx:args.start_idx + args.n_images]
        meta = {"fragments": args.fragments, "tau": tau, "strength": args.strength, "items": []}
        for i, fp in enumerate(imgs):
            image_id = f"ext_{i:05d}"; tx = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))
            img = Image.open(fp).convert("RGB").resize((512, 512))
            for name in args.fragments:
                m = frag[name]; cover = img
                if name == "vine" and args.vine_nested:
                    perm, M = m.get_perm_M(image_id)
                    scales = tuple(float(x) for x in args.vine_scales.split(","))
                    img = nested_vine_embed(m, cover, apply_crypto(tx, perm, M), scales=scales)   # nested rings, full strength
                    continue
                if SPEC[name][2] == "id_tx":
                    wm = m.embed(cover, image_id, tx)
                else:
                    perm, M = m.get_perm_M(image_id); wm = m.embed_with_target(cover, apply_crypto(tx, perm, M))
                img = scale_resid(cover, wm, args.strength)
            if use_or_tm:
                tmb = tmbits_for(image_id, tm.n_bits); cover = img
                img = scale_resid(cover, tm.embed(cover, tmb), args.strength)
            if args.syncseal or args.geo_cascade:
                img = sync_embed(sync, img, dev)          # add the geometric-sync watermark on top (+43.5dB)
            img.save(os.path.join(args.embed_dir, f"img_{i:05d}.png"))
            meta["items"].append({"i": i, "image_id": image_id})
            if (i + 1) % 10 == 0: print(f"  embedded [{i+1}/{len(imgs)}]", flush=True)
        json.dump(meta, open(os.path.join(args.embed_dir, "meta.json"), "w"), indent=2)
        print(f"[embed done] {len(imgs)} (strength={args.strength}) -> {args.embed_dir}\nEMBED_DONE")
        return

    # decode
    meta = json.load(open(os.path.join(args.embed_dir, "meta.json")))
    adir = args.attacked_dir or args.embed_dir
    R = {"fused_verify": [], "fused_zerobit": [], "composite_or": [], "fused_ba": [], "tm": [], "bestpath": [],
         "resync": [], "syncseal": [], "geo_cascade": [], "n_decodes": []}
    resync_angles = np.arange(-args.resync_range, args.resync_range + 0.01, args.resync_step)
    stage_ct = {"syncseal": 0, "refine": 0, "vine_scale": 0, "blind": 0, "gated": 0, "none": 0}   # cascade stage that carried/ended each detection

    def _frag_llr(name, pil, iid):
        kind, getter, _ = SPEC[name]; m = frag[name]; perm, M = m.get_perm_M(iid)
        return method_soft_to_codeword_llr(getattr(m, getter)(pil), perm, M, kind=kind, n_codeword=sb.n)
    def _cv(rl, iid): return bool(decode_and_verify(rl, iid, codec=sb)["detected"])

    def geo_cascade(att, iid, tx):
        """Cheap->expensive geometric fallback, every accept crypto-verify-gated. Returns (det, stage, n_decodes)."""
        nd = 0
        # A. SyncSeal one-shot: rectify ALL geometry (rot/scale/crop/perspective), any fragment
        rect, _ = sync_rectify(sync, att, dev)
        for name in args.fragments:
            rl = _frag_llr(name, rect, iid); nd += 1
            if _cv(rl, iid): return 1.0, "syncseal", nd
        # B. local rotation refine around the rectified frame (TM carrier) -> pushes the near-misses to a hit
        for d in (-3.0, 3.0, -6.0, 6.0):
            rl = _frag_llr("trustmark", rot(rect, d), iid); nd += 1
            if _cv(rl, iid): return 1.0, "refine", nd
        # B2. blind SCALE search for the nested-VINE layers. A crop/zoom leaves each layer at a NON-canonical
        #     scale (layer K under an attack that kept fraction c sits at K/c); VINE's capture width is <1%, so
        #     this grid must stay fine (see --vine_scale_step). crypto-verify is the accept gate as everywhere.
        if "vine" in args.fragments:
            for f in np.arange(0.34, 1.0001, args.vine_scale_step):
                if f >= 0.999:
                    view = att
                else:
                    s = int(round(512 * float(f))); o = (512 - s) // 2
                    view = att.crop((o, o, o + s, o + s))
                rl = _frag_llr("vine", view, iid); nd += 1
                if _cv(rl, iid): return 1.0, "vine_scale", nd
        # C. blind coarse-to-fine angle search on the ORIGINAL image (regen fallback when the sync mark is dead).
        #    COARSE probe with TM only (the rotation carrier); crypto-verify is the accept gate, TM bit-acc guides.
        has_tm = "trustmark" in args.fragments
        best_d, best_ba = 0.0, -1.0
        if has_tm:
            for d in np.arange(-args.resync_range, args.resync_range + 0.01, args.blind_coarse):
                rl = _frag_llr("trustmark", rot(att, float(d)), iid); nd += 1
                if _cv(rl, iid): return 1.0, "blind", nd
                ba = float(np.mean((rl > 0).astype(np.uint8) == tx))
                if ba > best_ba: best_ba, best_d = ba, d
            # EARLY-ABORT GATE: no coarse angle lifts TM bit-acc above the gate -> the loss is INFORMATION
            # (crop/regen), not misalignment, so no fine angle can cross the crypto floor. Skip the fine scan.
            if best_ba < args.blind_gate: return 0.0, "gated", nd
        # FINE search around the best coarse angle (TM + VideoSeal), crypto-verify accept
        search = [n for n in ("trustmark", "videoseal") if n in args.fragments]  # skip slow VINE (never carries rotation)
        for d in np.arange(best_d - args.blind_coarse, best_d + args.blind_coarse + 0.01, args.resync_step):
            rimg = rot(att, float(d))
            for name in search:
                rl = _frag_llr(name, rimg, iid); nd += 1
                if _cv(rl, iid): return 1.0, "blind", nd
        return 0.0, "none", nd
    miss = 0
    for it in meta["items"]:
        i, image_id = it["i"], it["image_id"]
        fp = os.path.join(adir, f"img_{i:05d}.png")
        if not os.path.exists(fp): miss += 1; continue
        tx = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))
        att = Image.open(fp).convert("RGB")
        if att.size != (512, 512): att = att.resize((512, 512))
        al = {}
        for name in args.fragments:
            kind, getter, _ = SPEC[name]
            m = frag[name]; perm, M = m.get_perm_M(image_id)
            al[name] = method_soft_to_codeword_llr(getattr(m, getter)(att), perm, M, kind=kind, n_codeword=sb.n)
        # primary fused codeword: equal-MRC, the position-wise sum of the aligned log-likelihoods
        fused = fuse_llrs(al, weights=None, n_codeword=sb.n)
        fba = float(np.mean(llr_to_bits(fused) == tx))
        fver = float(decode_and_verify(fused, image_id, codec=sb)["detected"])
        fzb = 1.0 if fba >= tau else 0.0
        # best-of-paths: any single-fragment crypto-verify (FPR-safe 37-bit exact; recovers
        # the surviving fragment under attacks where the others die -> no dilution loss)
        bestpath = 0.0
        if use_bestpath:
            for name in args.fragments:
                if bool(decode_and_verify(al[name], image_id, codec=sb)["detected"]): bestpath = 1.0; break
        if use_or_tm:
            tmb = tmbits_for(image_id, tm.n_bits); tmrec = tm.decode(att); nb = min(len(tmrec), len(tmb))
            tmba = float(np.mean(tmrec[:nb] == tmb[:nb])); tmdet = 1.0 if tmba >= 0.75 else 0.0
        else:
            tmba = float("nan"); tmdet = 0.0
        # rotation-resync fallback (crypto-verify-guided angle search): only when @0deg paths all miss.
        # accept iff any fragment crypto-verifies at any candidate angle -> zero false-accept, no CNN.
        resync = 0.0; syncseal = 0.0; geo_casc = 0.0; ndec = 0; primary = (fver or fzb or bestpath or tmdet)
        if args.geo_cascade and not primary:                          # UNIFIED cascade (SyncSeal + refine + blind)
            geo_casc, stg, ndec = geo_cascade(att, image_id, tx)
            stage_ct[stg] += 1
        elif args.resync and not primary:                             # blind angle search only (ablation)
            for d in resync_angles:
                if d == 0.0: continue
                rimg = rot(att, float(d)); hit = False
                for name in args.fragments:
                    ndec += 1
                    if _cv(_frag_llr(name, rimg, image_id), image_id): hit = True; break
                if hit: resync = 1.0; break
        elif args.syncseal and not primary:                           # SyncSeal one-shot only (ablation)
            rect, _ = sync_rectify(sync, att, dev)
            for name in args.fragments:
                ndec += 1
                if _cv(_frag_llr(name, rect, image_id), image_id): syncseal = 1.0; break
        R["fused_verify"].append(fver); R["fused_zerobit"].append(fzb); R["bestpath"].append(bestpath)
        R["resync"].append(resync); R["syncseal"].append(syncseal); R["geo_cascade"].append(geo_casc); R["n_decodes"].append(ndec)
        R["composite_or"].append(1.0 if (primary or resync or syncseal or geo_casc) else 0.0)
        R["fused_ba"].append(fba); R["tm"].append(tmba)
    n = len(R["composite_or"])
    summ = {k: (float(np.mean(v)) if v else None) for k, v in R.items()}
    out = {"attack": args.attack_name, "fragments": args.fragments, "n": n, "missing": miss, "tau": meta["tau"],
           "summary": summ, "cascade_stages": stage_ct}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\n=== composite vs {args.attack_name} (n={n}, missing={miss}) fragments={args.fragments} ===")
    print(f"COMPOSITE_or={summ['composite_or']:.3f}  fused_verify={summ['fused_verify']:.3f}  "
          f"fused_zerobit={summ['fused_zerobit']:.3f}  fused_ba={summ['fused_ba']:.3f}  TM={summ['tm']:.3f}"
          + (f"  bestpath={summ['bestpath']:.3f}  resync={summ['resync']:.3f}" if args.resync else "")
          + (f"  syncseal={summ['syncseal']:.3f}" if args.syncseal else "")
          + (f"  geo_cascade={summ['geo_cascade']:.3f}" if args.geo_cascade else ""))
    if args.resync or args.syncseal or args.geo_cascade:
        fired = [x for x in R["n_decodes"] if x > 0]
        print(f"SPEED: avg fragment-decodes/img over ALL={np.mean(R['n_decodes']):.1f}  over FIRED-only={np.mean(fired) if fired else 0:.1f}  "
              f"(n_fired={len(fired)})" + (f"  stages={stage_ct}" if args.geo_cascade else ""))
    print(f"[done] -> {args.output}\nDECODE_DONE")


if __name__ == "__main__":
    main()
