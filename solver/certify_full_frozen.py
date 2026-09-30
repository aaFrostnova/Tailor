"""Live certification of EVERY request of a class (user 2026-09-06), at the cost of one measurement per
DISTINCT configuration.

Two requests that received the same configuration (fragments, embed order, strengths, front-end) embed
the same images and, under the same attack, produce the same attacked pixels, so measuring once and
letting both read the cells at their own thresholds is the same experiment as measuring twice. Each
distinct configuration is embedded on N cross-source images (pool offset 300: A/B/C 25%, D 15%, E 10%,
disjoint from every fitting slice) and every attack any of its requests named is applied; per image we
keep each fragment's bit accuracy on the view the deployed decoder accepts (primary, or the cascade's
view after a verification miss), its keyed-verification flag, and the decoder's verdict.
  measure <class> <shard> <nshard> [N=100]       in-process + diffusion columns; embeds kept on disk
                                                  for the cross-environment chain when a group names them
  decode_xenv <class> <gidx>                      read att_ctrlregen_s0x / att_unmarker for one group
Verdicts per request are derived afterwards by certify_full_verdicts.py.
"""
import sys, os, json, glob, time, hashlib
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT as CF, WORKSPACE as SC, WMATTACKER_REPO as WMA
for p in (CF, f"{CF}/solver", WMA, SC): sys.path.insert(0, p)
MODE = sys.argv[1]; CLS = int(sys.argv[2]); ARGS = sys.argv[3:]; sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
from eval_matrix import OursComposite
from src.attacks import attack_pil_any
from src.image_pool import sample as _pool_sample, composition_of
from src.soft_bch import decode_and_verify
FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}
LIVE_OK = {"jpeg25", "blur", "noise", "bright", "contrast", "crop75", "crop50", "rot9", "rs256", "hflip",
           "crop_jpeg", "border20", "vaeB", "vaeC", "regen", "rinse2x"}
XENV = {"ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07", "unmarker"}
KEYS = ["C1", "C2", "C3", "C4", "C5"]; K = KEYS[CLS]; dev = "cuda"
OUT = f"{SC}/certify_full/{K}"; os.makedirs(OUT, exist_ok=True)


def groups_of(K):
    """SAT records of the class grouped by exact configuration; deterministic order."""
    recs = []
    for f in sorted(glob.glob(f"{SC}/class_eval_{K}_shard*.json")): recs += json.load(open(f))["records"]
    g = {}
    for r in recs:
        s = r["solver"]
        if not s: continue
        key = (tuple(s["order"]), tuple(sorted(k for k, v in s["fe"].items() if v)), tuple(round(s["s"][f], 3) for f in s["order"]))
        g.setdefault(key, []).append(r)
    keys = sorted(g, key=lambda k: (-len(g[k]), k))
    return [(i, k, g[k]) for i, k in enumerate(keys)]


def composite_for(order, fe_on, strengths):
    fe = {k: (k in fe_on) for k in ("resync", "scale", "angle", "tile")}
    c = OursComposite(dev, tm_variant="B", vine_variant="R",
                      config={"frags": [FKEY[f] for f in order], "order": [FKEY[f] for f in order], "strengths": {}, **W.frontend_config(fe)})
    c.strength = dict(c.DEFAULT_STRENGTH); c.strength.update({FKEY[f]: float(v) for f, v in zip(order, strengths)}); return c


def read(comp, order, img, iid, tx):
    """Two views per image: the PRIMARY read (what the deployed decoder sees first) and, when no keyed test
    passes there, the crypto-verify-gated cascade view (what the decoder falls back to on a miss).

    OursComposite.decode accepts on (any fragment verifies) or (the fused codeword verifies) or (the zero-bit
    presence test at its budget fires), and runs the cascade only on a miss. Whether the primary read is a
    miss depends on the request's budget, so the cell records both views and the verdict combines them at
    each request's own threshold: accepted = keyed pass on the primary view, or primary best-path >= tau, or
    a verified cascade view; the mean is taken over the view the decoder accepts. (Until 2026-09-08 the
    cascade ran only when the 1% presence test failed, so a request at 2^-37 read the unrescued primary view
    wherever a partner fragment cleared 1%: crop50 under the ring read 0.42 of images at 0.90 while the
    deployed decoder accepted 0.98.) `det` stays the decoder's own verdict at its 1% default."""
    from eval_matrix import presence_detected
    from src.soft_fusion import fuse_llrs, llr_to_bits
    def _read(v):
        L = {f: comp._frag_llr(FKEY[f], v, iid) for f in order}
        ba = {f: float(np.mean((L[f] > 0).astype(np.uint8) == tx)) for f in order}
        ver = {f: bool(decode_and_verify(L[f], iid, codec=comp.sb)["detected"]) for f in order}
        fused = fuse_llrs({FKEY[f]: L[f] for f in order}, weights=None, n_codeword=comp.sb.n)
        fba = float(np.mean(llr_to_bits(fused) == tx))
        fver = bool(decode_and_verify(fused, iid, codec=comp.sb)["detected"])
        idv = fver or any(ver.values())
        return ba, ver, idv, (idv or presence_detected(ba, fba, len(order)))
    ba, ver, idv, det = _read(img)
    cas = None
    if comp.geo and not idv:
        ok, vw = comp.geo_cascade(img, iid, tx, return_view=True)
        if ok and vw is not None:
            ba2, ver2, idv2, _ = _read(vw); cas = (ba2, ver2)
    return ba, ver, bool(idv), float(det or cas is not None), cas


def cell_of(order, rows):
    """Pack per-image reads into the two-view cell format (views = 2)."""
    ba = {f: [r[0][f] for r in rows] for f in order}; ver = {f: [r[1][f] for r in rows] for f in order}
    return {"views": 2, "ba": ba, "ver": ver, "idv": [r[2] for r in rows], "det": [r[3] for r in rows],
            "cas_ok": [r[4] is not None for r in rows],
            "cas_ba": {f: [(r[4][0][f] if r[4] else None) for r in rows] for f in order},
            "cas_ver": {f: [(r[4][1][f] if r[4] else None) for r in rows] for f in order}}


def is_measured(rec, a, geo_cfg):
    """A cell counts as measured when present and, for a configuration with a geometric stage, in the two-view format."""
    c = rec["attacks"].get(a)
    return c is not None and (c.get("views") == 2 or not geo_cfg)


def r512(img): return img if img.size == (512, 512) else img.resize((512, 512))


# Image set. In-domain = the pool's held-out slice (offset 300; the five sources the offline table was
# fitted on, different images). Out-of-domain = a directory of images from a source the pool never
# contained (CERT_IMAGES=<dir>); its results land in certify_full_<domain>/ so the two domains can be
# compared cell by cell to show how far the offline table transfers.
DOMAIN = os.environ.get("CERT_DOMAIN", "indomain"); IMG_DIR = os.environ.get("CERT_IMAGES", "")
if DOMAIN != "indomain": OUT = f"{SC}/certify_full_{DOMAIN}/{K}"; os.makedirs(OUT, exist_ok=True)

if MODE == "measure":
    SHARD, NSHARD = int(ARGS[0]), int(ARGS[1]); N = int(ARGS[2]) if len(ARGS) > 2 else 100
    G = groups_of(K); mine = [g for g in G if g[0] % NSHARD == SHARD]
    if IMG_DIR:
        files = sorted(glob.glob(f"{IMG_DIR}/*.png") + glob.glob(f"{IMG_DIR}/*.jpg"))[:N]
        comp_desc = f"{DOMAIN}:{os.path.basename(IMG_DIR)} n={len(files)}"
    else:
        files = _pool_sample(N, offset=300); comp_desc = composition_of(files)
    covers = [Image.open(f).convert("RGB").resize((512, 512), Image.BICUBIC) for f in files]
    print(f"{K}: {len(G)} distinct configurations, shard {SHARD}/{NSHARD} takes {len(mine)}; N={N} {comp_desc}", flush=True)
    for gi, key, members in mine:
        order, fe_on, strengths = key
        path = f"{OUT}/g{gi:04d}.json"
        rec = json.load(open(path)) if os.path.exists(path) else {"gidx": gi, "cls": K, "order": list(order), "fe": list(fe_on),
              "s": {f: v for f, v in zip(order, strengths)}, "members": [m["i"] for m in members], "n_img": N,
              "images": comp_desc, "domain": DOMAIN, "attacks": {}}
        want = sorted({a for m in members for a in m["attacks"]})
        geo_cfg = any(st in fe_on for st in ("resync", "scale", "angle", "tile"))
        todo = [a for a in want if a in LIVE_OK and not is_measured(rec, a, geo_cfg)]
        need_x = [a for a in want if a in XENV]
        if not todo and not need_x: print(f"  g{gi:04d} already measured", flush=True); continue
        comp = composite_for(order, fe_on, strengths); t0 = time.time()
        embeds = []
        for i, cov in enumerate(covers):
            emb, sec = comp.embed(cov, i); embeds.append((r512(emb), sec))
        if need_x:                                             # keep the embeds for the cross-environment chain
            ed = f"{OUT}/g{gi:04d}/embed"; os.makedirs(ed, exist_ok=True)
            for i, (emb, _) in enumerate(embeds): emb.save(f"{ed}/i{i:05d}.png")
            rec["xenv_needed"] = need_x
        rec["psnr_db"] = float(np.mean([10 * np.log10(255.0 ** 2 / max(np.mean((np.asarray(c, np.float64) - np.asarray(e, np.float64)) ** 2), 1e-9))
                                        for c, (e, _) in zip(covers, embeds)]))
        for a in todo:
            rows = []
            for emb, (iid, tx) in embeds:
                v = r512(attack_pil_any(a, emb, dev=dev)); rows.append(read(comp, order, v, iid, tx))
            cell = cell_of(order, rows); rec["attacks"][a] = cell
            json.dump(rec, open(path, "w"))
            print(f"  g{gi:04d} {'+'.join(order)}{'/' + '+'.join(fe_on) if fe_on else ''} s={list(strengths)} "
                  f"n_req={len(members)} {a:13s} best-path {np.mean([max(x) for x in zip(*[cell['ba'][f] for f in order])]):.3f} "
                  f"det {np.mean(cell['det']):.2f} cascade {np.mean(cell['cas_ok']):.2f} [{time.time()-t0:.0f}s]", flush=True)
        json.dump(rec, open(path, "w"))
    print("MEASURE_DONE", flush=True)

elif MODE == "decode_xenv":
    gi = int(ARGS[0]); path = f"{OUT}/g{gi:04d}.json"; rec = json.load(open(path))
    order, fe_on = rec["order"], rec["fe"]; comp = composite_for(order, fe_on, [rec["s"][f] for f in order])
    for a in rec.get("xenv_needed", []):
        files = sorted(glob.glob(f"{OUT}/g{gi:04d}/att_{a}/i*.png"))
        if not files: print(f"  g{gi:04d} {a}: no attacked images", flush=True); continue
        rows = []
        for fp in files:
            i = int(os.path.basename(fp)[1:6]); iid, tx = comp.secret_for(i)
            rows.append(read(comp, order, r512(Image.open(fp).convert("RGB")), iid, tx))
        cell = cell_of(order, rows); cell["n"] = len(files); rec["attacks"][a] = cell
        print(f"  g{gi:04d} {a:13s} n={len(files)} best-path {np.mean([max(x) for x in zip(*[cell['ba'][f] for f in order])]):.3f} det {np.mean(cell['det']):.2f} cascade {np.mean(cell['cas_ok']):.2f}", flush=True)
    json.dump(rec, open(path, "w")); print("XENV_DECODE_DONE", flush=True)
else:
    raise SystemExit(f"unknown mode {MODE}")
