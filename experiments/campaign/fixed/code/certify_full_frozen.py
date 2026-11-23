"""Live certification of EVERY request of a class (user 2026-09-06), at the cost of one measurement per
DISTINCT configuration.

Two requests that received the same configuration (fragments, embed order, strengths, front-end) embed
the same images and, under the same attack, produce the same attacked pixels, so measuring once and
letting both read the cells at their own thresholds is the same experiment as measuring twice. Each
distinct configuration is embedded on N cross-source images (pool offset 300: A/B/C 25%, D 15%, E 10%,
disjoint from every fitting slice) and every attack any of its requests named is applied; per image we
keep threshold-specific strict BA and each threshold's first accepted primary/geometric view.
No BCH/ID verifier is an acceptance route. Geometric search is charged in the full K budget.
  measure <class> <shard> <nshard> [N=100]       in-process + diffusion columns; embeds kept on disk
                                                  for the cross-environment chain when a group names them
  decode_xenv <class> <gidx>                      read att_ctrlregen_s0x / att_unmarker for one group
Verdicts per request are derived afterwards by certify_full_verdicts.py.
"""
import sys, os, json, glob, time, hashlib
import numpy as np
from pathlib import Path
RUN_ROOT = Path(__file__).resolve().parents[2]
from PIL import Image
CF = os.environ.get('TAILOR_PROJECT', '.')
SC = os.environ.get('TAILOR_WORKSPACE', 'workspace')
for p in (CF, f"{CF}/solver", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
sys.path.insert(0, str(RUN_ROOT / "code"))
sys.path.insert(0, str(RUN_ROOT / "fixed/code"))
MODE = sys.argv[1]; CLS = int(sys.argv[2]); ARGS = sys.argv[3:]; sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(__file__))
import watermark_smt_v2 as W
import capacity_protocol as CP
from unified_detector import decode_thresholds_detailed
from unified_measurement import cell_of_detailed
from eval_matrix import OursComposite
from src.attacks import attack_pil_any
from src.image_pool import sample as _pool_sample, composition_of
FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}
LIVE_OK = {"jpeg25", "blur", "noise", "bright", "contrast", "crop75", "crop50", "rot9", "rs256", "hflip",
           "crop_jpeg", "border20", "vaeB", "vaeC", "regen", "rinse2x"}
XENV = {"ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07", "unmarker", 'editing_ip2p_s20_v1'}
KEYS = ["C1", "C2", "C3", "C4", "C5"]; K = KEYS[CLS]; dev = "cuda"
OUT = f"{RUN_ROOT}/certify_full/{K}"; os.makedirs(OUT, exist_ok=True)


def groups_of(K):
    """SAT records of the class grouped by exact configuration; deterministic order."""
    recs = []
    for f in sorted(glob.glob(f"{RUN_ROOT}/class_eval_{K}_shard*.json")): recs += json.load(open(f))["records"]
    g = {}
    for r in recs:
        s = r["solver"]
        if not s: continue
        key = (tuple(s["order"]), tuple(sorted(k for k, v in s["fe"].items() if v)), tuple(float(s["s"][f]) for f in s["order"]))
        g.setdefault(key, []).append(r)
    keys = sorted(g, key=lambda k: (-len(g[k]), k))
    return [(i, k, g[k]) for i, k in enumerate(keys)]


def composite_for(order, fe_on, strengths):
    fe = {k: (k in fe_on) for k in ("resync", "scale", "angle", "tile")}
    c = OursComposite(dev, tm_variant="B", vine_variant="R",
                      config={"frags": [FKEY[f] for f in order], "order": [FKEY[f] for f in order], "strengths": {}, **W.frontend_config(fe)})
    c.strength = dict(c.DEFAULT_STRENGTH); c.strength.update({FKEY[f]: float(v) for f, v in zip(order, strengths)}); return c


def read(comp, order, img, iid, tx):
    """Fresh strict-BA primary and complete geometry reads at all six budgets."""
    return decode_thresholds_detailed(comp, img, (iid, tx), CP.REQUEST_ALPHAS)


def cell_of(order, rows):
    return cell_of_detailed(order, rows)


def is_measured(rec, attack, geo_cfg):
    n = 30 if attack == 'unmarker' else rec['n_img']
    return CP.valid_cell(rec['attacks'].get(attack), rec['order'], n,
                         cfg=dict(order=rec['order'], fe_on=rec['fe']))


def r512(img): return img if img.size == (512, 512) else img.resize((512, 512))


# Image set. In-domain = the pool's held-out slice (offset 300; the five sources the offline table was
# fitted on, different images). Out-of-domain = a directory of images from a source the pool never
# contained (CERT_IMAGES=<dir>); its results land in certify_full_<domain>/ so the two domains can be
# compared cell by cell to show how far the offline table transfers.
DOMAIN = os.environ.get("CERT_DOMAIN", "indomain"); IMG_DIR = os.environ.get("CERT_IMAGES", "")
if DOMAIN != "indomain": OUT = f"{RUN_ROOT}/certify_full_{DOMAIN}/{K}"; os.makedirs(OUT, exist_ok=True)

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
                  f"n_req={len(members)} {a:13s} thresholds={list(cell['by_threshold'])} n={cell['n']} [{time.time()-t0:.0f}s]", flush=True)
        json.dump(rec, open(path, "w"))
    print("MEASURE_DONE", flush=True)

elif MODE == "decode_xenv":
    gi = int(ARGS[0]); path = f"{OUT}/g{gi:04d}.json"; rec = json.load(open(path))
    order, fe_on = rec["order"], rec["fe"]; comp = composite_for(order, fe_on, [rec["s"][f] for f in order])
    for a in rec.get("xenv_needed", []):
        files = sorted(glob.glob(f"{OUT}/g{gi:04d}/att_{a}/i*.png"))
        expected = 30 if a == 'unmarker' else rec['n_img']
        indices = [int(os.path.basename(fp)[1:6]) for fp in files]
        if indices != list(range(expected)):
            raise RuntimeError(f'incomplete attacked image set for {a}: {len(files)}/{expected}')
        rows = []
        for fp in files:
            i = int(os.path.basename(fp)[1:6]); iid, tx = comp.secret_for(i)
            rows.append(read(comp, order, r512(Image.open(fp).convert("RGB")), iid, tx))
        cell = cell_of(order, rows); cell["n"] = len(files); rec["attacks"][a] = cell
        print(f"  g{gi:04d} {a:13s} n={len(files)} unified thresholds={list(cell['by_threshold'])}", flush=True)
    tmp = path + f".tmp{os.getpid()}"
    json.dump(rec, open(tmp, "w")); os.replace(tmp, path)
    print("XENV_DECODE_DONE", flush=True)
else:
    raise SystemExit(f"unknown mode {MODE}")
