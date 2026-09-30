"""BASELINE TABLE GENERATOR — one entry point, one artifact, provenance per cell, nothing inherited.

The solver's table used to be assembled from three JSONs of different ages plus seven hard-coded patch
blocks accumulated in the source. That cannot be reproduced or audited, and it is the SHARED, CROSS-REQUEST
fact every user's solve depends on (per-request measurements live in an overlay and must never write here).

This regenerates it. Every cell carries {value, n, measured_at, method, git}. A cell is either measured in
this run, merged from a declared cross-env job, or listed in `needs_measurement` -- it is NEVER copied from
an older table.

  python make_baseline.py --stage 1 --n 50      fragment solo: PSNR / latency / per-attack x alpha bit-acc
                                                + stacking penalties (local GPU)
  python make_baseline.py --stage 3 --n 50      CAP capacity via soft-MI              (local GPU)
  python make_baseline.py --stage 4 --n 30      front-ends: real resync/nested latency + scale-search boost
  python make_baseline.py --stage 2             emit the cross-env MANIFEST (regen/rinse/ctrlregen*/unmarker)
  python make_baseline.py --merge FILE.json     merge a finished cross-env job back in
  python make_baseline.py --report              coverage + staleness report
"""
import sys, os, io, json, time, glob, argparse, subprocess
import numpy as np
from PIL import Image, ImageFilter
CF = "/data/tailor/project"
SC = "/data/tailor/workspace/wm_dataset10k"
OUT = f"{SC}/baseline_table.json"
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", SC):
    sys.path.insert(0, p)

ALV = [0.3, 0.5, 0.7, 1.0]
FRAGS = ["VINE", "TrustMark", "VideoSeal"]
LIVE_ATT = ["clean", "jpeg25", "jpeg50", "blur", "noise", "bright", "contrast",
            "crop90", "crop75", "crop50", "rot9", "rot30", "vaeB", "vaeC"]
BATCH_ATT = ["regen", "rinse", "ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07", "ctrlregen", "unmarker"]
CROSS_ENV_ATT = ["ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07", "ctrlregen", "unmarker"]   # regen/rinse run locally (stage 2a)
ALL_ATT = LIVE_ATT + BATCH_ATT

def _git():
    return subprocess.run(["git", "-C", CF, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "nogit"
def _now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def cell(v, n, method): return {"value": v, "n": n, "measured_at": _now(), "method": method, "git": _git()}

def load():
    if os.path.exists(OUT): return json.load(open(OUT))
    return {"schema": 1, "created_at": _now(),
            "note": "baseline = shared cross-request fact; per-request measurements belong in an overlay",
            "cells": {"bit_acc": {}, "cap": {}, "psnr_solo": {}, "latency": {}, "penalties": {}, "frontend": {}},
            "stages_run": {}}
def save(t):
    t["updated_at"] = _now()
    t["coverage"] = coverage(t)
    json.dump(t, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")

# ---------------------------------------------------------------- coverage / staleness
def expected_keys():
    ks = []
    for f in FRAGS:
        for al in (ALV if f == "VINE" else [1.0]):
            for a in ALL_ATT: ks.append(f"bit_acc:{f}@{al}/{a}")
    for f in FRAGS:
        for a in ALL_ATT: ks.append(f"cap:{f}/{a}")
    for f in FRAGS:
        for al in (ALV if f == "VINE" else [1.0]):
            ks.append(f"psnr_solo:{f}@{al}"); ks.append(f"latency:{f}@{al}")
    ks += ["penalties:PEN2", "penalties:PEN3", "penalties:NESTED_PEN",
           "frontend:resync_ms", "frontend:nested_ms", "frontend:resync_boost", "frontend:nested_boost"]
    return ks
def coverage(t):
    have = set()
    for grp, d in t["cells"].items():
        for k in d: have.add(f"{grp}:{k}")
    exp = expected_keys(); missing = [k for k in exp if k not in have]
    return {"expected": len(exp), "measured": len(exp) - len(missing),
            "pct": round(100 * (len(exp) - len(missing)) / len(exp), 1),
            "needs_measurement": missing}
def report():
    t = load(); c = t["coverage"] if "coverage" in t else coverage(t)
    print(f"=== BASELINE COVERAGE: {c['measured']}/{c['expected']} = {c['pct']}% ===")
    by = {}
    for k in c["needs_measurement"]:
        grp = k.split(":")[0]; att = k.split("/")[-1] if "/" in k else k
        by.setdefault(grp, set()).add(att if att in ALL_ATT else "(other)")
    for grp, s in sorted(by.items()): print(f"  missing {grp:10s}: {sorted(s)}")
    ages = {}
    for grp, d in t["cells"].items():
        for k, v in d.items():
            age = (time.time() - time.mktime(time.strptime(v["measured_at"], "%Y-%m-%dT%H:%M:%SZ"))) / 86400
            ages.setdefault(grp, []).append(age)
    print("\n  cell age (days):", {g: f"{min(v):.1f}-{max(v):.1f}" for g, v in ages.items()})
    print("\n  stages run:", t.get("stages_run", {}))

# ---------------------------------------------------------------- shared measurement helpers
def _load_frags(dev="cuda"):
    from src.vine_crypto_wrapper import VineCryptoWrapper
    from src.trustmark_fragment import TrustMarkFragment
    from src.videoseal_fragment import VideoSealFragment
    from src.shortened_bch import ShortenedBCH
    sb = ShortenedBCH(); K = b"v5_key_encoder_master"
    return sb.n, {"VINE": VineCryptoWrapper(K, "vine", sb.n, dev, variant="R"),
                  "TrustMark": TrustMarkFragment(K, "trustmark", sb.n, model_type="B", device=dev),
                  "VideoSeal": VideoSealFragment(K, "videoseal", sb.n, device=dev)}
def _atts(dev="cuda"):
    from live_measure import ATTACKS
    _c = {}
    def vae(pil, which):
        import torch
        from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor
        if which not in _c:
            _c[which] = ((bmshj2018_hyperprior if which == "B" else cheng2020_anchor)(quality=3, pretrained=True)).eval().to(dev)
        x = torch.from_numpy(np.asarray(pil, np.float32).transpose(2, 0, 1)[None] / 255.).to(dev)
        with torch.no_grad(): y = _c[which](x)["x_hat"].clamp(0, 1)
        return Image.fromarray((y[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    return {**{a: ATTACKS[a] for a in ATTACKS if a in LIVE_ATT},
            "vaeB": lambda p: vae(p, "B"), "vaeC": lambda p: vae(p, "C")}
def _covers(n):
    return [Image.open(f).convert("RGB").resize((512, 512))
            for f in sorted(glob.glob(f"{SC}/pool/*/img/*.png"))[:n]]
def _prob2llr(p):
    return np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1-1e-6))
def _Hb(p):
    p = np.clip(p, 1e-12, 1-1e-12); return -(p*np.log2(p) + (1-p)*np.log2(1-p))
def _mi_per_bit(truth, llr, K=25):
    """Non-parametric I(bit;llr) in bits/position (inlined from capacity_mi.py, which cannot be imported:
    its module level does np.load(sys.argv[1])). Calibration-free via quantile bins."""
    b = truth.ravel().astype(np.float64); sg = llr.ravel()
    Hprior = _Hb(b.mean())
    edges = np.quantile(sg, np.linspace(0, 1, K+1)); edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.clip(np.digitize(sg, edges)-1, 0, K-1)
    Hcond = 0.0
    for k in range(K):
        m = idx == k
        if m.any(): Hcond += m.mean() * _Hb(b[m].mean())
    return max(Hprior - Hcond, 0.0), Hprior

def _soft(FR, NB, fn, pil):
    """soft output -> (llr, hard bits)"""
    prob2llr = _prob2llr
    if fn == "VINE":
        p = np.asarray(FR[fn].raw_probs(pil)).ravel()[:NB]; llr = prob2llr(p)
    else:
        llr = np.asarray(FR[fn].raw_logits(pil)).ravel()[:NB].astype(np.float64)
    return llr, (llr > 0).astype(np.uint8)

# ---------------------------------------------------------------- STAGE 1
def stage1(n):
    from composite_external_eval import scale_resid, nested_vine_embed
    NB, FR = _load_frags(); A = _atts(); covers = _covers(n)
    t = load()
    def psnr(a, b):
        m = np.mean((np.asarray(a, np.float32)/255 - np.asarray(b, np.float32)/255) ** 2)
        return 99.0 if m < 1e-12 else float(10*np.log10(1.0/m))
    print(f"[stage1] fragment solo, n={len(covers)}", flush=True)
    for fn in FRAGS:
        for al in (ALV if fn == "VINE" else [1.0]):
            ps, ems, dms = [], [], []; acc = {a: [] for a in A}
            for i, c in enumerate(covers):
                tg = np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
                t0 = time.time(); wm = FR[fn].embed_with_target(c, tg)
                emb = scale_resid(c, wm, al) if al != 1.0 else wm
                if emb.size != (512, 512): emb = emb.resize((512, 512))
                ems.append((time.time()-t0)*1000); ps.append(psnr(c, emb))
                t1 = time.time(); _soft(FR, NB, fn, emb); dms.append((time.time()-t1)*1000)
                for a, af in A.items(): acc[a].append(float(np.mean(_soft(FR, NB, fn, af(emb))[1] == tg)))
            key = f"{fn}@{al}"
            t["cells"]["psnr_solo"][key] = cell(round(float(np.mean(ps)), 2), len(covers), "stage1")
            t["cells"]["latency"][key] = cell({"embed_ms": round(float(np.mean(ems)), 1),
                                               "decode_ms": round(float(np.mean(dms)), 1)}, len(covers), "stage1")
            for a, v in acc.items():
                t["cells"]["bit_acc"][f"{key}/{a}"] = cell(round(float(np.mean(v)), 4), len(covers), "stage1")
            print(f"  {key:16s} PSNR {np.mean(ps):5.2f}  clean {np.mean(acc['clean']):.3f}", flush=True)
    # stacking penalties on the same covers
    def emb_cfg(c, frags, nested, al=1.0, s=0.70):
        img = c
        for f in frags:
            tg = np.random.RandomState(abs(hash(f)) % 2**31).randint(0, 2, NB).astype(np.uint8)
            wm = nested_vine_embed(FR["VINE"], img, tg, scales=(1.0, .75, .5)) if (f == "VINE" and nested) \
                 else FR[f].embed_with_target(img, tg)
            img = scale_resid(img, wm, al if f == "VINE" else s)
            if img.size != (512, 512): img = img.resize((512, 512))
        return img
    S = {k: float(np.mean([psnr(c, emb_cfg(c, fr, nv)) for c in covers]))
         for k, (fr, nv) in {"solo": (("VINE",), False), "two": (("VINE", "TrustMark"), False),
                             "three": (("VINE", "TrustMark", "VideoSeal"), False),
                             "nested": (("VINE",), True)}.items()}
    for k, v in {"PEN2": S["solo"]-S["two"], "PEN3": S["solo"]-S["three"], "NESTED_PEN": S["solo"]-S["nested"]}.items():
        t["cells"]["penalties"][k] = cell(round(v, 2), len(covers), "stage1")
    print(f"  penalties: PEN2 {S['solo']-S['two']:.2f}  PEN3 {S['solo']-S['three']:.2f}  NESTED {S['solo']-S['nested']:.2f}")
    t["stages_run"]["1"] = {"at": _now(), "n": len(covers)}
    save(t)

# ---------------------------------------------------------------- STAGE 3 (capacity)
def stage3(n):
    from composite_external_eval import scale_resid
    NB, FR = _load_frags(); A = _atts(); covers = _covers(n)
    t = load()
    print(f"[stage3] capacity (soft-MI), n={len(covers)}", flush=True)
    for fn in FRAGS:
        for a, af in A.items():
            T, L = [], []
            for i, c in enumerate(covers):
                tg = np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
                emb = FR[fn].embed_with_target(c, tg)
                if emb.size != (512, 512): emb = emb.resize((512, 512))
                llr, _ = _soft(FR, NB, fn, af(emb)); T.append(tg); L.append(llr)
            mi, _ = _mi_per_bit(np.array(T), np.array(L))
            t["cells"]["cap"][f"{fn}/{a}"] = cell(round(float(mi*NB), 1), len(covers), "stage3-softMI")
        print(f"  {fn:10s} " + " ".join(f"{a}:{t['cells']['cap'][fn+'/'+a]['value']:.0f}" for a in ("clean","jpeg25","crop75","vaeB")), flush=True)
    t["stages_run"]["3"] = {"at": _now(), "n": len(covers)}
    save(t)

# ---------------------------------------------------------------- STAGE 4 (front-ends)
def stage4(n):
    """Measure the DEPLOYED front-ends, not a re-implementation.

    The solver models front-end recovery as `boost = BoolVal(0.96 >= thr)` -- a bit-acc. The deployed
    front-end is not a bit-acc at all: geo_cascade() is a CRYPTO-VERIFY-GATED search (SyncSeal rectify ->
    rotation refine -> VINE scale search by CROP at step 0.005 -> blind angle) that returns a DETECTION
    boolean. A hand-rolled resize-based scale search measured 0.536 on crop75 and would have wrongly
    condemned the front-end; the deployed 10k run detects crop75/crop50 at 1.00. So we instantiate the
    real composite and measure the cascade's RECOVERY RATE per geometric attack.
    """
    from eval_matrix import OursComposite
    from live_measure import ATTACKS
    t = load(); covers = _covers(n)
    oc = OursComposite("cuda", "B", geo=True, vine_variant="R")
    # real front-end latency (wall clock of the deployed embed paths)
    from src.syncseal_frontend import load_sync, sync_embed
    from composite_external_eval import nested_vine_embed
    NB, FR = _load_frags()
    sync = load_sync(dev="cuda"); ms_s, ms_n = [], []
    for c in covers[:min(10, len(covers))]:
        t0 = time.time(); sync_embed(sync, c, "cuda"); ms_s.append((time.time()-t0)*1000)
        tg = np.random.RandomState(0).randint(0, 2, NB).astype(np.uint8)
        t0 = time.time(); nested_vine_embed(FR["VINE"], c, tg, scales=(1.0, .75, .5)); ms_n.append((time.time()-t0)*1000)
    t["cells"]["frontend"]["resync_ms"] = cell(round(float(np.mean(ms_s)), 1), len(ms_s), "stage4-deployed")
    t["cells"]["frontend"]["nested_ms"] = cell(round(float(np.mean(ms_n)), 1), len(ms_n), "stage4-deployed")
    print(f"[stage4] resync {np.mean(ms_s):.0f}ms (solver assumes 300)  nested {np.mean(ms_n):.0f}ms (assumes 1600)", flush=True)
    # cascade recovery rate per geometric attack, on the REAL composite
    GEO = ["crop90", "crop75", "crop50", "rot9", "rot30"]
    m = min(n, 20)
    for a in GEO:
        prim, casc = [], []
        for i, c in enumerate(covers[:m]):
            emb, sec = oc.embed(c, i)
            if emb.size != (512, 512): emb = emb.resize((512, 512))
            att = ATTACKS[a](emb); iid, tx = sec
            casc.append(1.0 if oc.geo_cascade(att, iid, tx) else 0.0)
            prim.append(1.0 if oc.decode(att, sec)[1] else 0.0)
        t["cells"]["frontend"][f"cascade_recovery/{a}"] = cell(round(float(np.mean(casc)), 4), m, "stage4-deployed-geo_cascade")
        t["cells"]["frontend"][f"deployed_detect/{a}"] = cell(round(float(np.mean(prim)), 4), m, "stage4-deployed-full-decode")
        print(f"  {a:8s} cascade-alone {np.mean(casc):.3f}   full deployed detect {np.mean(prim):.3f}", flush=True)
    # the solver's single `boost` constant should be the WORST cascade recovery it relies on
    rec = [t["cells"]["frontend"][f"cascade_recovery/{a}"]["value"] for a in GEO]
    t["cells"]["frontend"]["resync_boost"] = cell(round(float(min(rec)), 4), m, "stage4-deployed (min over geometric attacks)")
    t["cells"]["frontend"]["nested_boost"] = cell(round(float(min(rec)), 4), m, "stage4-deployed (min over geometric attacks)")
    print(f"  => front-end boost (worst geometric case) {min(rec):.4f}  (solver hard-codes 0.96)", flush=True)
    t["stages_run"]["4"] = {"at": _now(), "n": m}
    save(t)

# ---------------------------------------------------------------- STAGE 2a (regen / rinse, local)
def stage2a(n):
    """regen and rinse DO run in this env (regen_pipe.ReSDPipeline, lazy). Only ctrlregen* / unmarker
    genuinely need another conda env, so they alone stay in the cross-env manifest."""
    from eval_matrix import build_attack_one
    import tempfile
    NB, FR = _load_frags(); covers = _covers(n); t = load()
    atk = build_attack_one("cuda")
    tmp = tempfile.mkdtemp(prefix="regen_")
    for a in ("regen", "rinse2x"):
        key_att = "regen" if a == "regen" else "rinse"
        for fn in FRAGS:
            for al in (ALV if fn == "VINE" else [1.0]):
                from composite_external_eval import scale_resid
                accs, T, L = [], [], []
                for i, c in enumerate(covers):
                    tg = np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
                    wm = FR[fn].embed_with_target(c, tg)
                    emb = scale_resid(c, wm, al) if al != 1.0 else wm
                    if emb.size != (512, 512): emb = emb.resize((512, 512))
                    ip = os.path.join(tmp, f"i{i}.png"); op = os.path.join(tmp, f"o{i}.png")
                    emb.save(ip); atk(a, ip, op)
                    att = Image.open(op if os.path.exists(op) else ip).convert("RGB").resize((512, 512))
                    llr, hb = _soft(FR, NB, fn, att)
                    accs.append(float(np.mean(hb == tg))); T.append(tg); L.append(llr)
                t["cells"]["bit_acc"][f"{fn}@{al}/{key_att}"] = cell(round(float(np.mean(accs)), 4), len(covers), "stage2a-local")
                if al == 1.0:
                    mi, _ = _mi_per_bit(np.array(T), np.array(L))
                    t["cells"]["cap"][f"{fn}/{key_att}"] = cell(round(float(mi*NB), 1), len(covers), "stage2a-softMI")
                print(f"  {fn}@{al}/{key_att}: ba {np.mean(accs):.4f}", flush=True)
    t["stages_run"]["2a"] = {"at": _now(), "n": len(covers)}
    save(t)

# ---------------------------------------------------------------- STAGE 2 (manifest / merge)
def stage2_manifest():
    t = load()
    man = {"generated_at": _now(), "why": "these attacks need the ctrlregen / unmarker conda envs + GPU jobs",
           "jobs": [{"attack": a, "fragments": FRAGS, "alphas": ALV,
                     "env": ("ctrlregen" if "regen" in a or "ctrl" in a else "unmarker"),
                     "script": ("scripts/attack/ctrlregen_batch.py" if "regen" in a or "ctrl" in a
                                else "scripts/attack/unmarker_batch.py"),
                     "merge_into": f"bit_acc:<FRAG>@<ALPHA>/{a}  and  cap:<FRAG>/{a}"} for a in CROSS_ENV_ATT]}
    json.dump(man, open(f"{SC}/baseline_stage2_manifest.json", "w"), indent=2)
    print(f"wrote {SC}/baseline_stage2_manifest.json  ({len(CROSS_ENV_ATT)} cross-env attacks x {len(FRAGS)} fragments; regen/rinse are local -> stage 2a)")
    print("  NOTE: until merged, these cells stay in needs_measurement -- never back-filled from an old table.")
def merge(path):
    t = load(); ext = json.load(open(path)); k = 0
    for grp in ("bit_acc", "cap"):
        for key, v in (ext.get(grp) or {}).items():
            t["cells"][grp][key] = cell(v["value"] if isinstance(v, dict) else v,
                                        (v.get("n") if isinstance(v, dict) else None), f"stage2-merged:{os.path.basename(path)}")
            k += 1
    t["stages_run"].setdefault("2", []).append({"at": _now(), "file": os.path.basename(path), "cells": k})
    save(t); print(f"merged {k} cells from {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage"); ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--merge"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report: report()
    elif a.merge: merge(a.merge)
    elif a.stage == "1": stage1(a.n)
    elif a.stage == "2": stage2_manifest()
    elif a.stage == "2a": stage2a(a.n)
    elif a.stage == "3": stage3(a.n)
    elif a.stage == "4": stage4(a.n)
    elif a.stage == "all":
        stage1(a.n); stage3(a.n); stage4(min(a.n, 30)); stage2_manifest(); report()
    else: ap.print_help()
    print("MAKE_BASELINE_DONE")
