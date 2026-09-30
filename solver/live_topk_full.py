"""Top-k candidates validated by the live loop (user 2026-09-09).

The offline margin decides how often the certified optimum still holds on the user's images; the margin
run asks the table for 0.96 where the deployment needs 0.90 and pays 1.0 to 1.2 dB per class for it.
Here the margin is reduced and the solver returns its k best distinct skeletons (fragments, embed order,
stages), each at its certified optimum. The live loop measures the candidates in fidelity order on the
held-out images with the certification's own two-view read and deploys the first one that passes every
column at the request's bare threshold (rate >= 0.90 per image, mean >= tau; capacity is a reported output) and the
fidelity floor on the measured embed.

  enumerate <class> <n_req> [k=3] [shard nshard]   sample requests of the class, enumerate top-k at RATE_MARGIN
  measure   <class> [N=100]                          walk the candidates on the GPU; cells cached per configuration
  measure_rank <class> <rank> <shard> <nshard> [N]   batch form of the walk for arrays: measure the cells of the rank-r
                                                     candidates of every request still open after rank r-1 (this shard's
                                                     configurations); run `walk` between ranks
  walk      <class> [N=100]                          verdicts from the cached cells (no GPU): deploys the first passing
                                                     rank per request, leaves a request open where a cell is missing
  patch     <class> <shard> <nshard>                 rank-4 candidate for every request all of whose candidates failed:
                                                     the measured cells patch the table (mean offset gated by the live
                                                     standard error, per-image values shifted to the live rate) and the
                                                     request is re-solved on the patched table; written to patched_s*.json
  verdict   <class>                                  pass@rank, delivered fidelity, comparison with the margin run
  xenv_count <class> / xenv_mark <class>             cross-environment columns (CtrlRegen+, UnMarker; class C4): measure_rank
                                                     exports each configuration's embeds as a certify_full group under
                                                     $SC/certify_full_topk_<K>_<tag>/<K>/ so certify_full_xenv.sbatch (with
                                                     CERT_DOMAIN=topk_<K>_<tag>) attacks and decodes them; xenv_count prints
                                                     the groups still owing the chain, xenv_mark retires the decoded ones
Env: ALLOW_XENV=1 lets `enumerate` sample requests that name cross-environment columns (C4). RATE_MARGIN (default 0.04), MEAN_MARGIN (default W.DEFAULT_MARGIN), TAG (default rm<margin>), SEED (default 0).
Outputs under $SC/live_topk/<class>_<tag>/; cells under $SC/live_topk/cells/ (shared by every tag and class).
"""
import sys, os, json, glob, time, hashlib, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT as CF, WORKSPACE as SC, WMATTACKER_REPO as WMA
for p in (CF, f"{CF}/solver", WMA, SC): sys.path.insert(0, p)
MODE = sys.argv[1]; CLS = int(sys.argv[2]); ARGS = sys.argv[3:]; sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
import watermark_smt_topk as T
import capacity_protocol as CP
from pathlib import Path
RUN_INPUTS = Path(__file__).resolve().parent.parent / "inputs"
FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}
LIVE_OK = {"jpeg25", "blur", "noise", "bright", "contrast", "crop75", "crop50", "rot9", "rs256", "hflip",
           "crop_jpeg", "border20", "vaeB", "vaeC", "regen", "rinse2x"}
XENV = {"ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07", "unmarker"}
KEYS = ["C1", "C2", "C3", "C4", "C5"]; K = KEYS[CLS]
RATE_MARGIN = float(os.environ.get("RATE_MARGIN", "0.04")); MEAN_MARGIN = float(os.environ.get("MEAN_MARGIN", W.DEFAULT_MARGIN))
TAG = os.environ.get("TAG", f"rm{int(round(RATE_MARGIN * 100)):02d}"); SEED = int(os.environ.get("SEED", "0"))
DET_MIN = 0.9
OUT = f"{SC}/live_topk/{K}_{TAG}"; CELLS = f"{SC}/live_topk/cells"; os.makedirs(OUT, exist_ok=True); os.makedirs(CELLS, exist_ok=True)


def cfg_key(order, fe_on, strengths):
    return (tuple(order), tuple(sorted(fe_on)), tuple(round(float(s), 3) for s in strengths))


def cfg_id(key):
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:16]


def records():
    return json.loads((RUN_INPUTS / (K + '.json')).read_text())['records']


# ---------------------------------------------------------------------------------------------------
if MODE == "enumerate":
    raise RuntimeError("Use campaign.py enum_one: full frozen four-input requests are required")
elif MODE in ("measure", "measure_rank", "walk", "verdict", "patch", "xenv_count", "xenv_mark"):
    from PIL import Image
    if MODE == "measure_rank":
        RANK, SHARD, NSHARD = int(ARGS[0]), int(ARGS[1]), int(ARGS[2]); N = int(ARGS[3]) if len(ARGS) > 3 else 100
    elif MODE == "patch":
        SHARD, NSHARD = int(ARGS[0]), int(ARGS[1]); N = int(ARGS[2]) if len(ARGS) > 2 else 100
    else:
        N = int(ARGS[0]) if ARGS else 100
    rows = []
    for f in sorted(glob.glob(f"{OUT}/requests_s*.json")): rows += json.load(open(f))["rows"]
    rows.sort(key=lambda r: r["i"])
    if os.environ.get("FULL_PREFETCH_ROWS"):
        assert MODE == "measure_rank", "prefetch cannot publish a partial walk or verdict"
        rows = json.load(open(os.environ["FULL_PREFETCH_ROWS"]))
        assert rows and len({r["i"] for r in rows}) == len(rows) and all(0 <= r["i"] < 2000 for r in rows)
    else:
        assert [r["i"] for r in rows] == list(range(2000)), "full run requires all 2000 original requests"
    # rank-4 candidates from the patch stage (re-solved on the live-patched table), appended to their requests
    patched = {}
    for f in sorted(glob.glob(f"{OUT}/patched_s*.json")): patched.update({int(k): v for k, v in json.load(open(f))["candidates"].items()})
    for r in rows:
        c = patched.get(r["i"])
        if c and not any(x["rank"] == c["rank"] for x in r["candidates"]): r["candidates"].append(c)
    IMGSET = f"pool300_n{N}"

    def cell_path(key, a): return f"{CELLS}/{cfg_id(key)}_{a}_{IMGSET}.json"

    XDOM = f"topk_{K}_{TAG}"; XDIR = f"{SC}/certify_full_{XDOM}/{K}"     # groups exported for the cross-environment chain

    _group_index = None
    _cell_cache = {}

    def valid_cell(cell, attack):
        n = 30 if attack == "unmarker" else N
        return CP.valid_cell(cell, list((cell or {}).get('ba', {})), n)

    def certified_cell(key, a):
        global _group_index
        if _group_index is None:
            _group_index = {}
            for p in sorted(glob.glob(f"{SC}/certify_full/*/g*.json")) + sorted(glob.glob(f"{XDIR}/g*.json")):
                g = json.load(open(p))
                gkey = cfg_key(g["order"], g["fe"], [g["s"][f] for f in g["order"]])
                _group_index.setdefault(gkey, []).append((p, g))
        for p, g in _group_index.get(key, []):
            c = g["attacks"].get(a)
            if valid_cell(c, a):
                return {"cell": c, "psnr_db": g.get("psnr_db"), "source": p, "n": len(c["det"])}
        return None

    def load_cell(key, a):
        ck = (key, a)
        if ck in _cell_cache: return _cell_cache[ck]
        p = cell_path(key, a)
        if os.path.exists(p):
            c = json.load(open(p))
            if valid_cell(c.get("cell"), a):
                _cell_cache[ck] = c
                return c
        c = certified_cell(key, a)
        if c is not None:
            tmp = p + f".tmp{os.getpid()}"; json.dump(c, open(tmp, "w")); os.replace(tmp, p)
            _cell_cache[ck] = c
        return c

    def combine(cell, order, tau, line=None):
        return CP.combine(cell, order, tau, DET_MIN)

    def judge(row, cand, cells):
        return CP.judge(row, cand, cells, DET_MIN)

    def gpu_measurer():
        """(ensure, comp_desc): `ensure(key, order, fe_on, strengths, attacks)` measures the missing cells of one
        configuration on the held-out images and returns {attack: cell record or None}."""
        import torch
        from eval_matrix import OursComposite, presence_detected
        from src.attacks import attack_pil_any
        from src.image_pool import sample as _pool_sample, composition_of
        from src.soft_bch import decode_and_verify
        from src.soft_fusion import fuse_llrs, llr_to_bits
        dev = "cuda"
        files = _pool_sample(N, offset=300); comp_desc = composition_of(files)
        covers = [Image.open(f).convert("RGB").resize((512, 512), Image.BICUBIC) for f in files]
        print(f"{K}/{TAG}: {len(rows)} requests, N={N} {comp_desc}; free GPU {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB", flush=True)

        def r512(img): return img if img.size == (512, 512) else img.resize((512, 512))

        def composite_for(order, fe_on, strengths):
            fe = {k: (k in fe_on) for k in ("resync", "scale", "angle", "tile")}
            c = OursComposite(dev, tm_variant="B", vine_variant="R",
                              config={"frags": [FKEY[f] for f in order], "order": [FKEY[f] for f in order], "strengths": {}, **W.frontend_config(fe)})
            c.strength = dict(c.DEFAULT_STRENGTH); c.strength.update({FKEY[f]: float(v) for f, v in zip(order, strengths)}); return c

        def read(comp, order, img, iid, tx):
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

        def cell_of(order, rs):
            ba = {f: [r[0][f] for r in rs] for f in order}; ver = {f: [r[1][f] for r in rs] for f in order}
            return {"views": 2, "ba": ba, "ver": ver, "idv": [r[2] for r in rs], "det": [r[3] for r in rs],
                    "cas_ok": [r[4] is not None for r in rs],
                    "cas_ba": {f: [(r[4][0][f] if r[4] else None) for r in rs] for f in order},
                    "cas_ver": {f: [(r[4][1][f] if r[4] else None) for r in rs] for f in order}}

        state = {"key": None, "comp": None, "embeds": None, "psnr": None}

        def ensure(key, order, fe_on, strengths, attacks):
            """Measure the missing cells of this configuration; returns {attack: cell record or None}."""
            got = {a: load_cell(key, a) for a in attacks}
            todo = [a for a in attacks if got[a] is None and a in LIVE_OK]
            if todo:
                if state["key"] != key:
                    state["comp"] = None; torch.cuda.empty_cache()
                    comp = composite_for(order, fe_on, strengths); embeds = []
                    for i, cov in enumerate(covers):
                        emb, sec = comp.embed(cov, i); embeds.append((r512(emb), sec))
                    psnr = float(np.mean([10 * np.log10(255.0 ** 2 / max(np.mean((np.asarray(c, np.float64) - np.asarray(e, np.float64)) ** 2), 1e-9))
                                          for c, (e, _) in zip(covers, embeds)]))
                    state.update(key=key, comp=comp, embeds=embeds, psnr=psnr)
                comp, embeds, psnr = state["comp"], state["embeds"], state["psnr"]
                for a in todo:
                    # Several workers (local and array tasks) may hold overlapping shards: re-check before measuring,
                    # and write atomically so a concurrent writer can never leave a half-written cell behind.
                    got[a] = load_cell(key, a)
                    if got[a] is not None: continue
                    t0 = time.time(); rs = []
                    for emb, (iid, tx) in embeds:
                        v = r512(attack_pil_any(a, emb, dev=dev)); rs.append(read(comp, order, v, iid, tx))
                    cell = cell_of(order, rs)
                    rec = {"cell": cell, "psnr_db": psnr, "source": "live_topk", "key": list(map(list, key)), "images": comp_desc}
                    tmp = cell_path(key, a) + f".tmp{os.getpid()}"; json.dump(rec, open(tmp, "w")); os.replace(tmp, cell_path(key, a)); got[a] = rec
                    print(f"    {'+'.join(order)}{'/' + '+'.join(fe_on) if fe_on else ''} s={[round(s, 2) for s in strengths]} {a:12s} "
                          f"best-path {np.mean([max(x) for x in zip(*[cell['ba'][f] for f in order])]):.3f} rate@1% {np.mean(cell['det']):.2f} "
                          f"psnr {psnr:.2f} [{time.time()-t0:.0f}s]", flush=True)
            return got

        def export_group(key, order, fe_on, strengths, xattacks):
            """Write this configuration as a certify_full group (config, embeds of all covers, xenv_needed) so the
            cross-environment chain can attack and decode it; a group already exported is left alone."""
            gi = int(cfg_id(key)[:12], 16); path = f"{XDIR}/g{gi:04d}.json"
            if os.path.exists(path):
                existing = json.load(open(path))
                assert existing["cfg_id"] == cfg_id(key), "configuration group collision"
                missing = [a for a in xattacks if not valid_cell(existing["attacks"].get(a), a)]
                if missing:
                    existing["xenv_needed"] = sorted(set(existing.get("xenv_needed", [])) | set(missing))
                    tmp = path + f".tmp{os.getpid()}"; json.dump(existing, open(tmp, "w")); os.replace(tmp, path)
                return gi
            if state["key"] != key:
                state["comp"] = None; torch.cuda.empty_cache()
                comp = composite_for(order, fe_on, strengths); embeds = []
                for i, cov in enumerate(covers):
                    emb, sec = comp.embed(cov, i); embeds.append((r512(emb), sec))
                psnr = float(np.mean([10 * np.log10(255.0 ** 2 / max(np.mean((np.asarray(c, np.float64) - np.asarray(e, np.float64)) ** 2), 1e-9))
                                      for c, (e, _) in zip(covers, embeds)]))
                state.update(key=key, comp=comp, embeds=embeds, psnr=psnr)
            ed = f"{XDIR}/g{gi:04d}/embed"; os.makedirs(ed, exist_ok=True)
            for i, (emb, _) in enumerate(state["embeds"]): emb.save(f"{ed}/i{i:05d}.png")
            rec = {"gidx": gi, "cls": K, "order": list(order), "fe": list(fe_on), "s": {f: float(v) for f, v in zip(order, strengths)}, "members": [],
                   "n_img": N, "images": comp_desc, "domain": XDOM, "attacks": {}, "psnr_db": state["psnr"], "xenv_needed": sorted(xattacks), "cfg_id": cfg_id(key)}
            tmp = path + f".tmp{os.getpid()}"; json.dump(rec, open(tmp, "w")); os.replace(tmp, path)
            print(f"    exported group g{gi:04d} for the cross-environment chain: {'+'.join(order)} s={[round(x, 2) for x in strengths]} needs {sorted(xattacks)}", flush=True)
            return gi
        return ensure, comp_desc, export_group

    def cand_key(cand): return cfg_key(cand["order"], cand["fe_on"], [cand["s"][f] for f in cand["order"]])

    if MODE == "measure":
        ensure, comp_desc, export_group = gpu_measurer()
        t0 = time.time(); n_done = 0
        for row in rows:
            walk = []
            for cand in row["candidates"]:
                key = cfg_key(cand["order"], cand["fe_on"], [cand["s"][f] for f in cand["order"]])
                cells = ensure(key, cand["order"], cand["fe_on"], [cand["s"][f] for f in cand["order"]], row["attacks"])
                v = judge(row, cand, cells); v["rank"] = cand["rank"]; v["psnr_table"] = cand["psnr_db"]; v["cfg_id"] = cfg_id(key)
                walk.append(v)
                if v["pass"]: break
            row["walk"] = walk
            dep = next((v for v in walk if v["pass"]), None)
            row["deployed_rank"] = dep["rank"] if dep else None
            row["open"] = dep is None and any(v["pending"] for v in walk)
            n_done += 1
            print(f"  q{row['i']:04d} fpr={row['fpr']:.0e}: " + " ".join(
                  f"#{v['rank']}{'PASS' if v['pass'] else ('pend' if v['pending'] else 'fail')}({v['psnr_table']:.1f}/{(v['psnr_live'] or 0):.1f})" for v in walk)
                  + (f"  -> deploy #{dep['rank']}" if dep else "  -> none of the candidates passed") + f"  [{n_done}/{len(rows)} {time.time()-t0:.0f}s]", flush=True)
            json.dump({"cls": K, "tag": TAG, "N": N, "images": comp_desc, "rows": rows}, open(f"{OUT}/walk.json", "w"), indent=1)
        print("MEASURE_DONE", flush=True)

    elif MODE in ("measure_rank", "walk"):

        def walk_rows(rows, get_cells):
            return CP.walk_rows(rows, lambda cand, attacks: get_cells(cand_key(cand), attacks),
                                DET_MIN, lambda cand: cfg_id(cand_key(cand)))

        cached = lambda key, attacks: {a: load_cell(key, a) for a in attacks}
        if MODE == "walk":
            rows = walk_rows(rows, cached)
            n_dep = sum(1 for r in rows if r["deployed_rank"]); n_open = sum(1 for r in rows if r["open"])
            tmp = f"{OUT}/walk.json.tmp{os.getpid()}"; json.dump({"cls": K, "tag": TAG, "N": N, "images": IMGSET, "rows": rows}, open(tmp, "w"), indent=1); os.replace(tmp, f"{OUT}/walk.json")
            print(f"{K}/{TAG}: {len(rows)} requests: deployed {n_dep}, open (a cell still to measure) {n_open}, "
                  f"exhausted {len(rows) - n_dep - n_open}; by rank {dict(collections.Counter(r['deployed_rank'] for r in rows if r['deployed_rank']))}", flush=True)
            print("WALK_DONE", flush=True)
        else:
            # the rank-r candidates of the requests still open after the ranks below (cells present -> judged by walk)
            rows = walk_rows(rows, cached)
            need = {}
            for r in rows:
                if r["deployed_rank"] is not None: continue
                cand = next((c for c in r["candidates"] if c["rank"] == RANK), None)
                if cand is None: continue
                prev = [c["rank"] for c in r["candidates"] if c["rank"] < RANK]
                if prev and not any(v["rank"] == max(prev) and not v["pending"] for v in r["walk"]): continue   # a lower rank is still unmeasured
                key = cand_key(cand); need.setdefault(key, {"cand": cand, "attacks": set()})["attacks"].update(r["attacks"])
            if os.environ.get("FULL_RANK_PLAN"):
                plan = json.load(open(os.environ["FULL_RANK_PLAN"]))
                assert plan["cls"] == K and plan["rank"] == RANK
                need = {tuple(tuple(v) for v in x["key"]): {"cand": x["cand"], "attacks": set(x["attacks"])} for x in plan["configs"]}
            keys = sorted(need, key=lambda k: json.dumps(k)); mine = [k for i, k in enumerate(keys) if i % NSHARD == SHARD]
            todo = [k for k in mine if any(load_cell(k, a) is None and a in LIVE_OK for a in need[k]["attacks"])]
            xtodo = [k for k in mine if any(load_cell(k, a) is None and a in XENV for a in need[k]["attacks"])
]
            print(f"{K}/{TAG} rank {RANK}: {len(keys)} configurations to measure, shard {SHARD}/{NSHARD} takes {len(mine)} "
                  f"({len(todo)} with missing cells, {len(xtodo)} to export for the cross-environment chain)", flush=True)
            if todo or xtodo:
                ensure, comp_desc, export_group = gpu_measurer()
                for key in todo:
                    cand = need[key]["cand"]
                    ensure(key, cand["order"], cand["fe_on"], [cand["s"][f] for f in cand["order"]], sorted(need[key]["attacks"]))
                for key in xtodo:
                    cand = need[key]["cand"]
                    export_group(key, cand["order"], cand["fe_on"], [cand["s"][f] for f in cand["order"]], [a for a in need[key]["attacks"] if a in XENV])
            print("MEASURE_RANK_DONE", flush=True)

    elif MODE in ("xenv_count", "xenv_mark"):
        n_open = 0; n_done = 0
        for p in sorted(glob.glob(f"{XDIR}/g*.json")):
            g = json.load(open(p)); need_x = g.get("xenv_needed") or []
            if not need_x: continue
            if all(a in g["attacks"] and g["attacks"][a].get("views") == 2 for a in need_x):
                if MODE == "xenv_mark":
                    g["xenv_done"] = need_x; g.pop("xenv_needed"); tmp = p + f".tmp{os.getpid()}"; json.dump(g, open(tmp, "w")); os.replace(tmp, p)
                n_done += 1
            else:
                n_open += 1
        print(f"{K}/{TAG}: {n_open} groups still owe the cross-environment chain, {n_done} decoded{' and retired' if MODE == 'xenv_mark' else ''}")
        print(f"XENV_OPEN {n_open}")

    elif MODE == "patch":
        import math
        import solver_eval_continuous as SEC
        from live_calibration import table_value
        sg0 = SEC.sg; MEAS = set(sg0.attacks)
        _pw_path = f"{SC}/prior_width.json"     # pipeline/measure_prior_width.py writes it
        pw = json.load(open(_pw_path)) if os.path.exists(_pw_path) else {}
        PRIOR_SD = float(pw.get("prior_sd_p90") or pw.get("prior_sd_median") or 0.02)
        state = {r["i"]: r for r in json.load(open(f"{OUT}/walk.json"))["rows"]}

        def rate_offset(f, a, stage, s_f, tau, live_rate, n):
            """The shift of the per-image values that brings the table's acceptance rate at this strength down to
            the live rate; 0 when the two agree within two standard errors of a rate at n images."""
            st = stage if (stage and sg0.has_perimage(f, a, stage)) else None
            if not st and not sg0.has_perimage(f, a): return 0.0, None
            key = ("fe", st, f, a) if st else (f, a)
            tr = float(sg0.rate_curve(f, a, tau, st).eval(s_f))
            se = math.sqrt(max(live_rate * (1 - live_rate), 0.09) / n)
            # A column that failed the deployment floor live is a measured failure, not noise: the shift is applied
            # whenever the live rate sits below the floor the table promised to clear; elsewhere it needs 2 SE.
            if tr <= live_rate or (live_rate >= DET_MIN - 1e-9 and tr - live_rate < 2 * se): return 0.0, key
            lo, hi = -0.4, 0.0
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                if float(sg0.with_live({key: mid}).rate_curve(f, a, tau, st).eval(s_f)) > live_rate: hi = mid
                else: lo = mid
            return lo, key

        targets = [r for r in rows if r['i'] in state and CP.needs_patch(state[r['i']])]
        if os.environ.get("FULL_PATCH_IDS"):
            ids = {int(x) for x in os.environ["FULL_PATCH_IDS"].split(",")}
            mine = [r for r in targets if r["i"] in ids and r["candidates"]]
        else:
            mine = targets[SHARD::NSHARD]
        print(f"{K}/{TAG} patch: {len(targets)} exhausted requests, shard {SHARD}/{NSHARD} takes {len(mine)}; prior sd {PRIOR_SD:.4f}", flush=True)
        out = {}
        for r in mine:
            wr = state[r["i"]]; offsets = {}; t1 = time.time()
            for v in wr["walk"]:
                cand = next(c for c in r["candidates"] if c["rank"] == v["rank"])
                cfg = {"order": cand["order"], "s": cand["s"], "fe": cand["fe"], "fe_on": cand["fe_on"]}
                key = cand_key(cand)
                for a in r["attacks"]:
                    c = load_cell(key, a)
                    if c is None: continue
                    cell = c["cell"]; n = len(cell["det"])
                    observed, observed_ver, _, _ = CP.selected_views(cell, cand["order"], v["tau"])
                    for fi, f in enumerate(cand["order"]):
                        ba, ver = observed[fi], observed_ver[fi]
                        mean = float(ba.mean()); se = float(ba.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
                        stage = table_value(sg0, cfg, f, a)[1]
                        d_mean = sg0.live_offset(f, a, cfg["s"][f], mean, se_live=se, prior_sd=PRIOR_SD, k=2.0, asymmetric=True, fe=stage)
                        if d_mean < -1e-12:
                            k_m = ("fe", stage, f, a) if stage else (f, a); offsets[k_m] = min(offsets.get(k_m, 0.0), d_mean)
                        live_rate = float(np.mean(ver | (ba >= v["tau"] - 1e-12)))
                        d_rate, k_r = rate_offset(f, a, stage, cfg["s"][f], v["tau"], live_rate, n)
                        if k_r is not None and d_rate < -1e-12: offsets[k_r] = min(offsets.get(k_r, 0.0), d_rate)
            sg = sg0.with_live(offsets)
            scen = CP.solver_scenario(r, margin=r.get('mean_margin', MEAN_MARGIN))
            cands = T.enumerate_topk(scen, k=1, enable_order=True, continuous_strength=True, surrogate=sg, rate_margin=r.get("rate_margin", RATE_MARGIN))
            offs = sorted(((("/".join(str(x) for x in k)), round(d, 4)) for k, d in offsets.items()), key=lambda t: t[1])
            if cands:
                c = cands[0]
                out[r["i"]] = {"rank": 4, "skeleton": c["skeleton"].as_dict(), "order": c["cfg"]["order"], "fe": c["cfg"]["fe"], "fe_on": c["cfg"]["fe_on"],
                               "s": c["cfg"]["s"], "psnr_db": c["psnr_db"], "ms": c["cfg"]["ms"], "certified": c["certified"], "patched": offs}
                print(f"  q{r['i']:04d}: {len(offsets)} cells patched (largest {offs[0] if offs else None}) -> #4 {c['skeleton']!r} "
                      f"s={ {f: round(x, 2) for f, x in c['cfg']['s'].items()} } {c['psnr_db']:.2f} dB [{time.time()-t1:.0f}s]", flush=True)
            else:
                out[r["i"]] = None
                print(f"  q{r['i']:04d}: {len(offsets)} cells patched -> UNSAT on the patched table [{time.time()-t1:.0f}s]", flush=True)
        path = f"{OUT}/patched_s{SHARD:02d}.json"
        tmp = path + f".tmp{os.getpid()}"; json.dump({"cls": K, "tag": TAG, "shard": SHARD, "n_shards": NSHARD, "candidates": {str(k): v for k, v in out.items() if v}, "evaluated": {str(k): v is not None for k, v in out.items()}}, open(tmp, "w"), indent=1); os.replace(tmp, path)
        print(f"wrote {path} ({sum(1 for v in out.values() if v)} candidates, {sum(1 for v in out.values() if not v)} UNSAT) PATCH_DONE", flush=True)

    else:   # verdict
        summary = json.load(open(f"{OUT}/full_summary.json"))
        assert summary['protocol'] == CP.PROTOCOL
        print(json.dumps(summary, indent=2))
