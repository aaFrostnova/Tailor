"""The five threat classes and their request sampler, shared by class_scenarios.py (solve) and
certify_classes.py (measure). Sampling is deterministic per class (seed 1000 + class index) and the
call order inside sample_class is the request's identity: do not reorder the draws."""
import json, os, random
SIG = ["jpeg25", "blur", "noise", "bright", "contrast"]
# What the table can reach per column and budget (make_request_feasibility.py): the sampler draws each
# request's budget, payload and fidelity floor CONDITIONED on the hardest column it names, so that the
# classes measure the method rather than the fragments' known limits (user 2026-09-07).
# FEAS_MATRIX=<path> selects another matrix file (the v1 matrix the 2026-09-08 C1-C4 shards were drawn from is
# kept as request_feasibility_matrix.v1_20260908.json; the default is the current, payload-aware one).
_BUNDLED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inputs")
_FEAS_PATH = os.environ.get("FEAS_MATRIX") or os.path.join(_BUNDLED, "request_feasibility_matrix.json")
_FEAS = json.load(open(_FEAS_PATH))
NEEDS_2ND = set(_FEAS["needs_second_fragment"])          # columns VINE cannot carry: a second fragment, k = 2
# Sampler version. v3 (2026-09-08): after the fidelity floor is drawn against the presence line, it is capped by
# what the payload's capacity line costs (matrix need_cap; the capacity clause holds the carrier to
# bits_to_ba(min_bits) whatever the budget). A post-draw clamp: the random stream and every other field are
# those of v2. SAMPLER_V=2 with FEAS_MATRIX=request_feasibility_matrix.v2.json reproduces the 2026-09-08 shards.
SAMPLER_V = int(os.environ.get("SAMPLER_V", "3"))
# A deliberate minority of requests asks for a budget BEYOND the hardest column's ceiling. They are
# infeasible by construction (the fragments' limits, not composition) and keep real UNSAT in scope;
# they are marked `_stress` so the report can separate them.
STRESS = 0.06
CR = ["ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07"]
def classes(table_attacks):
    UNM = ["unmarker"] if "unmarker" in table_attacks else []
    return [
     # Neural re-encoding (vaeB/vaeC) and crop-then-JPEG are re-encodings too, and drawing them here is
     # what lets the request distribution reach the one region where TrustMark is the cheapest fragment
     # (identity grade, VAE compression, crop+JPEG, no regeneration): with vaeB/vaeC bound to regen in
     # C3/C5 that region was never sampled and TrustMark was selected on 0.6% of requests. Sampled
     # rather than given a class of its own (user 2026-09-06).
     dict(key="C1", name="Signal / re-encoding",           core=SIG,                                        optional=["rs256", "hflip", "vaeB", "vaeC", "crop_jpeg"],
          psnr=[34, 35, 36, 37, 38, 40], ms=[500, 1000, 2000, 4000], bits=([0, 10, 20, 37], [3, 2, 2, 1])),
     dict(key="C2", name="+ Geometry (crop / rotation)",   core=SIG + ["crop75", "rot9"],                   optional=["rs256", "hflip", "crop_jpeg", "crop50", "border20"],
          psnr=[32, 33, 34, 35, 36, 38], ms=[1000, 2000, 4000, 8000], bits=([0, 10, 20, 37], [3, 2, 2, 2])),
     dict(key="C3", name="+ AI regeneration",              core=SIG + ["vaeB", "vaeC", "regen"],            optional=["rinse2x", "rs256", "hflip"],
          psnr=[30, 31, 32, 33, 34, 36], ms=[1000, 2000, 4000, 8000], bits=([0, 10, 20, 37], [3, 2, 2, 2])),
     dict(key="C4", name="+ Adversarial removal",          core=SIG + ["regen"],                            optional=["vaeB", "vaeC", "rinse2x"],
          psnr=[30, 31, 32, 33, 34, 36], ms=[2000, 4000, 8000], bits=([0, 10, 20], [4, 2, 1]),
          # one CtrlRegen+ strength per request (user 2026-09-04): the strongest step is kept in scope but
          # held to a minority of the requests, so the class reports satisfaction against attack strength
          # rather than against the union of three strengths in every request
          # The budget, payload and fidelity floor follow the hardest column's ceiling (sample_class), so the
          # strongest step is answerable at the budgets the table says it can reach and infeasible only by design.
          # step 0.7 is beyond every fragment at the 90% per-image floor at ANY budget (VINE 0.64 mean, 88% of
          # images at 0.57): kept in scope as a stress row (10%), not as a fifth of the class
          cr=(CR, [0.45, 0.45, 0.10]),
          # UnMarker on a QUARTER of the requests rather than all of them (user 2026-09-06). It is the
          # one attack the offline table may not settle (adversarial, per image), so every request that
          # names it owes a live measurement at 3.5 GPU-minutes per image; keeping it in every request
          # made the class's full live certification cost 1,400 GPU-hours. Sampled last, after the
          # CtrlRegen+ step, so the step sequence is unchanged.
          unm=(UNM, 0.25)),
     dict(key="C5", name="Broad + identity (provenance)",  core=SIG + ["crop75", "rot9", "vaeB", "vaeC", "regen"], optional=["rinse2x", "crop50", "hflip", "rs256", "crop_jpeg", "border20"],
          psnr=[32, 33, 34, 35, 36, 38], ms=[4000, 8000, 16000], bits=([20, 37, 50], [2, 3, 1])),
    ]
# The false-positive budgets a deployment asks for, and how often. The budget is the dominant
# difficulty axis -- it sets the bit accuracy every column must reach -- so the mix decides what the
# class average means. The earlier weights [1,3,2,2,1,2] put 27% of every class at the two tightest
# budgets, where several columns are infeasible for any configuration (VINE under CtrlRegen+ at step
# 0.3 tops out at 0.830 bit accuracy and 1e-9 asks for 0.800 per image), so the average was dominated
# by requests no method could serve. These weights model demand instead: presence-grade detection at
# 1e-2 and 1e-4 is the common case, identity-grade attribution at 2^-37 is rare but kept in scope so
# the classes still contain genuinely infeasible requests. Satisfaction is reported per budget as well
# as pooled, and the per-budget numbers do not depend on this mix (user 2026-09-07).
FPR = ([1e-1, 1e-2, 1e-4, 1e-6, 1e-9, 2.0 ** -37], [3, 5, 4, 2, 1, 1])
def _k(attacks): return 2 if any(a in NEEDS_2ND for a in attacks) else 1
def _need(a, k, bi, bits=0):
    """the reference cell of column a at k fragments and budget index bi for a payload of `bits`: the cheapest
    fragment whose mean clears max(tau + margin, capacity line of bits) with the per-image rate at tau; a
    matrix without the payload dimension (v1) or a payload no cell reaches falls back to the payload-0 cell"""
    if bits: return _FEAS["need"].get(f"{a}|k{k}|{bi}|b{bits}") or _FEAS["need"].get(f"{a}|k{k}|{bi}")
    return _FEAS["need"].get(f"{a}|k{k}|{bi}")
def _reach(a, k): return max([bi for bi in range(len(FPR[0])) if _need(a, k, bi)] or [-1])   # tightest reachable budget index
def sample_class(C, cls_index, N, beta_from_fpr, capacity_output=False):
    random.seed(1000 + cls_index)
    out = []
    for i in range(N):
        # the attack set first (optional columns, the CtrlRegen+ step, UnMarker), then everything that depends on it
        opt = [a for a in C["optional"] if random.random() < 0.5]
        cr = random.choices(*C["cr"])[0] if C.get("cr") else None
        unm = bool(C.get("unm") and C["unm"][0] and random.random() < C["unm"][1])
        attacks = C["core"] + opt + ([cr] if cr else []) + (list(C["unm"][0]) if unm else [])
        k = _k(attacks); reach = {a: _reach(a, k) for a in attacks}; ceil = min(reach.values())
        hardest = sorted(a for a in attacks if reach[a] == ceil)
        # budget: the demand weights, restricted to what the hardest column can reach; a STRESS minority beyond it.
        # A column no fragment reaches at any budget makes the request stress "by column" whatever it draws.
        beyond = [bi for bi in range(len(FPR[0])) if bi > ceil]
        stress_kind = "column" if ceil < 0 else ("budget" if (beyond and random.random() < STRESS) else None)
        stress = stress_kind is not None
        original_stress_kind = stress_kind
        pool = beyond if stress else [bi for bi in range(len(FPR[0])) if bi <= ceil]
        bi = random.choices(pool, weights=[FPR[1][b] for b in pool])[0]; fpr = FPR[0][bi]
        # payload: within the capacity ceiling of every named column
        capmin = min(_FEAS["capacity_bits"][a] for a in attacks)
        bw = [w if b <= capmin else 0 for b, w in zip(*C["bits"])]
        bits = int(random.choices(C["bits"][0], weights=bw)[0]) if sum(bw) else 0
        # fidelity floor: at or below what the hardest column needs at this budget (at the ceiling budget for a
        # stress request), less an allowance for the composite (one fragment 1 dB, two fragments 2 dB)
        # (v2, 2026-09-08) the reference cells are taken at the request's own line, payload included: a 50-bit
        # payload asks every column for a mean of 0.89, far above the presence line at loose budgets, and the
        # fragment that reaches it is stronger and costlier than the presence reference (C5 drew 35-38 dB floors
        # for 50 bits through rinse2x, which needs VINE 0.935 at 35.4 dB solo: 125 requests infeasible by the table)
        ref = [_need(a, k, min(bi, ceil), bits) for a in attacks] if ceil >= 0 else []
        need_db = (min(x[0] for x in ref) - (1.0 if k == 1 else 2.0)) if (ref and all(ref)) else None
        pw = [1 if (need_db is None or p <= need_db) else 0 for p in C["psnr"]]
        psnr_rng_state = random.getstate()
        psnr = float(random.choices(C["psnr"], weights=pw)[0]) if sum(pw) else float(min(C["psnr"]))
        if SAMPLER_V >= 3 and bits > 0 and ceil >= 0 and _FEAS.get("need_cap") and not stress:
            refc = [_FEAS["need_cap"].get(f"{a}|k{k}|{min(bi, ceil)}|{bits}") for a in attacks]
            if all(refc):
                allowed = [p for p in C["psnr"] if p <= min(x[0] for x in refc) - (1.0 if k == 1 else 2.0)]
                if allowed: psnr = float(min(psnr, max(allowed)))
                else: psnr = float(min(C["psnr"])); stress, stress_kind = True, "fidelity"     # no floor of the class is reachable with this payload
            else: stress, stress_kind = True, "capacity"       # the payload's line is beyond the per-image floor on some column
        # latency: at or above what the cheapest cells cost. A stage a column needs runs once and is charged
        # its worst latency over the requested columns (the ring's scale search on a CtrlRegen+ or UnMarker
        # image is 1.8 to 2.5 s), plus the fragments' own decode; a 2 s budget with the ring on is unmeetable.
        stages = {x[2] for x in ref if x and x[2] != "plain"}
        need_ms = sum(_FEAS["fragment_ms"].values()) + max([_FEAS["stage_latency_ms"].get(f"latency_ms_{st}|{a}", 0.0) for st in stages for a in attacks] or [0.0])
        mw = [1 if m >= need_ms else 0 for m in C["ms"]]
        ms_rng_state = random.getstate()
        ms = float(random.choices(C["ms"], weights=mw)[0]) if sum(mw) else float(max(C["ms"]))
        if capacity_output:
            # Replay only the quality/latency draws against capacity-free references.
            # The legacy draws above advance the original stream so paired request
            # IDs keep exactly the same attacks and FPR. Their payload, quality
            # clamps and capacity-derived stress flags never enter the new request.
            bits = 0
            stress_kind = original_stress_kind
            stress = stress_kind is not None
            ref = [_need(a, k, min(bi, ceil), 0) for a in attacks] if ceil >= 0 else []
            need_db = (min(x[0] for x in ref) - (1.0 if k == 1 else 2.0)) if (ref and all(ref)) else None
            pw = [1 if (need_db is None or p <= need_db) else 0 for p in C['psnr']]
            replay = random.Random(); replay.setstate(psnr_rng_state)
            psnr = float(replay.choices(C['psnr'], weights=pw)[0]) if sum(pw) else float(min(C['psnr']))
            stages = {x[2] for x in ref if x and x[2] != 'plain'}
            need_ms = sum(_FEAS['fragment_ms'].values()) + max([
                _FEAS['stage_latency_ms'].get(f'latency_ms_{st}|{a}', 0.0)
                for st in stages for a in attacks] or [0.0])
            mw = [1 if m >= need_ms else 0 for m in C['ms']]
            replay.setstate(ms_rng_state)
            ms = float(replay.choices(C['ms'], weights=mw)[0]) if sum(mw) else float(max(C['ms']))
        out.append(dict(min_psnr=psnr, max_ms=ms, attacks=attacks, min_ba=beta_from_fpr(fpr), fpr=fpr,
                        allow_resync=True, allow_nested=True, min_bits=bits,
                        _aset=C["key"], _optional=opt, _cr=cr, _unm=unm, _stress=stress, _stress_kind=stress_kind, _k=k,
                        _ceiling_fpr=FPR[0][ceil] if ceil >= 0 else None, _hardest=hardest))
        if capacity_output:
            out[-1].pop('min_bits')
    return out
