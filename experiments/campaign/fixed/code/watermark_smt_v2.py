"""Unified-BA selection using explicitly legacy measured search priors.

The frozen historical BA/geometry/latency tables guide selection only. Their
keyed detector scores are not evidence for the new strict-BA protocol. All
returned candidates require fresh threshold-specific per-image live checks;
final positive acceptance and actual latency are independently remeasured.
The original request FPR and full primary/geometric K set the SMT threshold.
"""
import os
import json, sys, argparse, math
from pathlib import Path
_RUN_INPUTS = Path(__file__).resolve().parent.parent / "inputs"
from z3 import (Optimize, Bool, Int, Real, If, Or, And, Not, Implies, Sum, BoolVal, sat, is_true,
                AtMost)
CF=os.environ.get('TAILOR_PROJECT', '.')
D=json.load(open(CF+"/inputs/smt_inputs.json"))              # solo: VINE alpha-sweep + per-frag solo PSNR + timing
DC=json.load(open(CF+"/inputs/smt_inputs_composite.json"))    # composite PSNR stack anchors
FULL=json.load(open(CF+"/inputs/smt_inputs_full.json"))       # n=100 end-to-end: per-attack composite bit-acc + DEPLOYED detection (validated fused>=best-path)
# F = validated n=100 composite bit-acc (VINE/TM/VideoSeal) + regen/rinse from hidden_big composite
PA=FULL["per_attack"]; REGC={"VINE":{"regen":0.93,"rinse":0.88},"TrustMark":{"regen":0.50,"rinse":0.50},"VideoSeal":{"regen":0.50,"rinse":0.50}}
F={"VINE":{"bit_acc":{a:PA[a]["vine"] for a in PA}},"TrustMark":{"bit_acc":{a:PA[a]["tm"] for a in PA}},"VideoSeal":{"bit_acc":{a:PA[a]["vs"] for a in PA}}}
for f in F: F[f]["bit_acc"].update(REGC[f])
# CALIB: correct optimistic PA cells to the measured 10k standalone bit-acc (calibration_regression.py):
# TrustMark photometric was +0.13/+0.19 optimistic (bright 0.96->0.84, contrast 0.95->0.77), crossing
# every min_ba threshold and flipping 25% of recommendations. Recalibrate to the larger 10k measurement.
CALIB={"TrustMark":{"bright":0.84,"contrast":0.77}}
for f in CALIB: F[f]["bit_acc"].update(CALIB[f])
# Advanced attacks. CtrlRegen+ is STRENGTH-parameterized (attacker's choice, an INPUT axis): VINE
# 0.82/0.71/0.64 @ s0.3/0.5/0.7 (measured 10k t3, n=200); spatial marks (TrustMark/VideoSeal) die to
# regeneration at every strength.
#
# UnMarker: values below are BARE-decode bit-acc; the nested/fine-scale-search front-end recovers VINE
# (modeled in frag_defends via UNMK_REC, mirroring the crop/rot recovery clauses).
#   VINE bare 0.53      -- inputs/unified_bitacc.json unmarker_default raw_ba 0.533 (n=25),
#                          unmarker_strong 0.521 (n=15). The former 0.49 was eval10k/t3/vine_r.json, the
#                          same pre-scale-fix bare decode.
#   VINE recovered 0.98 -- unified_bitacc recovered_ba 0.984/0.976 with detect_crypto 1.0; corroborated by
#                          unmarker100_s200_100/_s300_150/_s400_200 (n=100 per strength) VINE_only = 1.00.
#   TrustMark 0.51      -- measured bare (eval10k/t3/trustmark_b.json 0.508); no scale-search path -> dead.
#   VideoSeal 0.50      -- NOT standalone-measured. The previous 0.85 was INFERRED by attributing composite
#                          UnMarker survival to VideoSeal; that attribution is refuted (the survival is
#                          VINE + scale search). Conservative placeholder until a standalone n>=200 run.
UNMK_REC=0.98            # VINE bit-acc under UnMarker WITH the nested/fine-scale-search decode (measured)
# crop50 was measured in the 10k run but never entered the table, so the solver could not reason about it
# (any scenario naming crop50 was silently unsatisfiable). BARE values from eval10k/t1_merged: VINE 0.500,
# TrustMark 0.501 (contrast: TrustMark survives crop75 at 0.826 but dies at crop50), VideoSeal 0.50
# conservative. The nested ring recovers it -- Ours+geo measures crop50 ba 1.000 / tpr 1.00 -- so crop50
# joins the crop clause in frag_defends and the nested capacity recovery in achievable_bits.
CROP50={"VINE":0.500,"TrustMark":0.501,"VideoSeal":0.500}
for _f,_v in CROP50.items(): F[_f]["bit_acc"]["crop50"]=_v
# MEASURED 0.607 (standalone, n=200, std=0.068) -- replaces the
# inferred 0.85 / conservative 0.50 placeholder. CAP['VideoSeal']['unmarker'] stays 0.0: this is a
# bit-acc measurement, not a soft-MI capacity measurement.
ADV={"VINE":     {"ctrlregen_s03":0.82,"ctrlregen_s05":0.71,"ctrlregen_s07":0.64,"ctrlregen":0.64,"unmarker":0.53},
     "TrustMark":{"ctrlregen_s03":0.50,"ctrlregen_s05":0.50,"ctrlregen_s07":0.50,"ctrlregen":0.50,"unmarker":0.51},
     "VideoSeal":{"ctrlregen_s03":0.50,"ctrlregen_s05":0.50,"ctrlregen_s07":0.50,"ctrlregen":0.50,"unmarker":0.607}}
for f in ADV: F[f]["bit_acc"].update(ADV[f])
PSTACK=DC["psnr_stack"]                                               # {"1frag":37.88,"2frag":36.65,"3frag":36.25}
# Stacking / front-end fidelity penalties, RE-CALIBRATED against the measured end-to-end catalog
# (wm_dataset10k/config_psnr.json, n=40 embed-only PSNR) instead of the older PSTACK anchors. Implied
# penalties vs the VINE-R solo anchor 36.19 dB: +TM 35.26 -> 0.93 | +TM+VS 34.40 -> 1.79 |
# nested 33.40 -> 2.79 | nested+TM+VS 32.31 -> 3.88. The previous constants understated the nested
# front-end by 1.79 dB (modeled 1.0 vs measured 2.79), i.e. by more than the fidelity effects reported
# from this model. Residual non-additivity is 0.70 dB (3-frag + nested predicts 31.61 vs measured 32.31),
# which stands as the model's known error bar for the fidelity axis.
PEN2, PEN3 = 0.93, 1.79
PEN4 = PEN3 + 0.4
NESTED_PEN = 2.79            # measured; was hard-coded 1.0
PSNR_MODEL_ERR = 0.70        # measured non-additivity of the stack+nested combination
PSNR_SOLO={f:D["fragments"][f]["psnr"] for f in D["fragments"]}       # per-fragment solo PSNR (base quality)
# NOTE: VINE_C3 (no-LPIPS ablation) candidate REMOVED — decoder-only finetune kill-test (n=100) confirmed it
# can't lift rinse (encoder-limited); deployed carrier = released VINE-R. See memory c3-decoder-finetune.
ALPHA=D["alpha_sweep"]           # {"0.3":..,"0.5":..,"0.7":..,"1.0":..}
ALV=[0.3,0.5,0.7,1.0]            # discrete alpha choices (measured)
SWEPT={"clean","jpeg25","crop75","noise"}
BA_full=lambda f,a: F[f]["bit_acc"].get(a,0.0)

# ---- reliable-bit CAPACITY axis (MEASURED soft-MI of 100 embedded, capacity_mi.py n=3000) ----
# Orthogonal to min_ba (per-bit accuracy). Robust-ID capacity under a set of attacks =
#   min_a ( max over chosen frags CAP[f][a] )   — UEP: same ID repeated across frags, best survivor carries it.
# NOTE: 11 attacks measured (hidden_big 6 + hidden_ext 5, capacity_combos.py). jpeg50/jpeg25 share "jpeg"; bright/contrast≈clean.
# rot9/rot30/vaeB/vaeC/rinse added from hidden_ext (N=1000). rot9/rot30 are NATIVE (no resync); resync recovers rotation
# toward clean (modeled below, mirrors the bit-acc front-end boost). rinse=28 (VINE) is the true double-regen info floor.
_CR_V={"ctrlregen_s03":20.0,"ctrlregen_s05":10.0,"ctrlregen_s07":6.0,"ctrlregen":6.0}   # VINE capacity by CtrlRegen+ strength
_CR_0={"ctrlregen_s03":0.0,"ctrlregen_s05":0.0,"ctrlregen_s07":0.0,"ctrlregen":0.0}
CAP={"VINE":     {"clean":96.0,"jpeg50":95.9,"jpeg25":95.9,"blur":80.4,"noise":95.8,"bright":96.0,"contrast":96.0,"crop90":0.0, "crop75":0.0, "crop50":0.0, "rot9":0.0, "rot30":0.0,"vaeB":89.9,"vaeC":90.2,"regen":52.3,"rinse":28.4,"unmarker":0.0, **_CR_V},
     "TrustMark":{"clean":94.3,"jpeg50":78.1,"jpeg25":78.1,"blur":62.1,"noise":80.2,"bright":94.3,"contrast":94.3,"crop90":85.7,"crop75":85.7,"crop50":0.0, "rot9":0.8, "rot30":0.0,"vaeB":76.1,"vaeC":65.0,"regen":0.0, "rinse":0.0, "unmarker":0.0, **_CR_0},
     # VideoSeal unmarker 0.0: the former 20.0 rested on the refuted "VideoSeal is the UnMarker survivor"
     # attribution. No standalone measurement exists; conservative until one is run.
     "VideoSeal":{"clean":59.6,"jpeg50":31.8,"jpeg25":31.8,"blur":12.9,"noise":12.2,"bright":59.6,"contrast":59.6,"crop90":13.5,"crop75":13.5,"crop50":0.0, "rot9":41.7,"rot30":0.0,"vaeB":9.9, "vaeC":7.5, "regen":0.0, "rinse":0.0, "unmarker":0.0, **_CR_0}}
# UnMarker capacity for VINE is a MEASURED LOWER BOUND, not a soft-MI number: with the nested/scale-search
# decode the full 37-bit ID crypto-verifies (unified_bitacc detect_crypto 1.0 at 2^-37), so >=37 bits get
# through; the true soft-MI is un-measured and likely higher. Applied only when nested is enabled.
UNMK_CAP_VINE=37.0

# ================================================================================================
# FRESH BASELINE OVERRIDE. Everything above assembled the table from smt_inputs*.json (2026-08-10,
# n=24) plus hand-patched blocks. Replace every VALUE with the freshly-measured, provenance-stamped
# baseline_table.json (100% coverage, std attached). The structures (F, ALPHA, CAP, PSNR_SOLO,
# penalties, latency) are kept; only the numbers change, so the rest of the solver is untouched.
# Per-cell image-to-image std for the live gate now comes from ba_sd_profile.json (BA_SD_COL).
import json as _bjson, os as _bos
_BLP = str(_RUN_INPUTS / "baseline_table.json")
BASELINE_PROVENANCE = None
if _bos.path.exists(_BLP):
    _BT = _bjson.load(open(_BLP)); _C = _BT["cells"]
    BASELINE_PROVENANCE = {"updated_at": _BT.get("updated_at"), "coverage": _BT.get("coverage")}
    _ALL_ATT = sorted(set(k.split("/")[-1] for k in _C["bit_acc"]))
    def _bv(grp, key):
        c = _C[grp].get(key); return (c.get("value") if isinstance(c, dict) else c) if c is not None else None
    # --- solo PSNR (fragment @1.0) and VINE alpha PSNR ---
    for f in list(F):
        v = _bv("psnr_solo", f"{f}@1.0");  PSNR_SOLO[f] = v if v is not None else PSNR_SOLO.get(f)
    for a in ALV:
        v = _bv("psnr_solo", f"VINE@{a}")
        if v is not None: ALPHA[str(a)]["psnr"] = v
    # --- VINE per-alpha bit-acc for EVERY attack (baseline now has all of them) ---
    for a in ALV:
        for at in _ALL_ATT:
            v = _bv("bit_acc", f"VINE@{a}/{at}")
            if v is not None: ALPHA[str(a)][at] = v
    # --- F = deployed per-fragment bit-acc: VINE at alpha=1.0 as the non-swept default; TM/VS @1.0 ---
    for f in list(F):
        for at in _ALL_ATT:
            v = _bv("bit_acc", f"{f}@1.0/{at}")
            if v is not None: F[f]["bit_acc"][at] = v
    # --- CAP (fragment / attack) ---
    for f in list(F):
        for at in _ALL_ATT:
            v = _bv("cap", f"{f}/{at}")
            if v is not None: CAP[f][at] = v
    # --- penalties + real front-end latency ---
    for nm in ("PEN2", "PEN3", "NESTED_PEN"):
        v = _bv("penalties", nm)
        if v is not None: globals()[nm] = v
    _rs = _bv("frontend", "resync_ms"); _ns = _bv("frontend", "nested_ms")
    FE_MS = {"resync": (_rs if _rs is not None else 300.0), "nested": (_ns if _ns is not None else 1600.0)}
    # --- per-cell std for the variance gate ---
    # BA_STD_FRESH (per-cell std for the retired live_required gate) moved to the attic with it.
else:
    FE_MS = {"resync": 300.0, "nested": 1600.0}
    BASELINE_PROVENANCE = {"warning": "baseline_table.json NOT FOUND -- using inherited smt_inputs values"}

CAP_UNMEASURED=set()   # all 11 SMT attacks now measured (rot/vae/rinse via hidden_ext N=1000)
# The 9-knot diffusion campaign renamed three columns the capacity measurements still carry under their
# old names, and a capacity cell that matches nothing drops out of the clause entirely rather than
# binding: a 90-bit identity under `rinse2x` came back SAT while the same request under `regen`, whose
# capacity is known (80.9 bit), is correctly UNSAT. `rinse` was measured as the double regeneration that
# `rinse2x` names, so that one is an alias. `rinse4x` and `ctrlregen_s05_x2` are strictly stronger than
# anything measured, and capacity falls with attack strength, so borrowing the weaker column's number
# would credit bits nobody measured; they take 0 until measured, which is the same rule the table already
# applies to VideoSeal under UnMarker.
CAP_ALIAS = {"rinse2x": "rinse"}
CAP_UNMEASURED_ZERO = {"rinse4x", "ctrlregen_s05_x2"}


def cap_constant(frag, attack):
    """Measured reliable-bit capacity for a cell, or None when nothing was measured for it.

    Returns 0.0 for a column known to be strictly harder than the hardest measured one: unmeasured
    capacity is never credited, and a request that needs bits there is infeasible rather than silently
    unconstrained."""
    tbl = CAP.get(frag, {})
    if attack in tbl:
        return tbl[attack]
    alias = CAP_ALIAS.get(attack)
    if alias is not None and alias in tbl:
        return tbl[alias]
    if attack in CAP_UNMEASURED_ZERO:
        return 0.0
    return None
NATIVE_BITS = 100   # every fragment carries the same 100-bit BCH codeword


def _h2(p):
    """Binary entropy in bits, exactly 0 at the endpoints."""
    p = float(p)
    if p <= 0.0 or p >= 1.0: return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def ba_to_bits(ba, n=NATIVE_BITS):
    """Reliable bits a fragment carries at mean bit accuracy `ba`: the binary-symmetric-channel bound
    n(1 - H(1 - ba)) of its n-bit codeword, zero at or below chance.

    This is what a bit-accuracy measurement can say about capacity, and it is the capacity the solver
    uses (see the capacity clause in add_strength_order). Checked against the measured soft-information
    capacity curves at all 244 knots of the 42 measured cells (capacity_bound_check.json): for every
    payload size up to 77 bits, no knot where the measurement refuses the payload has the bound granting
    it (the bound sits a median 7 bits below the measurement, the soft-decoding gain); the only excesses
    are +0.3 bits at 78 bits and up to +3.9 bits at saturation, where the soft estimator tops out at 96.
    Requests ask for at most 50 bits."""
    ba = float(ba)
    return 0.0 if ba <= 0.5 else n * (1.0 - _h2(1.0 - ba))


def bits_to_ba(bits, n=NATIVE_BITS):
    """The mean bit accuracy at which ba_to_bits reaches `bits`: 0.5 at or below zero bits, 1.0 at n
    bits or more. A payload constraint `capacity >= bits` is the strength condition `ba >= bits_to_ba(bits)`."""
    bits = float(bits)
    if bits <= 0.0: return 0.5
    if bits >= n: return 1.0
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ba_to_bits(mid, n) < bits: lo = mid
        else: hi = mid
    return hi


def achievable_bits(chosen, attacks, resync=False, nested=False):
    def eff(f,a):   # mirror the constraint's front-end capacity recovery
        c=CAP[f].get(a,0.0)
        if a in ("rot9","rot30") and f in ("TrustMark","VideoSeal") and resync: return max(c,CAP[f]["clean"])
        if a in ("crop90","crop75","crop50") and f=="VINE" and nested: return max(c,CAP[f]["clean"])
        if a=="unmarker" and f=="VINE" and nested: return max(c,UNMK_CAP_VINE)   # measured >=37-bit lower bound
        return c
    meas=[a for a in attacks if a in CAP["VINE"]]
    if not meas: return None
    return min(max(eff(f,a) for f in chosen) for a in meas)

def frag_defends(f, a, thr, alpha_ge07, resync, nested):
    """Z3 Bool: fragment f (a real fragment name or 'VINE_C3') decodes attack a >= thr under front-ends."""
    boost=BoolVal(0.96>=thr)
    if f=="VINE":
        # alpha affects PSNR + signal robustness; regen-family needs alpha>=0.7
        if a in SWEPT and a!="crop75":
            # min over allowed alpha is enforced elsewhere; here use worst-case? -> use per-alpha handled in caller
            base=BoolVal(True)  # replaced by alpha-specific constraint in caller
            return base
        v=BA_full("VINE",a)
        if a in ("regen","rinse","vaeB","vaeC","ctrlregen","ctrlregen_s03","ctrlregen_s05","ctrlregen_s07"):
            return And(BoolVal(v>=thr), alpha_ge07)      # regen-family (incl CtrlRegen+ at any strength) requires alpha>=0.7
        if a in ("crop90","crop75","crop50"): return Or(BoolVal(v>=thr), And(nested,boost))
        # UnMarker: bare decode dies (0.53) but the nested/fine-scale-search decode recovers it to
        # UNMK_REC=0.98 (measured, unified_bitacc + unmarker100 sweeps) -- same shape as the crop clause.
        if a=="unmarker": return Or(BoolVal(v>=thr), And(nested,BoolVal(UNMK_REC>=thr)))
        return BoolVal(v>=thr)
    # pixel fragments TrustMark / VideoSeal
    base=BoolVal(BA_full(f,a)>=thr)
    if a in ("rot9","rot30") and f in ("TrustMark","VideoSeal"): return Or(base, And(resync,boost))
    if a=="crop75" and f in ("TrustMark","VideoSeal"): return Or(base, And(resync,boost))
    return base

def solo_psnr_expr(f, a_lvl):
    if f=="VINE": return If(a_lvl==0,ALPHA["0.3"]["psnr"],If(a_lvl==1,ALPHA["0.5"]["psnr"],If(a_lvl==2,ALPHA["0.7"]["psnr"],ALPHA["1.0"]["psnr"])))
    return PSNR_SOLO.get(f,40.0)
def beta_from_fpr(alpha, n_tx=100, t_corr=10, k_data=37):
    """Single-test raw BA floor for legacy clean-strength priors only.

    Selection acceptance uses the original fpr and full config in build().
    BCH parameters are retained solely for source-compatible legacy callers.
    """
    if n_tx != 100:
        raise ValueError('unified protocol requires 100 bits')
    from unified_detector import threshold_for_config
    return threshold_for_config(dict(order=['VINE'], fe_on=[]), alpha)


def cost_ms(f):
    return D["fragments"][f]["embed_ms"]+D["fragments"][f]["decode_ms"]

def presence_threshold(min_ba, k, n_tx=100, t_corr=10):
    """Retired: a quantized BA floor cannot recover original FPR or geometry K."""
    raise ValueError('use unified_detector.threshold_for_config(cfg, original_fpr); m-only thresholds are retired')


FR=["VINE","TrustMark","VideoSeal"]
# ---------------------------------------------------------------------------------------------
# RESOLUTION. Every table above (F, CAP, PSNR_SOLO, ALPHA, cost_ms, PEN*) is measured at 512x512 only.
# `resolution` is therefore a DECLARED-BUT-GATED input: build() accepts it, and refuses any value that
# has no measured table rather than silently answering with 512 numbers. Populate RES_TABLES (and drop
# the guard) once the per-resolution campaign lands -- it must supply per-res attack bit-acc AND
# per-res PSNR, latency and stacking penalty, since quality and runtime constraints consume those too.
# Evidence (res_invariance_pilot.json n=50 + res_campaign.json n=50, TOST margin 0.02):
#   * GEOMETRY (crop75/crop50/rot9) is resolution-INVARIANT at every R tested -- the 512 cells transfer.
#   * SIGNAL (jpeg/blur/noise/clean) transfers for R>=512 but BREAKS at 256 (VINE/blur -0.044,
#     TrustMark/jpeg -0.036, VideoSeal -0.14..-0.17): the mark is small relative to fixed-PIXEL attacks.
#   * VAE breaks at 256 for every fragment (-0.03 VINE .. -0.29 VideoSeal); fine for R>=1024.
#   * DIFFUSION/ADVERSARIAL (regen/rinse/ctrlregen*/unmarker) are NOT measured at any R != 512.
#   * VideoSeal has NO native-resolution path (0/50 preserved at every R != 512), so its non-512 cells
#     are a forced-resize artifact; VINE and TrustMark preserve native resolution 50/50.
# ---------------------------------------------------------------------------------------------
# IMMUTABLE BASELINE + REQUEST-SCOPED OVERLAY.
# Live/CEGAR measurements are taken on ONE user's images, so writing them back into the shared table
# would let that user's data distribution bias every later request (and the bias would compound).
# The baseline measured table is therefore frozen here, and per-request refinements are applied only
# inside `request_overlay(...)`, which restores the baseline on exit -- including on exception.
# Every request starts from the same pristine baseline.
import copy as _copy
from contextlib import contextmanager as _ctx
_BASELINE = {"F": _copy.deepcopy(F), "CAP": _copy.deepcopy(CAP), "PSNR_SOLO": _copy.deepcopy(PSNR_SOLO)}

def baseline_ba(f, a):
    """Bit-acc from the FROZEN baseline, ignoring any active overlay."""
    return _BASELINE["F"][f]["bit_acc"].get(a, 0.0)

@_ctx
def request_overlay(overlay=None):
    """Scope a per-request table refinement.

    overlay = {"bit_acc": {(frag, attack): value}, "psnr_solo": {frag: value}, "cap": {(frag, attack): value}}
    Applies on top of the frozen baseline for the duration of the block, then restores it. Nothing a
    request measures can leak into the next request.
    """
    snap = {"F": _copy.deepcopy(F), "CAP": _copy.deepcopy(CAP), "PSNR_SOLO": _copy.deepcopy(PSNR_SOLO)}
    try:
        if overlay:
            for (fr, at), v in (overlay.get("bit_acc") or {}).items(): F[fr]["bit_acc"][at] = v
            for (fr, at), v in (overlay.get("cap") or {}).items():     CAP[fr][at] = v
            for fr, v in (overlay.get("psnr_solo") or {}).items():     PSNR_SOLO[fr] = v
        yield
    finally:                                    # restore the baseline no matter what happened
        F.clear();        F.update(snap["F"])
        CAP.clear();      CAP.update(snap["CAP"])
        PSNR_SOLO.clear(); PSNR_SOLO.update(snap["PSNR_SOLO"])

def assert_baseline_intact():
    """Guard for tests/harnesses: the shared table must equal the frozen baseline between requests."""
    bad = [(f, a) for f in F for a in F[f]["bit_acc"]
           if abs(F[f]["bit_acc"][a] - _BASELINE["F"][f]["bit_acc"].get(a, -1)) > 1e-12]
    if bad: raise AssertionError(f"baseline table was mutated outside request_overlay: {bad[:5]}")
    return True

# ---------------------------------------------------------------------------------------------
# VARIANCE-AWARE LIVE REQUIREMENT.
# The offline table stores a MEAN bit-acc. For some attacks that mean does not represent a specific
# user's image, so a table verdict on those attacks is not trustworthy -- the request MUST be settled by
# a live measurement on the user's OWN image:
#   * ADVERSARIAL attacks (UnMarker) optimise per-image worst-case; the offline mean is the attacker's
#     success on OTHER images, not this one -> ALWAYS live.
#   * HIGH-VARIANCE attacks: when the table mean sits within k*std of the required threshold theta, the
#     specific image can land on either side -> live to resolve (variance_profile.json supplies std).
ADVERSARIAL = {"unmarker"}                 # per-image optimised -> mean is never the point estimate
# Diffusion-family attacks are stochastic (per-image regeneration) and were measured at small n, so a
# point estimate that sits near theta must not decide feasibility alone -> give them a conservative
# default std so live_required_at() fires near the threshold where ba_sd_profile has no entry.
DIFFUSION = {"regen", "rinse", "ctrlregen", "ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07"}
DEFAULT_STD_DIFFUSION = 0.08
# BA_STD (variance_profile.json) served only the retired live_required(); the gate in force reads BA_SD_COL.
# Clean-image floors (follow-up item 3). `measure_clean_minimum.py` sweeps each fragment on unattacked
# images and records, per strength, the mean/sd of the hard bit accuracy and the fraction of images that
# clear the presence level; below the floor a fragment cannot be read reliably even before any attack,
# so the solver must not propose it there whatever the mean curve says. Two levels: presence = the
# zero-bit test (tau 0.63; floor = smallest strength with >=99% of images above tau) and identity = the
# soft-decision crypto verify, which needs about 0.83 hard bit accuracy on the image at hand (floor =
# smallest strength whose mean - z_0.99 * sd clears 0.83; the sweep stored no per-image values).
# An identity request (min_bits > 0) takes the identity floor; a presence request interpolates in its
# threshold between the two levels. Measured 2026-09-03 on the N=100 fitting slice:
CLEAN_FLOOR_DEFAULT = {"VINE": (0.14, 0.30), "TrustMark": (0.55, 0.70), "VideoSeal": (0.50, 0.625)}
CLEAN_FLOOR_JSON = _bos.environ.get("CLEAN_MIN_JSON",
                                    str(_RUN_INPUTS / "clean_minimum_strength.json"))
_CLEAN_IDENTITY_BA, _CLEAN_Z99, _CLEAN_PRESENCE_TAU = 0.83, 2.326, 0.63
def clean_floor_levels(path=None):
    """{fragment: (presence_floor, identity_floor)} from the clean sweep, or the baked defaults."""
    path = path or CLEAN_FLOOR_JSON
    try:
        d = _bjson.load(open(path))
    except (OSError, ValueError):
        return dict(CLEAN_FLOOR_DEFAULT)
    out = {}
    for f, rows in d["rows"].items():
        pres = next((r["s"] for r in rows if r["frac_presence"] >= 0.99), None)
        ident = next((r["s"] for r in rows if r["ba_mean"] - _CLEAN_Z99 * r["ba_sd"] >= _CLEAN_IDENTITY_BA), None)
        if pres is None or ident is None:
            pres, ident = CLEAN_FLOOR_DEFAULT.get(f, (0.0, 0.0))
        out[f] = (float(pres), float(max(pres, ident)))
    return out
def clean_floors(min_ba, min_bits=0, levels=None):
    """Per-fragment minimum strength for a request: identity floor when identity bits are required,
    otherwise linear in the threshold between the presence and identity levels."""
    levels = levels if levels is not None else clean_floor_levels()
    t = 1.0 if min_bits > 0 else min(1.0, max(0.0, (float(min_ba) - _CLEAN_PRESENCE_TAU) /
                                                  (_CLEAN_IDENTITY_BA - _CLEAN_PRESENCE_TAU)))
    return {f: round(p + t * (i - p), 4) for f, (p, i) in levels.items()}

# A certified optimum sits exactly ON the binding constraint: the solver spends no strength it does not
# have to, so the configuration it returns is predicted to clear the threshold by nothing at all. The
# table's curve is a mean over 100 images and a deployment reads one image, so half the live samples of a
# cell that predicts exactly the threshold land below it. Measured on 60 certified requests against 50
# unseen cross-source images each: with no allowance, 7 of 59 configurations missed the requirement, every
# one of them by 0.002 to 0.005, inside one standard error of the live mean (median 0.0074). An allowance
# of two standard errors removes all seven and turns 4 of the 60 requests infeasible; 0.05 costs 9 more
# infeasible requests and prevents nothing further. The allowance does not make the prediction better (the
# error is 0.004 at every setting) -- it moves the line the optimum sits on.
DEFAULT_MARGIN = 0.02


# live_required() (the legacy gate at a fixed operating point) is retired: see attic/repo/solver/
# retired_from_watermark_smt_v2.py. live_required_at() below is the gate in force.


# Per-column image-to-image SD, measured (ba_sd_profile.json: sd = se*sqrt(n) pooled over the certified
# configurations, plus the ring campaign's per-knot spread on the columns the scale stage replaces).
# variance_profile.json covered 14 classical columns at one fixed operating point; the gate below has to
# answer for every column a request can name, and the widest spreads are exactly the columns the offline
# mean settles least well: crop_jpeg 0.133, crop50 0.112, regen 0.097, rinse2x 0.098, unmarker 0.084
# (0.228 at the ring curve's widest knot). A key may be qualified by the front-end stage the solve read,
# because the stage changes the embed and with it the spread.
BA_SD_COL = {}
try:
    _sdp = _bos.environ.get("BA_SD_PROFILE",
                            str(_RUN_INPUTS / "ba_sd_profile.json"))
    if _bos.path.exists(_sdp):
        for _k, _v in _bjson.load(open(_sdp)).items():
            BA_SD_COL[_k] = float(_v.get("sd", 0.0))
except Exception:
    pass


def live_required_at(attack, table_value, threshold, sd=None, k=2.0, stage=None):
    """Whether the OFFLINE TABLE may settle this (attack, configuration, threshold), or a live
    measurement on the user's own image must.

    The retired live_required() answered the same question from a fixed operating point (BA_full), which
    is not the number a request's verdict rests on. This takes `table_value`: what the solve actually
    read for the configuration it returned, under the front-end stage it turned on. Two reasons to
    refuse the table:

      adversarial      UnMarker optimises against the image in front of it, so the offline mean is the
                       attacker's success rate on OTHER images. Measured on the certified C4 requests:
                       four of nine cleared the mean threshold with per-image detection 0.30 to 0.80,
                       i.e. the request passed while most single images did not.
      near-threshold   a certified optimum sits ON the constraint, so when the table value is within
                       k image-to-image standard deviations of the threshold the specific image can land
                       on either side. Measured: of the cells that clear the mean, 9% of the benign ones
                       have per-image detection below 0.90, against 52% of the diffusion ones.

    Returns (required, reason). The caller decides what to do with a required cell it cannot execute:
    a verdict that says the table settled it would be the failure this gate exists to prevent.
    """
    if attack in ADVERSARIAL:
        return True, "adversarial (per-image worst-case; the offline mean is the attacker's score on other images)"
    if sd is None:
        for key in ((f"{attack}@{stage}",) if stage else ()) + (attack,):
            if key in BA_SD_COL and BA_SD_COL[key] > 0:
                sd = BA_SD_COL[key]; break
    if sd is None and attack in DIFFUSION:
        sd = DEFAULT_STD_DIFFUSION
    if not sd or sd <= 0:
        return False, "no variance data"
    gap = abs(float(table_value) - float(threshold))
    if gap < k * sd:
        return True, f"near-threshold: |{table_value:.3f}-{threshold:.3f}|={gap:.3f} < {k}*{sd:.3f}"
    return False, f"table value is more than {k} sd from the threshold"

RES_MEASURED = {256, 512, 1024, 2048}
_GEOM = {"crop90", "crop75", "crop50", "rot9", "rot30"}
_SIGNAL = {"clean", "jpeg25", "jpeg50", "blur", "noise", "bright", "contrast"}
_VAE = {"vaeB", "vaeC"}
RES_OK = {512: None,                      # None = the measurement basis; everything allowed
          1024: _GEOM | _SIGNAL | _VAE,   # verified transferable at R>=1024
          2048: _GEOM | _SIGNAL | _VAE,
          256:  _GEOM}                    # only geometry survives the 256 check
def _check_resolution(res, attacks=()):
    if res not in RES_MEASURED:
        raise ValueError(f"resolution={res} was never measured (have {sorted(RES_MEASURED)}).")
    ok = RES_OK.get(res)
    if ok is None: return
    missing = [a for a in attacks if a not in ok]
    if missing:
        raise ValueError(
            f"resolution={res} has no validated table for {sorted(set(missing))}. Validated at this "
            f"resolution: {sorted(ok)}. The diffusion/adversarial family was measured only at 512, and "
            f"signal/VAE cells do not transfer to 256. Measure those cells or pass resolution=512.")

# Per-image acceptance floor (see eff_det in add_strength_order): a selected fragment must clear the
# request's threshold on at least this fraction of the measured images, not only on average. On by
# default (user 2026-09-06); det_min=0 restores the mean-only clause.
DEFAULT_DET_MIN = 0.90
# Margin on that floor (2026-09-08). The floor is a rate estimated from the table's images, and the optimum sits
# exactly on it, so on fresh images the rate falls below it half the time: in the class certification every
# failing cell read 0.80..0.89 live where the table said 0.90..0.94, with the table's means accurate to 0.001.
# The solver therefore asks the TABLE for det_min + DEFAULT_RATE_MARGIN so that the DEPLOYED floor det_min
# holds on new images: two standard errors of a rate near 0.9 estimated from 100 images. The capacity line
# gets the request's mean margin (DEFAULT_MARGIN) for the same reason (334 C3 requests missed it live by a
# median 0.004). Both are charged where the floor binds and nowhere else.
DEFAULT_RATE_MARGIN = 0.06


def build(min_psnr,max_ms,attacks,min_ba,allow_resync,allow_nested,min_bits=0,resolution=512,
          enable_order=False,continuous_strength=False,surrogate=None,margin=DEFAULT_MARGIN,clean_floor=None,
          fe_gain_min=0.05, pair_min_cascade=0.02, det_min=DEFAULT_DET_MIN, rate_margin=DEFAULT_RATE_MARGIN, fpr=None):
    _check_resolution(resolution, attacks)
    surrogate_mode = (enable_order or continuous_strength) and surrogate is not None
    if not surrogate_mode or fpr is None or min_bits != 0:
        raise ValueError('unified selection requires continuous surrogate, original fpr and capacity-output min_bits=0')
    opt=Optimize()
    opt._fpr = float(fpr)
    opt._prior_protocol = 'legacy_keyed_decoder_ba_prior_requires_fresh_unified_live'
    use={f:Bool(f) for f in FR}; resync=Bool("resync"); nested=Bool("nested")
    # The deployed cascade exposes four stages. `resync` covers SyncSeal rectify plus the residual
    # tilt (they share an embed-side mark and never run apart); `scale` is the VINE ring search,
    # which is what `nested` used to stand for; `angle` is the blind probe, a decode-side front-end
    # with no embed-side mark and so no PSNR cost at all -- it was never separately selectable.
    fe_angle=Bool("fe_angle"); fe_tile=Bool("fe_tile")
    FEV={"resync":resync, "scale":nested, "angle":fe_angle, "tile":fe_tile}
    # At most one at a time. Three of them change what gets embedded, so a pair is an embed nobody
    # measured -- the table holds a curve for "tiled TrustMark" and one for "TrustMark under a sync
    # mark", and none for both at once. The bundled overlay says the restriction is cheap: running
    # every stage beat the best single stage by a median 0.015 detection and never by more than
    # 0.180, because the overlapping stages rescue the same images.
    opt.add(AtMost(*FEV.values(), 1))
    a_lvl=Int("alpha_lvl")                                   # 0->0.5, 1->0.7, 2->1.0 (VINE only)
    opt.add(a_lvl>=0, a_lvl<=3)
    if not allow_resync: opt.add(Not(resync))
    if not allow_resync and not allow_nested: opt.add(Not(fe_angle)); opt.add(Not(fe_tile))
    opt.add(Implies(fe_tile, use["TrustMark"]))   # the tiled ring is a TrustMark-side embedding
    if not allow_nested: opt.add(Not(nested))
    opt.add(Implies(nested, use["VINE"]))
    opt.add(Implies(Not(use["VINE"]), a_lvl==3))            # alpha only meaningful with VINE
    opt.add(Or(*[use[f] for f in FR]))
    nfrag=Sum([If(use[f],1,0) for f in FR])
    # PSNR = lowest solo PSNR among included  - stacking - nested
    base=Real("basepsnr"); opt.add(base<=If(use["VINE"],solo_psnr_expr("VINE",a_lvl),99))
    opt.add(base<=If(use["TrustMark"],PSNR_SOLO["TrustMark"],99))
    opt.add(base<=If(use["VideoSeal"],PSNR_SOLO["VideoSeal"],99))
    # base must EQUAL the min (tie it down): base >= each-or-not handled by maximizing psnr later
    pen=If(nfrag<=1, 0.0, If(nfrag==2, PEN2, If(nfrag==3, PEN3, PEN4)))   # MEASURED stacking penalty (~0.6dB/frag)
    psnr=base - pen - If(nested,NESTED_PEN,0.0)
    if not surrogate_mode:                 # in surrogate_mode the fidelity floor is stated once,
        opt.add(psnr>=min_psnr)            # on the surrogate objective (see add_strength_order);
                                           # binding the discrete expression as well would let a
                                           # tighter floor select a worse continuous optimum.
    # Each stage is charged its own measured wall time. Under the bundled flags the solver paid
    # 300 ms for resync and 1600 ms for nested no matter which stages actually ran.
    def _fe_ms(cname, default):
        if surrogate_mode and surrogate is not None:
            v = [surrogate.latency(f"latency_ms_{cname}|{a}") for a in attacks]
            # A stage's cost is measured as (decode with it) minus (decode without it). On a column
            # where the primary decode already succeeds the stage never fires, so that difference is
            # timing noise around zero and comes out negative about half the time. Left signed, a
            # negative cost is a time REBATE for switching on a front-end that does nothing, which
            # makes the useless stage attractive. A stage cannot make the decode faster.
            v = [max(0.0, float(x)) for x in v if isinstance(x, (int, float))]
            if v: return max(v)                # the cascade runs once; charge the worst requested column
        return default
    time=Sum([If(use[f],cost_ms(f),0.0) for f in FR]) \
         + If(resync, _fe_ms("resync", FE_MS['resync']), 0.0) \
         + If(nested, _fe_ms("scale",  FE_MS['nested']), 0.0) \
         + If(fe_angle, _fe_ms("angle", 0.0), 0.0) \
         + If(fe_tile,  _fe_ms("tile",  0.0), 0.0)
    opt.add(time<=max_ms)
    a_ge07 = a_lvl>=2
    if not surrogate_mode:
        # discrete per-attack min_ba feasibility from the measured tables. Skipped in surrogate_mode:
        # the necessity experiment compares z3 against a grid enumerator that evaluates ONLY the
        # surrogate, so z3's feasible set must be driven by the surrogate alone (added below via
        # add_strength_order), not the real measured tables AND the surrogate at once.
        for a in attacks:
            opts=[]
            for f in list(F):
                if f=="VINE" and a in SWEPT and a!="crop75":
                    # alpha-specific measured bit-acc for swept signal attacks
                    ok=Or(*[And(a_lvl==i, BoolVal(ALPHA[str(ALV[i])][a]>=min_ba)) for i in range(4)])
                    opts.append(And(use["VINE"], ok))
                else:
                    opts.append(And(use[f], frag_defends(f,a,min_ba,a_ge07,resync,nested)))
            opt.add(Or(*opts))
    # CAPACITY constraint: need >= min_bits robust ID bits under each MEASURED in-scope attack
    # (some chosen fragment must carry >= min_bits reliable bits). Unmeasured attacks are skipped (warned in caller).
    if min_bits>0 and not surrogate_mode:      # surrogate_mode gets capacity from the measured
        for a in attacks:                      # per-strength curves inside add_strength_order
            if a not in CAP["VINE"]: continue   # capacity unmeasured for this attack -> skip
            capopts=[]
            for f in list(F):
                c=CAP[f].get(a,0.0); cl=CAP[f]["clean"]      # resync un-warps rotation -> capacity recovers toward clean
                if a in ("rot9","rot30") and f in ("TrustMark","VideoSeal"):
                    capopts.append(And(use[f], Or(BoolVal(c>=min_bits), And(resync, BoolVal(cl>=min_bits)))))
                elif a in ("crop90","crop75","crop50") and f=="VINE":  # nested-ring recovers VINE crop capacity
                    capopts.append(And(use[f], Or(BoolVal(c>=min_bits), And(nested, BoolVal(cl>=min_bits)))))
                elif a=="unmarker" and f=="VINE":            # nested/scale-search recovers VINE under UnMarker
                    # measured lower bound: the full 37-bit ID crypto-verifies at 2^-37 (unified_bitacc
                    # detect_crypto 1.0), so >=UNMK_CAP_VINE bits get through when nested is enabled.
                    capopts.append(And(use[f], Or(BoolVal(c>=min_bits), And(nested, BoolVal(UNMK_CAP_VINE>=min_bits)))))
                else:
                    capopts.append(And(use[f], BoolVal(c>=min_bits)))
            opt.add(Or(*capopts))
    if surrogate_mode:
        s_vars, p_vars, psnr_expr = add_strength_order(opt, use, attacks, surrogate, min_ba, enable_order,
                                               resync=resync, nested=nested, min_bits=min_bits,
                                               min_psnr=min_psnr, frontends=FEV, margin=margin,
                                               clean_floor=clean_floor, fe_gain_min=fe_gain_min,
                                               pair_min_cascade=pair_min_cascade, det_min=det_min, rate_margin=rate_margin, fpr=fpr)
        psnr = psnr_expr          # override the discrete-PSNR expression with the surrogate PSNR
        opt._svars = s_vars; opt._pvars = p_vars    # expose for callers/tests
        opt._fevars = FEV                            # the front-end decisions, one per cascade stage
    return opt,use,resync,nested,a_lvl,nfrag,psnr,time

def solve_exact(scen, eps=1e-9, max_rounds=32, **build_kw):
    """Solve to the CERTIFIED optimum, returning (value, rounds, certified).

    z3's `Optimize.maximize` is not complete on this encoding. On the measured surrogate it returns a
    satisfying but sub-optimal model, and it does so non-deterministically: repeated identical calls
    disagree by as much as 2 dB. Trusting it breaks the property the method rests on -- that the
    returned configuration is the best one meeting the request -- and it also understates the solver
    against a grid baseline, because a grid that lands on a better point then appears to beat an
    "exact" optimum.

    `check` itself is sound, so the optimizer is used only to propose a candidate, which is then
    certified: a fresh instance is asked whether any strictly better objective is satisfiable. Unsat
    proves optimality; sat yields a better candidate and the query repeats. Each round improves the
    objective by more than eps, and the objective is bounded, so this terminates -- in practice after
    zero or one extra round.

    Returns (None, rounds, True) when the request itself is unsatisfiable.
    """
    import z3
    o, u, rs, ns, al, nf, ps, tm = build(**scen, **build_kw)
    o.maximize(ps)
    if o.check() != z3.sat:
        return None, 0, True
    best = float(o.model().eval(ps).as_fraction())
    for rounds in range(max_rounds):
        probe = build(**scen, **build_kw)
        o2, ps2 = probe[0], probe[6]
        o2.add(ps2 > best + eps)
        if o2.check() != z3.sat:
            return best, rounds, True                            # certificate: nothing strictly better
        o2.maximize(ps2)                                         # improve inside the restricted region
        o2.check()
        best = float(o2.model().eval(ps2).as_fraction())
    return best, max_rounds, False                               # gave up before certifying


def solve_exact_model(scen, eps=1e-9, max_rounds=32, **build_kw):
    """As solve_exact, but also returns an Optimize pinned to the certified optimum.

    The certified value comes from a probe instance, so the configuration behind it must be recovered
    on an instance the caller can read variables from. We rebuild once and constrain the objective to
    the certified value, which leaves exactly the optimal configurations satisfiable.
    """
    import z3
    best, rounds, certified = solve_exact(scen, eps=eps, max_rounds=max_rounds, **build_kw)
    if best is None:
        return None, None, rounds, certified
    built = build(**scen, **build_kw)
    o, ps = built[0], built[6]
    o.add(ps >= best - eps)
    assert o.check() == z3.sat, "the certified optimum became unsatisfiable when pinned"
    # Among the configurations that hit the certified optimum, return one that switches on the
    # fewest front-ends. A front-end costs no PSNR unless it changes the embed, and the latency
    # budget only binds sometimes, so on many requests a stage that does nothing on the requested
    # columns is free in the objective and z3 sets it either way. The optimum is the same; the
    # configuration is not, and the method's claim is that what comes back is what the request
    # needs. Ties among the remaining choices stay arbitrary -- this settles only this one.
    fev = getattr(o, "_fevars", None)
    if fev:
        o.minimize(z3.Sum([z3.If(v, 1, 0) for v in fev.values()]))
        assert o.check() == z3.sat, "pinning the optimum and minimising front-ends became unsat"
    return built, o.model(), rounds, certified


def frontend_decisions(opt, model, resync=None, nested=None):
    """The front-end half of the solver's answer, as {stage: bool}.

    Read the stages off `opt._fevars` rather than off the two booleans the cascade used to collapse
    into. Callers that read only resync/nested deploy a configuration the solver did not return:
    the stages they drop are off in the deployment while the table credits them.
    """
    import z3
    fev = getattr(opt, "_fevars", None)
    if not fev:
        fev = {}
        if resync is not None: fev["resync"] = resync
        if nested is not None: fev["scale"] = nested
    return {k: bool(z3.is_true(model.eval(v, True))) for k, v in fev.items()}


def frontend_config(fe, alpha=None):
    """Map {stage: bool} onto the keys OursComposite reads, so one place owns the correspondence.
    `alpha` (the request's false-positive budget) is passed through as the decoder's presence budget."""
    out = {"resync": fe.get("resync", False),
           "nested": fe.get("scale", False),          # the ring is the embed side of the scale search
           "scale_search": fe.get("scale", False),
           "angle_sweep": fe.get("angle", False),
           "tile": fe.get("tile", False)}
    if alpha is not None: out["alpha"] = float(alpha)
    return out


def add_strength_order(opt, u, sel_attacks, surrogate, min_ba, order,
                       resync=None, nested=None, min_bits=0, min_psnr=None, frontends=None, margin=DEFAULT_MARGIN,
                       clean_floor=None, fe_gain_min=0.05, pair_min_cascade=0.02, det_min=DEFAULT_DET_MIN,
                       rate_margin=DEFAULT_RATE_MARGIN, fpr=None):
    import z3
    assert set(surrogate.fragments) <= set(u.keys()), "surrogate fragments must be a subset of the solver's fragment vars"
    # One decision variable per separately selectable geometric front-end. `resync` and `nested`
    # stay accepted under their old names so existing callers keep working; a caller that passes
    # `frontends` gets the full set the deployed cascade actually exposes.
    FE_VARS = dict(frontends) if frontends else {}
    if not FE_VARS:
        if resync is not None: FE_VARS["resync"] = resync
        if nested is not None: FE_VARS["scale"] = nested
    FE_GAIN = set()          # stages that supplied a measured gain on at least one cell
    # A stage is RESPONSIBLE for a requested column when its replacement curve beats the plain curve
    # by at least `fe_gain_min` somewhere in the strength range for some fragment (0.05 is about
    # 2.5 standard errors of a mean bit accuracy at N=100). A stage responsible for none of the
    # requested columns is fixed off: its non-geometric replacement curves differ from the plain ones
    # only by measurement noise, and left free the optimizer buys that noise (resync was selected on a
    # signal+VAE+regeneration request for a 0.05 dB gain). This is also what keeps the instance small
    # on requests without geometry.
    FE_RESP = {}
    FRs = surrogate.fragments
    s = {f: z3.Real(f"s_{f}") for f in FRs}
    # Strength range = [max(table range, clean-image floor for this request's level), table top].
    # `clean_floor=None` reads the measured floors; `{}` switches them off; a dict pins them.
    floors = clean_floors(min_ba, min_bits) if clean_floor is None else dict(clean_floor)
    opt._clean_floors = {}
    for f in FRs:
        lo,hi = surrogate.range(f)
        lo = max(lo, floors.get(f, lo))
        assert lo <= hi, f"clean floor {lo} above the measured top {hi} for {f}"
        opt._clean_floors[f] = lo
        opt.add(z3.Implies(u[f], z3.And(s[f] >= lo, s[f] <= hi)))
        opt.add(z3.Implies(z3.Not(u[f]), s[f] == 0))
    # precedence booleans (only meaningful among selected)
    p = {(f,g): z3.Bool(f"p_{f}_{g}") for f in FRs for g in FRs if f!=g}
    if order:
        for f in FRs:
            for g in FRs:
                if f>=g: continue
                both = z3.And(u[f], u[g])
                opt.add(z3.Implies(both, p[(f,g)] != p[(g,f)]))          # exactly one direction
        for f in FRs:                                                    # transitivity
            for g in FRs:
                for h in FRs:
                    if len({f,g,h})<3: continue
                    opt.add(z3.Implies(z3.And(p[(f,g)],p[(g,h)]), p[(f,h)]))
    else:
        for (f,g) in p: opt.add(p[(f,g)] == False)                       # fixed canonical order
    # feasibility: base_f(s_f,a) - sum_g after f delta_{g->f}(s_g,a) >= min_ba
    # Composite detection is BEST-PATH (max over fragments): a config survives attack a iff SOME
    # selected fragment clears it, not iff EVERY selected fragment clears it. So the per-fragment
    # feasibility clause is OR'd across fragments (at least one clears), not asserted independently
    # per fragment (which wrongly required every selected fragment to survive every attack).
    # The PWL domain constraints below are added unconditionally rather than gated on selection.
    # add_to_z3 clamps an out-of-range input to the endpoint value, so a fragment that is not
    # selected (its strength pinned to 0, outside its native range) still yields a well-defined
    # curve value instead of an unsatisfiable branch -- which is what the gating used to be for.
    # Leaving them gated has a cost that is easy to miss: a gate that is false leaves its curve
    # variable free, and a free real sitting inside an ite term is enough to make the optimizer
    # report a suboptimal point as optimal.
    def after_of(f, g):
        """z3 condition: g is selected and embedded after f (overwrites f)."""
        return z3.And(p[(f,g)], u[g]) if order else z3.And(u[g], z3.BoolVal(FRs.index(g)>FRs.index(f)))
    from composite_prediction import composite_expression, Z3Backend, isotonic, floor_strength

    def full_mean(f, a):
        # This exact formula is shared with the numeric residual/combination
        # audit. Mean offsets are applied AFTER already-modelled deltas; rate
        # corrections live only in the independent per-image curves.
        backend = Z3Backend()
        result = composite_expression(
            surrogate, f, a, s, after_of, FE_VARS, backend,
            legacy_frontends=(("scale", nested), ("resync", resync)),
            pair_min_cascade=pair_min_cascade)
        for cname, gain in result["frontend_gains"].items():
            FE_GAIN.add(cname)
            if gain >= fe_gain_min:
                FE_RESP.setdefault(cname, set()).add(a)
        return result["mean"], backend.constraints

    # Which fragment's EMBED a stage changes: the sync mark is laid over every fragment, the ring
    # re-embeds VINE, the tiled grid replaces TrustMark's embed; the angle probe changes no embed.
    # Read by the acceptance floor (det_cond) and by the admissibility rule further down.
    FE_AFFECTS = {"resync": list(FRs), "scale": ["VINE"], "tile": ["TrustMark"]}

    _isotonic = isotonic
    _floor_strength = floor_strength

    from unified_detector import threshold_for_config
    if fpr is None:
        raise ValueError('original fpr is required; never invert quantized min_ba')
    def tau_for(k, stage=None):
        # For the solver's one-FE domain, K depends on m and stage only.
        # Pick a valid representative subset with the stage's required host.
        host = 'TrustMark' if stage == 'tile' else 'VINE'
        names = [host] + [g for g in ('VINE', 'TrustMark', 'VideoSeal') if g != host]
        return threshold_for_config(dict(order=names[:k], fe_on=[] if stage is None else [stage]), fpr)

    def stage_threshold(k, margin_value=0.0):
        def target(stage):
            tau = tau_for(k, stage)
            # An impossible count 101 stays impossible even with margin zero.
            return z3.RealVal(tau if tau > 1 else min(1.0, tau + float(margin_value)))
        expr = target(None)
        for stage, enabled in FE_VARS.items():
            if enabled is not None:
                expr = z3.If(enabled, target(stage), expr)
        return expr

    def det_cond(f, a, nsel):
        """Legacy BA-rate prior evaluated at the same full-K threshold as mean.

        Stored keyed acceptance flags never contribute. These rates remain
        search priors; fresh unified per-image live gates decide deployment.
        """
        def bound(curve):
            if curve is None:
                return z3.BoolVal(True)
            strength = _floor_strength(curve, min(1.0, float(det_min) + float(rate_margin)))
            return z3.BoolVal(False) if strength is None else s[f] >= z3.RealVal(strength)
        def at(k):
            expr = bound(surrogate.rate_curve(f, a, tau_for(k)))
            for stage, enabled in FE_VARS.items():
                if enabled is None:
                    continue
                tau = tau_for(k, stage)
                curve = surrogate.rate_curve(f, a, tau, stage=stage)
                if curve is None and f not in FE_AFFECTS.get(stage, []):
                    curve = surrogate.rate_curve(f, a, tau)
                if curve is None and f in FE_AFFECTS.get(stage, []) and getattr(surrogate, '_perimage', None):
                    condition = z3.BoolVal(False)
                else:
                    condition = bound(curve)
                expr = z3.If(enabled, condition, expr)
            return expr
        return z3.If(nsel <= 1, at(1), z3.If(nsel == 2, at(2), at(3)))

    # NOTE for anyone comparing this solver against the grid enumerator in the necessity
    # experiment: that enumerator evaluates surrogate.base() directly and models no front-end
    # variables, so the two only solve the SAME problem when the front-ends are disabled. Run
    # necessity scenarios with allow_resync=False, allow_nested=False (which forces both booleans
    # false, collapsing eff_base to the plain curve) or the z3-vs-grid comparison is not like-for-like.

    # An embed-changing stage changes what the affected fragment looks like on EVERY column. Where a
    # requested column has no replacement curve for it, eff_base would fall back to the plain curve and
    # credit an embed that was never measured -- tiled TrustMark under VAE compression read the plain
    # TrustMark curve, promised 0.94, delivered 0.82, and was certified a false SAT live. So on such a
    # request the stage may not be enabled together with the fragment it changes. Tables that predate
    # the per-stage curves (no base_fe_* cell at all) keep the legacy bundled behaviour.
    # FE_AFFECTS is defined above det_cond, which reads it too.
    if any(k.startswith("base_fe_") for k in getattr(surrogate, "_frontend", {})):
        for cname, affected in FE_AFFECTS.items():
            cvar = (FE_VARS or {}).get(cname)
            if cvar is None: continue
            for f in affected:
                if f not in FRs: continue
                # An ADVERSARIAL column (UnMarker) is live-only: the table never settles it, whatever it
                # holds there is a prior for the search, and the live measurement of the returned
                # configuration is the verdict. So a missing replacement curve on it does not make the
                # stage inadmissible; the plain curve stands in as the prior and live decides.
                unmeasured = [a for a in sel_attacks if a not in ADVERSARIAL
                              and surrogate.frontend(f"base_fe_{cname}_{f}|{a}") is None]
                if unmeasured:
                    opt.add(z3.Implies(cvar, z3.Not(u[f])))

    missing = [a for a in sel_attacks if a not in surrogate.attacks]
    if missing:
        # Silently skipping an unmeasured attack is the worst available answer: the request comes back
        # SAT with that constraint never having entered the formula, so a user who asked for protection
        # against it is handed a configuration with no evidence behind it. This is the same rule
        # _check_resolution already applies to the resolution axis, applied to the attack axis.
        raise ValueError(
            f"no measured surrogate for {sorted(set(missing))}; the request cannot be answered. "
            f"Measured attacks: {sorted(surrogate.attacks)}")
    # The same original-alpha, full-geometry-K threshold drives mean and rate.
    # Margin is a search buffer only; live gates use the bare configured tau.
    nsel = z3.Sum([z3.If(u[g], 1, 0) for g in FRs])
    thr = z3.If(nsel <= 1, stage_threshold(1, margin),
                z3.If(nsel == 2, stage_threshold(2, margin), stage_threshold(3, margin)))
    opt._threshold_expr = z3.If(nsel <= 1, stage_threshold(1),
                               z3.If(nsel == 2, stage_threshold(2), stage_threshold(3)))
    opt._mean_target_expr = thr
    eff_ba = {}          # (f, a) -> the bit-accuracy expression coverage is judged on; capacity reads it too
    opt._eff_ba = eff_ba  # numeric/Z3 audits inspect the actual coverage expression
    cover_of = {}        # (f, a) -> "fragment f clears attack a" (strength condition, selection not included);
    opt._cover = cover_of   # read by the candidate enumeration (watermark_smt_topk), which asks that every
    for a in sel_attacks:   # selected fragment clear at least one requested column
        clears = []
        for f in FRs:
            expression, constraints = full_mean(f, a)
            for constraint in constraints:
                opt.add(constraint)
            eff_ba[(f, a)] = expression
            cover = eff_ba[(f, a)] >= thr
            if det_min and det_min > 0.0:
                cover = z3.And(cover, det_cond(f, a, nsel))
            cover_of[(f, a)] = cover
            clears.append(z3.And(u[f], cover))                             # fragment f (if selected) clears attack a
        opt.add(z3.Or(*clears))                                            # at least one selected fragment clears a
    # distortion D = sum_f d_f(s_f) + sum_{f<g} e_{fg}(s_f+s_g) [gated by co-select]; PSNR = -D proxy
    dterms=[]
    for f in FRs:
        dexpr,dc = surrogate.d(f).add_to_z3(s[f], f"d_{f}")
        for c in dc: opt.add(c)
        dterms.append(z3.If(u[f], dexpr, z3.RealVal(0)))
    for i,f in enumerate(FRs):
        for g in FRs[i+1:]:
            ssum = z3.Real(f"ssum_{f}_{g}"); opt.add(ssum == s[f]+s[g])
            eexpr,ec = surrogate.e(f,g).add_to_z3(ssum, f"e_{f}_{g}")
            for c in ec: opt.add(c)
            dterms.append(z3.If(z3.And(u[f],u[g]), eexpr, z3.RealVal(0)))
    # nested-ring front-end: its distortion cost is a MEASURED function of VINE's strength
    # (it rises ~13x across the grid), not the single constant the discrete path uses.
    npc = surrogate.frontend("nested_penalty")
    if npc is not None and nested is not None and "VINE" in FRs:
        npe, npcons = npc.add_to_z3(s["VINE"], "nestpen")
        gate = z3.And(nested, u["VINE"])
        for c in npcons: opt.add(c)
        dterms.append(z3.If(gate, npe, z3.RealVal(0)))
    # Every EMBED-side front-end costs fidelity, not just the nested ring. The SyncSeal mark goes on
    # top of the image and the tiled ring replaces TrustMark's embed; left uncharged they are free,
    # and a front-end that is free and can only raise coverage will always be switched on -- the
    # solver would return front-ends it has no reason to return. The cost is a measured function of
    # the carrying fragment's strength, on the same MSE scale as d and e.
    FE_HOST = {"resync": "TrustMark", "scale": "VINE", "tile": "TrustMark"}
    # The tiled grid REPLACES TrustMark's embed, so it costs nothing unless TrustMark is selected. The
    # SyncSeal mark is its own embed on top of the image: it is paid whenever resync is on, whatever the
    # fragment set (2026-09-08: charged only with TrustMark selected, VINE+VideoSeal+resync solutions
    # reported 1.7 dB more than they delivered; live 38.97 dB against 40.68 in the table, the missing
    # 2.68 MSE being exactly penalty_fe_resync). Its curve is indexed by the host's strength and is flat,
    # so without the host it is charged at the curve's mid-range value.
    FE_MARK_INDEPENDENT = {"resync"}
    for cname, host in FE_HOST.items():
        if cname == "scale": continue                    # already charged as nested_penalty above
        cvar = (FE_VARS or {}).get(cname)
        pc = surrogate.frontend(f"penalty_fe_{cname}")
        if cvar is None or pc is None: continue
        if host not in FRs and cname not in FE_MARK_INDEPENDENT: continue
        # A front-end is bought for coverage, never as a fidelity rebate. The tiled grid measures a
        # slightly NEGATIVE cost (it writes a little less energy than the full-frame embed, -0.1 to
        # -1.6 MSE) and, left signed, that rebate alone made the solver attach the grid to a TrustMark
        # that covered nothing -- a structural choice driven by 0.04 dB. The lower energy is already
        # accounted for where it matters, in the stage's lower replacement curves.
        pe_pos = None
        if host in FRs:
            pe, pcons = pc.add_to_z3(s[host], f"fepen_{cname}")
            for c in pcons: opt.add(c)
            pe_pos = z3.If(pe > 0, pe, z3.RealVal(0))
        if cname in FE_MARK_INDEPENDENT:
            const = z3.RealVal(max(0.0, float(pc.eval(0.5 * (pc.xs[0] + pc.xs[-1])))))
            cost = z3.If(u[host], pe_pos, const) if pe_pos is not None else const
            dterms.append(z3.If(cvar, cost, z3.RealVal(0)))
        else:
            dterms.append(z3.If(z3.And(cvar, u[host]), pe_pos, z3.RealVal(0)))
    # Prune: a stage with no responsible column among the requested attacks is fixed off.
    opt._fe_responsible = {k: sorted(v) for k, v in FE_RESP.items()}
    for cname, cvar in (FE_VARS or {}).items():
        if cvar is None or cname not in FE_GAIN: continue
        if not FE_RESP.get(cname):
            opt.add(z3.Not(cvar))
    # A stage that supplies a measured GAIN but carries no measured COST is free, and a free stage
    # that can only raise coverage is switched on in every optimal model. Silently skipping the
    # missing cost is how it stayed free; the request cannot be answered without it.
    for cname, host in FE_HOST.items():
        if cname == "scale": continue                    # charged above as nested_penalty
        if cname not in FE_GAIN or host not in FRs: continue
        if (FE_VARS or {}).get(cname) is None: continue
        if surrogate.frontend(f"penalty_fe_{cname}") is None:
            raise ValueError(
                f"front-end '{cname}' has a measured effect but no measured fidelity cost "
                f"(penalty_fe_{cname}); it would be free and always selected. Measure it, or drop "
                f"the stage from the request.")
    D = z3.Real("D_total"); opt.add(D == z3.Sum(dterms))
    psnr = z3.Real("psnr_surro"); opt.add(psnr == -D)     # monotone proxy; real dB mapping applied post-hoc

    # The caller's fidelity floor is stated in dB, but the objective here is a distortion in MSE.
    # Converting the floor instead of the objective keeps the constraint linear: PSNR >= q is
    # exactly D <= 255^2 / 10^(q/10). Without this the floor would bind the discrete PSNR
    # expression that surrogate_mode replaces, i.e. a quantity that is no longer the objective.
    if min_psnr is not None and min_psnr > 0:
        opt.add(D <= (255.0 ** 2) / (10.0 ** (float(min_psnr) / 10.0)))

    # CAPACITY. The payload a fragment carries through a column is derived from the same bit-accuracy
    # expression coverage is judged on: ba_to_bits gives the reliable bits of the codeword at that
    # accuracy, so `capacity >= min_bits` is the strength condition `ba >= bits_to_ba(min_bits)`, with the
    # same best-path (OR) semantics as coverage (the ID is repeated across fragments, the best survivor
    # carries it). Deriving it from bit accuracy rather than reading the measured capacity curves gives
    # three things the curves could not: every column has it (18 cells had no curve and fell back to a
    # strength-flat constant), it follows the front-end the solver enables (the replacement curve
    # describes the changed embed, so the tiled-TrustMark carve-out that once lived here is no longer
    # needed), and a live measurement of bit accuracy re-certifies it through with_live (a soft-information
    # capacity cannot be measured live). It is a lower bound on the measured curves at every requested
    # payload size (see ba_to_bits), so it never grants a payload the measurement refused.
    if min_bits > 0:
        ba_star = z3.RealVal(min(1.0, bits_to_ba(min_bits) + float(margin)))     # the capacity line plus the request's margin
        for a in sel_attacks:
            opt.add(z3.Or(*[z3.And(u[f], eff_ba[(f, a)] >= ba_star) for f in FRs]))
    return s, p, psnr

def report(tag,opt,use,resync,nested,a_lvl,psnr,time,attacks,min_ba):
    if opt.check()!=sat:
        print(f"  ❌ UNSAT — no combination satisfies these conditions (relax quality/speed, lower bit-acc, or L4-undefendable)."); return None
    m=opt.model(); ev=lambda e:m.eval(e,model_completion=True)
    chosen=[f for f in FR if is_true(ev(use[f]))]; rv=is_true(ev(resync)); nv=is_true(ev(nested))
    al=ALV[int(str(ev(a_lvl)))]
    fe=[x for x,b in [("resync",rv),("nested",nv)] if b]
    disp=[(f+f"(α={al})" if f=="VINE" else f) for f in chosen]
    print(f"  ✅ {' + '.join(disp)}"+(f"  +[{', '.join(fe)}]" if fe else "")+f"   PSNR≈{float(ev(psnr).as_fraction()):.1f}dB · {float(ev(time).as_fraction()):.0f}ms")
    return chosen,al,rv,nv

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--min_psnr",type=float,default=32.0); ap.add_argument("--max_ms",type=float,default=5000)
    ap.add_argument("--attacks",nargs="+"); ap.add_argument("--min_ba",type=float,default=0.90)
    ap.add_argument("--no_resync",action="store_true"); ap.add_argument("--no_nested",action="store_true")
    ap.add_argument("--min_bits",type=int,default=0,help="required robust ID bits (capacity axis; 0=off). 37=deployed ID")
    ap.add_argument("--pareto",action="store_true")
    a=ap.parse_args()
    SIG=["jpeg50","jpeg25","blur","noise","bright","contrast"]; GEO=["crop90","crop75","rot9","rot30"]; REG=["vaeB","vaeC","regen","rinse"]
    if a.pareto:
        # Pareto front via quality-threshold sweep: for each min PSNR, the minimal combo that still defends.
        atk=a.attacks or SIG+GEO+REG
        print(f"\nPARETO front (quality ↔ minimal combo): attacks={atk} ba>={a.min_ba}")
        seen=set()
        for q in [30,32,34,36,38,40,42,44,46]:
            opt,use,resync,nested,a_lvl,nfrag,psnr,time=build(float(q),9e9,atk,a.min_ba,not a.no_resync,not a.no_nested)
            opt.minimize(nfrag); opt.maximize(psnr); opt.minimize(time)
            if opt.check()!=sat: continue
            m=opt.model(); ev=lambda e:m.eval(e,model_completion=True)
            ch=tuple(f for f in FR if is_true(ev(use[f]))); al=ALV[int(str(ev(a_lvl)))]
            pv=float(ev(psnr).as_fraction()); key=(ch,al,is_true(ev(resync)),is_true(ev(nested)))
            if key in seen: continue
            seen.add(key)
            disp=[(f+f"(α={al})" if f=="VINE" else f) for f in ch]
            fe=[x for x,b in [("resync",ev(resync)),("nested",ev(nested))] if is_true(b)]
            print(f"  PSNR≥{q}: {' + '.join(disp)}"+(f" +[{','.join(fe)}]" if fe else "")+f"  → PSNR≈{pv:.1f}dB · {len(ch)}frag · {float(ev(time).as_fraction()):.0f}ms")
        print("\nV2_DONE"); sys.exit(0)
    if a.attacks:
        opt,*rest=build(a.min_psnr,a.max_ms,a.attacks,a.min_ba,not a.no_resync,not a.no_nested,a.min_bits)
        opt.minimize(rest[4]); opt.maximize(rest[5]); opt.minimize(rest[6])
        print(f"\nQUERY custom: PSNR>={a.min_psnr} ms<={a.max_ms} ba>={a.min_ba} bits>={a.min_bits} {a.attacks}")
        res=report("custom",opt,rest[0],rest[1],rest[2],rest[3],rest[5],rest[6],a.attacks,a.min_ba)
        unm=[x for x in a.attacks if x in CAP_UNMEASURED]
        if res:
            ab=achievable_bits(res[0],a.attacks,res[2],res[3])   # res[2]=resync, res[3]=nested
            if ab is not None: print(f"    robust-ID capacity ≈ {ab:.0f} bit (min over in-scope attacks; front-ends applied)"+(f"   [unmeasured: {unm}]" if unm else ""))
        if a.min_bits>0 and unm: print(f"    ⚠ capacity for {unm} is UNMEASURED — those attacks were skipped in the --min_bits check")
    else:
        DEMO=[("ID photo (signal, identity .90, HQ .90 quality)",36,5000,SIG,0.90),
              ("Social (signal+geo, identity .90)",34,5000,SIG+GEO,0.90),
              ("Web/AI (+regen, identity .90)",32,6000,SIG+GEO+REG,0.90),
              ("Web/AI (+regen, PRESENCE .63)",34,6000,SIG+GEO+REG,0.63),
              ("Highest-quality (signal only, presence, maximize PSNR)",44,5000,SIG,0.63),
              ("Adversarial (+CtrlRegen s0.9 proxy via rinse-hard? use regen)",30,7000,SIG+GEO+REG,0.90)]
        for lab,mp,mm,at,mb in DEMO:
            opt,*rest=build(mp,mm,at,mb,True,True)
            opt.minimize(rest[4]); opt.maximize(rest[5]); opt.minimize(rest[6])
            print(f"\nQUERY: {lab}\n  need PSNR>={mp} ms<={mm} ba>={mb}")
            report(lab,opt,rest[0],rest[1],rest[2],rest[3],rest[5],rest[6],at,mb)
    print("\nV2_DONE")
