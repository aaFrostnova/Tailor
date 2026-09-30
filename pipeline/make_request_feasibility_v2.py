"""What the offline table says each column can reach, per false-positive budget AND payload: the request
sampler's ceilings.

For every column, budget (threshold tau(k) for k = 1 or 2 selected fragments) and payload (0, 10, 20, 37,
50 bits), the cheapest single fragment, with any admissible front-end stage, whose MEAN bit accuracy
clears the request's line, max(tau + margin, bits_to_ba(bits)), AND whose per-image acceptance rate at
tau reaches det_min (the three conditions the solver's coverage clause, capacity clause and acceptance
floor impose), with the strength that first does so and the fragment's solo PSNR there. A column with no
such fragment at a budget is beyond the table's reach at that budget for every configuration: a request
naming it there is infeasible by the fragments' limits, not by composition.
Also the capacity ceiling of every column (reliable bits at its best mean, W.ba_to_bits) and the columns
that VINE cannot carry even with the ring (they force a second fragment, k = 2).

v2 (2026-09-08): the payload dimension (the C5 run drew 35-38 dB floors for 50-bit requests whose capacity
line 0.89 needs VINE 0.935 on rinse2x, 35.4 dB solo: 125 requests infeasible by the table, labelled
"within ceiling"), and the live-only column's reference is the ring (VINE + scale, the deployed default),
never the plain TrustMark/VideoSeal prior cells (n=30, chance-level): with those as reference the sampler
gave UnMarker requests no scale-search latency and a one-fragment fidelity allowance (3 C4 requests).
Keys: need[column|k<k>|<budget index>] (payload 0, as before) and need[column|k<k>|<budget index>|b<bits>].
Writes request_feasibility_matrix.json, read by class_defs.py. Regenerate after the table changes.
"""
import json, sys, os, math, subprocess
import numpy as np
CF = "/data/tailor/project"
SC = "/data/tailor/workspace/wm_dataset10k"
sys.path.insert(0, f"{CF}/scripts/defense")
import watermark_smt_v2 as W
sg = json.load(open(f"{SC}/surrogate_canonical.json")); base, fe, pi, D = sg["base"], sg["frontend"], sg["perimage"], sg["d"]
FR = ["VINE", "TrustMark", "VideoSeal"]; A = sg["attacks"]
BUDGETS = [1e-1, 1e-2, 1e-4, 1e-6, 1e-9, 2.0 ** -37]
BITS = [0, 10, 20, 37, 50]
MARGIN, DET_MIN = W.DEFAULT_MARGIN, W.DEFAULT_DET_MIN
LIVE_ONLY = {"unmarker"}                      # settled live; the table holds a search prior, the ring is the reference
def ev(c, x): return float(np.interp(x, c["xs"], c["ys"]))
# A stage is an embed of its own and costs fidelity: the SyncSeal mark (penalty_fe_resync, ~2.6 MSE flat)
# whenever resync is on, the nested ring (nested_penalty, indexed by VINE's strength) whenever the ring is,
# the tiled grid (penalty_fe_tile) with TrustMark. The solver charges all of them in its distortion sum, so a
# reference PSNR that omits them promises a fidelity the configuration cannot deliver (2.6 dB on the resync
# cells, 2.8 to 3.8 dB on the ring cells); the sampler then draws floors the solver cannot meet.
STAGE_PENALTY = {"resync": "penalty_fe_resync", "tile": "penalty_fe_tile", "scale": "nested_penalty", "angle": None}
def stage_mse(stage, s):
    k = STAGE_PENALTY.get(stage)
    c = fe.get(k) if k else None
    return max(ev(c, s), 0.0) if c else 0.0
def psnr(f, s, stage="plain"): return 10 * math.log10(255.0 ** 2 / (ev(D[f], s) + stage_mse(stage, s)))
def rate_at(rec, s, tau):
    r = [float(np.mean((np.asarray(b) >= tau - 1e-12) | (np.asarray(v) if v is not None else False)))
         for b, v in zip(rec["ba"], rec["ver"] or [None] * len(rec["xs"]))]
    return float(np.interp(s, rec["xs"], r))
LAT = sg.get("latency", {})
def cells(f, a):
    """the plain cell and the stage cells of (f, a) that the solver can actually use for this column: a stage
    is selectable only where it is responsible for some column (its replacement curve gains >= 0.05 over the
    plain one there, watermark_smt_v2 fe_gain_min), so for the ceiling of a column only stages responsible on
    THAT column count; a stage enabled for another column may still apply here, but a ceiling that rests on
    that is one noisy knot away from being wrong (CtrlRegen+ at step 0.7 sat exactly on 0.90 that way).
    A stage cell without per-image values reads the plain ones. Each cell carries the latency it costs.
    On a live-only column only the ring (VINE under scale) is a reference cell."""
    plain = base[f"{f}|{a}"]
    out = [] if a in LIVE_ONLY else [("plain", plain, pi.get(f"{f}|{a}"), 0.0)]
    for st in ("resync", "scale", "angle", "tile"):
        if a in LIVE_ONLY and not (f == "VINE" and st == "scale"): continue
        c = fe.get(f"base_fe_{st}_{f}|{a}")
        if c is None: continue
        grid = sorted(set(c["xs"]) | set(plain["xs"]))
        if max(ev(c, x) - ev(plain, x) for x in grid) < 0.05: continue
        out.append((st, c, pi.get(f"{st}:{f}|{a}") or pi.get(f"{f}|{a}"), float(LAT.get(f"latency_ms_{st}|{a}", 0.0))))
    return out
need = {}
for a in A:
    for k in (1, 2):
        for bi, b in enumerate(BUDGETS):
            tau = W.presence_threshold(W.beta_from_fpr(b), k)
            for bits in BITS:
                line = max(tau + MARGIN, W.bits_to_ba(bits) if bits > 0 else 0.0); cands = []
                for f in FR:
                    lo, hi = sg["ranges"][f]; grid = np.linspace(lo, hi, 400)
                    for st, c, rec, lat in cells(f, a):
                        ok = [s for s in grid if ev(c, s) >= line and (rec is None or rate_at(rec, s, tau) >= DET_MIN)]
                        if ok: cands.append((round(psnr(f, min(ok), st), 2), f, st, round(float(min(ok)), 3), round(lat, 1)))
                entry = list(max(cands)) if cands else None      # [solo PSNR, fragment, stage, strength, stage latency ms]
                need[f"{a}|k{k}|{bi}|b{bits}"] = entry
                if bits == 0: need[f"{a}|k{k}|{bi}"] = entry
capbits = {a: round(W.ba_to_bits(max(max(c["ys"]) for f in FR for _, c, _, _ in cells(f, a))), 1) for a in A}
# columns VINE cannot carry with or without the ring but another fragment can: a request naming one needs
# a second fragment for it, and the solver then holds every column to the two-fragment threshold. A column
# no fragment carries (CtrlRegen+ at step 0.7) is not one of them: VINE is still the best there, k stays 1.
def _best(f, a): return max([max(c["ys"]) for _, c, _, _ in cells(f, a)] or [0.0])
needs2 = [a for a in A if _best("VINE", a) < 0.7 and max(_best("TrustMark", a), _best("VideoSeal", a)) >= 0.7]
try: git = subprocess.run(["git", "-C", CF, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
except Exception: git = "unknown"
out = {"budgets": BUDGETS, "bits": BITS, "tau": {f"k{k}": [round(W.presence_threshold(W.beta_from_fpr(b), k), 3) for b in BUDGETS] for k in (1, 2, 3)},
       "capacity_line": {f"b{b}": (round(W.bits_to_ba(b), 4) if b > 0 else 0.0) for b in BITS},
       "margin": MARGIN, "det_min": DET_MIN, "live_only": sorted(LIVE_ONLY), "need": need, "capacity_bits": capbits, "needs_second_fragment": needs2,
       "stage_latency_ms": {k: v for k, v in LAT.items()}, "fragment_ms": {f: W.cost_ms(f) for f in FR},
       "source": {"table": "surrogate_canonical.json", "table_measured_at": (sg.get("source") or {}).get("measured_at"), "git": git, "version": 2, "stage_penalty_charged": True,
                  "note": "need[column|k<k>|<budget index>|b<bits>] = [solo PSNR dB, fragment, stage, strength, stage latency ms] of the cheapest fragment whose "
                          "mean clears max(tau+margin, bits_to_ba(bits)) with per-image rate >= det_min at tau; null = beyond every fragment there; "
                          "need[column|k<k>|<budget index>] = the payload-0 entry; on the live-only column only the ring is a reference cell"}}
OUT = os.environ.get("FEAS_OUT") or f"{SC}/request_feasibility_matrix.v2_payload.json"
json.dump(out, open(OUT, "w"), indent=1)
print("tightest reachable budget per column (k=1 / k=2), capacity bits:")
for a in A:
    c1 = max([bi for bi in range(6) if need[f"{a}|k1|{bi}"]] or [-1]); c2 = max([bi for bi in range(6) if need[f"{a}|k2|{bi}"]] or [-1])
    print(f"  {a:14s} {BUDGETS[c1] if c1 >= 0 else 'none':>8} / {BUDGETS[c2] if c2 >= 0 else 'none':>8}   {capbits[a]:5.1f} bits" + ("   needs a second fragment" if a in needs2 else ""))
print(f"wrote {OUT}")
