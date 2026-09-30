"""M2 solver-level evaluation against the CONTINUOUS solver and the measured canonical surrogate.

Mirrors solver_eval.py's scenario distribution (same grid, weights and seed) so the two are
comparable, but every configuration -- the solver's and every fixed baseline's -- is scored through
the measured surrogate, with strengths continuous for the solver and pinned for the baselines.

Two things differ from the discrete harness, both forced by the model rather than chosen:
  * There is no brute-force oracle. A strength is a real number, so "enumerate every configuration"
    does not exist; the enumeration comparison lives in the necessity experiment instead.
  * Fidelity comes back as a distortion in MSE and is converted to dB here, since the surrogate
    objective is -D and only the conversion makes it comparable to a dB floor.

Usage: python solver_eval_continuous.py <shard_idx> <n_shards> [N=2000]
Writes one shard of per-scenario records; merge with merge_solver_eval.py.
"""
import sys, os, json, math, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT as CF, WORKSPACE as SC, require
sys.path.insert(0, f"{CF}/solver")
import z3
from surrogate_model import Surrogate
import watermark_smt_v2 as W

SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NSHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

sg = Surrogate.from_dict(json.load(open(require(f"{SC}/surrogate_canonical.json", "the measured database",
    "pipeline/make_canonical_surrogate.py writes it; see Measuring your own fragments in the README",
    "TAILOR_WORKSPACE"))))
MEASURED = set(sg.attacks)

def db(D):
    return 10.0 * math.log10(255.0 ** 2 / max(D, 1e-12))

# ---- scenario space: identical to solver_eval.py (same sets, weights, seed) ----
SIG = ["jpeg25", "blur", "noise", "bright", "contrast"]
# The threat model bounds cropping at keeping 75% of the frame. This is a DECLARED scope, not a
# limit we measured: the matrix shows crop50 is defensible with a geometric front-end (RivaGAN reaches
# 0.929 and our geo configuration 1.000 there), so nothing here justifies calling a deeper crop
# unrecoverable. Earlier revisions named "crop90", which was never measured and was silently dropped at
# run time; it is removed rather than replaced by a deeper crop. The flip and resample columns are
# included now that the surrogate measures them.
GEO = ["crop75", "rot9", "rs256", "hflip"]
ATTACK_SETS = {
    "signal": SIG, "geo-crop": SIG + ["crop75"], "geo-rot": SIG + ["rot9"],
    "geo": SIG + GEO, "vae": SIG + ["vaeB", "vaeC"], "regen": SIG + ["regen"],
    "geo+vae": SIG + GEO + ["vaeB", "vaeC"],
    "geo+regen": SIG + GEO + ["regen"], "vae+regen": SIG + ["vaeB", "vaeC", "regen"],
    "signal+regen": SIG + ["regen"],
    "geo+vae+regen": SIG + GEO + ["vaeB", "vaeC", "regen"],
    "geo+rinse": SIG + GEO + ["regen", "rinse2x"],
    "+rinse": SIG + ["vaeB", "vaeC", "regen", "rinse2x"],
    "full": SIG + GEO + ["vaeB", "vaeC", "regen", "rinse2x"],
}
ASET_W = {"signal":3,"geo-crop":3,"geo-rot":2,"geo":3,"vae":2,"regen":3,"geo+vae":2,"geo+regen":2,
          "vae+regen":2,"signal+regen":2,"geo+vae+regen":2,"geo+rinse":1,"+rinse":1,"full":1}

random.seed(0)
akeys, aw = list(ASET_W), list(ASET_W.values())
SCEN = []
for _ in range(N):
    aset = random.choices(akeys, weights=aw, k=1)[0]
    SCEN.append(dict(min_psnr=random.choice([30.0,31.0,32.0,33.0,34.0,35.0]),
                     max_ms=random.choice([1000.0,2000.0,4000.0,8000.0]),
                     attacks=ATTACK_SETS[aset],
                     # beta is not sampled: the user states a false-positive budget and the
                     # required bit accuracy follows from it (W.beta_from_fpr), so tightening the
                     # security requirement propagates into feasibility instead of being a free knob.
                     min_ba=W.beta_from_fpr(random.choices(
                         [1e-1, 1e-2, 1e-4, 1e-6, 1e-9, 2.0**-37],
                         weights=[1, 3, 2, 2, 1, 2])[0]),
                     allow_resync=True, allow_nested=True,
                     min_bits=random.choices([0,10,20,37,50], weights=[3,2,2,2,1])[0],
                     _aset=aset))

# The grid's "rinse" is the repeated application of the mild regeneration; it is measured as
# rinse2x (and rinse4x), so the sets name the measured column directly.
# crop90 was never swept, so it is dropped from the request the surrogate is asked about. It is a
# milder crop than crop75, which is present in every set that contains it, so the binding crop
# constraint is unchanged; the drop is recorded rather than left implicit.
DROPPED = sorted({a for sc in SCEN for a in sc["attacks"] if a not in MEASURED})

# ---- fixed baselines: a subset, front-ends and strengths all pinned (no solving over them) ----
FIXED = {
    "VINE-only@1.0":        ({"VINE": 1.0}, False, False),
    "TrustMark-only":       ({"TrustMark": 1.0}, False, False),
    "VINE+TM@.7":           ({"VINE": 0.7, "TrustMark": 1.0}, False, False),
    # The always-full point: every fragment plus the most general front-end. It used to pin resync AND
    # the nested ring together, which the solver's at-most-one-front-end constraint now rejects outright
    # (a pair of embed-changing front-ends is an embed nobody measured), so that baseline satisfied
    # nothing by construction rather than by measurement.
    "3frag@.7+resync":      ({"VINE": 0.7, "TrustMark": 1.0, "VideoSeal": 1.0}, True, False),
}

def _build(sc, order=True, prune=True):
    att = [a for a in sc["attacks"] if a in MEASURED]
    return W.build(min_psnr=sc["min_psnr"], max_ms=sc["max_ms"], attacks=att, min_ba=sc["min_ba"],
                   allow_resync=sc["allow_resync"], allow_nested=sc["allow_nested"],
                   min_bits=sc["min_bits"], resolution=512,
                   enable_order=order, continuous_strength=True, surrogate=sg, fe_gain_min=(0.05 if prune else -1.0))

def solve_free(sc):
    """Certified optimum, not just whatever Optimize.maximize happens to return: on this encoding it
    is incomplete and non-deterministic, so an uncertified call reports a configuration that is merely
    feasible and understates the delivered fidelity."""
    att = [a for a in sc["attacks"] if a in MEASURED]
    scen = dict(min_psnr=sc["min_psnr"], max_ms=sc["max_ms"], attacks=att, min_ba=sc["min_ba"],
                allow_resync=sc["allow_resync"], allow_nested=sc["allow_nested"],
                min_bits=sc["min_bits"], resolution=512)
    # Embed order is part of the configuration the method returns (Sec. Method), so the evaluation
    # that reports what the solver delivers has to optimise over it too. Leaving it off here reported
    # a solver that never used a decision variable the paper credits it with.
    built, m, _rounds, certified = W.solve_exact_model(
        scen, enable_order=True, continuous_strength=True, surrogate=sg)
    if built is None: return None
    assert certified, "optimum not certified within the round budget"
    o, u, rs, ns, al, nf, ps, tm = built
    D = -float(m.eval(ps).as_fraction())
    frags = [f for f in W.FR if str(m.eval(u[f])) == "True"]
    # The full configuration, not only its shape: live certification of EVERY request (user 2026-09-06)
    # measures each distinct (order, strengths, front-end) once and lets every request that received it
    # read the same cells, so the record has to carry what was returned, strengths included.
    pv = getattr(o, "_pvars", {})
    def _before(x, y):
        for k, v in pv.items():
            if k == (x, y): return str(m.eval(v)) == "True"
            if k == (y, x): return str(m.eval(v)) != "True"
        return None
    order = sorted(frags, key=lambda f: sum(1 for g in frags if g != f and _before(g, f) is True))
    return {"psnr_db": db(D), "frags": frags, "order": order,
            "s": {f: float(m.eval(o._svars[f]).as_fraction()) for f in frags},
            "fe": W.frontend_decisions(o, m, rs, ns)}

def solve_fixed(sc, strengths, rv, nv):
    # A pinned baseline is a question about a fixed operating point, not a search: the responsible-column
    # pruning (a stage is fixed off where no requested column measurably benefits from it) is a device
    # for the solver's search and must not apply here. Left on, it declared the always-full baseline
    # with the SyncSeal mark infeasible on every request without geometry, because the pinned resync
    # contradicted the pruning clause, and the baseline's satisfaction fell from 0.75 to 0.35 for a
    # reason that had nothing to do with coverage.
    o, u, rs, ns, al, nf, ps, tm = _build(sc, prune=False)
    for f in W.FR: o.add(u[f] == (f in strengths))
    o.add(rs == rv); o.add(ns == nv)
    for f, v in strengths.items(): o.add(o._svars[f] == v)
    o.maximize(ps)
    if o.check() != z3.sat: return None
    return {"psnr_db": db(-float(o.model().eval(ps).as_fraction()))}

def main():
    """Solve this shard and write it. Guarded because certify_solver_eval.py and
    solve_live_continuous.py import this module for SCEN/MEASURED/FIXED: unguarded, the import
    re-solved all 2000 scenarios (2.7 h of CPU inside a GPU job) and then overwrote
    solver_eval_cont_shard00.json with a 1-shard, 2000-record file on top of the 20-shard array's
    100-record shard 0."""
    lo = (N * SHARD) // NSHARD; hi = (N * (SHARD + 1)) // NSHARD
    print(f"shard {SHARD}/{NSHARD}: scenarios [{lo},{hi})  dropped-unmeasured={DROPPED}", flush=True)
    recs = []; t0 = time.time()
    for i in range(lo, hi):
        sc = SCEN[i]
        r = {"i": i, "aset": sc["_aset"], "min_ba": sc["min_ba"], "min_bits": sc["min_bits"],
             "min_psnr": sc["min_psnr"], "max_ms": sc["max_ms"]}
        s = solve_free(sc)
        r["solver"] = s
        r["fixed"] = {k: solve_fixed(sc, st, rv, nv) for k, (st, rv, nv) in FIXED.items()}
        recs.append(r)
        if (i - lo + 1) % 5 == 0 or i == hi - 1:
            print(f"  {i-lo+1}/{hi-lo} [{time.time()-t0:.0f}s]", flush=True)

    out = f"{SC}/solver_eval_cont_shard{SHARD:02d}.json"
    json.dump({"shard": SHARD, "n_shards": NSHARD, "N": N, "range": [lo, hi],
               "dropped_unmeasured_attacks": DROPPED,
               "fixed_baselines": {k: {"strengths": st, "resync": rv, "nested": nv}
                                   for k, (st, rv, nv) in FIXED.items()},
               "records": recs}, open(out, "w"), indent=1)
    print(f"wrote {out}  ({len(recs)} scenarios, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
