"""Per-class solver-level evaluation: five threat classes, 2,000 sampled requests each.

The classes are the five deployment scenarios (C1 signal, C2 +geometry, C3 +AI
regeneration, C4 +adversarial removal, C5 broad+identity). A class fixes the attacks a deployment must
survive; a request within the class samples what the deployment can afford: the false-positive
budget (which sets the bit-accuracy target), the fidelity floor, the latency budget, the payload, and
which of the class's optional attacks are also in scope. Every request is solved to the certified
optimum and posed to the four fixed baselines (subset, strengths and front-ends pinned, pruning off).
C4 carries ONE sampled CtrlRegen+ strength (0.3/0.5/0.7 at 0.4/0.4/0.2) and UnMarker on a quarter of the requests.
Usage: python class_scenarios.py <class 0-4> <shard> <n_shards> [N=2000]
"""
import sys, os, json, random, time
SC = "/data/tailor/workspace/wm_dataset10k"
CF = "/data/tailor/project"
for p in (f"{CF}/scripts/defense", SC): sys.path.insert(0, p)
CLS = int(sys.argv[1]); SHARD = int(sys.argv[2]); NSHARD = int(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
sys.argv = [sys.argv[0]]                         # solver_eval_continuous reads argv at import
import watermark_smt_v2 as W
import solver_eval_continuous as SEC             # sg, MEASURED, FIXED, solve_free, solve_fixed, db
sg = SEC.sg
from class_defs import classes, sample_class
CLASSES = classes(sg.attacks)
C = CLASSES[CLS]
missing = [a for a in C["core"] + C["optional"] if a not in sg.attacks]; assert not missing, (C["key"], missing)
SCEN = sample_class(C, CLS, N, W.beta_from_fpr)
lo = (N * SHARD) // NSHARD; hi = (N * (SHARD + 1)) // NSHARD
# Two arrays may race on the same shard (cpu partition + gpu partition, user 2026-09-08): a shard whose
# complete output already exists is skipped, so a duplicate task costs nothing once the other has finished.
_out = f"{SC}/class_eval_{C['key']}_shard{SHARD:02d}.json"
if os.path.exists(_out):
    try:
        _d = json.load(open(_out))
        if _d.get("n_shards") == NSHARD and len(_d.get("records", [])) == hi - lo:
            print(f"shard {SHARD}/{NSHARD} of {C['key']} already complete ({_out}); nothing to do", flush=True); sys.exit(0)
    except Exception:
        pass
print(f"class {C['key']} {C['name']}: core={C['core']} optional={C['optional']}; shard {SHARD}/{NSHARD} -> [{lo},{hi})", flush=True)
recs = []; t0 = time.time()
for i in range(lo, hi):
    sc = SCEN[i]
    r = {"i": i, "cls": C["key"], "aset": C["key"], "attacks": sc["attacks"], "optional": sc["_optional"], "fpr": sc["fpr"],
         "min_ba": sc["min_ba"], "min_bits": sc["min_bits"], "min_psnr": sc["min_psnr"], "max_ms": sc["max_ms"], "cr": sc.get("_cr"), "unm": bool(sc.get("_unm")),
         "stress": bool(sc.get("_stress")), "stress_kind": sc.get("_stress_kind"), "k": sc.get("_k"), "ceiling_fpr": sc.get("_ceiling_fpr"), "hardest": sc.get("_hardest")}
    r["solver"] = SEC.solve_free(sc)
    r["fixed"] = {k: SEC.solve_fixed(sc, st, rv, nv) for k, (st, rv, nv) in SEC.FIXED.items()}
    recs.append(r)
    if (i - lo + 1) % 10 == 0:
        print(f"  {i-lo+1}/{hi-lo}  sat={sum(1 for x in recs if x['solver'])}/{len(recs)} [{time.time()-t0:.0f}s]", flush=True)
out = f"{SC}/class_eval_{C['key']}_shard{SHARD:02d}.json"
_tmp = out + ".tmp"          # written whole, then renamed: a half-written file is never taken as complete
json.dump({"cls": C["key"], "name": C["name"], "core": C["core"], "optional": C["optional"], "shard": SHARD, "n_shards": NSHARD, "N": N,
           "range": [lo, hi], "table_assembled_at": json.load(open(f"{SC}/surrogate_canonical.json"))["source"].get("assembled_at"),
           "margin": W.DEFAULT_MARGIN, "records": recs}, open(_tmp, "w"), indent=1)
os.replace(_tmp, out)
print(f"wrote {out}  CLASS_SHARD_DONE", flush=True)
