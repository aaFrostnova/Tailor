"""Per-attack image-to-image SD of the bit accuracy, measured on the certification runs.

The live gate (W.live_required_at) asks whether the table's point estimate is far enough from the
threshold that the specific image cannot land on the other side. That question needs the IMAGE-TO-IMAGE
sd, not the standard error of the mean; the certification stored `live_se` at a known n, so sd = se*sqrt(n).
variance_profile.json covered 14 classical columns at one operating point; this covers every column the
classes name, including the diffusion and adversarial ones, pooled over the certified configurations.
"""
import json, glob, math, collections, statistics as st
SC = "/data/tailor/workspace/wm_dataset10k"
acc = collections.defaultdict(list)
for K in ["C1", "C2", "C3", "C4", "C5"]:
    R = json.load(open(f"{SC}/certify_classes_{K}.json"))
    n_default = R["n_img"]
    for r in R["rows"]:
        if r["verdict"] != "SAT": continue
        se = r.get("live_se") or {}; ln = r.get("live_n") or {}
        for a, s in se.items():
            n = ln.get(a) or n_default
            if s is None or n < 2: continue
            acc[a].append(float(s) * math.sqrt(n))
out = {a: {"sd": round(st.median(v), 4), "sd_max": round(max(v), 4), "cells": len(v)} for a, v in sorted(acc.items())}
# the ring campaign measured the adversarial columns at nine strengths on the embed the stage produces;
# its low-strength knots carry the widest spread, which is exactly where the gate has to fire
ring = f"{SC}/unmarker9_ring"
for p in sorted(glob.glob(f"{ring}/knot_*/knot.json")):
    d = json.load(open(p))
    for a, c in d["cols"].items():
        k = f"{a}@scale"
        out.setdefault(k, {"sd": 0.0, "sd_max": 0.0, "cells": 0})
        out[k]["sd"] = max(out[k]["sd"], 0.0)
        out[k]["sd_max"] = round(max(out[k]["sd_max"], c["sd"]), 4)
        out[k]["cells"] += 1
for a in list(out):
    if a.endswith("@scale"):
        sds = [json.load(open(p))["cols"][a.split("@")[0]]["sd"] for p in sorted(glob.glob(f"{ring}/knot_*/knot.json"))]
        out[a]["sd"] = round(st.median(sds), 4)
json.dump(out, open(f"{SC}/ba_sd_profile.json", "w"), indent=1)
print(f"wrote {SC}/ba_sd_profile.json")
for a, v in out.items():
    print(f"  {a:20s} sd {v['sd']:.4f}  max {v['sd_max']:.4f}  ({v['cells']} cells)")
