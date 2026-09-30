"""Merge every measured surrogate piece into ONE canonical table the solver and the live layer read.

Inputs (all measured on pool E, N=100 for the in-process family and N=50 for the diffusion family):
  surrogate_table_ext.json              base/delta/d/e for the 10 in-process attacks, plus the
                                        strength-dependent capacity curves and the front-end curves
  surrogate_regen_overlay_ext.json      regen (6/6 ordered-pair deltas) and rinse (2/6, see below)
  surrogate_ctrlregen_s03_overlay.json  ctrlregen at diffusion step 0.3
  surrogate_ctrlregen_s07_overlay.json  ctrlregen at diffusion step 0.7

Any ordered pair an overlay did not measure is filled with an explicit ZERO interference curve and
listed under "zero_filled_delta" -- the solver needs a value for every (g,f,a) it can query, and
"no measured interference" is the honest default, but it must be visible rather than silent.
"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.paths import PROJECT as CF, WORKSPACE as SC
sys.path.insert(0, f"{CF}/solver")
from surrogate_model import Surrogate

# Order matters: later overlays override earlier ones key-by-key. The geometry overlay is LAST
# because it deliberately replaces crop75/crop50/rot9 -- the in-process campaign had re-implemented
# those attacks locally and its crop read the fraction as an area rather than a side fraction, while
# its rotation left black corners instead of reflecting. The overlay imports the reported matrix's
# own operators, so after this merge the solver and the matrix name the same attacks.
OVERLAYS = ["surrogate_mildregen_overlay_n100.json",
            "surrogate_regen_overlay_ext.json",
            "surrogate_ctrlregen_s03_overlay.json",
            "surrogate_ctrlregen_s07_overlay.json",
            "surrogate_geo_overlay.json",
            # front-end cells LAST: they are measured by driving the DEPLOYED cascade (SyncSeal
            # rectify, residual-angle retry, nested scale search, and the fragment-independent blind
            # angle sweep) against a same-run control with the cascade suppressed, so the stored value
            # is the cascade's own contribution rather than a re-implementation of one of its stages.
            "surrogate_frontend_overlay.json",
            "surrogate_signal_overlay.json"]
# The per-component cells: one file per (front-end stage, attack), each holding that stage's own
# effect against a same-run control. They come after the bundled overlay above so that where both
# describe a cell, the separated measurement wins -- the bundle credits whichever flag was set with
# what all four stages did together.
import glob as _glob
OVERLAYS = OVERLAYS + [os.path.basename(q) for q in
                       sorted(_glob.glob(f"{SC}/surrogate_fe_components_*of28.json"))]
# The same stages on the non-geometric columns (FE_ATTACKS run of make_frontend_components.py): an
# embed-changing stage needs a replacement curve on every column a request can name, or the solver
# would read the plain curve for an embed it never measured.
OVERLAYS = OVERLAYS + [os.path.basename(q) for q in
                       sorted(_glob.glob(f"{SC}/surrogate_fe_components_nongeo_*of30.json"))]
# Two-fragment front-end curves (host swept with a partner embedded after it, stage on) and the
# interference curves measured over the overwriting fragment's strength; both override the
# single-point/single-fragment approximations of the same cells.
OVERLAYS = OVERLAYS + [os.path.basename(q) for q in sorted(_glob.glob(f"{SC}/surrogate_fe_pairs_*of24.json"))]
# The range-extension shards of the same campaign (surrogate_delta_curves_ext_*of22.json) hold only the
# new knots. They reach the table through surrogate_range_ext.json, which is SPLICED into the existing
# curves below; taken as overlays here they sort after the full sweeps and replace a nine-knot curve
# with its two-knot extension piece (20 VINE/VideoSeal interference curves lost their measured range).
OVERLAYS = OVERLAYS + [os.path.basename(q) for q in sorted(_glob.glob(f"{SC}/surrogate_delta_curves_*of22.json"))
                       if "_ext_" not in os.path.basename(q)]

# The adversarial column is optional: UnMarker costs ~3.5 GPU-minutes per image, so its nine-knot sweep
# runs at N=30 per knot, and the solver treats the attack as ADVERSARIAL (a request naming it forces a
# live measurement; the curve only bounds the search). It is merged only when the nine-knot sweep has
# been decoded; an older three-knot probe file with the same name is skipped, loudly.
_um = f"{SC}/surrogate_unmarker_overlay.json"
if os.path.exists(_um):
    _o = json.load(open(_um)); _kn = {k: len(v["xs"]) for k, v in _o["base"].items()}
    if all(k == 9 for k in _kn.values()) and int((_o.get("source") or {}).get("n", 0)) >= 30:
        OVERLAYS.append("surrogate_unmarker_overlay.json")
    else:
        print(f"SKIPPING surrogate_unmarker_overlay.json: knots {_kn}, n={(_o.get('source') or {}).get('n')} "
              f"(needs 9 knots and n>=30; decode the unmarker9 sweep first)", flush=True)
# The scale stage's replacement curves on the adversarial columns (UnMarker, CtrlRegen+ x3), measured
# on the nested-ring embed the stage actually decodes (merge_unmk_ring.py); N=30 per knot like the
# adversarial column itself. Without them the stage is inadmissible on any request naming those
# columns, and the solver can only read the bare-decode UnMarker column (0.49 at every strength).
_ring = f"{SC}/surrogate_fe_ring_adv_overlay.json"
if os.path.exists(_ring):
    OVERLAYS.append("surrogate_fe_ring_adv_overlay.json")
MIN_N_PER_OVERLAY = {"surrogate_unmarker_overlay.json": 30, "surrogate_fe_ring_adv_overlay.json": 30}
# The table-completion campaigns (2026-09-07): interference sweeps over the overwriting fragment's
# strength for the pairs and columns the 22-cell campaign did not cover (make_delta_curves.py with
# DELTA_CELLS_FILE, make_delta_xenv.py for CtrlRegen+), and the resync/tile replacement curves with
# per-image cells and latency on the CtrlRegen+ and UnMarker columns (fe_xenv_campaign.py,
# fe_xenv_merge.py). Both carry full-range curves, so they enter as overlays, and they come LAST so a
# mid-strength constant never overrides a sweep. The UnMarker knots hold N=30 like the column itself.
OVERLAYS = OVERLAYS + [os.path.basename(q) for q in sorted(_glob.glob(f"{SC}/surrogate_delta_sweep_*of*.json"))]
_fex = sorted(os.path.basename(q) for q in _glob.glob(f"{SC}/surrogate_fe_xenv_*.json"))
OVERLAYS = OVERLAYS + _fex
MIN_N_PER_OVERLAY.update({q: 30 for q in _fex})

# 9 strength knots rather than 5: a leave-one-out check put the 5-knot interpolation error at several
# times the measurement noise for the cheap attack family, i.e. the curves were sampling-limited.
MIN_N = 50          # smallest image count any campaign cell may contribute
d = json.load(open(f"{SC}/surrogate_table_ext9.json"))
frags = d["fragments"]; ranges = d["ranges"]
merged_from = {"base": "surrogate_table_ext9.json (9 strength knots, N=100)"}
for name in OVERLAYS:
    o = json.load(open(f"{SC}/{name}"))
    added = sorted({k.split("|")[1] for k in o["base"]})
    d["base"].update(o["base"])
    d.setdefault("delta", {}).update(o.get("delta", {}))
    if o.get("cap"):                                   # capacity curves, when the overlay measured them
        d.setdefault("cap", {}).update(o["cap"])
    if o.get("frontend"):                              # front-end effect curves
        nan_keys = [k for k, v in o["frontend"].items() if isinstance(v, dict) and "ys" in v
                    and any(isinstance(y, float) and y != y for y in v["ys"])]
        if nan_keys:
            # a curve with a NaN knot cannot enter the solver (z3 rejects it) and must not shadow a
            # measured curve of the same name from an earlier overlay
            print(f"  {name}: skipping {len(nan_keys)} NaN curve(s): {nan_keys}", flush=True)
        d.setdefault("frontend", {}).update({k: v for k, v in o["frontend"].items() if k not in nan_keys})
    if o.get("latency"):                               # per-stage wall time, one scalar per cell
        d.setdefault("latency", {}).update(o["latency"])
    # A smoke run writes to the same filename as the real cell it smoke-tested. Merging one would
    # put a two-image mean into the table beside hundred-image ones, at the same weight and with
    # nothing downstream able to tell them apart.
    n = (o.get("source") or {}).get("n")
    if isinstance(n, int) and n < MIN_N_PER_OVERLAY.get(name, MIN_N):
        raise ValueError(f"{name} holds N={n}, below the campaign's N={MIN_N}; "
                         f"it is a smoke run, not a measurement. Re-run that cell before merging.")
    new_a = [a for a in added if a not in d["attacks"]]
    over  = [a for a in added if a in d["attacks"]]
    d["attacks"].extend(new_a)                          # never duplicate an attack already present
    for a in added:
        merged_from[a] = f"{name} (N={o['source']['n']})" + (" [OVERRIDES]" if a in over else "")
    print(f"merged {name}: added {new_a}" + (f", overrode {over}" if over else ""), flush=True)

# Columns the request scope does not use are dropped before anything downstream sees them. A column
# that stays in the table is one the solver may be asked about, and every such column needs capacity:
# `rinse4x` and `ctrlregen_s05_x2` are strictly stronger than the hardest column whose reliable-bit
# capacity was measured, so an identity request on them could only be answered by crediting bits nobody
# measured. The robustness measurements themselves are kept in the overlays; this drops them from the
# canonical table so no request can reach them.
DROP_ATTACKS = {"rinse4x", "ctrlregen_s05_x2"}
if DROP_ATTACKS:
    kept = [a for a in d["attacks"] if a not in DROP_ATTACKS]
    dropped = [a for a in d["attacks"] if a in DROP_ATTACKS]
    d["attacks"] = kept
    for blk in ("base", "delta", "d", "e", "cap", "frontend", "latency"):
        if blk not in d: continue
        for k in [k for k in d[blk] if any(f"|{a}" in k or k.endswith(a) for a in DROP_ATTACKS)]:
            d[blk].pop(k)
    print(f"dropped {dropped} from the canonical table (not in any request scope)", flush=True)

# RANGE EXTENSION (merge_range_ext.py): new knots are spliced INTO the existing curves rather than
# replacing them, and the fragments' strength ranges widen to what was measured. Curves the extension
# measured but the table never had are not created from the new knots alone.
_ext = f"{SC}/surrogate_range_ext.json"
if os.path.exists(_ext):
    _e = json.load(open(_ext)); _n = 0
    for blk in ("base", "d", "frontend", "delta"):
        for k, c in _e.get(blk, {}).items():
            if k not in d.get(blk, {}): continue
            cur = d[blk][k]
            if not (isinstance(cur, dict) and "xs" in cur): continue
            pts = {float(x): float(y) for x, y in zip(cur["xs"], cur["ys"])}
            for x, y in zip(c["xs"], c["ys"]):
                pts.setdefault(float(x), float(y))                 # an existing knot keeps its value
            xs = sorted(pts); d[blk][k] = {"xs": xs, "ys": [pts[x] for x in xs]}; _n += 1
    for F, r in (_e.get("ranges") or {}).items():
        if r:
            lo, hi = ranges[F]; ranges[F] = [min(lo, r[0]), max(hi, r[1])]
    d["ranges"] = ranges
    merged_from["range_extension"] = f"surrogate_range_ext.json ({_n} curves extended; ranges {ranges})"
    print(f"range extension: {_n} curves gained knots; ranges now {ranges}", flush=True)

# every (g,f,a) the solver can query must exist
zero_filled = []
for a in d["attacks"]:
    for g in frags:
        for f in frags:
            if g == f: continue
            k = f"{g}|{f}|{a}"
            if k not in d["delta"]:
                lo, hi = ranges[g]
                d["delta"][k] = {"xs": [lo, hi], "ys": [0.0, 0.0]}
                zero_filled.append(k)
print(f"zero-filled {len(zero_filled)} unmeasured delta cells: {zero_filled}", flush=True)

try:
    git_rev = subprocess.run(["git", "-C", CF, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    git_rev = "unknown"
d["source"] = {**d.get("source", {}),
               "canonical": True,
               "merged_from": merged_from,
               "zero_filled_delta": zero_filled,
               "n_attacks": len(d["attacks"]),
               "assembled_at": os.environ.get("TS", "unknown"),
               "git": git_rev,
               "capacity_coverage": "cap curves exist for the 10 in-process attacks only; the solver "
                                    "falls back to the discrete measured constant for the diffusion "
                                    "family, so capacity stays constrained on every attack."}
for blk in ("base", "delta", "d", "e", "cap", "frontend"):
    for k, v in d.get(blk, {}).items():
        if isinstance(v, dict) and "ys" in v:
            assert not any(isinstance(y, float) and y != y for y in v["ys"]), f"NaN in {blk}:{k}"
out = f"{SC}/surrogate_canonical_raw.json"   # the merged measurements, kept auditable
json.dump(d, open(out, "w"), indent=2)

sg = Surrogate.from_dict(json.load(open(out)))
missing = [(g, f, a) for a in sg.attacks for g in frags for f in frags
           if g != f and sg.delta(g, f, a) is None]
assert not missing, f"delta still missing: {missing[:5]}"
print(f"\nwrote {out}")
print(f"  attacks({len(sg.attacks)}): {sg.attacks}")
print(f"  base={len(d['base'])} delta={len(d['delta'])} cap={len(d.get('cap',{}))} frontend={len(d.get('frontend',{}))} latency={len(d.get('latency',{}))}")
print("  loads via Surrogate.from_dict, every (g,f,a) present  OK")

# The solver does not read the raw knots. Interpolating them directly bakes each measurement error
# into the model and spends a case split on every segment; fitting pools the knots and thinning
# spends segments only where a curve bends. Consumers read the result of both.
import subprocess as _sp
_sp.run([sys.executable, f"{SC}/fit_surrogate_curves.py", out, f"{SC}/surrogate_fitted.json"], check=True)
_sp.run([sys.executable, f"{SC}/thin_surrogate_curves.py", f"{SC}/surrogate_fitted.json",
         f"{SC}/surrogate_canonical.json"], check=True)
print(f"  -> fitted + thinned into {SC}/surrogate_canonical.json (what the solver reads)")
# The per-image values behind the curves (merge_perimage.py) ride along as a block of their own. They
# are not curves, so they go in AFTER fitting and thinning, which only know curves; the solver derives
# the acceptance rate at each request's own threshold from them (Surrogate.rate_curve).
_pi = f"{SC}/surrogate_perimage.json"
if os.path.exists(_pi):
    for _f in (out, f"{SC}/surrogate_canonical.json"):
        _d = json.load(open(_f)); _d["perimage"] = json.load(open(_pi)); json.dump(_d, open(_f, "w"), indent=1)
    print(f"  attached {len(json.load(open(_pi)))} per-image cells to the canonical table")
