"""Replace raw piecewise-linear interpolation of the measured knots with a fitted curve.

Interpolating the knots directly makes every measurement error a permanent feature of the model:
the curve is forced through each noisy point. Fitting instead pools the knots, so a curve is
estimated from all of its measurements at once and the noise partly averages out.

Rather than assume a functional form, each curve is fitted with several candidates and the one that
predicts HELD-OUT knots best is kept (leave-one-out over the knots). Straight interpolation is one
of the candidates, so a curve whose shape no model captures keeps the behaviour it has today; the
selection can only match or beat it.

Candidates, all monotone-aware and all cheap to evaluate:
  pwl        the current piecewise-linear interpolation (baseline)
  isotonic   PWL after projecting the knots onto the monotone direction the curve already trends in
  linear     a straight line (maximum denoising for the many near-straight curves)
  power      a + b*s^p  (distortion against strength behaves this way: scaling a residual by s
             scales its squared error by s^2)
  logistic   lo + (hi-lo)/(1+exp(-k(s-s0)))  (a bit-accuracy saturating between chance and perfect)

Usage: python fit_surrogate_curves.py [in.json] [out.json]
"""
import sys, json, math
import numpy as np

IN = sys.argv[1] if len(sys.argv) > 1 else "/data/tailor/workspace/wm_dataset10k/surrogate_canonical.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/data/tailor/workspace/wm_dataset10k/surrogate_fitted.json"

def _iso(ys, inc):
    """Pool-adjacent-violators: nearest monotone sequence in least squares."""
    y = np.array(ys, float) if inc else -np.array(ys, float)
    n = len(y); v = y.copy(); w = np.ones(n)
    i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1] + 1e-12: i += 1; continue
        nv = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
        nw = w[i] + w[i + 1]
        v = np.concatenate([v[:i], [nv], v[i + 2:]]); w = np.concatenate([w[:i], [nw], w[i + 2:]])
        i = max(i - 1, 0)
    out = np.repeat(v, w.astype(int))
    return list(out if inc else -out)

def _fit_linear(x, y):
    b, a = np.polyfit(x, y, 1); return lambda t: a + b * np.asarray(t, float)

def _fit_power(x, y):
    best = None
    for p in np.linspace(0.5, 3.0, 26):          # exponent on a small grid; a,b closed-form given p
        X = np.vstack([np.ones_like(x), x ** p]).T
        try: c, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception: continue
        r = float(np.sum((X @ c - y) ** 2))
        if best is None or r < best[0]: best = (r, c, p)
    if best is None: return None
    _, c, p = best
    return lambda t: c[0] + c[1] * np.asarray(t, float) ** p

def _fit_logistic(x, y):
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi - lo < 1e-6: return lambda t: np.full_like(np.asarray(t, float), lo)
    span = hi - lo; best = None
    for k in np.linspace(1.0, 30.0, 30):
        for s0 in np.linspace(float(x[0]), float(x[-1]), 21):
            f = lo + span / (1.0 + np.exp(-k * (x - s0)))
            r = float(np.sum((f - y) ** 2))
            if best is None or r < best[0]: best = (r, k, s0)
    _, k, s0 = best
    return lambda t: lo + span / (1.0 + np.exp(-k * (np.asarray(t, float) - s0)))

def candidates(x, y, inc):
    c = {"pwl": (lambda t: np.interp(t, x, y)),
         "isotonic": (lambda t, yy=_iso(y, inc): np.interp(t, x, yy))}
    if len(x) >= 2: c["linear"] = _fit_linear(x, y)
    if len(x) >= 3:
        p = _fit_power(x, y)
        if p is not None: c["power"] = p
        c["logistic"] = _fit_logistic(x, y)
    return c

def loo_error(x, y, name, inc):
    """Mean |held-out knot - prediction from the other knots| for one candidate."""
    errs = []
    for j in range(len(x)):
        xt = np.delete(x, j); yt = np.delete(y, j)
        if len(xt) < 2: continue
        try: f = candidates(xt, yt, inc)[name]
        except KeyError: return None
        errs.append(abs(float(f(x[j])) - y[j]))
    return float(np.mean(errs)) if errs else None

d = json.load(open(IN))
EMIT = 33            # dense re-sampling of the fitted curve; step 3 thins this adaptively
report = {}; picked = {}
for blk in ("base", "delta", "d", "e", "cap"):
    if blk not in d: continue
    for key, v in d[blk].items():
        x = np.array(v["xs"], float); y = np.array(v["ys"], float)
        if len(x) < 3:                       # 2-knot flat cells carry no shape to fit
            picked[f"{blk}|{key}"] = "pwl"; continue
        inc = bool(y[-1] >= y[0])
        scores = {}
        for name in candidates(x, y, inc):
            e = loo_error(x, y, name, inc)
            if e is not None: scores[name] = e
        best = min(scores, key=scores.get)
        report[f"{blk}|{key}"] = {"scores": {k: round(s, 5) for k, s in scores.items()}, "picked": best,
                                  "pwl": round(scores.get("pwl", float("nan")), 5)}
        picked[f"{blk}|{key}"] = best
        if best != "pwl":
            f = candidates(x, y, inc)[best]
            nx = np.linspace(x[0], x[-1], EMIT)
            ny = np.clip(np.asarray(f(nx), float), 0.0, None)
            if blk in ("base",):     ny = np.clip(ny, 0.0, 1.0)
            d[blk][key] = {"xs": [float(t) for t in nx], "ys": [round(float(t), 5) for t in ny]}

pw = np.array([r["pwl"] for r in report.values() if not math.isnan(r["pwl"])])
bs = np.array([r["scores"][r["picked"]] for r in report.values()])
from collections import Counter
print(f"curves fitted: {len(report)}")
print(f"  model picked: {dict(Counter(picked.values()))}")
print(f"  LOO error  PWL baseline: median {np.median(pw):.5f}  mean {pw.mean():.5f}")
print(f"  LOO error  selected    : median {np.median(bs):.5f}  mean {bs.mean():.5f}")
print(f"  improvement: median {100*(1-np.median(bs)/np.median(pw)):.1f}%   mean {100*(1-bs.mean()/pw.mean()):.1f}%")
d["source"] = {**d.get("source", {}), "curve_fit": {
    "emit_points": EMIT, "picked": picked,
    "loo_pwl_median": float(np.median(pw)), "loo_selected_median": float(np.median(bs)),
    "note": "each curve fitted with several candidates; the one predicting held-out knots best was "
            "kept, with plain interpolation among the candidates so no curve gets worse."}}
json.dump(d, open(OUT, "w"), indent=1)
json.dump(report, open(OUT.replace(".json", "_report.json"), "w"), indent=1)
print(f"wrote {OUT}")
