"""Exact numeric/Z3 evaluation of the campaign's composite *surrogate*.

This is a model prediction, not a measured joint per-image distribution. Pair
curves retain their original fixed-partner-strength interpretation, and several
applicable pair curves retain the solver's last-in-fragment-order precedence.
Live mean residuals are added AFTER the already-modelled directional deltas.
They transfer across strengths/configurations sharing a selected source curve;
that transfer is a refinement heuristic, not a uniform error bound.
"""
import math

FRONTEND_ORDER = ("resync", "scale", "angle", "tile")
FE_AFFECTS = {"resync": None, "scale": ("VINE",), "tile": ("TrustMark",)}


class NumericBackend:
    def curve(self, curve, strength, name):
        return curve.eval(strength)

    @staticmethod
    def choose(condition, yes, no):
        return yes if condition else no

    source = choose

    @staticmethod
    def both(*conditions):
        return all(conditions)

    @staticmethod
    def either(*conditions):
        return any(conditions)

    @staticmethod
    def negate(condition):
        return not condition

    @staticmethod
    def cap(value):
        return min(1.0, value)


class Z3Backend:
    """Accumulate PWL constraints; no Optimize call or GPU work is performed."""
    def __init__(self):
        import z3
        self.z3 = z3
        self.constraints = []

    def curve(self, curve, strength, name):
        result, constraints = curve.add_to_z3(strength, name)
        self.constraints.extend(constraints)
        return result

    def choose(self, condition, yes, no):
        return self.z3.If(condition, yes, no)

    @staticmethod
    def source(condition, yes, no):
        return None  # explanatory strings are only resolved by the numeric backend

    def both(self, *conditions):
        return self.z3.And(*conditions)

    def either(self, *conditions):
        return self.z3.Or(*conditions)

    def negate(self, condition):
        return self.z3.Not(condition)

    def cap(self, value):
        return self.z3.If(value > 1.0, self.z3.RealVal(1), value)


def composite_expression(sg, fragment, attack, strengths, after, frontends,
                         backend, *, legacy_frontends=None,
                         pair_min_cascade=0.02):
    """Shared branch/delta formula used by both the solver and numeric audit.

    ``after(f,g)`` means selected g is embedded after host f. A pair curve is
    indexed by host strength; a delta(g,f,a) is indexed by overwriter strength.
    No final [0,1] clipping is applied: the original solver does not clip its
    base-minus-delta prediction either.
    """
    f, a, b = fragment, attack, backend
    offsets = getattr(sg, "_composite_mean_offsets", {})
    source = ("base", f, a)
    base = b.curve(sg.base(f, a), strengths[f], f"base_{f}_{a}")
    correction = float(offsets.get(source, 0.0))
    selected_source, selected_stage = source, None
    levels, pair_used, gains = [], {}, {}
    for stage, enabled in frontends.items():
        if enabled is None:
            continue
        name = f"base_fe_{stage}_{f}|{a}"
        curve = sg.frontend(name)
        if curve is None:
            continue
        plain = sg.base(f, a)
        gains[stage] = max(curve.eval(x) - plain.eval(x)
                           for x in sorted(set(plain.xs) | set(curve.xs)))
        level = b.curve(curve, strengths[f], f"feon_{stage}_{f}_{a}")
        level_source = ("frontend", name)
        level_correction = float(offsets.get(level_source, 0.0))
        stage_pairs = {}
        for g in sg.fragments:
            if g == f:
                continue
            pair_name = f"base_fe_{stage}_{f}|{a}|with_{g}"
            pair = sg.frontend(pair_name)
            if pair is None:
                continue
            control = sg.frontend(f"ctrl_pair_{f}|{a}|with_{g}")
            if control is not None:
                grid = sorted(set(pair.xs) | set(control.xs))
                if max(pair.eval(x) - control.eval(x) for x in grid) < pair_min_cascade:
                    continue
            pair_value = b.curve(pair, strengths[f], f"fepair_{stage}_{f}_{g}_{a}")
            use_pair = b.both(enabled, after(f, g))
            pair_source = ("frontend", pair_name)
            level = b.choose(use_pair, pair_value, level)
            level_correction = b.choose(use_pair, float(offsets.get(pair_source, 0.0)), level_correction)
            level_source = b.source(use_pair, pair_source, level_source)
            # Only the LAST selected pair is represented by level. Earlier
            # overlays must retain their directional interference deductions.
            stage_pairs = {h: b.both(chosen, b.negate(use_pair))
                           for h, chosen in stage_pairs.items()}
            stage_pairs[g] = use_pair
        levels.append((stage, enabled, level, level_correction, level_source, stage_pairs))
    took_effect = bool(levels)
    for stage, enabled, level, level_correction, level_source, stage_pairs in levels:
        for g in sg.fragments:
            pair_used[g] = b.choose(enabled, stage_pairs.get(g, False), pair_used.get(g, False))
        base = b.choose(enabled, level, base)
        correction = b.choose(enabled, level_correction, correction)
        selected_source = b.source(enabled, level_source, selected_source)
        selected_stage = b.source(enabled, stage, selected_stage)
    if not levels:
        legacy = legacy_frontends if legacy_frontends is not None else (
            ("scale", frontends.get("scale")), ("resync", frontends.get("resync")))
        for stage, enabled in legacy:
            if enabled is None:
                continue
            old_name = "nested" if stage == "scale" else stage
            name = f"base_{old_name}_{f}|{a}"
            curve = sg.frontend(name)
            if curve is None:
                continue
            level = b.curve(curve, strengths[f], f"legacyfe_{f}_{a}_{old_name}")
            source = ("frontend", name)
            base = b.choose(enabled, level, base)
            correction = b.choose(enabled, float(offsets.get(source, 0.0)), correction)
            selected_source = b.source(enabled, source, selected_source)
            selected_stage = b.source(enabled, stage, selected_stage)
            took_effect = True
    if took_effect:
        base = b.cap(base)
    drops, terms = [], []
    for g in sg.fragments:
        if g == f:
            continue
        value = b.curve(sg.delta(g, f, a), strengths[g], f"del_{g}_{f}_{a}")
        later = after(f, g)
        replaced = pair_used.get(g, False)
        applied = b.both(later, b.negate(replaced))
        drop = b.choose(applied, value, 0.0)
        drops.append(drop)
        terms.append({"overwriter": g, "host": f, "strength": strengths[g],
                      "curve_value": value, "after": later,
                      "pair_used": replaced, "applied": applied, "drop": drop})
    uncorrected = base - sum(drops)
    return {"mean": uncorrected + correction, "mean_uncorrected": uncorrected,
            "mean_offset": correction, "base": base, "delta_terms": terms,
            "curve_key": selected_source, "stage": selected_stage,
            "frontend_gains": gains}


def isotonic(ys):
    """Equal-weight PAVA, identical to the solver's acceptance-floor fit."""
    vals, weights = [], []
    for y in ys:
        vals.append(float(y)); weights.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1] + 1e-15:
            v2, w2 = vals.pop(), weights.pop()
            v1, w1 = vals.pop(), weights.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); weights.append(w1 + w2)
    return [value for value, weight in zip(vals, weights) for _ in range(int(round(weight)))]


def floor_strength(curve, level):
    xs, ys = curve.xs, isotonic(curve.ys)
    j = next((i for i, y in enumerate(ys) if y >= level - 1e-12), None)
    if j is None:
        return None
    if j == 0:
        return xs[0]
    x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
    return x1 if y1 == y0 else x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def rate_source(sg, f, a, tau, stage=None):
    """Return (curve, actual stage, missing-changed-embed) as det_cond does."""
    plain = sg.rate_curve(f, a, tau)
    if stage is None:
        return plain, None, False
    curve = sg.rate_curve(f, a, tau, stage)
    if curve is not None:
        return curve, stage, False
    affected = FE_AFFECTS.get(stage, ())
    if affected is None or f in affected:
        return None, stage, bool(getattr(sg, "_perimage", None))
    return plain, None, False


def predict_composite(sg, cfg, attack, *, tau=None, mean_margin=0.0,
                      det_min=0.9, rate_margin=0.0, pair_min_cascade=0.02):
    """Return exact model mean and the solver's *solo/FE* rate proxy per fragment.

    Pass tau=unified_detector.threshold_for_config(cfg, request_fpr); the
    caller must charge the full primary and geometric search budget. ``attack_pass`` is OR_f(mean_pass AND rate_pass).
    It excludes other solver rules (ranges, fidelity, latency, FE responsibility,
    skeleton usefulness etc.), so full_admissibility_checked is always False.
    """
    order = list(cfg["order"])
    if not order or len(order) != len(set(order)) or not set(order) <= set(sg.fragments):
        raise ValueError("order must contain distinct selected surrogate fragments")
    active = set(cfg.get("fe_on", [k for k, value in cfg.get("fe", {}).items() if value]))
    if "nested" in active:
        active.remove("nested"); active.add("scale")
    if not active <= set(FRONTEND_ORDER) or len(active) > 1:
        raise ValueError("the solver permits at most one known frontend")
    frontends = {name: name in active for name in FRONTEND_ORDER}
    strengths = {f: float(cfg["s"][f]) if f in order else 0.0 for f in sg.fragments}
    if not all(math.isfinite(value) for value in strengths.values()):
        raise ValueError("non-finite strength")
    positions = {f: i for i, f in enumerate(order)}
    after = lambda f, g: g in positions and positions[g] > positions[f]
    stage = next(iter(active), None)
    mean_threshold = (float(tau) if tau > 1 else min(1.0, float(tau) + float(mean_margin))) if tau is not None else None
    rate_threshold = min(1.0, float(det_min) + float(rate_margin))
    result = {}
    for f in order:
        item = composite_expression(sg, f, attack, strengths, after, frontends,
                                    NumericBackend(), pair_min_cascade=pair_min_cascade)
        item.update(rate_raw=None, rate_isotonic=None, rate_floor_strength=None,
                    rate_stage=None, mean_pass=None, rate_pass=None, cover_pass=None)
        if tau is not None:
            curve, actual_stage, missing = rate_source(sg, f, attack, tau, stage)
            item["rate_stage"] = actual_stage
            item["rate_missing_changed_embed"] = missing
            if curve is not None:
                item["rate_raw"] = curve.eval(strengths[f])
                fitted = type(curve)(curve.xs, isotonic(curve.ys))
                item["rate_isotonic"] = fitted.eval(strengths[f])
                item["rate_floor_strength"] = floor_strength(curve, rate_threshold)
            item["mean_pass"] = item["mean"] >= mean_threshold
            item["rate_pass"] = (True if not det_min or det_min <= 0 else
                                 False if missing else True if curve is None else
                                 item["rate_floor_strength"] is not None and
                                 strengths[f] >= item["rate_floor_strength"])
            item["cover_pass"] = item["mean_pass"] and item["rate_pass"]
        result[f] = item
    return {"fragments": result, "attack": attack, "tau": tau,
            "mean_threshold": mean_threshold, "rate_threshold": rate_threshold,
            "attack_pass": any(v["cover_pass"] for v in result.values()) if tau is not None else None,
            "rate_semantics": "legacy_ba_prior_no_identity_credit_not_unified_joint_perimage",
            "requires_fresh_unified_live": True,
            "full_admissibility_checked": False}


def estimate_rate_offset(sg, f, a, stage, strength, tau, live_rate, n, det_min=0.9):
    """Original conservative rate gate/search, applied ONLY to per-image values."""
    curve, actual_stage, missing = rate_source(sg, f, a, tau, stage)
    evidence = {"key": None, "offset": 0.0, "predicted_rate": None,
                "measured_rate": float(live_rate), "shifted_rate": None,
                "missing_changed_embed": missing, "target_attained": None}
    if curve is None:
        return evidence
    key = ("fe", actual_stage, f, a) if actual_stage else (f, a)
    predicted = float(curve.eval(strength))
    se = math.sqrt(max(live_rate * (1 - live_rate), 0.09) / n)
    evidence.update(key=key, predicted_rate=predicted, rate_se=se, shifted_rate=predicted)
    if predicted <= live_rate or (live_rate >= det_min - 1e-9 and predicted - live_rate < 2 * se):
        evidence["target_attained"] = predicted <= live_rate + 1e-12
        return evidence
    lo, hi = -0.4, 0.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        shifted = sg.with_composite_offsets(rate_offsets={key: mid})
        if shifted.rate_curve(f, a, tau, actual_stage).eval(strength) > live_rate:
            hi = mid
        else:
            lo = mid
    shifted = sg.with_composite_offsets(rate_offsets={key: lo})
    rate = shifted.rate_curve(f, a, tau, actual_stage).eval(strength)
    evidence.update(offset=lo, shifted_rate=rate, target_attained=rate <= live_rate + 1e-12)
    return evidence
