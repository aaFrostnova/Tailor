"""H1/H2 proposals with historical oracle and configuration-aware raw-BA thresholds.

This module reads no measurements and imports no historical evaluator.  It accepts
only four-input requests and an explicit, frozen oracle value.  In particular,
unknown historical columns do not constitute a measured failure.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from unified_detector import threshold_for_config

METHODS = ("H1_rule", "H2_rule")
POLICY_VERSION = "historical_H1_H2_unified_raw_ba_v2"
FRAGMENTS = ("VINE", "TrustMark", "VideoSeal")
ROT = {"rot9", "rot30"}
NEST_ATT = {"crop75", "crop90", "crop50", "unmarker"}
REGEN = {"regen", "rinse", "vaeB", "vaeC", "ctrlregen", "ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07"}
REQUEST_FIELDS = {"attacks", "fpr", "min_psnr", "max_ms"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_request(request):
    if set(request) != REQUEST_FIELDS:
        raise ValueError("policy requires exactly attacks, fpr, min_psnr, max_ms; capacity is output")
    if (not request["attacks"] or len(set(request["attacks"])) != len(request["attacks"])
            or any(not isinstance(a, str) or not a for a in request["attacks"])):
        raise ValueError("invalid attack set")
    if any(a.startswith("nfpa") for a in request["attacks"]):
        raise ValueError("NFPA is excluded from this revision")
    if not math.isfinite(request["fpr"]) or not 0 < request["fpr"] < 1:
        raise ValueError("invalid FPR")
    if not math.isfinite(request["max_ms"]) or request["max_ms"] <= 0:
        raise ValueError("invalid latency budget")
    if request["min_psnr"] is not None and not math.isfinite(request["min_psnr"]):
        raise ValueError("invalid PSNR floor")


def beta_from_fpr(alpha, n_tx=100, t_corr=10, k_data=37):
    """Compatibility helper: raw-BA threshold for one plain fragment; no BCH cap."""
    if n_tx != 100:
        raise ValueError('unified protocol requires a 100-bit word')
    return threshold_for_config(dict(S=['VINE'], order=['VINE'], s={'VINE': 1.0}, fe_on=[]), alpha)


class HistoricalOracle:
    """Boolean mirror of solve_frontier._attack_defended + frag_defends.

    An absent BA column retains the historical numeric fallback of 0 only for
    reproducing the proposal.  ``unknown_attacks`` exposes the missing evidence
    separately, and no historical oracle value is a selection/final result.
    """

    def __init__(self, value):
        self.value = copy.deepcopy(value)
        if tuple(value["fragments"]) != FRAGMENTS:
            raise ValueError("historical fragment order changed")
        self.alpha = value["alpha"]
        self.ba = value["bit_acc"]
        self.swept = set(value["swept"])
        self.solo_q = value["solo_q"]
        self.alphas = value["alpha_levels"]

    def defended(self, fragment, attack, threshold, alpha_index, resync, nested):
        if fragment == "VINE" and attack in self.swept and attack != "crop75":
            return self.alpha[str(self.alphas[alpha_index])][attack] >= threshold
        native = self.ba[fragment].get(attack, 0.0) >= threshold
        boost = self.value["frontend_recovery_ba"] >= threshold
        if fragment == "VINE":
            if attack in REGEN:
                return native and alpha_index >= 2
            if attack in {"crop90", "crop75", "crop50"}:
                return native or (nested and boost)
            if attack == "unmarker":
                return native or (nested and self.value["unmarker_recovery_ba"] >= threshold)
            return native
        if attack in {"rot9", "rot30", "crop75"}:
            return native or (resync and boost)
        return native

    def unknown_attacks(self, attacks):
        return sorted(a for a in attacks if not any(a in self.ba[f] for f in FRAGMENTS))


def greedy_subset(attacks, min_ba, oracle, fpr=None):
    """Historical gain/tie rules; re-evaluate coverage at each trial configuration threshold."""
    attack_set = set(attacks)
    resync = bool(attack_set & ROT)

    def nested_of(subset):
        return "VINE" in subset and bool(attack_set & NEST_ATT)

    def covered_by(subset):
        if not subset:
            return set()
        nested = nested_of(subset)
        cfg = dict(S=sorted(subset), order=list(subset),
                   s={f: .7 if f == 'VINE' else 1.0 for f in subset},
                   fe_on=sorted((['resync'] if resync else []) + (['scale'] if nested else [])))
        threshold = threshold_for_config(cfg, fpr) if fpr is not None else min_ba
        return {a for a in attack_set if any(
            oracle.defended(f, a, threshold, alpha, resync, nested)
            for f in subset for alpha in ([0, 1, 2, 3] if f == "VINE" else [3]))}

    chosen = []
    uncovered = set(attack_set)
    trace = []
    stopped = "covered_all"
    while uncovered:
        base = covered_by(tuple(chosen))
        gains = {f: len(covered_by(tuple(chosen) + (f,)) - base)
                 for f in FRAGMENTS if f not in chosen}
        if not gains:
            stopped = "all_fragments_considered"
            break
        gain = max(gains.values())
        if gain == 0:
            trace.append(dict(chosen_before=list(chosen), gains=gains, selected=None,
                              uncovered_before=sorted(uncovered)))
            stopped = "zero_gain"
            break
        tied = [f for f in gains if gains[f] == gain]
        selected = max(tied, key=lambda f: oracle.solo_q[f])
        trace.append(dict(chosen_before=list(chosen), gains=gains, selected=selected,
                          tied=tied, tie_psnr={f: oracle.solo_q[f] for f in tied},
                          uncovered_before=sorted(uncovered)))
        chosen.append(selected)
        uncovered = attack_set - covered_by(tuple(chosen))
    subset = tuple(f for f in FRAGMENTS if f in chosen) or FRAGMENTS
    evidence = dict(trace=trace, stop_reason=stopped, empty_fallback=not chosen,
                    covered_attacks=sorted(covered_by(subset)),
                    uncovered_attacks=sorted(attack_set - covered_by(subset)))
    return subset, resync, nested_of(subset), evidence


def propose(request, method, oracle):
    validate_request(request)
    if method not in METHODS:
        raise ValueError(f"unsupported independent baseline: {method}")
    min_ba = beta_from_fpr(request["fpr"])
    if method == "H1_rule":
        subset, resync, nested = ("VINE", "TrustMark"), True, False
        trace = dict(trace=[], rule="fixed expert default; no oracle optimization")
    else:
        subset, resync, nested, trace = greedy_subset(request["attacks"], min_ba, oracle, fpr=request["fpr"])
    frontends = sorted((["resync"] if resync else []) + (["scale"] if nested else []))
    config = dict(S=sorted(subset), order=list(subset),
                  s={f: 0.7 if f == "VINE" else 1.0 for f in subset}, fe_on=frontends)
    evidence = dict(policy_version=POLICY_VERSION, oracle_version=oracle.value["schema"],
                    oracle_sha256=canonical_hash(oracle.value), min_ba=threshold_for_config(config, request["fpr"]),
                    acceptance_protocol="unified_raw_ba_v1", threshold_depends_on_candidate_subset_and_frontends=True,
                    coverage_unknown=oracle.unknown_attacks(request["attacks"]),
                    information_scope="historical oracle; does not share new SMT offline columns",
                    frontend_mapping={"resync": "resync", "nested": "scale"},
                    deployment_domain="dual_FE_outside_SMT_single_FE_domain" if resync and nested else "within_SMT_single_FE_domain",
                    margin_affects_policy=False, alpha_optimized=False,
                    live_policy="one fixed rule candidate; accept or reject; no top-k/patch/backoff",
                    **trace)
    return dict(config=config, policy_evidence=evidence)
