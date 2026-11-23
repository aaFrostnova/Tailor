"""Three separately deployed, fixed single-fragment baselines.

These reproduce single_asshipped(sc, f), for each fixed f separately. They do
not reproduce single_ship, the old best-among-single-fragments aggregate.
No selection/final result, offline table, or search oracle enters the policy.
"""
import copy
import hashlib
import json
import math

METHOD_FRAGMENTS = dict(S_VINE='VINE', S_TrustMark='TrustMark', S_VideoSeal='VideoSeal')
METHODS = tuple(METHOD_FRAGMENTS)
POLICY_VERSION = 'fixed_single_asshipped_unified_raw_ba_v2'
REQUEST_FIELDS = {'attacks', 'fpr', 'min_psnr', 'max_ms'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_request(request):
    if set(request) != REQUEST_FIELDS:
        raise ValueError('exactly four inputs required; capacity is output')
    attacks = request['attacks']
    if (not isinstance(attacks, list) or not attacks
            or any(not isinstance(a, str) or not a for a in attacks)
            or len(set(attacks)) != len(attacks)):
        raise ValueError('invalid attack set')
    if any(a.startswith('nfpa') for a in attacks):
        raise ValueError('NFPA excluded from editing-only revision')
    if not math.isfinite(request['fpr']) or not 0 < request['fpr'] < 1:
        raise ValueError('invalid FPR')
    if not math.isfinite(request['max_ms']) or request['max_ms'] <= 0:
        raise ValueError('invalid latency budget')
    if request['min_psnr'] is not None and not math.isfinite(request['min_psnr']):
        raise ValueError('invalid PSNR floor')


def method_specs():
    return dict(schema='fixed_single_fragment_policy_specs_v1', policy_version=POLICY_VERSION,
                historical_function='rigorous_baselines.single_asshipped(sc, f)',
                historical_alpha_index=3, historical_alpha_value=1.0,
                reproduces_single_ship_best_among_singles=False,
                methods={method: dict(S=[fragment], order=[fragment], s={fragment: 1.0}, fe_on=[])
                         for method, fragment in METHOD_FRAGMENTS.items()},
                acceptance_protocol='unified_raw_ba_v1', selection_rule='one fixed fragment per separately named method; accept or reject after live',
                policy_uses_offline_coverage=False, policy_uses_final_observations=False,
                margin_affects_policy=False, strength_optimized=False, frontend_optimized=False,
                fragment_optimized_per_request=False, capacity_is_output=True,
                no_topk_patch_or_margin_backoff=True, comparison_arms_are_paired_references=True)


def propose(request, method):
    validate_request(request)
    specs = method_specs()
    if method not in METHODS:
        raise ValueError(f'unknown fixed single baseline: {method}')
    return dict(config=copy.deepcopy(specs['methods'][method]), policy_evidence=dict(
        policy_version=POLICY_VERSION, acceptance_protocol='unified_raw_ba_v1', method_spec_sha256=canonical_hash(specs),
        rule='fixed single_asshipped for this method; not best-single selection',
        historical_alpha_index=3, fixed_strength=1.0, frontend_enabled=False,
        coverage_oracle_used=False, coverage_unknown=[],
        coverage_unknown_semantics='not queried: fixed rule has no coverage oracle',
        deployment_domain='within_SMT_single_FE_domain',
        margin_affects_policy=False, alpha_optimized=False,
        original_single_ship_aggregate_reproduced=False,
        live_policy='one fixed rule candidate; accept or reject; no top-k/patch/backoff'))
