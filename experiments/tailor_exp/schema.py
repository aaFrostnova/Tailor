"""Result schema of the follow-up runs and the canonical status vocabulary.

Every request of every method ends in exactly one canonical status. Terminal statuses are
outcomes; non-terminal statuses mean the request has no outcome yet and must be reported as
pending, never as a failure or a success:

  accepted          the deployed verifier passed every live gate for the selected configuration
  model_unsat       the offline model found no candidate (at every margin stage the policy allows)
  search_stalled    a heuristic search stopped without a candidate (not a proof of infeasibility)
  live_failed       the single configuration of a fixed method failed a live gate
  live_exhausted    every candidate the policy allows was measured live and failed
  timeout           the solver hit its wall-clock limit
  budget_exhausted  a budgeted search spent its candidate or time budget without a live pass
  pending / missing_input / incompatible_checkpoint / audit_incomplete   no outcome yet

RAW_STATUS maps the status strings of the frozen result files (census of 2026-11-09 over the
nine files that feed the main table) onto this vocabulary; the raw string is kept in each record.
"""
import math
import re

from . import protocol

SCHEMA_VERSION = 'tailor_exp_result_v1'
TERMINAL = ('accepted', 'model_unsat', 'search_stalled', 'live_failed', 'live_exhausted', 'timeout',
            'budget_exhausted')
NON_TERMINAL = ('pending', 'missing_input', 'incompatible_checkpoint', 'audit_incomplete')
STATUSES = TERMINAL + NON_TERMINAL
RAW_STATUS = {
    'unsat': 'model_unsat',
    'zero_margin_model_unsat': 'model_unsat',
    'greedy_stalled': 'search_stalled',
    'fixed_live_exhausted': 'live_exhausted',
    'fallback_live_exhausted': 'live_exhausted',
}
IMAGE_ROLES = ('selection', 'heldout', 'negative_clean', 'negative_wrong_identity', 'profiling')
LATENCY_BASES = ('modeled', 'measured')
LIVE_MEASURED = ('accepted', 'live_failed', 'live_exhausted')
# Statuses whose record may name a configuration (the one accepted, or the last one measured live).
CONFIGURABLE = ('accepted', 'live_failed', 'live_exhausted', 'budget_exhausted')
CONFIG_FIELDS = ('selected_configuration', 'strengths', 'order', 'geometry', 'T_c', 'threshold')
NUMERIC_FIELDS = ('predicted_distortion', 'actual_psnr', 'actual_ssim', 'embed_ms', 'decode_ms',
                  'wall_clock_search_ms')
REQUIRED = (
    'schema_version', 'run_id', 'protocol_version', 'cohort_hash', 'spec_id', 'method', 'image_role',
    'key_seed', 'decoder_hash', 'attack_param_id', 'margin_policy', 'latency_basis', 'status', 'raw_status',
    'alpha', 'selected_configuration', 'strengths', 'order', 'geometry', 'T_c', 'threshold',
    'predicted_distortion', 'actual_psnr', 'actual_ssim', 'embed_ms', 'decode_ms', 'wall_clock_search_ms',
    'live_candidate_count', 'cache_provenance', 'readout_ref',
)
_HEX64 = re.compile(r'[0-9a-f]{64}')


class SchemaError(ValueError):
    """A result record violates the schema."""


def canonical_status(raw):
    if raw in STATUSES:
        return raw
    if raw in RAW_STATUS:
        return RAW_STATUS[raw]
    raise SchemaError(f'unknown status {raw!r}')


def is_terminal(status):
    return canonical_status(status) in TERMINAL


def is_satisfied(status):
    return canonical_status(status) == 'accepted'


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _exact_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _names(value):
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _problems(r):
    missing = [k for k in REQUIRED if k not in r]
    if missing:
        return [f'missing fields {missing}']
    out = []
    if r['schema_version'] != SCHEMA_VERSION:
        out.append(f"schema_version {r['schema_version']!r}")
    if r['protocol_version'] != protocol.PROTOCOL:
        out.append(f"protocol_version {r['protocol_version']!r} is not {protocol.PROTOCOL}")
    for key in ('cohort_hash', 'spec_id', 'decoder_hash', 'attack_param_id'):
        if not isinstance(r[key], str) or not _HEX64.fullmatch(r[key]):
            out.append(f'{key} must be a 64-character lowercase hex digest')
    for key in ('run_id', 'method', 'key_seed', 'margin_policy'):
        if not isinstance(r[key], str) or not r[key]:
            out.append(f'{key} must be a non-empty string')
    if r['image_role'] not in IMAGE_ROLES:
        out.append(f"image_role {r['image_role']!r}")
    if r['latency_basis'] not in LATENCY_BASES:
        out.append(f"latency_basis {r['latency_basis']!r}")
    alpha_ok = _finite(r['alpha']) and 0.0 < r['alpha'] < 1.0
    if not alpha_ok:
        out.append('alpha must lie strictly between 0 and 1')
    count = r['live_candidate_count']
    if not _exact_int(count) or count < 0:
        out.append('live_candidate_count must be a non-negative integer')
        count = None
    for key in NUMERIC_FIELDS:
        if r[key] is not None and not _finite(r[key]):
            out.append(f'{key} must be a finite number or null')
    try:
        status = canonical_status(r['status'])
        if status != r['status']:
            out.append(f"status {r['status']!r} is not canonical")
        if canonical_status(r['raw_status']) != status:
            out.append(f"raw_status {r['raw_status']!r} does not map to {status!r}")
    except SchemaError as err:
        out.append(str(err))
        return out
    if status in LIVE_MEASURED and count is not None and count < 1:
        out.append(f'{status} needs at least one live candidate')
    if (status in LIVE_MEASURED or (count or 0) > 0) and not _readout_ok(r['readout_ref']):
        out.append(f'{status} with live candidates needs readout_ref with path and sha256 of the per-image readouts')
    configured = [k for k in CONFIG_FIELDS if r[k] is not None]
    if configured and status not in CONFIGURABLE:
        out.append(f'{status} records carry no configuration, found {configured}')
    elif configured or status == 'accepted':
        out.extend(_configuration_problems(r, alpha_ok, status == 'accepted'))
    if status == 'accepted':
        if not _finite(r['actual_psnr']):
            out.append('accepted record needs a finite actual_psnr')
        if not isinstance(r['cache_provenance'], str) or not r['cache_provenance']:
            out.append('accepted record needs cache_provenance')
    return out


def _readout_ok(ref):
    return isinstance(ref, dict) and bool(ref.get('path')) and bool(_HEX64.fullmatch(str(ref.get('sha256', ''))))


def _configuration_problems(r, alpha_ok, accepted):
    """A named configuration must be complete, deployable and judged at its own T(c) and threshold."""
    missing = [k for k in CONFIG_FIELDS if r[k] is None]
    if missing:
        return [f'configuration is incomplete, missing {missing}']
    if not _names(r['order']) or not _names(r['geometry']):
        return ['order and geometry must be lists of names']
    try:
        order, stages = protocol.check_deployable(r['order'], r['geometry'])
    except protocol.ProtocolError as err:
        return [f'configuration: {err}']
    out = []
    strengths = r['strengths']
    if not isinstance(strengths, dict) or set(strengths) != set(order):
        out.append('strengths must name exactly the selected fragments')
    elif not all(_finite(v) and v > 0 for v in strengths.values()):
        out.append('strengths must be positive finite numbers')
    if not isinstance(r['selected_configuration'], str) or not r['selected_configuration']:
        out.append('selected_configuration must be a non-empty string')
    if not _exact_int(r['T_c']) or not _exact_int(r['threshold']):
        return out + ['T_c and threshold must be exact integers']
    t_c = protocol.max_tests(order, stages)
    if r['T_c'] != t_c:
        out.append(f"T_c {r['T_c']} != {t_c} for order {order} and stages {stages}")
    elif alpha_ok:
        theta = protocol.threshold_for_tests(t_c, r['alpha'])
        if r['threshold'] != theta:
            out.append(f"threshold {r['threshold']} != {theta}")
        if accepted and protocol.is_impossible(theta):
            out.append('accepted under an impossible threshold')
    return out


def validate_record(record):
    problems = _problems(record)
    if problems:
        raise SchemaError('; '.join(problems))
    return record
