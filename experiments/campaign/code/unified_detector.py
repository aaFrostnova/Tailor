"""Unified raw-bit acceptance with a fixed Bonferroni search budget.

No BCH/identity decision is an acceptance route. A bit matches only when its
finite aligned LLR has the strictly correct sign; zero never matches. Under
the declared fresh-query ideal-PRF null this score's tail is bounded by
Binomial(100, 1/2), including ties. This is a conditional theoretical rule,
not an empirical guarantee for every fixed key or adaptive attacker.

Embedding is unchanged. Batch-threshold evaluation shares GPU reads but is
not a deployment latency measurement. Single-threshold decode follows the
actual early-stop path, with the same fixed worst-case test count.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import importlib
import math
from numbers import Real

import numpy as np


N_BITS = 100
PROTOCOL = 'unified_raw_ba_v1'
_FRAGMENTS = {'vine': 'vine', 'trustmark': 'trustmark', 'videoseal': 'videoseal'}
_GEOMETRIES = {'resync', 'scale', 'tile', 'angle'}
_CFG_KEYS = {'S', 'order', 's', 'fe_on', 'fe', 'frags', 'strengths', 'resync', 'nested',
             'scale_search', 'angle_sweep', 'tile', 'alpha', 'n_bits'}
_NATIVE_FLAGS = {'resync', 'nested', 'scale_search', 'angle_sweep', 'tile'}
_TAIL_NUMERATORS = tuple(sum(math.comb(N_BITS, j) for j in range(t, N_BITS + 1))
                         for t in range(N_BITS + 2))
_TAIL_DENOMINATOR = 1 << N_BITS


def _alpha(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('alpha must be a finite number strictly between zero and one')
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError('alpha must be a finite number strictly between zero and one')
    return value


def _fragment(value):
    if not isinstance(value, str) or value.lower() not in _FRAGMENTS:
        raise ValueError('unknown watermark fragment: %r' % (value,))
    return _FRAGMENTS[value.lower()]


def _sequence(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError('%s must be a list or tuple' % name)
    return list(value)


def _flag(cfg, name, default=False):
    value = cfg.get(name, default)
    if type(value) is not bool:
        raise ValueError('%s must be a boolean' % name)
    return value


def normalize_config(cfg):
    if not isinstance(cfg, Mapping) or set(cfg) - _CFG_KEYS:
        raise ValueError('unknown or invalid decoder configuration fields')
    if 'n_bits' in cfg and (type(cfg['n_bits']) is not int or cfg['n_bits'] != N_BITS):
        raise ValueError('this decoder requires exactly 100 transmitted bits')
    order_value = cfg.get('order', cfg.get('S', cfg.get('frags')))
    order = [_fragment(f) for f in _sequence(order_value, 'order')]
    if not order or len(order) != len(set(order)):
        raise ValueError('order must contain a nonempty set of distinct known fragments')
    for name in ('S', 'frags'):
        if name in cfg:
            selected = [_fragment(f) for f in _sequence(cfg[name], name)]
            if len(selected) != len(set(selected)) or set(selected) != set(order):
                raise ValueError('fragment subset and order disagree')
    mapped_stages = None
    if 'fe' in cfg:
        frontend = cfg['fe']
        if not isinstance(frontend, Mapping) or set(frontend) - _GEOMETRIES:
            raise ValueError('fe must map known geometric stages to booleans')
        if any(type(enabled) is not bool for enabled in frontend.values()):
            raise ValueError('fe values must be booleans')
        mapped_stages = {name for name, enabled in frontend.items() if enabled}
        if set(cfg) & _NATIVE_FLAGS:
            raise ValueError('do not mix fe and native frontend flags')
    if 'fe_on' in cfg:
        if set(cfg) & _NATIVE_FLAGS:
            raise ValueError('do not mix fe_on and native frontend flags')
        stages = _sequence(cfg['fe_on'], 'fe_on')
        if any(not isinstance(s, str) or s not in _GEOMETRIES for s in stages) or len(stages) != len(set(stages)):
            raise ValueError('unknown or duplicate geometric stage')
        if mapped_stages is not None and set(stages) != mapped_stages:
            raise ValueError('fe and fe_on disagree')
    elif mapped_stages is not None:
        stages = list(mapped_stages)
    else:
        resync = _flag(cfg, 'resync')
        nested = _flag(cfg, 'nested')
        stages = [name for name, enabled in (
            ('resync', resync), ('scale', _flag(cfg, 'scale_search', nested)),
            ('angle', _flag(cfg, 'angle_sweep', resync)), ('tile', _flag(cfg, 'tile')),
        ) if enabled]
    if 'scale' in stages and 'vine' not in order:
        raise ValueError('scale recovery requires VINE')
    if 'tile' in stages and 'trustmark' not in order:
        raise ValueError('tile recovery requires TrustMark')
    for name in ('s', 'strengths'):
        if name in cfg:
            if not isinstance(cfg[name], Mapping):
                raise ValueError('%s must be a strength mapping' % name)
            for fragment, strength in cfg[name].items():
                _fragment(fragment)
                if isinstance(strength, bool) or not isinstance(strength, Real) or not math.isfinite(strength) or strength <= 0:
                    raise ValueError('strengths must be finite positive numbers')
    if 'alpha' in cfg:
        _alpha(cfg['alpha'])
    return dict(order=order, fe_on=sorted(stages))


def budget_for_config(cfg):
    normalized = normalize_config(cfg)
    m, stages = len(normalized['order']), set(normalized['fe_on'])
    primary = 1 if m == 1 else m + 1
    geometry = dict(resync=5 * m if 'resync' in stages else 0,
                    scale=133 if 'scale' in stages else 0,
                    tile=81 if 'tile' in stages else 0,
                    angle=65 * m if 'angle' in stages else 0)
    return dict(protocol=PROTOCOL, n_bits=N_BITS, order=normalized['order'],
                fe_on=normalized['fe_on'], primary_tests=primary,
                geometry_tests=geometry, max_tests=primary + sum(geometry.values()),
                geometry_fusion_is_acceptance_test=False,
                budget_depends_on_early_stop=False)


@lru_cache(maxsize=4096)
def _threshold(max_tests, alpha):
    alpha_numerator, alpha_denominator = alpha.as_integer_ratio()
    # Integer comparison against the exact rational represented by alpha;
    # never use 1-alpha or a rounded inverse CDF at extreme tail levels.
    for count, numerator in enumerate(_TAIL_NUMERATORS):
        if max_tests * numerator * alpha_denominator <= alpha_numerator * _TAIL_DENOMINATOR:
            return count
    raise AssertionError('the impossible threshold 101 must always satisfy the bound')


def threshold_count(cfg, alpha):
    return _threshold(budget_for_config(cfg)['max_tests'], _alpha(alpha))


def threshold_for_config(cfg, alpha):
    return threshold_count(cfg, alpha) / N_BITS


def _word(tx):
    tx = np.asarray(tx)
    if tx.shape != (N_BITS,) or tx.dtype.kind not in 'biuf' or not np.all(np.isfinite(tx)) or not np.all((tx == 0) | (tx == 1)):
        raise ValueError('expected codeword must contain exactly 100 binary bits')
    return tx.astype(np.uint8, copy=False)


def _llrs(llrs):
    values = np.asarray(llrs)
    if values.shape != (N_BITS,) or values.dtype.kind not in 'iuf':
        raise ValueError('LLRs must be a numeric length-100 vector')
    values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError('non-finite LLRs are not accepted')
    return values


def strict_matches(llrs, tx):
    values, word = _llrs(llrs), _word(tx)
    return int(np.count_nonzero(((values > 0) & (word == 1)) | ((values < 0) & (word == 0))))


def strict_accuracy(llrs, tx):
    return strict_matches(llrs, tx) / N_BITS


def config_for_comp(comp):
    if getattr(getattr(comp, 'sb', None), 'n', None) != N_BITS:
        raise ValueError('composite must use a 100-bit transmitted word')
    stages = []
    for stage, attribute in (('resync', 'resync'), ('scale', 'fe_scale'), ('tile', 'fe_tile'), ('angle', 'fe_angle')):
        value = getattr(comp, attribute, False)
        if type(value) is not bool:
            raise ValueError('composite frontend flags must be booleans')
        if value:
            stages.append(stage)
    geo = getattr(comp, 'geo', bool(stages))
    if type(geo) is not bool:
        raise ValueError('composite geo flag must be boolean')
    if not geo:  # preserve the existing decode_no_cascade contract
        stages = []
    cfg = normalize_config(dict(order=list(comp.order), fe_on=stages))
    if 'scale' in stages and getattr(comp, '_vss', None) != .005:
        raise ValueError('unbudgeted scale-search step')
    if 'angle' in stages:
        required = {'_rr': 180.0, '_rs': 3.0, '_bc': 10.0}
        if any(getattr(comp, name, None) != value for name, value in required.items()):
            raise ValueError('unbudgeted angle-search parameters')
    if 'tile' in stages:
        tiled = getattr(comp, '_tiled', None)
        if (tiled is None or getattr(tiled, 'cell', None) != 256 or
                list(getattr(tiled, 'offsets', [])) != list(range(0, 257, 32))):
            raise ValueError('unbudgeted tile-search grid')
    return cfg


def _fuse(order, llrs):
    # Preserve the original order of float64 additions.
    result = np.zeros(N_BITS, dtype=np.float64)
    for fragment in order:
        result += _llrs(llrs[fragment])
    return _llrs(result)


def _score(llrs, tx):
    values = _llrs(llrs)
    count = strict_matches(values, tx)
    return dict(matches=count, accuracy=count / N_BITS, zero_count=int(np.count_nonzero(values == 0)))


def _view_scores(comp, view, iid, tx, available):
    llrs = dict(available)
    for fragment in comp.order:
        if fragment not in llrs:
            llrs[fragment] = _llrs(comp._frag_llr(fragment, view, iid))
    per_fragment = {f: _score(llrs[f], tx) for f in comp.order}
    fused = _score(_fuse(comp.order, llrs), tx)
    return dict(fragments=per_fragment, fusion=fused), llrs


def _geometry_views(comp, att, cfg):
    """Yield each original candidate view and ONLY its acceptance fragments."""
    stages = set(cfg['fe_on'])
    present = [f for f in ('vine', 'trustmark', 'videoseal') if f in cfg['order']]
    if 'resync' in stages:
        rect, _ = comp._sync_rectify(comp._sync, att, comp.dev)
        yield dict(path='resync:rectified', kind='resync', view=rect, fragments=present)
        for angle in (-3.0, 3.0, -6.0, 6.0):
            yield dict(path='resync:tilt:%g' % angle, kind='resync',
                       view=comp._rot(rect, angle), fragments=present)
    if 'scale' in stages:
        scales = np.arange(.34, 1.0001, comp._vss)
        if len(scales) != 133:
            raise ValueError('scale enumeration exceeds the declared budget')
        for index, scale in enumerate(scales):
            if scale >= .999:
                view = att
            else:
                size = int(round(512 * float(scale)))
                offset = (512 - size) // 2
                view = att.crop((offset, offset, offset + size, offset + size))
            yield dict(path='scale:%03d' % index, kind='scale', view=view, fragments=['vine'])
    if 'tile' in stages:
        tiled = comp._tiled
        image = tiled._to512(att)
        for oy in tiled.offsets:
            for ox in tiled.offsets:
                view = image.crop((ox, oy, ox + tiled.cell, oy + tiled.cell))
                yield dict(path='tile:%d:%d' % (ox, oy), kind='tile', view=view,
                           fragments=['trustmark'], tile=(image, ox, oy))
    if 'angle' in stages:
        from src.angle_probe import candidate_angles
        angles = list(candidate_angles(att, search=comp._rr, step=comp._bc, refine=comp._rs))
        if len(angles) > 65 or any(not math.isfinite(float(a)) for a in angles):
            raise ValueError('angle candidates exceed the declared finite budget or contain a nonfinite angle')
        for index, angle in enumerate(angles):
            yield dict(path='angle:%03d:%g' % (index, angle), kind='angle',
                       view=comp._rot(att, float(angle)), fragments=present)


def _acceptance_llr(comp, spec, fragment, iid):
    # The old tile fallback clipped this same window's TrustMark LLRs to
    # +/-15. Symmetric positive clipping does not change strict sign matches.
    # Retaining raw LLRs also preserves accepted-view equal fusion reporting
    # and exposes nonfinite model output rather than silently clipping Inf.
    return _llrs(comp._frag_llr(fragment, spec['view'], iid))


def _evaluate(comp, att, secret, alphas, *, trace_all=False, geometry_only=False):
    cfg = config_for_comp(comp)
    budget = budget_for_config(cfg)
    iid, tx = secret
    if not isinstance(iid, str) or not iid:
        raise ValueError('expected identity must be a nonempty string')
    tx = _word(tx)
    alphas = [_alpha(a) for a in alphas]
    if not alphas:
        raise ValueError('at least one alpha is required')
    thresholds = sorted({_threshold(budget['max_tests'], a) for a in alphas})
    primary, primary_llrs = _view_scores(comp, att, iid, tx, {})
    primary_tests = [(f, primary['fragments'][f]) for f in cfg['order']]
    if len(cfg['order']) > 1:
        primary_tests.append(('fused', primary['fusion']))
    best_primary_path, best_primary_score = max(primary_tests, key=lambda item: item[1]['matches'])
    history = []
    attempted = 0
    maxima = 0.0
    for name, score in primary_tests:
        if not geometry_only:
            attempted += 1
            maxima = max(maxima, score['accuracy'])
            history.append(dict(path='primary:' + name, kind='primary', fragment=name, **score))
    selected = {}
    selected_views = {}

    def finish(theta, detected, score, path, kind, view, view_scores, attempts, history_length):
        per_fragment = {f: value['accuracy'] for f, value in view_scores['fragments'].items()}
        result = dict(
            protocol=PROTOCOL, detected=bool(detected), threshold_count=theta,
            threshold_accuracy=theta / N_BITS, threshold_attainable=theta <= N_BITS,
            max_tests=budget['max_tests'], n_bits=N_BITS,
            accepted_score=float(score), bit_accuracy=float(score),
            best_fragment_ba=max(per_fragment.values()), per_fragment_ba=per_fragment,
            fused_ba=view_scores['fusion']['accuracy'], used_geometry=bool(detected and kind != 'primary'),
            accepted_path=path if detected else None, accepted_kind=kind if detected else None,
            primary_scores=primary, accepted_view_scores=view_scores if detected else None,
            attempted_tests=attempts, max_attempted_score=maxima,
            budget=budget, identity_verification_is_acceptance_route=False,
            zero_llr_matches=False, report_fusion_in_geometry_only=True,
            evaluated_trace_length=history_length,
        )
        selected[theta] = result
        selected_views[theta] = view if detected else None

    if not geometry_only:
        for theta in thresholds:
            if best_primary_score['matches'] >= theta:
                finish(theta, True, best_primary_score['accuracy'], 'primary:' + best_primary_path,
                       'primary', att, primary, attempted, len(history))
    unresolved = set(thresholds) - set(selected)
    geometry_views = _geometry_views(comp, att, cfg) if unresolved or trace_all else ()
    for spec in geometry_views:
        if not unresolved and not trace_all:
            break
        available = {}
        completed_scores = None
        for fragment in spec['fragments']:
            if fragment not in available:
                available[fragment] = _acceptance_llr(comp, spec, fragment, iid)
            score = _score(available[fragment], tx)
            attempted += 1
            if attempted > budget['max_tests']:
                raise AssertionError('executed acceptance tests exceed the declared budget')
            maxima = max(maxima, score['accuracy'])
            history.append(dict(path=spec['path'] + ':' + fragment, kind=spec['kind'], fragment=fragment, **score))
            accepted_here = [theta for theta in sorted(unresolved) if score['matches'] >= theta]
            if accepted_here:
                if completed_scores is None:
                    # Only accepted views receive supplementary report reads.
                    # These extra fragments/fusion NEVER become acceptance
                    # routes for scale or tile, nor for an already stopped test.
                    completed_scores, report_llrs = _view_scores(comp, spec['view'], iid, tx, available)
                    available.update(report_llrs)
                for theta in accepted_here:
                    finish(theta, True, score['accuracy'], spec['path'] + ':' + fragment,
                           spec['kind'], spec['view'], completed_scores, attempted, len(history))
                    unresolved.remove(theta)
            if not unresolved and not trace_all:
                break
        if not unresolved and not trace_all:
            break
    for theta in sorted(unresolved):
        # Failed recovery retains the primary-view score. Do not maximize the
        # reporting view post hoc over an unsuccessful geometric search.
        finish(theta, False, best_primary_score['accuracy'], None, 'primary',
               att, primary, attempted, len(history))
    for result in selected.values():
        # A per-threshold record ends at that threshold's logical first pass.
        # trace_all additionally exposes the common complete candidate trace.
        result['tests'] = history[:result['evaluated_trace_length']]
        if trace_all:
            result['all_tests'] = history
    return dict(protocol=PROTOCOL, max_tests=budget['max_tests'], budget=budget,
                alpha_to_threshold={repr(a): str(_threshold(budget['max_tests'], a)) for a in alphas},
                by_threshold={str(theta): selected[theta] for theta in thresholds},
                shared_measurement_is_deployment_latency=False), selected_views


def decode_thresholds_detailed(comp, att, secret, alphas, *, trace_all=False):
    """Share candidate extraction across targets; not a latency measurement."""
    result, _ = _evaluate(comp, att, secret, alphas, trace_all=trace_all)
    return result


def decode_detailed(comp, att, secret, alpha=None, *, trace_all=False):
    alpha = _alpha(getattr(comp, 'alpha', .01) if alpha is None else alpha)
    result, _ = _evaluate(comp, att, secret, [alpha], trace_all=trace_all)
    row = result['by_threshold'][result['alpha_to_threshold'][repr(alpha)]]
    return dict(row, alpha=alpha)


def decode(comp, att, secret, alpha=None):
    detail = decode_detailed(comp, att, secret, alpha)
    return detail['fused_ba'], detail['detected']


def geo_cascade(comp, att, iid, tx, return_view=False, alpha=None):
    """Compatibility entry point; geometry is BA-gated, never BCH-gated."""
    alpha = _alpha(getattr(comp, 'alpha', .01) if alpha is None else alpha)
    result, views = _evaluate(comp, att, (iid, tx), [alpha], geometry_only=True)
    theta = int(result['alpha_to_threshold'][repr(alpha)])
    ok = result['by_threshold'][str(theta)]['detected']
    return (ok, views[theta]) if return_view else ok


def install(target=None):
    """Patch only the current process's decoder class, preserving embedding."""
    if target is None:
        target = importlib.import_module('eval_matrix').OursComposite
    elif hasattr(target, 'OursComposite'):
        target = target.OursComposite
    if not isinstance(target, type):
        raise TypeError('install expects OursComposite or a module containing it')
    if not hasattr(target, '_unified_ba_original_decode'):
        target._unified_ba_original_decode = target.decode
        target._unified_ba_original_geo_cascade = target.geo_cascade
    target.decode = decode
    target.geo_cascade = geo_cascade
    target._unified_ba_protocol = PROTOCOL
    return target
