"""Four-input requests, per-image live checks, and capacity reported after selection.

Capacity is the existing 100-bit BSC model evaluated on measured bit accuracy.
It is an estimate, not a test of an alternative payload length or ECC decoder.
"""
import math
import numpy as np

PROTOCOL = 'four_inputs_capacity_output_unified_raw_ba_v1'
CELL_PROTOCOL = 'unified_raw_ba_v1'
REQUEST_ALPHAS = (0.1, 0.01, 1e-4, 1e-6, 1e-9, 2**-37)
REQUEST_FIELDS = ('attacks', 'fpr', 'min_psnr', 'max_ms')


def exact_cfg_key(order, fe_on, strengths):
    """The exact executed float strengths; never quantize a continuous solution."""
    order, stages, strengths = tuple(order), tuple(sorted(fe_on)), tuple(strengths)
    if (not order or len(order) != len(set(order)) or len(stages) != len(set(stages))
            or len(strengths) != len(order)):
        raise ValueError('invalid exact cache configuration')
    if any(isinstance(x, bool) or not math.isfinite(float(x)) or float(x) <= 0 for x in strengths):
        raise ValueError('strengths must be finite and positive')
    return order, stages, tuple(float(x) for x in strengths)


def exact_cfg_id(key):
    import hashlib
    import json
    # Python's float JSON representation roundtrips to the identical binary64.
    key = exact_cfg_key(*key)
    return hashlib.sha256(json.dumps(key, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def solver_scenario(request, margin=0.02):
    import watermark_smt_v2 as W
    if 'min_bits' in request:
        raise ValueError('min_bits is not an input in the capacity-output protocol')
    assert request['attacks'] and len(set(request['attacks'])) == len(request['attacks'])
    assert 0 < request['fpr'] < 1 and request['max_ms'] > 0
    return dict(attacks=request['attacks'], min_ba=W.beta_from_fpr(request['fpr']),
                min_psnr=request['min_psnr'], max_ms=request['max_ms'],
                allow_resync=True, allow_nested=True, resolution=512,
                min_bits=0, margin=margin, fpr=float(request['fpr']))


def capacity_from_ba(ba, n=100):
    p = 1.0 - max(0.5, min(1.0, float(ba)))
    h = 0.0 if p == 0 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return n * (1 - h)


def _threshold_key(tau):
    if isinstance(tau, bool) or not math.isfinite(float(tau)):
        raise ValueError('invalid unified BA threshold')
    count = round(float(tau) * 100)
    if not 0 <= count <= 101 or abs(float(tau) * 100 - count) > 1e-8:
        raise ValueError('threshold must be an exact 100-bit match count')
    return str(count)


def valid_cell(cell, order, n, cfg=None, alphas=REQUEST_ALPHAS):
    """Accept only fresh unified measurements; legacy ID/two-view cells are priors."""
    if not isinstance(cell, dict) or cell.get('protocol') != CELL_PROTOCOL:
        return False
    try:
        if type(n) is not int or n <= 0 or cell['n'] != n or cell['order'] != list(order) or not order:
            return False
        buckets = cell['by_threshold']
        if not isinstance(buckets, dict) or not buckets:
            return False
        if cfg is not None:
            from unified_detector import threshold_count
            if not {str(threshold_count(cfg, alpha)) for alpha in alphas}.issubset(buckets):
                return False
        for key, rec in buckets.items():
            theta = rec['threshold_count']
            if type(theta) is not int or str(theta) != key or not 0 <= theta <= 101:
                return False
            for name in ('detected', 'used_geometry'):
                if len(rec[name]) != n or any(type(v) is not bool for v in rec[name]):
                    return False
            arrays = [rec[name] for name in ('accepted_score', 'best_fragment_ba', 'fused_ba')]
            if set(rec['per_fragment_ba']) != set(order):
                return False
            arrays += [rec['per_fragment_ba'][f] for f in order]
            if any(len(vs) != n or any(isinstance(v, bool) or not math.isfinite(float(v))
                    or not 0 <= float(v) <= 1 for v in vs) for vs in arrays):
                return False
            if any(ok != (score >= theta / 100) for ok, score in zip(rec['detected'], rec['accepted_score'])):
                return False
            if any(abs(best - max(rec['per_fragment_ba'][f][i] for f in order)) > 1e-9
                   for i, best in enumerate(rec['best_fragment_ba'])):
                return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return True


def _bucket(cell, order, tau):
    if not valid_cell(cell, order, cell.get('n')):
        raise ValueError('incomplete or legacy detector cell; fresh unified measurement required')
    key = _threshold_key(tau)
    if key not in cell['by_threshold']:
        raise ValueError('unmeasured unified threshold ' + key)
    return cell['by_threshold'][key]


def selected_views(cell, order, tau):
    """The first accepted view at exactly this threshold; no keyed-ID bypass."""
    rec = _bucket(cell, order, tau)
    values = np.array([rec['per_fragment_ba'][f] for f in order], float)
    return (values, values >= tau, np.asarray(rec['detected'], bool),
            np.asarray(rec['used_geometry'], bool))


def combine(cell, order, tau, det_min=0.9):
    rec = _bucket(cell, order, tau)
    scores = np.asarray(rec['accepted_score'], float)
    best = np.asarray(rec['best_fragment_ba'], float)
    accepted = np.asarray(rec['detected'], bool)
    mean, rate = float(scores.mean()), float(accepted.mean())
    return dict(mean=mean, rate=rate, n=len(scores), threshold_count=int(_threshold_key(tau)),
                cascade_used=float(np.mean(rec['used_geometry'])),
                image_pass=accepted.tolist(), image_bit_accuracy=scores.tolist(),
                image_best_fragment_bit_accuracy=best.tolist(), mean_best_fragment_ba=float(best.mean()),
                passed_images=int(accepted.sum()), failed_images=int((~accepted).sum()),
                capacity_bits_estimate=capacity_from_ba(float(best.mean())),
                ok=bool(rate >= det_min - 1e-9 and mean >= tau - 1e-9))


def judge(request, candidate, cells, det_min=0.9):
    from unified_detector import threshold_for_config, budget_for_config
    solver_scenario(request)
    detector_cfg = dict(order=candidate['order'], fe_on=candidate['fe_on'])
    tau = threshold_for_config(detector_cfg, request['fpr'])
    result = dict(tau=tau, line=tau, threshold_count=int(_threshold_key(tau)),
                  max_tests=budget_for_config(detector_cfg)['max_tests'], protocol=PROTOCOL, attacks={}, pending=[])
    for attack in request['attacks']:
        cached = cells.get(attack)
        if (cached is None or not valid_cell(cached.get('cell'), candidate['order'], cached.get('cell', {}).get('n'))
                or _threshold_key(tau) not in cached['cell']['by_threshold']):
            result['pending'].append(attack)
            continue
        stats = combine(cached['cell'], candidate['order'], tau, det_min)
        stats['source'] = cached.get('source')
        result['attacks'][attack] = stats
    psnr = next((c['psnr_db'] for c in cells.values() if c is not None and c.get('psnr_db') is not None), None)
    result['psnr_live'] = psnr
    result['psnr_ok'] = psnr is not None and (request['min_psnr'] is None or psnr >= request['min_psnr'] - 1e-9)
    result['latency_model_ms'] = candidate['ms']
    result['latency_ok'] = candidate['ms'] <= request['max_ms'] + 1e-9
    result['cols_ok'] = not result['pending'] and all(v['ok'] for v in result['attacks'].values())
    result['pass'] = bool(result['cols_ok'] and result['psnr_ok'] and result['latency_ok'])
    result['capacity_bits_estimate'] = (None if result['pending'] else
        min(v['capacity_bits_estimate'] for v in result['attacks'].values()))
    result['capacity_bottleneck'] = (None if result['pending'] else
        min(result['attacks'], key=lambda a: result['attacks'][a]['capacity_bits_estimate']))
    result['capacity_basis'] = 'min over attacks of 100*(1-H2(1-mean best-fragment BA in the threshold-specific adopted view)); estimate only'
    return result


def walk_rows(rows, get_cells, det_min=0.9, identify=lambda c: None):
    for row in rows:
        row['walk'], row['deployed_rank'], row['open'] = [], None, False
        ranks = [c['rank'] for c in row['candidates']]
        assert ranks == sorted(set(ranks)), 'candidate ranks must be ordered and unique'
        for candidate in row['candidates']:
            verdict = judge(row, candidate, get_cells(candidate, row['attacks']), det_min)
            verdict.update(rank=candidate['rank'], psnr_table=candidate['psnr_db'], cfg_id=identify(candidate))
            row['walk'].append(verdict)
            if verdict['pending']:
                row['open'] = True
                break
            if verdict['pass']:
                row['deployed_rank'] = candidate['rank']
                break
    return rows


def needs_patch(row):
    return (bool(row['candidates']) and row['deployed_rank'] is None and not row['open']
            and bool(row.get('walk')) and all(not v['pending'] and not v['pass'] for v in row['walk'])
            and not any(c['rank'] >= 4 for c in row['candidates']))
