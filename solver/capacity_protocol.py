"""Four-input requests, per-image live checks, and capacity reported after selection.

Capacity is the existing 100-bit BSC model evaluated on measured bit accuracy.
It is an estimate, not a test of an alternative payload length or ECC decoder.
"""
import math
import numpy as np

PROTOCOL = 'four_inputs_capacity_output_v1'
REQUEST_FIELDS = ('attacks', 'fpr', 'min_psnr', 'max_ms')


def solver_scenario(request, margin=0.02):
    import watermark_smt_v2 as W
    if 'min_bits' in request:
        raise ValueError('min_bits is not an input in the capacity-output protocol')
    assert request['attacks'] and len(set(request['attacks'])) == len(request['attacks'])
    assert 0 < request['fpr'] < 1 and request['max_ms'] > 0
    return dict(attacks=request['attacks'], min_ba=W.beta_from_fpr(request['fpr']),
                min_psnr=request['min_psnr'], max_ms=request['max_ms'],
                allow_resync=True, allow_nested=True, resolution=512,
                min_bits=0, margin=margin)


def capacity_from_ba(ba, n=100):
    p = 1.0 - max(0.5, min(1.0, float(ba)))
    h = 0.0 if p == 0 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return n * (1 - h)


def valid_cell(cell, order, n):
    if not isinstance(cell, dict) or cell.get('views') != 2:
        return False
    try:
        for name in ('det', 'idv', 'cas_ok'):
            if len(cell[name]) != n:
                return False
        for name in ('ba', 'ver', 'cas_ba', 'cas_ver'):
            if not order or not set(order).issubset(cell[name]):
                return False
            for f in order:
                if len(cell[name][f]) != n:
                    return False
                if name in ('ba', 'cas_ba'):
                    for i, value in enumerate(cell[name][f]):
                        if value is None and name == 'cas_ba' and not cell['cas_ok'][i]:
                            continue
                        if value is None or not math.isfinite(float(value)) or not 0 <= value <= 1:
                            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def selected_views(cell, order, tau):
    """Match the deployed primary/cascade choice at THIS request's threshold."""
    n = len(cell['det'])
    assert valid_cell(cell, order, n), 'incomplete per-image cell'
    primary = np.array([cell['ba'][f] for f in order], float)
    bp = primary.max(axis=0)
    prim_accept = np.asarray(cell['idv'], bool) | (bp >= tau - 1e-12)
    cascade = np.array([[(-1 if v is None else v) for v in cell['cas_ba'][f]] for f in order], float)
    use_cascade = np.asarray(cell['cas_ok'], bool) & ~prim_accept
    values = np.where(use_cascade[None, :], cascade, primary)
    primary_ver = np.array([cell['ver'][f] for f in order], bool)
    cascade_ver = np.array([cell['cas_ver'][f] for f in order], bool)
    verified = np.where(use_cascade[None, :], cascade_ver, primary_ver)
    # cas_ok is the deployed cascade's successful keyed verification flag.
    accepted = prim_accept | np.asarray(cell['cas_ok'], bool)
    return values, verified, accepted, use_cascade


def combine(cell, order, tau, det_min=0.9):
    values, _, accepted, cascade = selected_views(cell, order, tau)
    ba = values.max(axis=0)
    mean, rate = float(ba.mean()), float(accepted.mean())
    return dict(mean=mean, rate=rate, n=len(ba), cascade_used=float(cascade.mean()),
                image_pass=accepted.tolist(), image_bit_accuracy=ba.tolist(),
                passed_images=int(accepted.sum()), failed_images=int((~accepted).sum()),
                capacity_bits_estimate=capacity_from_ba(mean),
                ok=bool(rate >= det_min - 1e-9 and mean >= tau - 1e-9))


def judge(request, candidate, cells, det_min=0.9):
    import watermark_smt_v2 as W
    scen = solver_scenario(request)
    tau = W.presence_threshold(scen['min_ba'], len(candidate['order']))
    result = dict(tau=tau, line=tau, attacks={}, pending=[])
    for attack in request['attacks']:
        cached = cells.get(attack)
        if cached is None:
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
    result['capacity_basis'] = 'min over attacks of 100*(1-H2(1-mean deployed best-path bit accuracy)); estimate only'
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
