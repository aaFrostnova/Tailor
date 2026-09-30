"""Immutable deployment choices and an independent, non-adaptive final evaluation.

``passed`` on the aggregate is an evidence/completeness check, not a claim that
every deployed request passed its final test. Scientific failures remain rows.
"""
import copy
import json
import math
import os
from pathlib import Path
import tempfile

from rigor_protocol import ROOT, canonical_hash, digest, read, validate_splits

FRAGMENTS = {'VINE', 'TrustMark', 'VideoSeal'}
FRONTENDS = {'resync', 'scale', 'angle', 'tile'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_once(path, value):
    """An identical retry is allowed; a changed frozen choice is never allowed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, f'immutable artifact differs: {path}')
        return value
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write('\n')
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(read(path) == value, f'immutable artifact differs: {path}')
    finally:
        os.unlink(temporary)
    return value


def signed(value, field):
    result = copy.deepcopy(value)
    result[field] = canonical_hash(value)
    return result


def check_signed(value, field):
    require(value.get(field) == canonical_hash({k: v for k, v in value.items() if k != field}),
            f'{field} mismatch')


def check_splits(path, verify_files=False):
    value = read(path)
    validate_splits(value, verify_files=verify_files)
    # Explicit final-role barrier even when historical development roles overlap.
    final = value['splits']['final']['images']
    others = [r for role, split in value['splits'].items() if role != 'final' for r in split['images']]
    for field in ('path', 'sha256', 'pixel_sha256'):
        a = [str(Path(r[field]).resolve()) if field == 'path' else r[field] for r in final]
        b = {str(Path(r[field]).resolve()) if field == 'path' else r[field] for r in others}
        require(len(set(a)) == len(a) and not set(a) & b, f'final image overlap: {field}')
    return value


def normalize_config(cfg):
    order = list(cfg['order'])
    require(order and len(set(order)) == len(order) and set(order) <= FRAGMENTS, 'invalid fragment order')
    require(set(cfg.get('S', order)) == set(order), 'configuration fragment set differs from order')
    strengths = {f: float(cfg['s'][f]) for f in order}
    ranges = {'VINE': (.14, 1), 'TrustMark': (.4, 2), 'VideoSeal': (.5, 3.2)}
    for f, x in strengths.items():
        require(math.isfinite(x) and ranges[f][0] <= x <= ranges[f][1], f'invalid strength: {f}')
    fe = cfg.get('fe_on')
    if fe is None:
        fe = [f for f, enabled in cfg.get('fe', {}).items() if enabled]
    require(len(set(fe)) == len(fe) and set(fe) <= FRONTENDS, 'unknown frontend')
    return dict(S=sorted(order), order=order, s=strengths, fe_on=sorted(fe))


def check_inputs(inputs):
    require(set(inputs) == {'attacks', 'fpr', 'min_psnr_db', 'max_ms'}, 'exactly four request inputs required')
    attacks = inputs['attacks']
    require(isinstance(attacks, list) and attacks and len(set(attacks)) == len(attacks), 'invalid attack set')
    require(all(isinstance(a, str) and a for a in attacks), 'invalid attack name')
    require('nfpa_sd21_xy40_s10_v1' not in attacks, 'NFPA is excluded from the editing-only revision')
    require(math.isfinite(inputs['fpr']) and 0 < inputs['fpr'] < 1, 'invalid FPR')
    require(math.isfinite(inputs['max_ms']) and inputs['max_ms'] > 0, 'invalid latency budget')
    require(inputs['min_psnr_db'] is None or math.isfinite(inputs['min_psnr_db']), 'invalid PSNR floor')


def row_key(row):
    return f"{row['class']}/{row['arm']}/{row['request_id']}"


def freeze_selection(selection_rows, split_manifest_path, output_path, mode='final', smoke_n=2):
    """Freeze all terminal choices before any final image is decoded or embedded.

    Rows: class, arm, request_id, four ``inputs``, terminal_status, selected.
    selected is null or {cfg, source:{path,sha256,...}, selection_metrics:{...}}.
    Empty selections need terminal evidence too, in row.source. No unknown or
    pending solver state is accepted as a terminal negative result.
    """
    require(mode in ('final', 'smoke'), 'invalid image role')
    splits = check_splits(split_manifest_path)
    rows = copy.deepcopy(selection_rows)
    require(rows, 'empty selection')
    seen, counts = set(), {}
    hashes = {}
    for row in rows:
        require(row['class'] in {'C1', 'C2', 'C3', 'C4', 'C5'} and row['arm'] in {'04', '06'}, 'invalid group')
        require(isinstance(row['request_id'], int) and 0 <= row['request_id'] < 2000, 'invalid request id')
        key = row_key(row)
        require(key not in seen, 'duplicate request')
        seen.add(key)
        group = f"{row['class']}/{row['arm']}"
        counts[group] = counts.get(group, 0) + 1
        check_inputs(row['inputs'])
        test_fixture = mode == 'smoke' and row['terminal_status'] == 'test_only_fixed_configuration'
        require(test_fixture or row['terminal_status'] in {'fixed_live_pass', 'fallback_live_pass', 'fixed_live_exhausted',
                'fallback_live_exhausted', 'zero_margin_model_unsat'}, 'nonterminal selection')
        selected = row.get('selected')
        require((selected is not None) == (test_fixture or row['terminal_status'].endswith('live_pass')), 'terminal status and choice disagree')
        source = selected['source'] if selected is not None else row.get('source')
        require(isinstance(source, dict) and 'path' in source and 'sha256' in source, 'missing terminal evidence')
        p = str(Path(source['path']).resolve())
        if p not in hashes:
            hashes[p] = digest(p)
        require(hashes[p] == source['sha256'], 'terminal evidence hash mismatch')
        if selected is not None:
            selected['cfg'] = normalize_config(selected['cfg'])
            selected['cfg_sha256'] = canonical_hash(selected['cfg'])
    if mode == 'final':
        require(len(rows) == 20000 and len(counts) == 10 and set(counts.values()) == {2000}, 'final requires all 20k terminal requests')
    require(isinstance(smoke_n, int) and 1 <= smoke_n <= 100, 'invalid smoke image count')
    rows.sort(key=lambda r: (r['class'], r['arm'], r['request_id']))
    value = signed(dict(schema='rigor_frozen_selection_v1', mode=mode,
        split_manifest=str(Path(split_manifest_path).resolve()), split_manifest_sha256=digest(split_manifest_path),
        images_sha256=splits['splits'][mode]['images_sha256'], n=100 if mode == 'final' else smoke_n,
        no_final_feedback=True, requests=rows, group_counts=counts,
        acceptance_protocol='unified_raw_ba_v1', original_requests=10000 if mode == 'final' else len({(r['class'], r['request_id']) for r in rows}),
        arm_request_evaluations=len(rows)), 'selection_sha256')
    guard = Path(split_manifest_path).resolve().parent.parent / 'final/measurement_started.json'
    if mode == 'final' and guard.exists():
        require(read(guard)['selection_sha256'] == value['selection_sha256'], 'final images already opened for a different selection')
    return write_once(output_path, value)


def check_frozen(frozen):
    check_signed(frozen, 'selection_sha256')
    require(frozen.get('acceptance_protocol') == 'unified_raw_ba_v1', 'legacy frozen selection cannot be deployed')
    require(frozen['schema'] == 'rigor_frozen_selection_v1' and frozen['no_final_feedback'] is True, 'unfrozen selection')
    require(digest(frozen['split_manifest']) == frozen['split_manifest_sha256'], 'split manifest changed')
    splits = check_splits(frozen['split_manifest'])
    require(frozen['images_sha256'] == splits['splits'][frozen['mode']]['images_sha256'], 'image role changed')
    keys, counts = set(), {}
    for row in frozen['requests']:
        require(row['class'] in {'C1', 'C2', 'C3', 'C4', 'C5'} and row['arm'] in {'04', '06'}, 'invalid frozen group')
        require(isinstance(row['request_id'], int) and 0 <= row['request_id'] < 2000, 'invalid frozen request id')
        require(row_key(row) not in keys, 'duplicate frozen request')
        keys.add(row_key(row))
        group = f"{row['class']}/{row['arm']}"
        counts[group] = counts.get(group, 0) + 1
        check_inputs(row['inputs'])
        if row.get('selected') is not None:
            cfg = row['selected']['cfg']
            require(cfg == normalize_config(cfg), 'frozen configuration is not canonical')
            require(row['selected']['cfg_sha256'] == canonical_hash(cfg), 'frozen configuration hash mismatch')
    require(counts == frozen['group_counts'] and len(keys) == frozen['arm_request_evaluations'], 'frozen coverage mismatch')
    if frozen['mode'] == 'final':
        require(frozen['n'] == 100 and frozen['original_requests'] == 10000 and len(keys) == 20000
                and len(counts) == 10 and set(counts.values()) == {2000}, 'incomplete final request population')
    return splits


def check_plan(plan, frozen):
    """Every requested cfg/attack/FPR appears exactly once, with its exact members."""
    check_signed(plan, 'plan_sha256')
    require(plan['schema'] == 'rigor_final_plan_v1' and plan['selection_sha256'] == frozen['selection_sha256'], 'plan selection mismatch')
    expected = {}
    for row in frozen['requests']:
        selected = row.get('selected')
        if selected is None:
            continue
        for attack in row['inputs']['attacks']:
            key = selected['cfg_sha256'], attack
            item = expected.setdefault(key, dict(cfg=selected['cfg'], fprs=set(), requests=set()))
            item['fprs'].add(row['inputs']['fpr'])
            item['requests'].add(row_key(row))
    actual = set()
    for task in plan['tasks']:
        check_signed(task, 'task_id')
        key = task['cfg_sha256'], task['attack']
        require(key in expected and key not in actual, 'unexpected or duplicate final task')
        actual.add(key)
        target = expected[key]
        require(task['cfg'] == target['cfg'], 'task changed exact deployment configuration')
        require(task['fprs'] == sorted(target['fprs']), 'task FPR coverage mismatch')
        require(len(task['requests']) == len(target['requests']) and set(task['requests']) == target['requests'], 'task request coverage mismatch')
        require(task['mode'] == frozen['mode'] and task['selection_sha256'] == frozen['selection_sha256']
                and task['images_sha256'] == frozen['images_sha256'], 'task image role mismatch')
        require(task['n'] == (min(30, frozen['n']) if task['attack'] == 'unmarker' else frozen['n']), 'wrong final sample size')
        require(task['warmup'] == 3, 'warmup protocol changed')
        require(task['required_gpu_model'] == 'NVIDIA A100-SXM4-80GB', 'wrong preregistered GPU model')
    require(actual == set(expected), 'missing final tasks')
    return True


def plan_tasks(frozen_path, output_path, specifications=None, measurement_sources=None):
    """One cfg/attack task, with all request FPRs; no rounded-strength cache keys."""
    frozen = read(frozen_path)
    check_frozen(frozen)
    from final_measurement import attack_specification, source_fingerprints
    if specifications is None:
        names = {a for row in frozen['requests'] if row.get('selected') for a in row['inputs']['attacks']}
        specifications = {a: attack_specification(a) for a in sorted(names)}
    if measurement_sources is None:
        measurement_sources = source_fingerprints()
    prereg_path = ROOT / 'plans/preregistration.json'
    gpu_model = read(prereg_path)['latency']['required_gpu_model'] if prereg_path.exists() else 'NVIDIA A100-SXM4-80GB'
    groups = {}
    for row in frozen['requests']:
        selected = row.get('selected')
        if selected is None:
            continue
        for attack in row['inputs']['attacks']:
            require(attack in specifications, f'unknown attack: {attack}')
            key = (selected['cfg_sha256'], attack)
            item = groups.setdefault(key, dict(cfg=selected['cfg'], cfg_sha256=key[0], attack=attack,
                attack_spec=specifications[attack], fprs=[], requests=[]))
            item['fprs'].append(row['inputs']['fpr'])
            item['requests'].append(row_key(row))
    tasks = []
    for _, task in sorted(groups.items()):
        task['fprs'] = sorted(set(task['fprs']))
        task.update(n=min(30, frozen['n']) if task['attack'] == 'unmarker' else frozen['n'],
                    mode=frozen['mode'], selection_sha256=frozen['selection_sha256'],
                    images_sha256=frozen['images_sha256'], warmup=3,
                    measurement_source_hashes=measurement_sources, required_gpu_model=gpu_model)
        tasks.append(signed(task, 'task_id'))
    plan = signed(dict(schema='rigor_final_plan_v1', frozen_path=str(Path(frozen_path).resolve()),
        frozen_file_sha256=digest(frozen_path), selection_sha256=frozen['selection_sha256'], tasks=tasks), 'plan_sha256')
    check_plan(plan, frozen)
    return write_once(output_path, plan)


def clopper_pearson_lower(k, n, confidence=.95):
    require(isinstance(k, int) and isinstance(n, int) and 0 <= k <= n and n > 0, 'invalid binomial sample')
    require(0 < confidence < 1, 'invalid confidence level')
    if k == 0:
        return 0.0
    from scipy.stats import beta
    return float(beta.ppf(1 - confidence, k, n - k + 1))


def describe(values):
    import numpy as np
    from scipy.stats import t
    x = np.asarray(values, float)
    require(x.ndim == 1 and len(x) > 0 and np.isfinite(x).all(), 'invalid measurement vector')
    mean = float(x.mean())
    half = float(t.ppf(.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None
    return dict(n=len(x), mean=mean, p50=float(np.quantile(x, .5)), p95=float(np.quantile(x, .95, method='higher')),
                maximum=float(x.max()), mean_ci95=None if half is None else [mean - half, mean + half],
                mean_ci_method='two-sided Student t interval; approximate, not distribution-free')


def detection_report(flags):
    require(flags and all(isinstance(x, bool) for x in flags), 'invalid detection flags')
    k, n = sum(flags), len(flags)
    lower = clopper_pearson_lower(k, n)
    return dict(n=n, passed_images=k, empirical_rate=k / n, empirical_floor=.9,
                empirical_pass=k / n >= .9 - 1e-12, binomial_one_sided_95_lower=lower,
                population_lcb_pass=lower >= .9, confidence_scope='one attack and one fixed configuration; not simultaneous')


def summarize(frozen_path, plan_path, cache_root, output_path=None):
    """Validate every scheduled measurement; report failures without new choices."""
    import sys
    from final_measurement import validate_result
    frozen, plan = read(frozen_path), read(plan_path)
    check_frozen(frozen)
    check_plan(plan, frozen)
    require(plan['frozen_file_sha256'] == digest(frozen_path) and plan['selection_sha256'] == frozen['selection_sha256'], 'plan selection changed')
    fixed_code = ROOT / 'fixed/code'
    sys.path.insert(0, str(fixed_code))
    import capacity_protocol as CP
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import unified_detector as UD
    task_results = {}
    for task in plan['tasks']:
        result = read(Path(cache_root) / task['task_id'] / 'result.json')
        validate_result(task, result)
        task_results[(task['cfg_sha256'], task['attack'])] = result
    require(len(task_results) == len(plan['tasks']), 'duplicate cfg/attack tasks')
    if task_results:
        require(len({r['metadata']['gpu_model'] for r in task_results.values()}) == 1, 'mixed final timing GPU models')
    rows = []
    for request in frozen['requests']:
        selected = request.get('selected')
        row = dict(key=row_key(request), inputs=request['inputs'], terminal_status=request['terminal_status'],
                   cfg_sha256=None if selected is None else selected['cfg_sha256'], final_feedback_used=False)
        if selected is None:
            row.update(final_status='not_deployed', final_pass=None, reason=request['terminal_status'])
            rows.append(row)
            continue
        cfg, inputs = selected['cfg'], request['inputs']
        tau = UD.threshold_for_config(cfg, inputs['fpr'])
        attacks, quality = {}, None
        for attack in inputs['attacks']:
            result = task_results[(selected['cfg_sha256'], attack)]
            deployment = result['deployment'][repr(inputs['fpr'])]
            live = CP.combine(result['cell'], cfg['order'], tau)
            actual = detection_report(deployment['detected'])
            diagnostic = detection_report(live['image_pass'])
            end_to_end = [e + d for e, d in zip(result['embed_ms'], deployment['decode_ms'])]
            attacks[attack] = dict(n=result['n'], unmarker_limited_sample=attack == 'unmarker',
                live=live, live_detection=diagnostic, actual_decoder=actual,
                differing_detection_images=sum(a != b for a, b in zip(live['image_pass'], deployment['detected'])),
                deployment_fused_ba=describe(deployment['fused_ba']), live_acceptance_score=describe(live['image_bit_accuracy']),
                capacity_best_fragment_ba=describe(live['image_best_fragment_bit_accuracy']),
                latency_composition=result.get('latency_composition', {'decode_remeasured': True, 'embedding_reused': False}),
                embedding_ms=describe(result['embed_ms']), decode_ms=describe(deployment['decode_ms']),
                end_to_end_ms=describe(end_to_end), evidence=result['task_id'])
            # All attacks reference the same full-100-image embedding measurement.
            if quality is None:
                quality = result['embedding_quality']
            require(quality == result['embedding_quality'], 'embedding quality differs across attack caches')
        latency = max(a['end_to_end_ms']['mean'] for a in attacks.values())
        psnr_ok = inputs['min_psnr_db'] is None or quality['mean'] >= inputs['min_psnr_db'] - 1e-9
        empirical = all(a['live']['ok'] and a['actual_decoder']['empirical_pass'] for a in attacks.values())
        row.update(final_status='measured', attacks=attacks, tau=tau, embedding_psnr_db=quality,
                   psnr_pass=psnr_ok, latency_max_attack_mean_ms=latency,
                   latency_pass=latency <= inputs['max_ms'] + 1e-9, empirical_detection_pass=empirical,
                   population_lcb_pass=all(a['actual_decoder']['population_lcb_pass'] and a['live_detection']['population_lcb_pass'] for a in attacks.values()),
                   final_pass=bool(psnr_ok and empirical and latency <= inputs['max_ms'] + 1e-9),
                   capacity_bits_estimate=min(a['live']['capacity_bits_estimate'] for a in attacks.values()),
                   capacity_label='BSC estimate, not demonstrated coded payload')
        rows.append(row)
    report = dict(schema='rigor_final_holdout_v1', passed=True, complete=True,
        original_requests=frozen['original_requests'], arm_request_evaluations=len(rows),
        selection_sha256=frozen['selection_sha256'], plan_sha256=plan['plan_sha256'],
        acceptance_protocol='unified_raw_ba_v1', rows=rows, deployed_requests=sum(r['final_status'] == 'measured' for r in rows),
        final_pass_requests=sum(r['final_pass'] is True for r in rows),
        all_deployed_requests_passed=all(r['final_pass'] is not False for r in rows),
        no_reselection=True, latency_budget_statistic='maximum requested-attack mean embedding plus actual decoder ms',
        tail_latency_guarantee=False, confidence_is_simultaneous=False)
    if output_path:
        write_once(output_path, report)
    return report
