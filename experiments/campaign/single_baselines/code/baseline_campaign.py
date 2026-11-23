"""Fixed single-fragment rules, full live gate, then independent final.

BASELINE_RUN_ROOT selects the isolated method root. No function consumes the
main experiment's final results. A failed rule proposal is not solver UNSAT.
"""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from rigor_protocol import B, ROOT, MAIN, atomic, canonical_hash, digest, read
from final_holdout import (check_frozen, check_inputs, check_plan, check_splits,
    describe, detection_report, freeze_selection, normalize_config, plan_tasks,
    require, row_key, signed, write_once)
import final_measurement as FM

METHODS = ('S_VINE', 'S_TrustMark', 'S_VideoSeal')
TEMPLATE = B.parent / 'baselines'
FRAGMENT_BY_METHOD = dict(S_VINE='VINE', S_TrustMark='TrustMark', S_VideoSeal='VideoSeal')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_manifest():
    """Freeze controller and science before preparing any image-task manifest."""
    paths = sorted((B / 'code').glob('*.py')) + sorted((B / 'code').glob('*.sbatch'))
    paths += [MAIN / 'code' / name for name in
              ('final_measurement.py', 'final_holdout.py', 'rigor_protocol.py', 'unified_detector.py', 'unified_measurement.py')]
    paths += [B / 'policy_freeze.json']
    template_paths = [TEMPLATE / 'code' / name for name in ('final_measurement.py', 'final_holdout.py')]
    for path in template_paths:
        require(digest(path) == digest(B / 'code' / path.name), 'measurement semantics changed from audited template')
    paths += template_paths + [TEMPLATE / 'protocol_sources.json']
    # GPU smoke evidence is produced after source/task preparation and is
    # required separately at formal controller startup, never frozen as code.
    paths += [FM.CF / 'src' / name for name in ('vine_crypto_wrapper.py', 'trustmark_fragment.py',
              'videoseal_fragment.py', 'shortened_bch.py', 'payload.py')]
    return dict(schema='baseline_executed_sources_v1',
                adaptation='New isolated roots and fixed single-fragment policies; final measurement/holdout sources byte-identical to audited H1/H2 template',
                hashes={str(p.resolve()): digest(p) for p in paths})


def freeze_sources():
    return write_once(B / 'protocol_sources.json', source_manifest())


def verify_sources():
    value = read(B / 'protocol_sources.json')
    for path, sha in value['hashes'].items():
        require(digest(path) == sha, f'baseline executed source changed: {path}')
    return value


def verify_smoke_preflight():
    """Allow CPU planning first; require current adapter smoke before formal GPU work."""
    for root, schema in [(B, 'single_fragment_gpu_preflight_v1'), (TEMPLATE, 'baseline_gpu_preflight_v1')]:
        path = root / 'smoke/audit.json'
        require(path.exists(), 'formal baseline launch awaits GPU smoke audit: ' + str(path))
        audit = read(path)
        require(audit.get('schema') == schema and audit.get('preflight_passed') is True
                and audit.get('complete') is True and audit.get('formal_data') is False,
                'missing or invalid current-protocol smoke audit: ' + str(path))
        require(audit.get('source_manifest_sha256') == digest(root / 'protocol_sources.json'),
                'smoke audit belongs to different executed sources')
        plan = read(root / 'smoke/plans/smoke_tasks.json')
        require(audit.get('plan_sha256') == plan['plan_sha256'], 'smoke audit belongs to a different plan')
        for evidence in audit['evidence'].values():
            require(digest(evidence['path']) == evidence['sha256'], 'smoke evidence changed')
    return True


def four_inputs(request):
    r = dict(request)
    if 'min_psnr' in r:
        require('min_psnr_db' not in r, 'ambiguous PSNR input')
        r['min_psnr_db'] = r.pop('min_psnr')
    check_inputs(r)
    return r


def proposal_rows(path):
    value = read(path)
    require(value['method'] == ROOT.name, 'wrong policy method')
    evidence = dict(path=str(path.resolve()), sha256=digest(path),
                    kind='frozen fixed single-fragment proposal; not live acceptance')
    rows = []
    for row in value['rows']:
        group = row.get('group', row.get('class'))
        if '/' in group:
            group = group.split('/')[0]
        cfg = normalize_config(row.get('config', row.get('cfg')))
        fragment = FRAGMENT_BY_METHOD[ROOT.name]
        require(cfg['order'] == [fragment] and cfg['s'] == {fragment: 1.0} and cfg['fe_on'] == [],
                'single-fragment rule must use the declared fragment at strength 1 with no frontends')
        rows.append(dict(**{'class': group}, arm=str(row['arm']),
            request_id=row.get('request_index', row.get('request_id')),
            inputs=four_inputs(row.get('request', row.get('inputs'))),
            terminal_status='baseline_rule_proposal',
            selected=dict(cfg=cfg,
                source=evidence, selection_metrics={'not_yet_measured': True}),
            method=ROOT.name, policy_evidence=row.get('policy_evidence', {})))
    return rows


def prepare():
    require(ROOT.name in METHODS, 'formal prepare requires a baseline method')
    freeze_sources()
    verify_sources()
    proposal = ROOT / 'plans/proposals.json'
    require(proposal.exists(), 'policy proposals must be prepared first')
    policy_freeze = read(B / 'policy_freeze.json')
    expected = policy_freeze['methods'][ROOT.name]
    require(str(proposal.resolve()) == expected['path'] and digest(proposal) == expected['sha256'],
            'proposal differs from frozen policy')
    require(expected['rows'] == 20000, 'incomplete frozen policy population')
    oracle = policy_freeze.get('oracle')
    if oracle is not None:
        require(digest(oracle['path']) == oracle['sha256'], 'frozen policy information artifact changed')
    policy = policy_freeze.get('policy')
    require(policy is not None and digest(policy['path']) == policy['sha256'], 'fixed method specification changed')
    for path, sha in policy_freeze['source_hashes'].items():
        require(digest(path) == sha, f'frozen policy provenance changed: {path}')
    source = read(proposal).get('source_hashes', {})
    for path, sha in source.items():
        require(digest(path) == sha, f'policy source changed: {path}')
    splits = ROOT / 'plans/image_splits.json'
    write_once(splits, read(MAIN / 'plans/image_splits.json'))
    check_splits(splits)
    prereg = dict(schema='baseline_measurement_protocol_v1', method=ROOT.name,
        proposals_sha256=digest(proposal), no_final_feedback=True,
        rule_has_one_candidate=True, no_topk_or_patch_search=True,
        population=dict(original_requests=10000, arm_request_evaluations=20000,
                        arms_reuse_same_rule_and_measurement=True),
        image_roles=dict(selection='same frozen main selection manifest', final='same held-out benchmark manifest as main; reused across detector protocols', standard_n=100, unmarker_n=30),
        policy_information='Fixed individual library fragment; no coverage oracle or strength/subset optimization',
        fixed_configuration=dict(fragment=FRAGMENT_BY_METHOD[ROOT.name], strength=1.0, frontend=[]),
        readout='Current common composite/deployment reader with 100 coded bits; not a claim of publisher-native payload or reader',
        dual_frontend='Not used: all frontends disabled for every single-fragment proposal',
        selection_gate=dict(per_attack_live_rate=.9, live_mean_at_least_tau=True,
            actual_decoder_is_diagnostic=False, acceptance_protocol='unified_raw_ba_v1', psnr='mean full-100 embedding PSNR',
            latency='max requested-attack measured mean embed + actual decode; no joint-FE model substitution'),
        selection_difference_from_smt='Baseline uses measured latency; SMT uses its model latency at selection. Both use the same unified raw-BA decoder at selection and final; no keyed OR route remains.',
        latency=dict(required_gpu_model='NVIDIA A100-SXM4-80GB', batch=1, warmup=3,
            synchronized=True, actual_fpr=True, attacks_excluded=True,
            p95_is_not_budget_guarantee=True),
        capacity='Output BSC estimate; no input payload constraint',
        scope='Three separate individual current-library fragments at strength 1.0; no best-among-singles selection',
        source_manifest_sha256=digest(B / 'protocol_sources.json'))
    write_once(ROOT / 'plans/preregistration.json', prereg)
    path = ROOT / 'plans/selection_proposals.json'
    frozen = freeze_selection(proposal_rows(proposal), splits, path, mode='selection')
    plan = plan_tasks(path, ROOT / 'plans/selection_tasks.json')
    write_once(ROOT / 'selection/prepared.json', dict(complete=True,
        method=ROOT.name, proposals_sha256=digest(proposal),
        selection_sha256=frozen['selection_sha256'], tasks=len(plan['tasks'])))
    return dict(method=ROOT.name, stage='selection_prepared', tasks=len(plan['tasks']))


def validated_results(plan_path, cache_root):
    """Read every full result exactly once; scientific validation precedes use."""
    plan = read(plan_path)
    frozen = read(plan['frozen_path'])
    check_frozen(frozen)
    check_plan(plan, frozen)
    require(digest(plan['frozen_path']) == plan['frozen_file_sha256'], 'frozen selection changed')
    results = {}
    evidence = {}
    for task in plan['tasks']:
        p = Path(cache_root) / task['task_id'] / 'result.json'
        result = read(p)
        FM.validate_result(task, result)
        results[(task['cfg_sha256'], task['attack'])] = result
        evidence[task['task_id']] = dict(path=str(p.resolve()), sha256=digest(p))
    require(len(results) == len(plan['tasks']), 'duplicate exact task key')
    return frozen, plan, results, evidence


def quarantine_contents(folder, target):
    """Preserve the lock inode and all invalid artifacts for inspection."""
    target.mkdir(parents=True, exist_ok=False)
    for path in folder.iterdir():
        if path.name != 'lock':
            path.rename(target / path.name)


def inspect_cache(role, force=False):
    """Validate new results before scheduling completion; retry corrupt artifacts.

    Selection evidence is never repaired after final has begun, because changing
    those measurements could change a frozen choice. Such a problem is explicit
    and requires scientific review rather than a fabricated negative/pass.
    """
    verify_sources()
    require(role in ('selection', 'final'), 'invalid cache role')
    plan_path = ROOT / 'plans' / f'{role}_tasks.json'
    plan = read(plan_path)
    frozen = read(plan['frozen_path'])
    check_frozen(frozen)
    check_plan(plan, frozen)
    cache = ROOT / role / 'cache'
    state_path = ROOT / role / 'inspection.json'
    state = read(state_path) if state_path.exists() else dict(validated={}, invalid=[])
    for index, task in enumerate(plan['tasks']):
        path = cache / task['task_id'] / 'result.json'
        if not path.exists():
            state['validated'].pop(str(index), None)
            continue
        stat = path.stat()
        fingerprint = [stat.st_size, stat.st_mtime_ns]
        if not force and state['validated'].get(str(index)) == fingerprint:
            continue
        try:
            FM.validate_result(task, read(path))
        except Exception as error:
            # Never treat source/protocol mutation as a repairable measurement.
            FM.check_sources(task['measurement_source_hashes'])
            require(not (role == 'selection' and (ROOT / 'final/measurement_started.json').exists()),
                    'selection evidence corrupted after final began; frozen choice cannot be repaired')
            folder = path.parent
            with (folder / 'lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                target = ROOT / 'quarantine' / f'{role}_{task["task_id"]}_{time.time_ns()}'
                ctx = FM.context(plan_path, index, cache)
                try:
                    FM.verify_embedding(ctx)
                except Exception:
                    embedding = ctx['embed_folder']
                    embedding.mkdir(parents=True, exist_ok=True)
                    with (embedding / 'lock').open('a') as embed_lock:
                        fcntl.flock(embed_lock, fcntl.LOCK_EX)
                        quarantine_contents(embedding, target / 'invalid_embedding')
                quarantine_contents(folder, target / 'invalid_task')
                record = dict(timestamp=utc(), index=index, task_id=task['task_id'],
                              error=repr(error), quarantine=str(target), retry_required=True)
                atomic(target / 'reason.json', record)
                state['invalid'].append(record)
                state['validated'].pop(str(index), None)
        else:
            state['validated'][str(index)] = fingerprint
        state.update(timestamp=utc(), role=role, plan_sha256=plan['plan_sha256'])
        atomic(state_path, state)
    state.update(timestamp=utc(), role=role, plan_sha256=plan['plan_sha256'])
    atomic(state_path, state)
    return dict(role=role, validated=len(state['validated']), invalid=len(state['invalid']))


def request_metrics(request, results, metrics_cache=None, role='final'):
    """Same image gates as final main table; actual latency also gates selection.

    Complete diagnostic arrays live once in task result evidence. Per-request
    records reference those arrays to avoid copying them 20,000 times.
    """
    sys.path.insert(0, str(MAIN / 'fixed/code'))
    import capacity_protocol as CP
    sys.path.insert(0, str(MAIN / 'code'))
    import unified_detector as UD
    cfg = request['selected']['cfg']
    cfg_sha = request['selected']['cfg_sha256']
    inputs = request['inputs']
    tau = UD.threshold_for_config(cfg, inputs['fpr'])
    attacks = {}
    quality = None
    for attack in inputs['attacks']:
        result = results[(cfg_sha, attack)]
        key = (cfg_sha, attack, inputs['fpr'])
        stats = None if metrics_cache is None else metrics_cache.get(key)
        if stats is None:
            deployment = result['deployment'][repr(inputs['fpr'])]
            live = CP.combine(result['cell'], cfg['order'], tau)
            actual = detection_report(deployment['detected'])
            diagnostic = detection_report(live['image_pass'])
            times = [e + d for e, d in zip(result['embed_ms'], deployment['decode_ms'])]
            stats = dict(evidence=result['task_id'], n=result['n'],
                unmarker_limited_sample=attack == 'unmarker',
                live={k: v for k, v in live.items() if k not in ('image_pass', 'image_bit_accuracy')},
                live_detection=diagnostic, actual_decoder=actual,
                differing_detection_images=sum(a != b for a, b in
                    zip(live['image_pass'], deployment['detected'])),
                end_to_end_ms=describe(times), embedding_ms=describe(result['embed_ms']),
                decode_ms=describe(deployment['decode_ms']),
                deployment_fused_ba=describe(deployment['fused_ba']))
            if metrics_cache is not None:
                metrics_cache[key] = stats
        attacks[attack] = stats
        if quality is None:
            quality = result['embedding_quality']
        require(quality == result['embedding_quality'], 'different embeddings across requested attacks')
    latency = max(x['end_to_end_ms']['mean'] for x in attacks.values())
    psnr_ok = inputs['min_psnr_db'] is None or quality['mean'] >= inputs['min_psnr_db'] - 1e-9
    require(role in ('selection', 'final'), 'unknown gate image role')
    actual_decoder_ok = all(x['actual_decoder']['empirical_pass'] for x in attacks.values())
    detection_ok = all(x['live']['ok'] for x in attacks.values()) and actual_decoder_ok
    require(all(x['differing_detection_images'] == 0 for x in attacks.values()), 'unified live and decoder decisions differ')
    latency_ok = latency <= inputs['max_ms'] + 1e-9
    return dict(acceptance_protocol='unified_raw_ba_v1', tau=tau, attacks=attacks, embedding_psnr_db=quality,
        psnr_pass=psnr_ok, latency_pass=latency_ok,
        latency_max_attack_mean_ms=latency, empirical_detection_pass=detection_ok,
        actual_decoder_diagnostic_pass=actual_decoder_ok,
        population_lcb_pass=all(x['actual_decoder']['population_lcb_pass']
             and x['live_detection']['population_lcb_pass'] for x in attacks.values()),
        passed=bool(psnr_ok and detection_ok and latency_ok),
        capacity_bits_estimate=min(x['live']['capacity_bits_estimate'] for x in attacks.values()),
        capacity_label='BSC estimate, not demonstrated coded payload')


def compressed_row_metrics(metrics):
    return {**{k: v for k, v in metrics.items() if k != 'attacks'},
            'attacks': {a: dict(evidence=x['evidence'], live_pass=x['live']['ok'],
                        actual_decoder_pass=x['actual_decoder']['empirical_pass'])
                        for a, x in metrics['attacks'].items()}}


def select_and_freeze_final():
    verify_sources()
    frozen, plan, results, evidence = validated_results(
        ROOT / 'plans/selection_tasks.json', ROOT / 'selection/cache')
    require(frozen['mode'] == 'selection', 'selection requires selection images')
    decisions, metric_cache = [], {}
    for request in frozen['requests']:
        metrics = request_metrics(request, results, metric_cache, role='selection')
        decisions.append(dict(key=row_key(request), passed=metrics['passed'],
                              metrics=compressed_row_metrics(metrics)))
    decision_path = ROOT / 'selection/decisions.json'
    decision_doc = dict(schema='baseline_single_rule_live_decisions_v1', complete=True,
        method=ROOT.name, proposal_selection_sha256=frozen['selection_sha256'],
        plan_sha256=plan['plan_sha256'], rows=decisions, task_evidence=evidence,
        shared_attack_metrics={f'{c}/{a}/{f}': v for (c, a, f), v in metric_cache.items()},
        no_final_feedback=True, actual_latency_used=True,
        selected=sum(x['passed'] for x in decisions), total=len(decisions))
    write_once(decision_path, decision_doc)
    source = dict(path=str(decision_path.resolve()), sha256=digest(decision_path),
                  kind='single fixed rule complete selection-set live gate')
    terminal = []
    for request, decision in zip(frozen['requests'], decisions):
        require(row_key(request) == decision['key'], 'decision request ordering mismatch')
        row = {k: request[k] for k in ('class', 'arm', 'request_id', 'inputs')}
        row.update(method=ROOT.name, terminal_status='fixed_live_pass' if decision['passed']
                   else 'fixed_live_exhausted', source=source, selected=None,
                   failure_is_not_solver_unsat=True)
        if decision['passed']:
            row['selected'] = dict(cfg=request['selected']['cfg'], source=source,
                                  selection_metrics=decision['metrics'])
        terminal.append(row)
    final_path = ROOT / 'plans/final_selection.json'
    final = freeze_selection(terminal, ROOT / 'plans/image_splits.json', final_path)
    final_plan = plan_tasks(final_path, ROOT / 'plans/final_tasks.json')
    write_once(ROOT / 'selection/complete.json', dict(complete=True, selected=decision_doc['selected'],
        total=len(decisions), final_selection_sha256=final['selection_sha256'],
        final_plan_sha256=final_plan['plan_sha256'], final_tasks=len(final_plan['tasks'])))
    return dict(method=ROOT.name, stage='final_prepared', selected=decision_doc['selected'],
                tasks=len(final_plan['tasks']))


def audit():
    verify_sources()
    frozen, plan, results, evidence = validated_results(
        ROOT / 'plans/final_tasks.json', ROOT / 'final/cache')
    require(frozen['mode'] == 'final', 'final audit requires final images')
    # Re-evaluate selection cells and ensure all selection choices are immutable.
    select_and_freeze_final()
    rows, metric_cache = [], {}
    for request in frozen['requests']:
        row = dict(key=row_key(request), inputs=request['inputs'],
            terminal_status=request['terminal_status'], final_feedback_used=False)
        if request['selected'] is None:
            row.update(final_status='not_deployed', final_pass=None, cfg_sha256=None)
        else:
            metrics = request_metrics(request, results, metric_cache)
            row.update(final_status='measured', final_pass=metrics['passed'],
                cfg_sha256=request['selected']['cfg_sha256'], **compressed_row_metrics(metrics))
        rows.append(row)
    report = dict(schema='baseline_independent_final_v1', passed=True, complete=True,
        method=ROOT.name, original_requests=10000, arm_request_evaluations=len(rows),
        two_arms_are_paired_duplicates_for_rule=True, rows=rows,
        selection_sha256=frozen['selection_sha256'], plan_sha256=plan['plan_sha256'],
        task_evidence=evidence,
        shared_attack_metrics={f'{c}/{a}/{f}': v for (c, a, f), v in metric_cache.items()},
        deployed_requests=sum(x['final_status'] == 'measured' for x in rows),
        final_pass_requests=sum(x['final_pass'] is True for x in rows),
        no_reselection=True, confidence_is_simultaneous=False, tail_latency_guarantee=False,
        capacity_label='BSC estimate, not demonstrated coded payload',
        selection_difference_from_smt='Baseline measured latency; SMT modeled latency. Same CP.combine selection detection; final uses live plus actual decoder and actual latency for both.',
        baseline_policy_scope='Fixed individual current-library fragment, strength 1.0, no frontend, common 100-coded-bit deployment readout; not publisher-native payload')
    path = ROOT / 'final/report.json'
    write_once(path, report)
    audit = dict(schema='baseline_pipeline_audit_v1', passed=True, complete=True,
        method=ROOT.name, terminal_requests=20000, no_final_feedback=True,
        selection_sha256=frozen['selection_sha256'], final_tasks=len(plan['tasks']),
        final_pass_requests=report['final_pass_requests'],
        evidence={str(path.resolve()): digest(path),
                  str(ROOT / 'selection/decisions.json'): digest(ROOT / 'selection/decisions.json'),
                  str(B / 'protocol_sources.json'): digest(B / 'protocol_sources.json')})
    write_once(ROOT / 'pipeline_audit.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'select', 'audit', 'inspect'))
    parser.add_argument('--role', choices=('selection', 'final'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    value = (inspect_cache(args.role, args.force) if args.command == 'inspect' else
             {'prepare': prepare, 'select': select_and_freeze_final, 'audit': audit}[args.command]())
    print(json.dumps(value))


if __name__ == '__main__':
    main()
