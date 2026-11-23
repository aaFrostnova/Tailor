"""Check six fixed-single smoke tasks without requiring scientific success."""
import json

from rigor_protocol import B, ROOT, digest, read, atomic
from baseline_campaign import FRAGMENT_BY_METHOD, METHODS, utc, verify_sources
from final_holdout import check_frozen, check_plan, require, write_once
import final_measurement as FM


def main():
    require(ROOT.name == 'smoke', 'use BASELINE_RUN_ROOT=SB/smoke')
    verify_sources()
    plan_path = ROOT / 'plans/smoke_tasks.json'
    plan = read(plan_path)
    frozen = read(plan['frozen_path'])
    check_frozen(frozen)
    check_plan(plan, frozen)
    require(frozen['mode'] == 'smoke' and frozen['n'] == 2 and len(plan['tasks']) == 6,
            'wrong smoke size/role')
    expected = {f'{method}/{attack}' for method in METHODS for attack in ('jpeg25', 'editing_ip2p_s20_v1')}
    require(len(frozen['requests']) == 36 and {x['smoke_case'] for x in frozen['requests']} == expected,
            'incomplete fragment/attack/FPR smoke coverage')
    evidence, missing, failures = {}, [], []
    for index, task in enumerate(plan['tasks']):
        require(len(task['fprs']) == 6, 'missing request FPR values in smoke task')
        require(len(task['cfg']['order']) == 1 and not task['cfg']['fe_on']
                and set(task['cfg']['s'].values()) == {1.0}, 'wrong single-fragment config')
        path = ROOT / 'smoke/cache' / task['task_id'] / 'result.json'
        if not path.exists():
            missing.append(index)
            continue
        try:
            result = read(path)
            FM.validate_result(task, result)
        except Exception as error:
            failures.append(dict(index=index, error=repr(error)))
        else:
            evidence[str(index)] = dict(path=str(path), sha256=digest(path), attack=task['attack'],
                cfg=task['cfg'], n=task['n'], fprs=task['fprs'], gpu=result['metadata']['gpu_model'])
    complete = not missing and not failures
    atomic(ROOT / 'progress.json', dict(timestamp=utc(), complete=complete, completed=len(evidence),
                                        total=6, missing=missing, failures=failures))
    if complete:
        write_once(ROOT / 'audit.json', dict(schema='single_fragment_gpu_preflight_v1',
            preflight_passed=True, complete=True, cases=6, request_fpr_records=36,
            images_per_task=2, formal_data=False, scientific_detection_success_required=False,
            source_manifest_sha256=digest(B / 'protocol_sources.json'),
            plan_sha256=plan['plan_sha256'], evidence=evidence,
            readout='current common 100-coded-bit deployment reader, not publisher-native payload'))
    print(json.dumps(dict(complete=complete, completed=len(evidence), total=6, missing=missing, failures=failures)))


if __name__ == '__main__':
    main()
