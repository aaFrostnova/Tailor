"""Check seven H1/H2 adapter cases without requiring scientific detection success."""
import json

from rigor_protocol import B, ROOT, digest, read, atomic
from baseline_campaign import utc, verify_sources
from final_holdout import check_frozen, check_plan, require, write_once
import final_measurement as FM


def main():
    require(ROOT.name == 'smoke', 'use BASELINE_RUN_ROOT=B/smoke')
    verify_sources()
    plan_path = ROOT / 'plans/smoke_tasks.json'
    plan = read(plan_path)
    frozen = read(plan['frozen_path'])
    check_frozen(frozen)
    check_plan(plan, frozen)
    require(frozen['mode'] == 'smoke' and frozen['n'] == 2 and len(plan['tasks']) == 7,
            'wrong smoke size/role')
    expected = {'single_fragment','resync','scale','dual_frontend','editing','ctrlregen','unmarker'}
    require(len(frozen['requests']) == 7 and {x['smoke_case'] for x in frozen['requests']} == expected,
            'incomplete baseline adapter smoke coverage')
    evidence, missing, failures = {}, [], []
    for index, task in enumerate(plan['tasks']):
        require(task['fprs'] == [.01], 'unexpected smoke request FPR')
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
                                        total=7, missing=missing, failures=failures))
    if complete:
        write_once(ROOT / 'audit.json', dict(schema='baseline_gpu_preflight_v1',
            preflight_passed=True, complete=True, cases=7, request_fpr_records=7,
            images_per_task=2, formal_data=False, scientific_detection_success_required=False,
            source_manifest_sha256=digest(B / 'protocol_sources.json'),
            plan_sha256=plan['plan_sha256'], evidence=evidence,
            readout='current common 100-coded-bit deployment reader, not publisher-native payload'))
    print(json.dumps(dict(complete=complete, completed=len(evidence), total=7, missing=missing, failures=failures)))


if __name__ == '__main__':
    main()
