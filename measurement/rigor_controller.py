"""Resume corrected selection, independent model assessment and final testing."""
import datetime
import fcntl
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rigor_protocol import ROOT, SOURCE, PY, atomic, digest, read
from prepare_campaign import verify_frozen


def progress(phase, stream='main', **values):
    record = dict(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
    atomic(ROOT / ('progress.json' if stream == 'main' else f'{stream}_progress.json'), record)
    print('RIGOR_PROGRESS', stream, json.dumps(record), flush=True)


def active_jobs():
    rows = subprocess.check_output(['squeue', '-u', '3529', '-h', '-r', '-o', '%i'], text=True).splitlines()
    return {row.split('_')[0] for row in rows}


def wait_job(job):
    while str(job) in active_jobs():
        time.sleep(30)


def wait_recorded(paths):
    recorded = set()
    for path in paths:
        if Path(path).exists():
            recorded.update(str(json.loads(row)['job']) for row in Path(path).read_text().splitlines() if row.strip())
    for job in sorted(recorded & active_jobs()):
        print('WAIT_EXISTING_ARRAY', job, flush=True)
        wait_job(job)


def index_spec(indices):
    parts = []
    start = end = indices[0]
    for value in indices[1:]:
        if value == end + 1:
            end = value
        else:
            parts.append(str(start) if start == end else f'{start}-{end}')
            start = end = value
    parts.append(str(start) if start == end else f'{start}-{end}')
    return ','.join(parts)


def missing_tasks(mode, plan_path):
    assert mode in ('final','combination')
    if mode == 'combination':
        from combination_evaluation import check_observation
        plan=read(plan_path)
        missing=[]
        for i,task in enumerate(plan['tasks']):
            path=ROOT/'combination'/plan['plan_id']/'raw'/f"{task['id']}.json"
            if path.exists():
                check_observation(plan,task,read(path))
            else:
                missing.append(i)
        return missing
    from final_measurement import validate_result
    missing = []
    for i, task in enumerate(read(plan_path)['tasks']):
        assert task['attack'] != 'nfpa_sd21_xy40_s10_v1'
        p = ROOT / 'final/cache' / task['task_id'] / 'result.json'
        if p.exists():
            validate_result(task, read(p))
        else:
            missing.append(i)
    return missing


def run_gpu_tasks(mode, plan_path):
    assert mode in ('final','combination')
    jobs_path = ROOT / f'{mode}_jobs.jsonl'
    wait_recorded([jobs_path])
    module = 'final_measurement.py' if mode == 'final' else 'combination_measurement.py'
    for attempt in range(3):
        verify_frozen()
        missing = missing_tasks(mode, plan_path)
        if not missing:
            return
        # Bound submitted job count and array text; one dedicated GPU per task.
        for start in range(0, len(missing), 400):
            indices = missing[start:start+400]
            command = ['sbatch', '--parsable', '-p', 'gpu-a100', '--constraint=a100-80g',
                       '--gres=gpu:1', '-c', '6', '--mem=64G', '-t', '24:00:00',
                       '--array=' + index_spec(indices) + ('%8' if mode == 'final' else '%4'), '-J', 'unified_ba_' + mode,
                       '-o', str(ROOT / f'logs/{mode}_%A_%a.log'), '--export=ALL',
                       str(ROOT / 'code/gpu_task.sbatch'), module, str(plan_path)]
            job = subprocess.check_output(command, text=True).strip().split(';')[0]
            with jobs_path.open('a') as handle:
                handle.write(json.dumps(dict(job=job, mode=mode, plan=str(plan_path),
                    plan_sha256=digest(plan_path), indices=indices, attempt=attempt)) + '\n')
            progress('measuring', stream=mode, job=job, total=len(read(plan_path)['tasks']),
                     pending=len(missing), wave_tasks=len(indices), concurrency=8, attempt=attempt)
            wait_job(job)
    remaining = missing_tasks(mode, plan_path)
    if remaining:
        raise RuntimeError(f'{mode}: incomplete after retries: {remaining}')


def run_combination():
    verify_frozen()
    folder = ROOT / 'combination_scope'
    if not (folder/'report.json').exists() or not (folder/'audit.json').exists():
        from combination_evaluation import evaluate
        plan=read(folder/'plan.json')
        run_gpu_tasks('combination',folder/'plan.json')
        observations=[read(ROOT/'combination'/plan['plan_id']/'raw'/f"{task['id']}.json") for task in plan['tasks']]
        report=evaluate(plan,read(plan['table_source']),observations)
        assert report['complete'] and report['measurement_evidence_verified'] is True
        report['projection_audit_passed']=True
        report['projection_meaning']='fixed editing-only preregistered design; no imported decoder scores'
        atomic(folder/'report.json',report)
        paths=[folder/'plan.json',folder/'report.json']
        for obs in observations:
            paths.extend(Path(e['path']) for e in obs['measurement']['evidence'])
        atomic(folder/'audit.json',dict(passed=True,acceptance_protocol='unified_raw_ba_v1',
            old_detector_outputs_reused=False,model_fitted_from_audit=False,
            evidence={str(p):digest(p) for p in paths}))
    report, plan, audit = (read(folder / name) for name in ('report.json', 'plan.json', 'audit.json'))
    assert report['complete'] and report['tasks'] == 126
    assert report['plan_id'] == plan['plan_id']
    assert report['measurement_evidence_verified'] is True and report['projection_audit_passed'] is True
    assert audit['passed'] is True
    for name, expected in audit['evidence'].items():
        assert digest(name) == expected, ('combination evidence changed', name)
    assert len(plan['tasks']) == 126 and len(report['rows']) == 1512
    assert all(task['attack'] == 'editing_ip2p_s20_v1' for task in plan['tasks'])
    assert all(row['attack'] == 'editing_ip2p_s20_v1' for row in report['rows'])
    progress('complete', stream='combination', tasks=126, old_detector_outputs_reused=False,
             model_fitted_from_audit=False, report=str(folder / 'report.json'))
    return report


def corrected_selection():
    fixed = ROOT / 'fixed'
    if not (fixed / 'progress.json').exists() or read(fixed / 'progress.json').get('phase') != 'complete':
        # Independent CPU solves can overlap the old, already submitted GPU work.
        wait_recorded([fixed / 'jobs.jsonl'])
        verify_frozen()
        progress('enumerating_restored_requests', pending_enumerations=read(fixed / 'prepared.json')['pending_enumerations'])
        subprocess.run([PY, '-u', str(fixed / 'code/controller.py'), 'enumerate-only'], check=True)
    for mode in ('fixed', 'fallback'):
        verify_frozen()
        folder = ROOT / mode
        status = read(folder / 'progress.json') if (folder / 'progress.json').exists() else {}
        if status.get('phase') == 'complete':
            continue
        if mode == 'fixed':
            wait_recorded([folder / 'jobs.jsonl'])
        progress('corrected_' + mode, run_dir=str(folder))
        subprocess.run([PY, '-u', str(folder / 'code/controller.py')], check=True)
        assert read(folder / 'progress.json')['phase'] == 'complete'


def final_testing():
    from selection_adapter import selection_rows
    from final_holdout import freeze_selection, plan_tasks, summarize
    progress('freezing_final_selection')
    frozen_path, plan_path = ROOT / 'plans/final_selection.json', ROOT / 'plans/final_tasks.json'
    freeze_selection(selection_rows(), ROOT / 'plans/image_splits.json', frozen_path)
    plan = plan_tasks(frozen_path, plan_path)
    progress('final_testing', selected_requests=sum(r['selected'] is not None for r in read(frozen_path)['requests']),
             tasks=len(plan['tasks']), frozen_selection_sha256=digest(frozen_path))
    run_gpu_tasks('final', plan_path)
    result = summarize(frozen_path, plan_path, ROOT / 'final/cache', ROOT / 'final_results.json')
    assert result['passed'] is True
    progress('final_testing_complete', result=str(ROOT / 'final_results.json'))
    return result


def main():
    verify_frozen()
    with ThreadPoolExecutor(max_workers=2) as pool:
        combination=pool.submit(run_combination)
        def report_combination_failure(done):
            error=done.exception()
            if error is not None:
                progress('failed',stream='combination',error=repr(error))
        combination.add_done_callback(report_combination_failure)
        corrected_selection()
        final_testing()
        report=combination.result()
    verify_frozen()
    audit = dict(passed=True, original_requests=10000, arm_request_evaluations=20000,
                 completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 corrected_selection=True, combination_tasks=126, independent_final_test=True,
                 heldout_reused_across_protocol_revisions=True, newly_unseen_holdout_claim=False,
                 acceptance_protocol='unified_raw_ba_v1',
                 active_extension_attacks=['editing_ip2p_s20_v1'], excluded_attacks=['nfpa_sd21_xy40_s10_v1'],
                 actual_latency_measurements_complete=True, final_feedback_to_selection=False,
                 meaning='all required measurements and audits complete; does not mean every request passed',
                 global_optimum_claimed=False,
                 evidence={str(p): digest(p) for p in [ROOT / 'fallback/pipeline_audit.json',
                    ROOT / 'plans/final_selection.json', ROOT / 'plans/final_tasks.json',
                    ROOT / 'combination_scope/report.json', ROOT / 'combination_scope/plan.json', ROOT / 'combination_scope/audit.json', ROOT / 'final_results.json', ROOT / 'source_hashes.json']})
    atomic(ROOT / 'pipeline_audit.json', audit)
    progress('complete', original_requests=10000, arm_request_evaluations=20000,
             audit=str(ROOT / 'pipeline_audit.json'))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'main'
    assert mode in ('main', 'combination')
    with (ROOT / f'{mode}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            (main if mode == 'main' else run_combination)()
        except Exception as exc:
            progress('failed', stream=mode, error=repr(exc))
            raise
