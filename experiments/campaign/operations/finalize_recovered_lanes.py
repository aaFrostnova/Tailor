"""Publish completion only after all frozen lanes pass, including recovered singles."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

U = Path(__file__).resolve().parents[1]
O = U / 'operations'
METHODS = [('baselines', 'H1_rule'), ('baselines', 'H2_rule'),
           ('single_baselines', 'S_VINE'), ('single_baselines', 'S_TrustMark'),
           ('single_baselines', 'S_VideoSeal')]


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def inspect():
    paths = [U/'pipeline_audit.json'] + [U/folder/method/'pipeline_audit.json' for folder,method in METHODS]
    pending = [str(p) for p in paths if not p.exists()]
    if pending: return pending, {}
    audits = {str(p):read(p) for p in paths}
    for name, audit in audits.items():
        if audit.get('passed') is not True:
            raise ValueError('completed lane audit failed: ' + name)
        if name != str(U/'pipeline_audit.json'):
            if audit.get('complete') is not True or audit.get('terminal_requests') != 20000:
                raise ValueError('incomplete baseline audit: ' + name)
        for source, expected in audit['evidence'].items():
            if digest(source) != expected:
                raise ValueError('lane audit evidence changed: ' + source)
    main = audits[str(U/'pipeline_audit.json')]
    if (main.get('acceptance_protocol') != 'unified_raw_ba_v1'
            or main.get('original_requests') != 10000 or main.get('arm_request_evaluations') != 20000
            or main.get('independent_final_test') is not True
            or main.get('actual_latency_measurements_complete') is not True
            or main.get('final_feedback_to_selection') is not False):
        raise ValueError('main audit lacks full unified pipeline completion')
    for source, expected in read(U/'source_hashes.json').items():
        if digest(source) != expected:
            raise ValueError('frozen source changed: ' + source)
    return [], {name:digest(name) for name in audits}


def main():
    with (O/'recovered_finalizer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            pending, evidence = inspect()
            if not pending: break
            atomic(O/'recovered_finalizer_state.json', dict(phase='waiting_for_lane_audits',
                updated_at=utc(), job=os.environ.get('SLURM_JOB_ID'), pending=pending))
            time.sleep(30)
        result = dict(complete=True, acceptance_protocol='unified_raw_ba_v1',completed_at=utc(),
            main_requests=10000, main_arm_evaluations=20000,baseline_methods=5,paper_export_pending=True,
            final_lane_audit_hashes=evidence,
            recovery=dict(single_supervisor_job='64465825', supersedes='operations/supervisor_state.json lanes.single_baselines failure at 2026-09-14T18:14:30Z',
                          original_main_supervisor='64460183', scientific_protocol_unchanged=True))
        atomic(U/'all_experiments_complete.json',result)
        atomic(O/'recovered_finalizer_state.json',dict(phase='complete',updated_at=utc(),audit_hashes=evidence))
        print(json.dumps(result),flush=True)


if __name__=='__main__': main()
