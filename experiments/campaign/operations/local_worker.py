"""Local GPU participant. Scientific entry points and native locks stay intact.

Run with --execute to process frozen U tasks; otherwise only report readiness.
Completed artifacts are validated using native validators. No local status is a
pipeline audit. This worker never submits/cancels cluster or editor jobs.
"""
from __future__ import annotations
import argparse
import datetime
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import secrets
import signal
import subprocess
import sys
import time

U = Path(__file__).resolve().parents[1]
OPS = U / 'operations'
PY = os.environ.get('TAILOR_PYTHON', 'python')
CF = Path(os.environ.get('TAILOR_PROJECT', '.'))
BUSY = 75
XENV = {'ctrlregen_s03', 'ctrlregen_s05', 'ctrlregen_s07', 'unmarker', 'editing_ip2p_s20_v1'}
_CACHE_OWNER = False
_CACHE_PATH_ENV = 'UNIFIED_SOURCE_CACHE_PATH'
_CACHE_TOKEN_ENV = 'UNIFIED_SOURCE_CACHE_TOKEN'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp%d' % os.getpid())
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    os.replace(tmp, path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def inside(path):
    return Path(path).resolve().is_relative_to(U.resolve())


def fingerprint(path):
    st = Path(path).stat()
    return dict(device=st.st_dev, inode=st.st_ino, size=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def document_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def initialize_source_cache():
    """A restarted participant performs a new first full verification."""
    global _CACHE_OWNER
    _CACHE_OWNER = True
    os.environ[_CACHE_PATH_ENV] = str(OPS / 'local_source_verification.json')
    os.environ[_CACHE_TOKEN_ENV] = secrets.token_hex(32)


def source_cache(freeze_sha):
    path, token = os.environ.get(_CACHE_PATH_ENV), os.environ.get(_CACHE_TOKEN_ENV)
    if not path or not token or not inside(path): return None
    try:
        st = Path(path).stat()
        if st.st_uid != os.getuid() or st.st_mode & 0o077: return None
        result = read(path)
        signature = result.pop('certificate_sha256')
        if (signature != document_hash(result) or result.get('freeze_sha256') != freeze_sha
                or result.get('token_sha256') != hashlib.sha256(token.encode()).hexdigest()):
            return None
        return result
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def save_source_cache(freeze_sha, files):
    if not _CACHE_OWNER: return
    token = os.environ[_CACHE_TOKEN_ENV]
    path = Path(os.environ[_CACHE_PATH_ENV])
    value = dict(schema='unified_local_source_stat_certificate_v1', freeze_sha256=freeze_sha,
                 token_sha256=hashlib.sha256(token.encode()).hexdigest(),
                 verified_at=utc(), owner_pid=os.getpid(), files=files)
    value['certificate_sha256'] = document_hash(value)
    temp = path.with_name(path.name + '.tmp%d' % os.getpid())
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write('\n')
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def verify_freeze(force_full=False):
    path = U / 'source_hashes.json'
    if not path.exists():
        return None
    hashes = read(path)
    required = [U / 'code/unified_detector.py', U / 'code/rigor_controller.py',
                U / 'code/combination_measurement.py', U / 'fixed/code/campaign.py']
    if not isinstance(hashes, dict) or not all(str(p) in hashes for p in required):
        raise ValueError('source freeze omits required unified scientific entry points')
    freeze_sha = sha(path)
    cached = None if force_full else source_cache(freeze_sha)
    old_files = cached.get('files', {}) if cached else {}
    verified, changed = {}, False
    for name, expected in hashes.items():
        before = fingerprint(name)
        previous = old_files.get(name)
        if previous != dict(expected_sha256=expected, stat=before):
            if sha(name) != expected:
                raise ValueError('frozen source changed: ' + name)
            if fingerprint(name) != before:
                raise ValueError('source changed while being hashed: ' + name)
            changed = True
        verified[name] = dict(expected_sha256=expected, stat=before)
    if sha(path) != freeze_sha:
        raise ValueError('freeze manifest changed during verification')
    if changed or not cached:
        save_source_cache(freeze_sha, verified)
    return freeze_sha


class NonblockingNativeLocks:
    """Only upgrade U's original exclusive flock calls to nonblocking calls."""
    def __enter__(self):
        self.original = fcntl.flock
        self.busy = False
        def lock(fd, operation):
            number = fd.fileno() if hasattr(fd, 'fileno') else fd
            try:
                path = Path(os.readlink('/proc/self/fd/%d' % number))
            except OSError:
                path = None
            if path is not None and inside(path) and operation & fcntl.LOCK_EX:
                operation |= fcntl.LOCK_NB
            try:
                return self.original(fd, operation)
            except BlockingIOError:
                self.busy = True
                raise
        fcntl.flock = lock
        return self

    def __exit__(self, *_args):
        fcntl.flock = self.original


def configure_env():
    os.environ.update(HF_HOME=os.environ.get('HF_HOME', '~/.cache/huggingface'),
        HUGGINGFACE_HUB_CACHE=os.environ.get('HF_HOME', '~/.cache/huggingface') + '/hub',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', FULL_LOCAL_WORKER='1')


def task_identity(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def spec_for(kind, manifest, index, stage=None):
    manifest = Path(manifest).resolve()
    if not inside(manifest):
        raise ValueError('foreign task manifest')
    result = dict(kind=kind, manifest=str(manifest), manifest_sha256=sha(manifest),
                  index=index, stage=stage)
    if kind == 'measure_task':
        item = read(manifest)[index]
        result['linked_plan_sha256'] = sha(item['plan'])
    return result


def native_setup(spec):
    kind = spec['kind']
    if kind in ('combination', 'final'):
        sys.path.insert(0, str(U / 'code'))
        return None
    folder = U / ('fallback' if spec.get('stage') else 'fixed')
    sys.path.insert(0, str(folder / 'code'))
    import campaign
    if spec.get('stage'):
        campaign.configure(spec['stage'])
    return campaign


def validate(spec):
    """True only for this manifest's complete scientific output, never a count."""
    if sha(spec['manifest']) != spec['manifest_sha256']:
        raise ValueError('task manifest changed')
    campaign = native_setup(spec)
    plan = read(spec['manifest'])
    if spec['kind'] == 'measure_task' and sha(plan[spec['index']]['plan']) != spec['linked_plan_sha256']:
        raise ValueError('linked configuration plan changed')
    if spec['kind'] == 'combination':
        from combination_evaluation import check_observation
        task = plan['tasks'][spec['index']]
        path = U / 'combination' / plan['plan_id'] / 'raw' / (task['id'] + '.json')
        if not path.exists(): return False
        check_observation(plan, task, read(path))
        return True
    if spec['kind'] == 'final':
        from final_measurement import validate_result
        task = plan['tasks'][spec['index']]
        path = U / 'final/cache' / task['task_id'] / 'result.json'
        if not path.exists(): return False
        validate_result(task, read(path))
        return True
    status = campaign.task_status(spec['manifest'], spec['index'])
    if not status.exists() or not read(status).get('complete'):
        return False
    item = plan[spec['index']]
    from capacity_protocol import valid_cell, exact_cfg_id, exact_cfg_key
    if spec['kind'] == 'xenv_task':
        group = read(item['path'])
        key = exact_cfg_key(group['order'], group['fe'], [group['s'][f] for f in group['order']])
        if exact_cfg_id(key) != group['cfg_id']:
            raise ValueError('wrong xenv group identity')
        for attack in item['required']:
            if not valid_cell(group.get('attacks', {}).get(attack), group['order'],
                              30 if attack == 'unmarker' else 100,
                              cfg=dict(order=group['order'], fe_on=group['fe'])):
                raise ValueError('complete xenv status has invalid required cell')
        return True
    if spec['kind'] != 'measure_task':
        raise ValueError('unknown local measurement mode')
    rank_plan = read(item['plan'])
    cfg = rank_plan['configs'][item['index']]
    cfgid = exact_cfg_id(cfg['key'])
    order, fe, strengths = cfg['key']
    for attack in set(cfg['attacks']) - XENV:
        path = U / 'live_topk/cells' / f'{cfgid}_{attack}_pool300_n100.json'
        if not path.exists() or not valid_cell(read(path).get('cell'), order, 100,
                                               cfg=dict(order=order, fe_on=fe)):
            raise ValueError('complete measurement status has invalid cell')
    required = set(cfg['attacks']) & XENV
    if required:
        domain = 'topk_' + rank_plan['cls'] + '_' + campaign.tag(rank_plan['arm'])
        folder = U / ('certify_full_' + domain) / rank_plan['cls']
        group = read(folder / ('g%04d.json' % int(cfgid[:12], 16)))
        if group.get('cfg_id') != cfgid or group.get('n_img') != 100:
            raise ValueError('complete measurement status has wrong export group')
        if exact_cfg_id(exact_cfg_key(group['order'], group['fe'], [group['s'][f] for f in group['order']])) != cfgid:
            raise ValueError('export group configuration differs')
        for attack in required:
            if attack not in group.get('xenv_needed', []) and not valid_cell(group.get('attacks', {}).get(attack),
                    order, 30 if attack == 'unmarker' else 100, cfg=dict(order=order, fe_on=fe)):
                raise ValueError('export neither queued nor measured required attack')
        for index in range(100):
            if not (folder / ('g%04d' % int(cfgid[:12], 16)) / 'embed' / ('i%05d.png' % index)).exists():
                raise ValueError('export group lacks expected embedding')
    return True


def candidates(stage):
    """Reverse array order; existing status is only a candidate-selection hint."""
    if stage == 'combination':
        path = U / 'combination_scope/plan.json'
        if not path.exists(): return []
        p = read(path)
        return [spec_for('combination', path, i) for i in reversed(range(len(p['tasks'])))
                if not (U / 'combination' / p['plan_id'] / 'raw' / (p['tasks'][i]['id'] + '.json')).exists()]
    if stage == 'final':
        path = U / 'plans/final_tasks.json'
        if not path.exists(): return []
        p = read(path)
        return [spec_for('final', path, i) for i in reversed(range(len(p['tasks'])))
                if not (U / 'final/cache' / p['tasks'][i]['task_id'] / 'result.json').exists()]
    if stage == 'prefetch':
        sys.path.insert(0, str(U / 'fixed/code'))
        import prefetch
        path = prefetch.next_task()
        return [spec_for('measure_task', path, 0)] if path else []
    folder = U / 'fixed' if stage == 'fixed' else U / 'fallback/stages' / stage
    active = folder / 'active_gpu.json'
    if not active.exists(): return []
    row = read(active)
    if row.get('mode') not in ('measure_task', 'xenv_task') or not row.get('manifest'): return []
    path = Path(row['manifest'])
    result = []
    for i in reversed(range(len(read(path)))):
        spec = spec_for(row['mode'], path, i, None if stage == 'fixed' else stage)
        if not validate(spec):
            result.append(spec)
        if len(result) >= 64:
            break
    return result


def execute_spec(path):
    spec = read(path)
    if not verify_freeze(): raise ValueError('cannot execute before freeze')
    campaign = native_setup(spec)
    if validate(spec):
        print('LOCAL_NATIVE_RESULT_ALREADY_VALIDATED', flush=True)
        return 0
    with NonblockingNativeLocks() as locks:
        try:
            if spec['kind'] == 'combination':
                from combination_measurement import task
                task(spec['manifest'], spec['index'])
            elif spec['kind'] == 'final':
                from final_measurement import run_task
                run_task(spec['manifest'], spec['index'], U / 'final/cache')
            else:
                campaign.dispatch_task(spec['kind'], spec['manifest'], spec['index'])
        except BlockingIOError:
            print('LOCAL_TASK_BUSY', flush=True)
            return BUSY
    if locks.busy:
        return BUSY
    if not validate(spec):
        raise RuntimeError('task returned without validated complete output')
    print('LOCAL_NATIVE_RESULT_VALIDATED', flush=True)
    return 0


def pid_matches(pid, substring):
    try:
        return substring in Path('/proc/%d/cmdline' % pid).read_bytes().replace(b'\x00', b' ').decode()
    except FileNotFoundError:
        return False


def loop(args):
    OPS.mkdir(exist_ok=True)
    with (OPS / 'local_worker.lock').open('a') as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = OPS / 'local_worker_state.json'
        state = read(state_path) if state_path.exists() else dict(attempts={}, validated_task_keys=[])
        previous_child = state.get('child_pid')
        previous_task_key = state.get('task_key')
        state.update(pid=os.getpid(), started_at=utc(), execute=args.execute, audit_claim=False)
        initialize_source_cache()
        stop = [False]
        signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
        busy_until = {}
        scan_failures = 0
        idle_since = None
        last_idle_alert = 0.0
        def status(phase, **more):
            state.update(phase=phase, heartbeat_at=utc(), **more)
            atomic(state_path, state)
        def alert(kind, **more):
            with (OPS / 'local_worker_alerts.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(at=utc(), kind=kind, **more)) + '\n')
        while not stop[0]:
            status('checking_dependencies')
            if (previous_child and previous_task_key and
                    pid_matches(previous_child, str(OPS / 'local_tasks' / (previous_task_key + '.json')))):
                status('waiting_for_previous_owned_task', child_pid=previous_child,
                       message='A previous participant process exited; its native task still owns the GPU.')
                time.sleep(args.poll); continue
            previous_child = None
            if any(pid_matches(pid, args.wait_command) for pid in args.wait_pid):
                status('waiting_for_gpu_preflight', wait_pids=args.wait_pid)
                time.sleep(args.poll); continue
            try:
                frozen = verify_freeze()
                if frozen is None:
                    status('waiting_for_source_freeze'); time.sleep(args.poll); continue
                if state.get('source_freeze_sha256', frozen) != frozen:
                    raise ValueError('freeze identity differs from this participant state')
                state['source_freeze_sha256'] = frozen
                if not args.execute:
                    status('frozen_ready_observe_only'); time.sleep(args.poll); continue
                selected = None
                for stage in ('combination', 'prefetch', 'fixed', '02', '00', 'final'):
                    pick_path = OPS / 'local_candidates.json'
                    command = [PY, '-B', str(Path(__file__)), '_candidates', '--stage', stage, '--output', str(pick_path)]
                    result = subprocess.run(command, cwd=CF, capture_output=True, text=True, timeout=120)
                    if result.returncode:
                        raise RuntimeError('candidate scan failed for %s: %s' % (stage, result.stderr[-3000:]))
                    for spec in read(pick_path):
                        key = task_identity(spec)
                        if state['attempts'].get(key, 0) >= 3 or busy_until.get(key, 0) > time.time(): continue
                        # Never elevate this existence hint into a completion or audit count.
                        selected = spec; break
                    if selected: break
                scan_failures = 0
                if selected is None:
                    idle_since = time.time() if idle_since is None else idle_since
                    if time.time() - idle_since >= 600 and time.time() - last_idle_alert >= 600:
                        alert('no_eligible_gpu_work', idle_seconds=time.time()-idle_since,
                              quarantined_tasks=sum(v >= 3 for v in state['attempts'].values()),
                              meaning='CPU/dependency work may be pending; this is not a completion verdict')
                        last_idle_alert = time.time()
                    status('waiting_for_eligible_work', all_complete_claim=False,
                           quarantined_tasks=sum(v >= 3 for v in state['attempts'].values()))
                    time.sleep(args.poll); continue
                idle_since = None
                key = task_identity(selected)
                task_path = OPS / 'local_tasks' / (key + '.json')
                atomic(task_path, selected)
                log_path = OPS / 'local_logs' / (key + '.log')
                log_path.parent.mkdir(exist_ok=True)
                command = [PY, '-B', '-u', str(Path(__file__)), '_execute', '--task', str(task_path)]
                # Persist before launch: parent crashes cannot reset retry counts.
                # A confirmed busy exit rolls back this non-executed attempt.
                state['attempts'][key] = state['attempts'].get(key, 0) + 1
                status('launch_intent', task=selected, task_key=key, child_pid=None, log=str(log_path))
                with log_path.open('a') as log:
                    child = subprocess.Popen(command, cwd=CF, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    start = time.monotonic()
                    status('running', task=selected, task_key=key, child_pid=child.pid, log=str(log_path), task_started_at=utc())
                    timed_out = False
                    while child.poll() is None:
                        if stop[0] or time.monotonic() - start > args.task_timeout:
                            timed_out = not stop[0]
                            os.killpg(child.pid, signal.SIGTERM)
                            try: child.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
                            break
                        status('running', elapsed_seconds=round(time.monotonic()-start, 1))
                        time.sleep(min(args.poll, 15))
                rc = child.returncode
                if stop[0]: break
                if rc == BUSY:
                    state['attempts'][key] -= 1
                    busy_until[key] = time.time() + 60
                    status('task_busy', child_pid=None, returncode=rc)
                elif rc == 0:
                    if key not in state['validated_task_keys']: state['validated_task_keys'].append(key)
                    status('task_native_validated', child_pid=None, returncode=rc)
                else:
                    busy_until[key] = time.time() + 60
                    alert('task_failed', task=selected, attempt=state['attempts'][key], returncode=rc,
                          timed_out=timed_out, log=str(log_path), quarantined=state['attempts'][key] >= 3)
                    status('task_failed', child_pid=None, returncode=rc)
            except Exception as exc:
                scan_failures += 1
                alert('worker_error', error=repr(exc), consecutive=scan_failures)
                status('error', error=repr(exc), consecutive_errors=scan_failures)
                if scan_failures >= 3:
                    raise RuntimeError('three consecutive worker errors; durable monitor must investigate') from exc
                time.sleep(args.poll)
        status('stopped', child_pid=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='loop', choices=['loop', '_execute', '_candidates', 'verify-sources-full'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--wait-pid', type=int, action='append', default=[])
    parser.add_argument('--wait-command', default='run_pipeline_smoke.py')
    parser.add_argument('--poll', type=float, default=15.)
    parser.add_argument('--task-timeout', type=float, default=21600.)
    parser.add_argument('--task'); parser.add_argument('--stage'); parser.add_argument('--output')
    args = parser.parse_args()
    configure_env()
    if args.mode == 'verify-sources-full':
        frozen = verify_freeze(force_full=True)
        if not frozen: raise ValueError('source freeze does not yet exist')
        print(json.dumps(dict(source_freeze_sha256=frozen, force_full_sha256=True)))
        return
    if args.mode == '_execute': raise SystemExit(execute_spec(args.task))
    if args.mode == '_candidates':
        if not verify_freeze(): raise ValueError('candidate creation waits for source freeze')
        atomic(args.output, candidates(args.stage)); return
    if not 0 < args.poll <= 60 or args.task_timeout <= 0: raise ValueError('invalid poll or timeout')
    loop(args)


if __name__ == '__main__':
    main()
