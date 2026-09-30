"""Submit, verify and resume the full top-k/live experiment on SLURM."""
import datetime
import json
import os
import subprocess
import time

from campaign import ROOT, CODE, INPUTS, CLASSES, ARMS, PY, atomic, read, checkpoint, assemble
from campaign import plan_rank, driver, xenv_plan, outdir, task_status, finish, needs_patch


def progress(phase, **kwargs):
    data = {'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'phase': phase, **kwargs}
    atomic(ROOT / 'progress.json', data)
    print('PROGRESS', json.dumps(data), flush=True)


def wait_array(job):
    while any(j == job or j.startswith(job + '_') for j in subprocess.check_output(
            ['squeue', '-u', '3529', '-h', '-r', '-o', '%i'], text=True).splitlines()):
        time.sleep(30)
    result = {}
    for _ in range(12):
        rows = subprocess.check_output(['sacct', '-X', '-j', job, '-n', '-P',
                                        '-o', 'JobID,State,ExitCode'], text=True).splitlines()
        result = {r.split('|')[0]: r.split('|')[1:3] for r in rows if r.strip()}
        if result and not any(v[0] in ('RUNNING', 'PENDING', 'COMPLETING') for v in result.values()):
            break
        time.sleep(10)
    return result


def array_spec(indices):
    spans = []
    start = end = indices[0]
    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            spans.append(str(start) if start == end else f'{start}-{end}')
            start = end = i
    spans.append(str(start) if start == end else f'{start}-{end}')
    return ','.join(spans)


def run_array(mode, manifest, indices=None):
    items = read(manifest)
    indices = list(range(len(items))) if indices is None else list(indices)
    if not indices:
        return
    # QoS permits 2000 submitted jobs per user, including unrelated jobs. Keep
    # each wave at <=1000 and compact contiguous indices to avoid Slurm's
    # parameter-length limit. The request manifest itself stays unchanged.
    if len(indices) > 1000 or len(array_spec(indices)) > 3000:
        size = 1000 if len(array_spec(indices)) <= 3000 else 400
        for start in range(0, len(indices), size):
            run_array(mode, manifest, indices[start:start + size])
        return
    gpu = mode in ('measure_task', 'xenv_task')
    partition = 'gpu-a100' if mode == 'xenv_task' else ('gpu' if gpu else 'cpu')
    concurrency = 8 if mode == 'xenv_task' else (10 if gpu else 40)
    remaining = indices
    for attempt in range(3):
        memory = ('64G' if mode == 'xenv_task' else '48G') if gpu else ('32G' if attempt == 0 else '64G')
        # Explicit indices, never a changing modulo partition of pending tasks.
        array = array_spec(remaining) + '%' + str(concurrency)
        cmd = ['sbatch', '--parsable', '-p', partition, '-c', '6' if gpu else '2', '--mem='+memory,
               '-t', '24:00:00' if gpu else '2-00:00:00', '--array='+array, '-J', 'capout_' + mode,
               '-o', str(ROOT / f'logs/{mode}_%A_%a.log'), '--export=ALL']
        if gpu:
            cmd += ['--gres=gpu:1']
        cmd += [str(CODE / 'task.sbatch'), mode, str(manifest)]
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        with (ROOT / 'jobs.jsonl').open('a') as handle:
            handle.write(json.dumps({'job': job, 'mode': mode, 'manifest': str(manifest),
                                     'indices': remaining, 'attempt': attempt, 'memory': memory}) + '\n')
        progress(mode, job=job, task_count=len(remaining), attempt=attempt, manifest=str(manifest))
        if gpu:
            atomic(ROOT / 'active_gpu.json', {'mode': mode, 'manifest': str(manifest), 'job': job})
        states = wait_array(job)
        failed = []
        for index in remaining:
            state = states.get(f'{job}_{index}')
            if mode == 'enum_batch':
                complete = all(checkpoint(k, arm, i).exists() for k, arm, i in items[index])
            else:
                p = task_status(manifest, index)
                complete = p.exists() and read(p).get('complete')
            if not complete or state != ['COMPLETED', '0:0']:
                failed.append(index)
        if gpu:
            atomic(ROOT / 'active_gpu.json', {})
        if not failed:
            print('ARRAY_VERIFIED', job, len(indices), flush=True)
            return
        print('ARRAY_RETRY', job, 'failed', failed, flush=True)
        remaining = failed
    raise RuntimeError(f'{mode}: tasks still failed after three attempts: {remaining}')


def main():
    progress('enumeration')
    enum_manifest = ROOT / 'plans/enumeration.json'
    batches = read(enum_manifest)
    missing = [j for j, batch in enumerate(batches) if any(not checkpoint(k, a, i).exists() for k, a, i in batch)]
    run_array('enum_batch', enum_manifest, missing)
    for k in CLASSES:
        for arm in ARMS:
            assemble(k, arm)
    progress('all_enumerated', arm_request_count=20000)
    for rank in (1, 2, 3, 4):
        if rank == 4:
            patches = []
            for k in CLASSES:
                for arm in ARMS:
                    driver(k, arm, 'walk', 100)
                    rows = read(outdir(k, arm) / 'walk.json')['rows']
                    assert not any(r['open'] for r in rows), (k, arm, 'unmeasured lower rank')
                    patches += [dict(cls=k, arm=arm, i=r['i']) for r in rows
                                if needs_patch(r)]
            manifest = ROOT / 'plans/patches.json'
            atomic(manifest, patches)
            run_array('patch_task', manifest)
        tasks = []
        for k in CLASSES:
            for arm in ARMS:
                path, configs = plan_rank(k, arm, rank)
                tasks += [dict(plan=str(path), index=i) for i in range(len(configs))]
        manifest = ROOT / f'plans/measure_rank{rank}.json'
        atomic(manifest, tasks)
        run_array('measure_task', manifest)
        xenv = []
        for k in CLASSES:
            for arm in ARMS:
                xenv += xenv_plan(k, arm, rank)
        manifest = ROOT / f'plans/xenv_rank{rank}.json'
        atomic(manifest, xenv)
        run_array('xenv_task', manifest)
        for k in CLASSES:
            for arm in ARMS:
                driver(k, arm, 'xenv_mark')
        progress('rank_completed', rank=rank)
    results = [finish(k, arm) for k in CLASSES for arm in ARMS]
    assert sum(r['completed_requests'] for r in results) == 20000
    atomic(ROOT / 'full_results.json', results)
    progress('complete', original_requests=10000, arm_request_evaluations=20000)
    print('FULL_CAMPAIGN_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        progress('failed', error=repr(exc))
        raise
