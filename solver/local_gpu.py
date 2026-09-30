"""Contribute the current allocation's GPU using the same task locks as SLURM."""
import os
import subprocess
import time
from campaign import ROOT, CODE, PY, read, task_status



def main():
    """Run the worker loop until the campaign reports complete or failed."""
    os.environ['FULL_LOCAL_WORKER'] = '1'
    while True:
        progress = ROOT / 'progress.json'
        if progress.exists() and read(progress)['phase'] in ('complete', 'failed'):
            print('LOCAL_GPU_STOP', read(progress), flush=True)
            break
        active = ROOT / 'active_gpu.json'
        if not active.exists() or not read(active):
            manifest = None
            if not progress.exists() or read(progress)['phase'] in ('enumeration', 'enum_batch'):
                from prefetch import next_task
                manifest = next_task()
            if manifest:
                print('LOCAL_GPU_PREFETCH', manifest, flush=True)
                subprocess.run([PY, '-u', str(CODE / 'campaign.py'), 'measure_task', manifest, '0'], check=True)
                continue
            time.sleep(20)
            continue
        task = read(active)
        items = read(task['manifest'])
        did = False
        for index in reversed(range(len(items))):
            status = task_status(task['manifest'], index)
            if status.exists() and read(status).get('complete'):
                continue
            print('LOCAL_GPU_TRY', task['mode'], index, flush=True)
            subprocess.run([PY, '-u', str(CODE / 'campaign.py'), task['mode'], task['manifest'], str(index)], check=True)
            if status.exists() and read(status).get('complete'):
                did = True
                break
        if not did:
            time.sleep(20)


if __name__ == '__main__':
    main()
