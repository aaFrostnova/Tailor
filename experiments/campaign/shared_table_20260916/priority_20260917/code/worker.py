"""Persistent workers claim immutable task files; failed measurements are retried."""
import datetime
import fcntl
import subprocess
import time
import traceback
from common import *

def complete(kind,task):
    if kind=='cpu': return output(task).exists()
    return all(load_cell(task['candidate'],a) is not None for a in task['attacks'])

def main(kind):
    verify_sources()
    start=time.monotonic()
    limit=(11.5 if kind=='cpu' else 23.5)*3600
    worker=os.environ.get('SLURM_JOB_ID','local')+'_'+os.environ.get('SLURM_ARRAY_TASK_ID','0')
    while time.monotonic()-start < limit:
        queue=RUN/f'queue_{kind}.json'
        paths=read(queue) if queue.exists() else []
        did_work=False
        for name in paths:
            task=read(name); ident=kind+'_'+task_id(task)
            if complete(kind,task): continue
            with (RUN/'claims'/(ident+'.lock')).open('a') as h:
                try: fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: continue
                if complete(kind,task): continue
                err=RUN/'errors'/(ident+'.json')
                previous=read(err) if err.exists() else {}
                # Retry with delay, retain unresolved status; never call an
                # exception an UNSAT or a failed live gate.
                if time.time()-previous.get('last_failure',0)<min(3600,60*2**min(previous.get('attempts',0),6)): continue
                atomic(RUN/'status'/(worker+'.json'),dict(worker=worker,task=name,started_at=time.time(),kind=kind))
                print('TASK_START',kind,name,flush=True)
                try:
                    subprocess.run([PY,'-B','-u',str(RUN/'code'/('propose.py' if kind=='cpu' else 'gpu.py')),name],
                                   check=True,timeout=1800 if kind=='cpu' else 22*3600)
                    assert complete(kind,task), 'worker returned without validated output'
                    atomic(RUN/'status'/(worker+'.json'),dict(worker=worker,last_complete=name,completed_at=time.time(),kind=kind))
                except Exception:
                    error=traceback.format_exc()
                    atomic(err,dict(task=name,attempts=previous.get('attempts',0)+1,last_failure=time.time(),error=error))
                    print('TASK_UNRESOLVED',name,error,flush=True)
                did_work=True
                break
        if not did_work:
            if (RUN/'progress.json').exists() and read(RUN/'progress.json').get('complete'): return
            time.sleep(20)

if __name__=='__main__': main(sys.argv[1])
