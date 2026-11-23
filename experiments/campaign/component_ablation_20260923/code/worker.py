"""Bounded workers: no GPU wait for CPU proposals, no busy configuration allocations."""
import datetime
import fcntl
import os
import signal
import subprocess
import time
import traceback
from common import *

def complete(kind,task):
    return output(task).exists() if kind=='cpu' else all(load_cell(task['candidate'],a) is not None for a in task['attacks'])

def stop(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def active_gpu():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
    rows=[r.split(',') for r in raw.splitlines() if r.strip()]
    assert len(rows)==1 and rows[0][0].strip().startswith('GPU-'),'ambiguous GPU telemetry'
    return float(rows[0][1])>0

def run_child(cmd,kind,deadline):
    proc=subprocess.Popen(cmd,start_new_session=True)
    started=last_gpu=time.time()
    limit=min(deadline,started+(1800 if kind=='cpu' else 4*3600))
    try:
        while proc.poll() is None and time.time()<limit:
            try:proc.wait(timeout=min(30,max(1,limit-time.time())))
            except subprocess.TimeoutExpired:
                if kind=='gpu':
                    if active_gpu():last_gpu=time.time()
                    if time.time()-last_gpu>1200:raise RuntimeError('no GPU compute observed for 20 minutes')
        if proc.poll() is None:raise TimeoutError('allocation or task deadline; outputs remain checkpointed')
        return proc.returncode
    finally:stop(proc)

def main(kind):
    verify_sources()
    deadline=float(os.environ.get('SLURM_JOB_END_TIME',time.time()+4*3600))-120
    worker=os.environ.get('SLURM_JOB_ID','local')+'_'+os.environ.get('SLURM_ARRAY_TASK_ID','0')
    while time.time()<deadline-60:
        queue=RUN/f'queue_{kind}.json';paths=read(queue) if queue.exists() else []
        worked=False
        for name in paths:
            if time.time()>=deadline-60:return
            task=read(name);ident=kind+'_'+task_id(task)
            if complete(kind,task):continue
            with (RUN/'claims'/(ident+'.lock')).open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if complete(kind,task):continue
                err=RUN/'errors'/(ident+'.json');previous=read(err) if err.exists() else {}
                if time.time()-previous.get('last_failure',0)<min(3600,60*2**min(previous.get('attempts',0),6)):continue
                atomic(RUN/'status'/(worker+'.json'),dict(kind=kind,task=name,started_at=time.time()))
                print('TASK_START',kind,name,flush=True)
                try:
                    code=run_child([PY,'-B','-u',str(RUN/'code'/('propose.py' if kind=='cpu' else 'gpu.py')),name],kind,deadline)
                    if code==75 and kind=='gpu':continue
                    assert code==0,('stage exit',code)
                    assert complete(kind,task),'incomplete output'
                    atomic(RUN/'status'/(worker+'.json'),dict(kind=kind,last_complete=name,completed_at=time.time()))
                except Exception:
                    detail=traceback.format_exc()
                    atomic(err,dict(task=name,attempts=previous.get('attempts',0)+1,last_failure=time.time(),error=detail))
                    print('TASK_UNRESOLVED',detail,flush=True)
                    if kind=='gpu':return
                worked=True;break
        if not worked:
            print('RELEASE_NO_RUNNABLE_TASK',kind,flush=True);return

if __name__=='__main__':main(sys.argv[1])
