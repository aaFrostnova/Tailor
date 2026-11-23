"""Supervise only this isolated revision, including prerequisite GPU smoke checks."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,json,os,subprocess,sys,threading,time,uuid
U=Path(__file__).resolve().parents[1]
PY=os.environ.get('TAILOR_PYTHON', 'python')
sys.path.insert(0,str(U/'code'))
from prepare_campaign import verify_frozen
from rigor_protocol import atomic,read,digest
from smoke_dependencies import pending_dependencies
LOCK=threading.RLock()
STATE=U/'operations/supervisor_state.json'
S=read(STATE) if STATE.exists() else dict(schema='unified_ba_supervisor_v1',lanes={},submissions=[])

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save():
 with LOCK:
  S.update(updated_at=utc(),pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'))
  atomic(STATE,S)
def status(lane,phase,**kw):
 with LOCK:
  S['lanes'][lane]=dict(phase=phase,updated_at=utc(),**kw);save()
 print(json.dumps(dict(lane=lane,phase=phase,**kw)),flush=True)
def cmd(args):return subprocess.check_output(args,text=True,timeout=90).strip()
def queue():
 return [r.split('|') for r in cmd(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%F|%T|%k|%r']).splitlines() if r]
def child(lane,args,env=None,progress=None):
 log=U/'logs'/f'{lane}.log'
 with log.open('a') as out:
  launched_at=time.time()
  p=subprocess.Popen(args,stdout=out,stderr=subprocess.STDOUT,env=env)
  try:
   while p.poll() is None:
    status(lane,'running',child_pid=p.pid,log=str(log))
    if progress and progress.exists():
     q=read(progress)
     # A restarted controller must not inherit an old child's error count.
     if (q.get('controller_pid')==p.pid and progress.stat().st_mtime>=launched_at
         and q.get('consecutive_errors',0)>=3):
      p.terminate();p.wait(timeout=30)
      raise RuntimeError(f'{lane}: repeated controller error: {q.get("error")}')
    time.sleep(20)
   if p.returncode:raise RuntimeError(f'{lane} exited {p.returncode}; see {log}')
  finally:
   if p.poll() is None:p.terminate()
def retry_controller(lane,args,env=None,progress=None):
 for attempt in range(3):
  try:
   child(lane,args,env,progress);return
  except Exception as e:
   status(lane,'retrying' if attempt<2 else 'failed',attempt=attempt+1,error=repr(e))
   if attempt==2:raise
   time.sleep(30)
def submit_smoke(lane,base,indices,attempt):
 token='unified-smoke-'+uuid.uuid4().hex
 plan=base/'smoke/plans/smoke_tasks.json'
 record=dict(token=token,lane=lane,indices=indices,attempt=attempt,job=None,plan=str(plan),plan_sha256=digest(plan))
 with LOCK:S['submissions'].append(record);save()
 args=['sbatch','--parsable','--hold','--comment',token,'-p','gpu','--constraint=a100-80g','--gres=gpu:1','-c','6','--mem=64G','-t','12:00:00','--array='+','.join(map(str,indices))+'%2','-J','unified_'+lane+'_smoke','-o',str(U/'logs'/f'{lane}_smoke_%A_%a.log'),'--export=ALL',str(base/'code/baseline_gpu.sbatch'),str(base/'smoke'),'smoke',str(plan)]
 job=cmd(args).split(';')[0]
 with LOCK:record['job']=job;save()
 cmd(['scontrol','release',job])
 with LOCK:record['released']=True;save()
 return job
def run_smoke(lane,base):
 plan_path=base/'smoke/plans/smoke_tasks.json';plan=read(plan_path)
 for attempt in range(3):
  # Recover this revision's held submissions before considering new work.
  rows=queue()
  for rec in S['submissions']:
   if rec['lane']!=lane:continue
   found={r[1] for r in rows if r[3]==rec['token']}
   if rec['job'] is None and found:
    if len(found)!=1:raise RuntimeError('ambiguous smoke submission')
    rec['job']=found.pop();save()
   if rec['job'] and any(r[1]==rec['job'] and r[2]=='PENDING' and r[4] in ('JobHeldUser','JobHeldAdmin') for r in rows):
    cmd(['scontrol','release',rec['job']])
  active={r[1] for r in rows}
  jobs={rec['job'] for rec in S['submissions'] if rec['lane']==lane and rec['job'] in active}
  while jobs:
   status(lane,'waiting_smoke',jobs=sorted(jobs));time.sleep(20)
   active={r[1] for r in queue()};jobs &= active
  cache=base/'smoke/smoke/cache'
  missing=[i for i,t in enumerate(plan['tasks']) if not (cache/t['task_id']/'result.json').exists()]
  if not missing:break
  job=submit_smoke(lane,base,missing,attempt)
  while job in {r[1] for r in queue()}:
   status(lane,'measuring_smoke',job=job,pending=len(missing));time.sleep(20)
 env=dict(os.environ,BASELINE_RUN_ROOT=str(base/'smoke'))
 child(lane+'_smoke_audit',[PY,'-B','-u',str(base/'code/audit_smoke.py')],env)
 audit=read(base/'smoke/audit.json')
 if not audit.get('complete') or not audit.get('preflight_passed'):raise RuntimeError('baseline GPU preflight incomplete')
def main_lane():
 retry_controller('main',[PY,'-B','-u',str(U/'code/rigor_controller.py'),'main'])
 assert read(U/'pipeline_audit.json')['passed']
 status('main','complete',audit=str(U/'pipeline_audit.json'))
def baseline_lane(name):
 base=U/name
 if not (base/'smoke/audit.json').exists():run_smoke(name,base)
 # Singles depend on both their own and the shared H1/H2 adapter audit.
 # Waiting consumes no controller retries and never reports scientific failure.
 while True:
  pending=pending_dependencies(U,name)
  if not pending:break
  status(name,'waiting_dependencies',pending=pending,retry_budget_consumed=False)
  time.sleep(20)
 retry_controller(name,[PY,'-B','-u',str(base/'code/baseline_controller.py'),'--slots','2','--max-attempts','3'],progress=base/'progress.json')
 methods=('H1_rule','H2_rule') if name=='baselines' else ('S_VINE','S_TrustMark','S_VideoSeal')
 assert all(read(base/m/'pipeline_audit.json')['passed'] for m in methods)
 status(name,'complete')
def main():
 verify_frozen();(U/'logs').mkdir(exist_ok=True)
 os.environ.update(UNIFIED_ATTACK_REUSE_INDEX=str(U/'plans/pixel_reuse_index.json'),UNIFIED_EMBED_REUSE_CERTIFICATE=str(U/'plans/embedding_migration.json'),HF_HOME=os.environ.get('HF_HOME', '~/.cache/huggingface'),HF_HUB_CACHE=os.environ.get('HF_HOME', '~/.cache/huggingface') + '/hub',HUGGINGFACE_HUB_CACHE=os.environ.get('HF_HOME', '~/.cache/huggingface') + '/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures={pool.submit(main_lane):'main',pool.submit(baseline_lane,'baselines'):'baselines',pool.submit(baseline_lane,'single_baselines'):'single_baselines'}
  for future in concurrent.futures.as_completed(futures):
   lane=futures[future]
   try:future.result()
   except Exception as e:status(lane,'failed',error=repr(e));errors.append((lane,repr(e)))
 verify_frozen()
 if errors:raise RuntimeError(errors)
 atomic(U/'all_experiments_complete.json',dict(complete=True,acceptance_protocol='unified_raw_ba_v1',completed_at=utc(),main_requests=10000,main_arm_evaluations=20000,baseline_methods=5,paper_export_pending=True))
if __name__=='__main__':
 with (U/'operations/supervisor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
