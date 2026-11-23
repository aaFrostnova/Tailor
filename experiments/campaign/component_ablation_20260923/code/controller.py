"""Rolling CPU proposals and GPU live queue; all missing evidence stays unresolved."""
import collections
import datetime
import fcntl
import subprocess
import time
import traceback
from common import *
import os
import common
from functools import lru_cache

_native_load_cell = common.load_cell
_native_combine = CP.combine
_native_valid = CP.valid_cell
_reductions, _validations = {}, {}

def _combined(cell,order,tau,det_min=.9):
    key=id(cell),tuple(order),float(tau),float(det_min)
    if key not in _reductions:_reductions[key]=(cell,_native_combine(cell,order,tau,det_min))
    return _reductions[key][1]

def _valid(cell,order,n,cfg=None,alphas=CP.REQUEST_ALPHAS):
    key=id(cell),tuple(order),n,json.dumps(cfg,sort_keys=True),tuple(alphas)
    if key not in _validations:_validations[key]=(cell,_native_valid(cell,order,n,cfg=cfg,alphas=alphas))
    return _validations[key][1]
@lru_cache(maxsize=50000)
def _cycle_cell(order,fe,strengths,attack):
    return _native_load_cell(dict(order=list(order),fe_on=list(fe),s=dict(zip(order,strengths))),attack)

def _cached_cell(candidate,attack):
    order,fe,strengths=cfgkey(candidate)
    return _cycle_cell(order,fe,strengths,attack)

def inspect_target(t, cpu, gpu):
    v,arm = t['variant'],t['arm']
    initial = dict(t,stage='initial')
    p = output(initial)
    if not p.exists():
        if v in ('S1','S3','S3_R') and not output(dict(initial,variant='Full')).exists():
            return dict(status='pending_dependency')
        cpu.append(initial)
        return dict(status='pending_proposal')
    initial_result = read(p)
    # Back off only after each stage has either no model candidate or complete
    # failed live evidence, including its request-local correction. Pending
    # measurements return above advancement and must never trigger backoff.
    eligible = v in FALLBACK
    stages = ['initial'] + ((['04','02','00'] if arm=='06' else ['02','00']) if eligible else [])
    walks = []
    for stage in stages:
        task = dict(t,stage=stage)
        if not output(task).exists():
            if stage=='04' and not output(dict(task,arm='04',stage='initial')).exists():
                return dict(status='pending_dependency',walk=walks)
            cpu.append(task)
            return dict(status='pending_fallback',walk=walks)
        proposal = read(output(task))
        cs = proposal['candidates']
        if not cs: continue
        for c in cs:
            verdict = judge(t['request'],c)
            walks.append(dict(stage=stage,rank=c['rank'],cfg_id=CP.exact_cfg_id(cfgkey(c)),
                              passed=verdict['pass'],pending=verdict['pending']))
            if verdict['pass']:
                return dict(status='accepted',stage=stage,rank=c['rank'],candidate=c,verdict=verdict,walk=walks)
            if verdict['pending']:
                key = cfgkey(c)
                e = gpu.setdefault(key,dict(candidate=c,attacks=set(),cls=t['cls']))
                e['attacks'].update(verdict['pending'])
                return dict(status='pending_live',walk=walks)
        if v in REFINE:
            patched = dict(t,stage=stage+'_patch')
            if not output(patched).exists():
                cpu.append(patched)
                return dict(status='pending_refinement',walk=walks)
            for c in read(output(patched))['candidates']:
                verdict = judge(t['request'],c)
                walks.append(dict(stage=stage,rank=4,cfg_id=CP.exact_cfg_id(cfgkey(c)),passed=verdict['pass'],pending=verdict['pending']))
                if verdict['pass']:
                    return dict(status='accepted',stage=stage,rank=4,candidate=c,verdict=verdict,walk=walks)
                if verdict['pending']:
                    key=cfgkey(c);e=gpu.setdefault(key,dict(candidate=c,attacks=set(),cls=t['cls']))
                    e['attacks'].update(verdict['pending'])
                    return dict(status='pending_live',walk=walks)
        if not eligible:
            return dict(status='live_exhausted',walk=walks)
    return dict(status=('greedy_stalled' if v=='G_shared' else ('live_exhausted' if walks else 'model_unsat')),walk=walks)

def task_file(kind,task):
    p = RUN/'tasks'/(kind+'_'+task_id(task)+'.json')
    if not p.exists(): atomic(p,task)
    return str(p)

def sweep():
    _cycle_cell.cache_clear()
    _reductions.clear();_validations.clear()
    CP.combine,CP.valid_cell=_combined,_valid
    common.load_cell = _cached_cell
    manifest = read(RUN/'manifest.json')
    cpu,gpu,results=[],{},[]
    for t in manifest['targets']:
        result=inspect_target(t,cpu,gpu)
        # Detailed per-image arrays stay in validated cells, not in every copy
        # of the periodic request report.
        if 'verdict' in result:
            result['verdict']={k:v for k,v in result['verdict'].items() if k!='attacks'}
        results.append(dict(cohort=t.get('cohort','main'),variant=t['variant'],arm=t['arm'],spec_id=t['spec_id'],cls=t['cls'],i=t['i'],**result))
    subset=set(manifest['priority_ablation_specs'])
    cpu = fair_cpu_tasks(cpu, subset)
    atomic(RUN/'queue_cpu.json',[task_file('cpu',t) for t in cpu])
    jobs=[]
    for _,g in sorted(gpu.items(),key=lambda x:CP.exact_cfg_id(x[0])):
        g['attacks']=sorted(g['attacks'])
        jobs.append(task_file('gpu',g))
    atomic(RUN/'queue_gpu.json',jobs)
    counts=collections.defaultdict(collections.Counter)
    for r in results: counts[r['variant']+'_'+r['arm']][r['status']]+=1
    status=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=POLICY,
                counts=dict(counts),pending_cpu=len(cpu),pending_gpu_configs=len(jobs),
                target_evaluations=len(results),complete=not any(r['status'].startswith('pending') for r in results))
    atomic(RUN/'results.json',dict(status=status,rows=results))
    from report import publish
    publish(results, status)
    atomic(RUN/'progress.json',status)
    print('PROGRESS',json.dumps(status),flush=True)
    return status

def fair_cpu_tasks(tasks, subset):
    # Keep correction/subset priority, but do not put every Full task ahead of greedy.
    tiers = collections.defaultdict(lambda: collections.defaultdict(collections.deque))
    for task in sorted(tasks,key=lambda t:(t['i'],t['arm'],t['stage'])):
        tier = (0 if task['stage'].endswith('_patch') else 1,
                0 if task['spec_id'] in subset else 1)
        tiers[tier][task['variant']].append(task)
    out = []
    preference = {'G_shared':0,'Full':1,'S1':2,'S3':3,'S3_R':4,'Grid':5,'NoFrontend':6,'NoRecoveryInteraction':7}
    for tier in sorted(tiers):
        groups = tiers[tier]
        names = sorted(groups,key=lambda v:preference.get(v,8))
        while any(groups.values()):
            for name in names:
                if groups[name]: out.append(groups[name].popleft())
    return out

def active_jobs():
    try:
        rows = subprocess.check_output(['squeue','-u',str(os.getuid()),'-h','-r','-o','%i|%j'],text=True,timeout=30)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        # Unknown queue state must not be treated as an empty queue: defer
        # submissions until the next controller sweep to avoid duplicate jobs.
        print('SCHEDULER_QUERY_RETRY', type(exc).__name__, str(exc), flush=True)
        return None
    return [dict(zip(('id','name'), r.split('|'))) for r in rows.splitlines() if r.strip()]

def runnable_gpu_count():
    # Count unique configurations with missing evidence and an available lock.
    # Queue variants differing only in requested attacks must not reserve extra GPUs.
    paths = read(RUN/'queue_gpu.json') if (RUN/'queue_gpu.json').exists() else []
    seen = set(); count = 0
    for name in paths:
        task = read(name); candidate = task['candidate']
        digest = CP.exact_cfg_id(cfgkey(candidate))
        if digest in seen: continue
        if all(load_cell(candidate,a) is not None for a in task['attacks']): continue
        seen.add(digest)
        errpath = RUN/'errors'/('gpu_'+task_id(task)+'.json')
        previous = read(errpath) if errpath.exists() else {}
        if time.time()-previous.get('last_failure',0)<min(3600,60*2**min(previous.get('attempts',0),6)): continue
        with (U/'fixed/checkpoints'/('cfg_'+digest+'.lock')).open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: continue
            count += 1
    return count

def ensure_workers(state,status):
    active = active_jobs()
    if active is None:
        return
    for kind,capacity in [('cpu',24),('gpu',4)]:
        occupied = sum(j['name'] == 'component_ablation_'+kind for j in active)
        if occupied >= capacity: continue
        pending = status['pending_cpu' if kind=='cpu' else 'pending_gpu_configs']
        if not pending: continue
        n = min(capacity-occupied, pending)
        if kind == 'cpu':
            runnable=0
            for name in read(RUN/'queue_cpu.json'):
                task=read(name)
                if output(task).exists():continue
                err=RUN/'errors'/('cpu_'+task_id(task)+'.json')
                previous=read(err) if err.exists() else {}
                if time.time()-previous.get('last_failure',0)<min(3600,60*2**min(previous.get('attempts',0),6)):continue
                with (RUN/'claims'/('cpu_'+task_id(task)+'.lock')).open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    runnable+=1
                    if runnable>=n:break
            n=min(n,runnable)
            if not n:continue
        if kind == 'gpu':
            n = min(n, runnable_gpu_count())
            if not n: continue
        cmd=['sbatch','--parsable','-p','cpu' if kind=='cpu' else 'gpu',
             '-c','1' if kind=='cpu' else '6','--mem=8G' if kind=='cpu' else '--mem=64G',
             '-t','04:00:00' if kind=='cpu' else '04:00:00',f'--array=0-{n-1}%{n}',
             '-J','component_ablation_'+kind,'-o',str(RUN/f'logs/{kind}_%A_%a.log'),
             '--chdir='+str(RUN),'--export=ALL']
        if kind=='gpu': cmd+=['--gres=gpu:1','--constraint=a100-80g']
        cmd += [str(RUN/'code/worker.sbatch'),kind]
        job=subprocess.check_output(cmd,text=True,timeout=60).strip().split(';')[0]
        state[kind]=job
        with (RUN/'jobs.jsonl').open('a') as f: f.write(json.dumps(dict(kind=kind,job=job,workers=n,scheduling_revision='gpu_idle_fix_20260921'))+'\n')
        atomic(RUN/'worker_jobs.json',state)

def main():
    verify_sources()
    with (RUN/'controller.lock').open('a') as h:
        fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(RUN/'worker_jobs.json') if (RUN/'worker_jobs.json').exists() else {}
        deadline=float(os.environ.get('SLURM_JOB_END_TIME',time.time()+10*3600))-120
        while time.time()<deadline:
            status=sweep()
            if status['complete']:
                atomic(RUN/'pipeline_audit.json',dict(source_binding=sha(RUN/'sources.json'),**status))
                return
            ensure_workers(state,status)
            time.sleep(45)

if __name__=='__main__':
    if sys.argv[1:] == ['sweep']: sweep()
    else: main()
