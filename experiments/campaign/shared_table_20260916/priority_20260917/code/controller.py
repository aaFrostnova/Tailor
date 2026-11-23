"""Rolling CPU proposals and GPU live queue; all missing evidence stays unresolved."""
import collections
import datetime
import fcntl
import subprocess
import time
import traceback
from common import *
import common
from functools import lru_cache

_native_load_cell = common.load_cell
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
        cpu.append(initial)
        return dict(status='pending_proposal')
    initial_result = read(p)
    eligible = v in FALLBACK and not initial_result['candidates']
    stages = ['initial'] + ((['04','02','00'] if arm=='06' else ['02','00']) if eligible else [])
    walks = []
    for stage in stages:
        task = dict(t,stage=stage)
        if not output(task).exists():
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
    common.load_cell = _cached_cell
    manifest = read(RUN/'manifest.json')
    cpu,gpu,results=[],{},[]
    for t in manifest['targets']:
        result=inspect_target(t,cpu,gpu)
        # Detailed per-image arrays stay in validated cells, not in every copy
        # of the periodic request report.
        if 'verdict' in result:
            result['verdict']={k:v for k,v in result['verdict'].items() if k!='attacks'}
        results.append(dict(variant=t['variant'],arm=t['arm'],spec_id=t['spec_id'],cls=t['cls'],i=t['i'],**result))
    subset=set(manifest['priority_ablation_specs'])
    order={'S1':0,'S3':1,'S3_R':2,'Full':3,'G_shared':4,'NoFrontend':5,'NoRecoveryInteraction':6,'Grid':7}
    cpu.sort(key=lambda t:(0 if t['stage'].endswith('_patch') else 1,
                          0 if t['spec_id'] in subset else 1,order[t['variant']],t['i'],t['arm']))
    atomic(RUN/'queue_cpu.json',[task_file('cpu',t) for t in cpu])
    jobs=[]
    for _,g in sorted(gpu.items(),key=lambda x:CP.exact_cfg_id(x[0])):
        g['attacks']=sorted(g['attacks'])
        jobs.append(task_file('gpu',g))
    atomic(RUN/'queue_gpu.json',jobs)
    counts=collections.defaultdict(collections.Counter)
    for r in results: counts[r['variant']+'_'+r['arm']][r['status']]+=1
    status=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                counts=dict(counts),pending_cpu=len(cpu),pending_gpu_configs=len(jobs),
                target_evaluations=len(results),complete=not cpu and not gpu)
    atomic(RUN/'results.json',dict(status=status,rows=results))
    atomic(RUN/'progress.json',status)
    print('PROGRESS',json.dumps(status),flush=True)
    return status

def active_jobs():
    return {r.split('|')[0] for r in subprocess.check_output(['squeue','-u','3529','-h','-r','-o','%i|%j'],text=True).splitlines()}

def ensure_workers(state,status):
    active=active_jobs()
    for kind,n in [('cpu',32),('gpu',4)]:
        job=state.get(kind)
        if job and any(x==job or x.startswith(job+'_') for x in active): continue
        if not status['pending_cpu' if kind=='cpu' else 'pending_gpu_configs']: continue
        cmd=['sbatch','--parsable','-p','cpu' if kind=='cpu' else 'gpu',
             '-c','1' if kind=='cpu' else '6','--mem=16G' if kind=='cpu' else '--mem=64G',
             '-t','12:00:00' if kind=='cpu' else '24:00:00',f'--array=0-{n-1}%{n}',
             '-J','shared_'+kind,'-o',str(RUN/f'logs/{kind}_%A_%a.log'),'--export=ALL']
        if kind=='gpu': cmd+=['--gres=gpu:1','--constraint=a100-80g']
        cmd += [str(RUN/'code/worker.sbatch'),kind]
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
        state[kind]=job
        with (RUN/'jobs.jsonl').open('a') as f: f.write(json.dumps(dict(kind=kind,job=job,workers=n))+'\n')
        atomic(RUN/'worker_jobs.json',state)

def main():
    verify_sources()
    with (RUN/'controller.lock').open('a') as h:
        fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(RUN/'worker_jobs.json') if (RUN/'worker_jobs.json').exists() else {}
        while True:
            status=sweep()
            ensure_workers(state,status)
            if status['complete']:
                atomic(RUN/'pipeline_audit.json',dict(source_binding=sha(RUN/'sources.json'),**status))
                return
            time.sleep(45)

if __name__=='__main__':
    if sys.argv[1:] == ['sweep']: sweep()
    else: main()
