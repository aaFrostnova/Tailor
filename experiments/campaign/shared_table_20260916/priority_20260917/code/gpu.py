"""Native unified live measurement, including all cross-environment attacks."""
import fcntl
import os
import runpy
import subprocess
import sys
from common import *

def run(task):
    c, attacks, cls = task['candidate'], task['attacks'], task['cls']
    key = cfgkey(c); digest = CP.exact_cfg_id(key)
    folder = RUN/'gpu_tasks'/task_id(task)
    folder.mkdir(parents=True,exist_ok=True)
    # Same configuration lock used by the existing main measurement workers.
    with (U/'fixed/checkpoints'/('cfg_'+digest+'.lock')).open('a') as h:
        fcntl.flock(h,fcntl.LOCK_EX)
        missing = [a for a in attacks if load_cell(c,a) is None]
        if missing:
            row = dict(i=0,attacks=missing,fpr=.01,min_psnr=0.,max_ms=1e20,candidates=[dict(c,rank=1)])
            atomic(folder/'rows.json',[row])
            atomic(folder/'plan.json',dict(cls=cls,arm='04',rank=1,configs=[dict(key=list(map(list,key)),cand=dict(c,rank=1),attacks=missing)]))
            # End the encoder/ordinary-attack process before loading large
            # cross-environment attack models on the same GPU.
            subprocess.run([PY,'-B','-u',__file__,'native',str(folder),cls],check=True)
            xm = [a for a in missing if a in XENV and load_cell(c,a) is None]
            if xm:
                p = group_path(cls,key)
                g = read(p)
                item = dict(cls=cls,arm='04',gidx=g['gidx'],domain=f'topk_{cls}_{tag()}',required=xm,path=str(p))
                atomic(folder/'xenv.json',[item])
                C.xenv_task(str(folder/'xenv.json'),0)
        for a in attacks:
            rec = load_cell(c,a)
            assert rec is not None, ('missing measurement',digest,a)
            # Publish native cross-environment records for exact reuse.
            p = U/'live_topk/cells'/f'{digest}_{a}_pool300_n100.json'
            if not p.exists(): atomic(p,dict(rec,key=list(map(list,key))))
    atomic(folder/'complete.json',dict(task=task,complete=True,protocol=CP.PROTOCOL))
    print('GPU_TASK_VERIFIED',digest,attacks,flush=True)

if __name__ == '__main__':
    verify_sources()
    if sys.argv[1]=='native':
        folder,cls=Path(sys.argv[2]),sys.argv[3]
        C.boot('04')
        os.environ.update(TAG=tag(),FULL_PREFETCH_ROWS=str(folder/'rows.json'),FULL_RANK_PLAN=str(folder/'plan.json'))
        source=U/'fixed/code/live_topk_full.py'
        sys.argv=[str(source),'measure_rank',str(C.CLASSES.index(cls)),'1','0','1','100']
        runpy.run_path(str(source),run_name='__main__')
    else:
        run(read(sys.argv[1]))
