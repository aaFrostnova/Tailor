"""CPU preparation for fresh timing on the three fixed qualitative requests."""
from pathlib import Path
import hashlib, json, os, sys, random
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
B=OUT.parent
U=B/'results/rigor_editing_only_20260911/unified_ba_20260914'
CF=Path(os.environ.get('TAILOR_PROJECT', '.'))
sys.path[:0]=[str(B/'qualitative_edge_20260925'),str(CF),str(CF/'external/WatermarkAttacker')]
from image_sources import load_watermarked,load_cover,configuration_id

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def prepare_task(payload):
    ti,t,total=payload
    from src.attacks import attack_pil_any
    folder=OUT/'tasks'/t['cfg_id'];folder.mkdir(parents=True,exist_ok=True)
    wp=folder/'watermarked.npy'
    if not wp.exists():np.save(wp,np.stack([load_watermarked(t['cfg_id'],i) for i in range(100)]),allow_pickle=False)
    t['watermarked_npy']=str(wp);t['watermarked_sha256']=sha(wp)
    watermarked=np.load(wp,mmap_mode='r')
    for a,ar in t['attacks'].items():
        ap=folder/f'attack_{a}.npy';mp=folder/f'attack_{a}.json'
        if a in ['ctrlregen_s03','unmarker']:
            original=read(ar['original_cells'][0]);gp=Path(original['source']).resolve();g=read(gp)
            assert g['cfg_id']==t['cfg_id']
            assert configuration_id(dict(order=g['order'],fe_on=g['fe'],s=g['s']))==t['cfg_id']
            fp=gp.with_suffix('')/f'att_{a}'
            files=[fp/f'i{i:05d}.png' for i in range(ar['n'])]
            assert all(p.is_file() for p in files)
            ar['image_paths']=[str(p) for p in files]
            ar['image_sha256']=[sha(p) for p in files]
            ar['source_group']=str(gp);ar['source_group_sha256']=sha(gp)
            ar['pixel_source']='frozen exact-configuration live attack images'
        elif a in ['vaeB','vaeC','regen','rinse2x']:
            ar['npy']=str(ap);ar['metadata']=str(mp);ar['needs_gpu_preparation']=True
            ar['pixel_source']='fresh attack realization on frozen native encodings, seed 42+image index'
        else:
            if not mp.exists():
                arr=np.empty((ar['n'],512,512,3),dtype=np.uint8)
                for i in range(ar['n']):
                    random.seed(42+i);np.random.seed(42+i)
                    im=attack_pil_any(a,Image.fromarray(watermarked[i]),dev='cpu')
                    if im.size!=(512,512):im=im.resize((512,512))
                    arr[i]=np.asarray(im.convert('RGB'))
                np.save(ap,arr,allow_pickle=False)
                atomic(mp,dict(complete=True,n=ar['n'],cfg_id=t['cfg_id'],attack=a,sha256=sha(ap),seed_policy='42+image index',prepared_on='CPU'))
            assert read(mp)['sha256']==sha(ap)
            ar['npy']=str(ap);ar['metadata']=str(mp);ar['needs_gpu_preparation']=False
            ar['pixel_source']='fresh deterministic attack realization on frozen native encodings'
    atomic(folder/'task.json',t)
    print(json.dumps(dict(task=ti,total=total,cfg=t['cfg_id'][:12],attacks=len(t['attacks']),phase='cpu_prepared')),flush=True)
    return t

def main():
    from src.attacks import attack_pil_any
    os.environ['TQDM_DISABLE']='1'
    tasks={};cases=[]
    for ci in (1,2,3):
        ep=B/f'qualitative_diverse_20260925/case_{ci:02d}/request_evidence.json'
        e=read(ep);cases.append({'case':ci,'evidence_path':str(ep),'evidence_sha256':sha(ep),'request':e['request']})
        for m in e['methods']:
            key=m['cfg_id'];assert configuration_id(m['cfg'])==key
            t=tasks.setdefault(key,dict(cfg_id=key,cfg=m['cfg'],bindings=[],attacks={}))
            t['bindings'].append({'case':ci,'method':m['method'],'alpha':e['request']['fpr']})
            for a,rv in m['attacks'].items():
                ar=t['attacks'].setdefault(a,dict(n=rv['n'],alphas=[],original_cells=[]))
                assert ar['n']==rv['n']
                if e['request']['fpr'] not in ar['alphas']:ar['alphas'].append(e['request']['fpr'])
                if rv['source'] not in ar['original_cells']:ar['original_cells'].append(rv['source'])
    tasks=list(tasks.values())
    images=read(B/'ssim_preview_20260924/index/ssim_main_both_extended_index.json')['images']
    sources=[U/'code/final_measurement.py',U/'code/unified_detector.py',CF/'solver/eval_matrix.py',CF/'solver/composite_external_eval.py',CF/'src/attacks.py']
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks=list(pool.map(prepare_task,[(i,t,len(tasks)) for i,t in enumerate(tasks)]))
    atomic(OUT/'plan.json',dict(schema='fresh_qualitative_latency_v1',tasks=tasks,cases=cases,images=images,
        source_sha256={str(p):sha(p) for p in sources},warmup=3,repeats=3,
        required_gpu='NVIDIA A100-SXM4-80GB',latency_scope='synchronized warm encoder + single-alpha deployed decoder',
        original_requests_and_budgets_unchanged=True,old_latency_values_reused=False,
        attack_generation_excluded_from_latency=True))
    print('PREPARATION_COMPLETE',len(tasks),flush=True)

if __name__=='__main__':main()
