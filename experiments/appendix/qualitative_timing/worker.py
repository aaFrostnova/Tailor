"""Fresh encoder/decoder timing; attack preparation runs in a separate process."""
from pathlib import Path
import argparse,hashlib,json,os,platform,random,sys,time,traceback
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
B=OUT.parent
U=B/'results/rigor_editing_only_20260911/unified_ba_20260914'
CF=Path(os.environ.get('TAILOR_PROJECT', '.'))
sys.path[:0]=[str(U/'code'),str(CF),str(CF/'scripts'),str(CF/'solver'),str(CF/'external/WatermarkAttacker')]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp'+str(os.getpid()));tmp.write_text(json.dumps(x,allow_nan=False)+'\n');tmp.replace(p)
def progress(folder,**fields):
    fields.update(utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),unix=time.time())
    atomic(folder/'progress.json',fields);print(json.dumps(fields),flush=True)

def prepare_gpu(plan,task,folder):
    pending=[(a,v) for a,v in task['attacks'].items() if v.get('needs_gpu_preparation') and not Path(v['metadata']).exists()]
    if not pending:print('ATTACK_PREPARATION_ALREADY_COMPLETE',flush=True);return
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.get_device_name()==plan['required_gpu']
    from src.attacks import attack_pil_any
    assert sha(task['watermarked_npy'])==task['watermarked_sha256']
    wm=np.load(task['watermarked_npy'],mmap_mode='r')
    for attack,ar in pending:
        arr=np.empty((ar['n'],512,512,3),dtype=np.uint8)
        start=time.monotonic()
        for i in range(ar['n']):
            random.seed(42+i);np.random.seed(42+i);torch.manual_seed(42+i)
            image=attack_pil_any(attack,Image.fromarray(wm[i]),dev='cuda')
            if image.size!=(512,512):image=image.resize((512,512))
            arr[i]=np.asarray(image.convert('RGB'))
            if i%10==0:progress(folder,phase='attack_preparation',attack=attack,images=i+1,total=ar['n'])
        torch.cuda.synchronize()
        ap=Path(ar['npy']);tmp=ap.with_suffix('.tmp.npy');np.save(tmp,arr,allow_pickle=False);tmp.replace(ap)
        atomic(Path(ar['metadata']),dict(complete=True,n=ar['n'],cfg_id=task['cfg_id'],attack=attack,
            sha256=sha(ap),seed_policy='42+image index',prepared_on=torch.cuda.get_device_name(),seconds=time.monotonic()-start,
            excluded_from_deployment_latency=True))
        progress(folder,phase='attack_prepared',attack=attack,seconds=time.monotonic()-start)

def load_attacks(ar):
    if 'image_paths' in ar:
        result=[]
        for p,h in zip(ar['image_paths'],ar['image_sha256']):
            assert sha(p)==h
            im=Image.open(p).convert('RGB')
            if im.size!=(512,512):im=im.resize((512,512))
            result.append(im)
        return result
    mp=read(ar['metadata']);assert mp['complete'] and mp['sha256']==sha(ar['npy'])
    a=np.load(ar['npy'],mmap_mode='r');assert a.shape==(ar['n'],512,512,3)
    return [Image.fromarray(np.array(x,copy=True)) for x in a]

def measure(plan,task,folder):
    import torch
    import final_measurement as FM
    import unified_detector as UD
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.get_device_name()==plan['required_gpu']
    if (folder/'result.json').exists():
        r=read(folder/'result.json');assert r['complete'] and r['plan_sha256']==sha(OUT/'plan.json')
        print('TIMING_ALREADY_COMPLETE',flush=True);return
    for p,h in plan['source_sha256'].items():assert sha(p)==h,('source changed',p)
    covers=[]
    for row in plan['images']['images']:
        assert sha(row['path'])==row['sha256']
        im=Image.open(row['path']).convert('RGB').resize((512,512),Image.Resampling.BICUBIC)
        assert hashlib.sha256(im.tobytes()).hexdigest()==row['pixel_sha256']
        covers.append(im)
    progress(folder,phase='loading_encoder_decoder',cfg_id=task['cfg_id'])
    ns=FM.measurement_namespace();comp=FM.composite(ns,task['cfg'])
    decoder_cfg={k:task['cfg'][k] for k in ['order','s','fe_on']}
    metadata=FM.device_metadata(plan['required_gpu'])
    metadata.update(worker_sha256=sha(__file__),gpu_task_index=os.environ.get('SLURM_ARRAY_TASK_ID'),
        timing_path='single-alpha unified_decoder.decode_detailed, including configured geometric and supplementary reads',
        excluded=['attack preparation','PNG/array I/O','model loading','warmup'])
    for i in range(plan['warmup']):comp.embed(covers[i],i)
    torch.cuda.synchronize()
    ep=folder/'encoder.json'
    if ep.exists():
        enc=read(ep);assert enc['plan_sha256']==sha(OUT/'plan.json')
    else:
        timings=[];psnr=[];pixels=[]
        for i,cover in enumerate(covers):
            tt=[]
            for rep in range(plan['repeats']):
                (image,secret),ms=FM.timed_call(comp.embed,cover,i)
                tt.append(ms)
                if rep==0:arr=np.asarray(image.convert('RGB')).copy()
            assert arr.shape==(512,512,3)
            timings.append(tt);pixels.append(arr)
            delta=arr.astype(np.float64)-np.asarray(cover,dtype=np.float64)
            psnr.append(float(10*np.log10(255**2/max(float(np.mean(delta**2)),1e-9))))
            if i%20==0:progress(folder,phase='encoder_timing',images=i+1,total=100)
        np.save(folder/'fresh_embeddings.npy',np.stack(pixels),allow_pickle=False)
        enc=dict(complete=True,plan_sha256=sha(OUT/'plan.json'),cfg_id=task['cfg_id'],n=100,
            embed_ms=timings,psnr_db=psnr,psnr_mean=float(np.mean(psnr)),warmup=plan['warmup'],repeats=plan['repeats'],metadata=metadata,
            fresh_embeddings_npy=str(folder/'fresh_embeddings.npy'),fresh_embeddings_sha256=sha(folder/'fresh_embeddings.npy'))
        atomic(ep,enc)
    output={}
    for attack,ar in task['attacks'].items():
        # Input pixels and secrets are loaded before timing, outside the measured function.
        images=load_attacks(ar);secrets=[comp.secret_for(i) for i in range(ar['n'])]
        output[attack]={}
        for alpha in ar['alphas']:
            key=repr(alpha);rp=folder/f'decode_{attack}_{key}.json'
            if rp.exists():
                rr=read(rp);assert rr['complete'] and rr['plan_sha256']==sha(OUT/'plan.json')
                output[attack][key]=rr;continue
            progress(folder,phase='decoder_timing',attack=attack,alpha=alpha,images=0,total=ar['n'])
            for i in range(plan['warmup']):UD.decode_detailed(comp,images[i%len(images)],secrets[i%len(images)],alpha=alpha)
            torch.cuda.synchronize()
            timings=[];scores=[];detected=[];attempts=[];paths=[];repeat_consistency=[]
            for i,(image,secret) in enumerate(zip(images,secrets)):
                tt=[];reads=[]
                for rep in range(plan['repeats']):
                    result,ms=FM.timed_call(UD.decode_detailed,comp,image,secret,alpha=alpha)
                    tt.append(ms);reads.append(result)
                r=reads[0]
                assert r['threshold_count']==UD.threshold_count(decoder_cfg,alpha)
                assert r['attempted_tests']<=r['max_tests']
                timings.append(tt);scores.append(r['accepted_score']);detected.append(r['detected'])
                attempts.append(r['attempted_tests']);paths.append(r['accepted_path'])
                repeat_consistency.append(all(x['accepted_score']==r['accepted_score'] and x['detected']==r['detected'] and x['attempted_tests']==r['attempted_tests'] for x in reads))
                if i%20==0:progress(folder,phase='decoder_timing',attack=attack,alpha=alpha,images=i+1,total=ar['n'])
            total=np.array(enc['embed_ms'][:ar['n']])+np.array(timings)
            rr=dict(complete=True,plan_sha256=sha(OUT/'plan.json'),cfg_id=task['cfg_id'],attack=attack,alpha=alpha,n=ar['n'],
                image_indices=list(range(ar['n'])),decode_ms=timings,embed_plus_decode_ms=total.tolist(),
                mean_ms=float(total.mean()),p95_ms=float(np.percentile(total,95)),
                mean_encode_ms=float(np.array(enc['embed_ms'][:ar['n']]).mean()),mean_decode_ms=float(np.array(timings).mean()),
                accepted_scores=scores,detected=detected,attempted_tests=attempts,accepted_paths=paths,
                repeated_decision_consistency=repeat_consistency,mean_ba=float(np.mean(scores)),rate=float(np.mean(detected)),
                threshold_count=UD.threshold_count(decoder_cfg,alpha),warmup=plan['warmup'],repeats=plan['repeats'],metadata=metadata,
                attacks_are_outside_timing=True,old_timing_reused=False,attack_input_provenance=ar)
            assert all(repeat_consistency),'nonrepeatable decoder result'
            atomic(rp,rr);output[attack][key]=rr
            progress(folder,phase='decoder_complete',attack=attack,alpha=alpha,mean_ms=rr['mean_ms'],rate=rr['rate'])
    atomic(folder/'result.json',dict(complete=True,plan_sha256=sha(OUT/'plan.json'),cfg_id=task['cfg_id'],cfg=task['cfg'],
        encoder=enc,decoders=output,bindings=task['bindings'],metadata=metadata))
    progress(folder,phase='complete',cfg_id=task['cfg_id'])

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','measure']);p.add_argument('--index',type=int,required=True)
    a=p.parse_args();plan=read(OUT/'plan.json');task=plan['tasks'][a.index];folder=OUT/'tasks'/task['cfg_id']
    try:
        if a.phase=='prepare':prepare_gpu(plan,task,folder)
        else:measure(plan,task,folder)
    except Exception as exc:
        atomic(folder/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),phase=a.phase,index=a.index,time=time.time()))
        raise

if __name__=='__main__':main()
