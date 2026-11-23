"""Direct primary-decoder superposition audit; complete ordered design, fixed unit strengths."""
import os,sys,json,hashlib,itertools,time,random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parent
CF=Path(os.environ.get('TAILOR_PROJECT', '.'))
U=Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/rigor_editing_only_20260911/unified_ba_20260914')
sys.path[:0]=[str(CF/'solver'),str(CF/'scripts'),str(CF),str(U/'code')]
from src.image_pool import sample,composition_of
from eval_matrix import OursComposite
import unified_detector as UD
FRAGS=['vine','trustmark','videoseal']
ORDERS=[list(p) for n in [1,2,3] for p in itertools.permutations(FRAGS,n)]
SEED=20260924
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(4)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 paths=sample(100,offset=300);assert len(paths)==len(set(paths))==100
 source=[CF/'solver/eval_matrix.py',CF/'src/vine_crypto_wrapper.py',CF/'src/trustmark_fragment.py',CF/'src/videoseal_fragment.py',CF/'src/soft_fusion.py',CF/'src/shortened_bch.py',U/'code/unified_detector.py',Path(__file__)]
 threshold=UD._threshold(3,.01)
 plan=dict(schema='direct_superposition_v1',n=100,seed=SEED,size=[512,512],orders=ORDERS,strengths={f:1. for f in FRAGS},attacks=['clean'],geometry=False,fusion=False,ecc_decoding=False,raw_bits=100,independent_decoder_family_alpha=.01,decoder_tests=3,common_threshold_count=threshold,composition=composition_of(paths),images=[dict(index=i,path=p,sha256=sha(p)) for i,p in enumerate(paths)],sources={str(p):sha(p) for p in source},comparison='Same image, secret and strength versus corresponding standalone fragment; all 6 pair and 6 triple orders, no selection by outcome.',scope='Clean-image coexistence at unit strength, not a guarantee under attacks or every strength.')
 if (ROOT/'plan.json').exists():assert json.loads((ROOT/'plan.json').read_text())==plan
 else:atomic(ROOT/'plan.json',plan)
 print('PLAN',json.dumps({k:plan[k] for k in ['n','composition','common_threshold_count']}),flush=True)
 comp=OursComposite('cuda',tm_variant='B',vine_variant='R',config=dict(order=FRAGS,strengths={f:1. for f in FRAGS},resync=False,nested=False,scale_search=False,angle_sweep=False,tile=False))
 assert not comp.geo and comp.sb.n==100
 start=time.time();(ROOT/'rows').mkdir(exist_ok=True)
 for i,path in enumerate(paths):
  out=ROOT/'rows'/f'{i:03d}.json'
  if out.exists():continue
  with Image.open(path) as im:cover=im.convert('RGB').resize((512,512),Image.Resampling.BICUBIC)
  iid,tx=comp.secret_for(i);cache={():cover};rows=[]
  def read(im,frags):
   result={}
   for f in frags:
    llr=comp._frag_llr(f,im,iid);matches=UD.strict_matches(llr,tx)
    result[f]=dict(matches=matches,bit_accuracy=matches/100,threshold_pass=matches>=threshold,exact_word=matches==100,llrs=np.asarray(llr).tolist())
   return result
  with torch.inference_mode():
   control=read(cover,FRAGS)
   for order in ORDERS:
    prefix=tuple(order[:-1]);comp.order=[order[-1]]
    # The no-frontend encoder is sequential. Reuse only exact shared prefix pixels.
    emb,secret=comp.embed(cache[prefix],i);assert iid==secret[0] and np.array_equal(tx,secret[1]);cache[tuple(order)]=emb
    obs=read(emb,order)
    mse=float(np.mean((np.asarray(cover,float)-np.asarray(emb,float))**2))
    rows.append(dict(order=order,pixel_sha256=hashlib.sha256(emb.tobytes()).hexdigest(),psnr=10*np.log10(255**2/max(mse,1e-9)),decoders=obs,all_selected_pass=all(v['threshold_pass'] for v in obs.values()),all_selected_exact=all(v['exact_word'] for v in obs.values())))
  atomic(out,dict(index=i,image_path=path,image_sha256=plan['images'][i]['sha256'],expected_word=tx.tolist(),control=control,measurements=rows))
  elapsed=time.time()-start
  print('IMAGE',i+1,'/100','seconds',round(elapsed,1),flush=True)
  atomic(ROOT/'progress.json',dict(completed_images=i+1,total_images=100,elapsed_seconds=elapsed,complete=i==99))
 atomic(ROOT/'complete.json',dict(complete=True,images=100,configurations=15,configuration_image_measurements=1500,fragment_readouts=3300,plan_sha256=sha(ROOT/'plan.json'),rows_sha256={p.name:sha(p) for p in sorted((ROOT/'rows').glob('*.json'))},seconds=time.time()-start))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
