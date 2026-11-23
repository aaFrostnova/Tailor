"""Audit and summarize all matched standalone/ordered-composition raw decoder reads."""
import hashlib,json,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 plan=json.loads((R/'plan.json').read_text());done=json.loads((R/'complete.json').read_text());assert done['complete'] and done['plan_sha256']==sha(R/'plan.json')
 assert all(sha(Path(p))==v for p,v in plan['sources'].items())
 assert all(sha(R/'rows'/p)==v for p,v in done['rows_sha256'].items())
 rows=[json.loads((R/'rows'/f'{i:03d}.json').read_text()) for i in range(100)];assert [r['index'] for r in rows]==list(range(100))
 frags=['vine','trustmark','videoseal'];out=[];rng=np.random.default_rng(20260924);boot=rng.integers(0,100,(10000,100))
 for row in rows:
  assert row['image_sha256']==plan['images'][row['index']]['sha256']
  assert [m['order'] for m in row['measurements']]==plan['orders']
  tx=np.array(row['expected_word'])
  for m in row['measurements']:
   assert set(m['decoders'])==set(m['order'])
   for f,v in m['decoders'].items():
    L=np.array(v['llrs']);assert np.isfinite(L).all() and L.shape==(100,)
    n=int((((L>0)&(tx==1))|((L<0)&(tx==0))).sum());assert n==v['matches'] and n/100==v['bit_accuracy'] and (n>=64)==v['threshold_pass'] and (n==100)==v['exact_word']
   assert m['all_selected_pass']==all(v['threshold_pass'] for v in m['decoders'].values())
   assert m['all_selected_exact']==all(v['exact_word'] for v in m['decoders'].values())
 for j,order in enumerate(plan['orders']):
  decoders={}
  for f in order:
   matches=np.array([r['measurements'][j]['decoders'][f]['matches'] for r in rows]);base=np.array([r['measurements'][frags.index(f)]['decoders'][f]['matches'] for r in rows]);delta=matches-base
   decoders[f]=dict(mean_ba_percent=float(matches.mean()),mean_delta_pp=float(delta.mean()),median_delta_pp=float(np.median(delta)),delta_bootstrap_ci95_pp=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),min_ba_percent=int(matches.min()),pass_count=int((matches>=64).sum()),exact_count=int((matches==100).sum()),n=100)
  out.append(dict(order=order,decoders=decoders,all_pass_count=sum(r['measurements'][j]['all_selected_pass'] for r in rows),all_exact_count=sum(r['measurements'][j]['all_selected_exact'] for r in rows),mean_psnr=float(np.mean([r['measurements'][j]['psnr'] for r in rows]))))
 report=dict(schema='direct_superposition_report_v1',complete=True,n_images=100,composition=plan['composition'],n_configurations=15,fragment_readouts=3300,settings={k:plan[k] for k in ['strengths','attacks','geometry','fusion','ecc_decoding','raw_bits','common_threshold_count','independent_decoder_family_alpha','decoder_tests']},results=out,plan_sha256=sha(R/'plan.json'),completion_sha256=sha(R/'complete.json'),row_hashes=done['rows_sha256'],interpretation='Clean-image independent raw readout coexistence at unit strengths; bit accuracy, threshold verification and exact-word recovery are distinct. No attack-wide orthogonality claim.',bootstrap='10,000 paired image bootstrap resamples, seed 20260924; individual config/decoder CI, not simultaneous bounds.')
 (R/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 for r in out:print('>'.join(r['order']),{f:(round(v['mean_ba_percent'],2),round(v['mean_delta_pp'],2),v['min_ba_percent'],v['exact_count']) for f,v in r['decoders'].items()},'allpass',r['all_pass_count'],'allexact',r['all_exact_count'])
 print('DONE',flush=True)
if __name__=='__main__':main()
