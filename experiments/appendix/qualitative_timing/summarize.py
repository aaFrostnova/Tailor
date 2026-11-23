"""Verify raw fresh timing arrays and construct three measured figure records."""
from pathlib import Path
import argparse,copy,hashlib,json,math
import numpy as np

OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--status-only',action='store_true');args=ap.parse_args()
    plan=read(OUT/'plan.json');plan_sha=sha(OUT/'plan.json');results={};status=[]
    for i,t in enumerate(plan['tasks']):
        folder=OUT/'tasks'/t['cfg_id'];rp=folder/'result.json'
        state=read(folder/'progress.json') if (folder/'progress.json').exists() else {'phase':'queued'}
        status.append({'task':i,'cfg_id':t['cfg_id'],'complete':rp.exists(),'progress':state})
        if rp.exists():results[t['cfg_id']]=read(rp)
    print(json.dumps({'complete_configs':len(results),'total_configs':len(plan['tasks']),'states':status},indent=2))
    if args.status_only:return
    assert len(results)==len(plan['tasks']),'fresh measurements are not complete'
    verified=0
    for key,result in results.items():
        assert result['complete'] and result['plan_sha256']==plan_sha
        enc=result['encoder'];assert enc['plan_sha256']==plan_sha and enc['n']==100
        e=np.array(enc['embed_ms']);assert e.shape==(100,plan['repeats']) and (e>0).all()
        assert sha(enc['fresh_embeddings_npy'])==enc['fresh_embeddings_sha256']
        for attack,byalpha in result['decoders'].items():
            for alpha,r in byalpha.items():
                assert r['complete'] and not r['old_timing_reused'] and r['plan_sha256']==plan_sha
                assert float(alpha)==r['alpha'] and r['cfg_id']==key
                d=np.array(r['decode_ms']);assert d.shape==(r['n'],plan['repeats']) and (d>0).all()
                cost=e[:r['n']]+d
                assert np.array_equal(cost,np.array(r['embed_plus_decode_ms']))
                assert abs(float(cost.mean())-r['mean_ms'])<1e-9
                assert all(r['repeated_decision_consistency'])
                assert r['metadata']['gpu_model']==plan['required_gpu'] and r['metadata']['synchronized']
                verified+=1
    summary=[];changes=[]
    for case in plan['cases']:
        assert sha(case['evidence_path'])==case['evidence_sha256']
        data=copy.deepcopy(read(case['evidence_path']));req=data['request'];alpha=repr(req['fpr'])
        folder=OUT/f"case_{case['case']:02d}";folder.mkdir(exist_ok=True)
        for m in data['methods']:
            old={'method':m['method'],'accepted':m['accepted'],'psnr_mean':m['psnr_mean'],'attacks_passed':m['attacks_passed']}
            result=results[m['cfg_id']];assert result['cfg']==m['cfg']
            rr={a:result['decoders'][a][alpha] for a in req['attacks']}
            m.pop('historical_timing',None)
            m['selection_latency_model_ms']=m.pop('latency_model_ms')
            m['measured_latency_ms']=max(x['mean_ms'] for x in rr.values())
            m['latency_limiting_attack']=max(rr,key=lambda a:rr[a]['mean_ms'])
            m['fresh_embeddings_npy']=result['encoder']['fresh_embeddings_npy']
            m['fresh_embeddings_sha256']=result['encoder']['fresh_embeddings_sha256']
            m['psnr_mean']=result['encoder']['psnr_mean']
            assert abs(m['psnr_mean']-old['psnr_mean'])<=.001,('embedding reproduction drift',m['method'],m['psnr_mean'],old['psnr_mean'])
            m['fresh_timing']={'complete':True,'gpu':plan['required_gpu'],'n_images_by_attack':{a:r['n'] for a,r in rr.items()},
                'repeats':plan['repeats'],'warmup':plan['warmup'],'mean_ms_by_attack':{a:r['mean_ms'] for a,r in rr.items()},
                'result_path':str(OUT/'tasks'/m['cfg_id']/'result.json'),
                'result_sha256':sha(OUT/'tasks'/m['cfg_id']/'result.json')}
            for a,r in rr.items():
                av=m['attacks'][a];assert av['n']==r['n']
                av.update(rate=r['rate'],mean_ba=r['mean_ba'],passed=r['rate']>=.9-1e-12 and r['mean_ba']>=m['tau']-1e-12,
                    rate_ok=r['rate']>=.9-1e-12,mean_ok=r['mean_ba']>=m['tau']-1e-12,
                    passed_images=sum(r['detected']),displayed_image_ba=r['accepted_scores'][data['displayed_image']['index']],
                    displayed_image_pass=r['detected'][data['displayed_image']['index']],
                    source=str(OUT/'tasks'/m['cfg_id']/f'decode_{a}_{alpha}.json'))
            m['attacks_passed']=sum(av['passed'] for av in m['attacks'].values())
            m['failed_attacks']=[a for a,v in m['attacks'].items() if not v['passed']]
            m['min_attack_rate']=min(v['rate'] for v in m['attacks'].values())
            m['all_attacks_passed']=m['attacks_passed']==len(req['attacks'])
            m['psnr_ok']=m['psnr_mean']>=req['psnr_min']-1e-9
            m['latency_ok']=m['measured_latency_ms']<=req['latency_max_ms']+1e-9
            m['accepted']=m['all_attacks_passed'] and m['psnr_ok'] and m['latency_ok']
            changes.append({'case':case['case'],'before':old,'after':{k:m[k] for k in old},'measured_latency_ms':m['measured_latency_ms']})
            summary.append({'case':case['case'],'scenario':req['scenario'],'method':m['method'],'latency_ms':m['measured_latency_ms'],
                            'psnr':m['psnr_mean'],'attacks_passed':m['attacks_passed'],'accepted':m['accepted']})
        data['schema']='qualitative_fresh_timing_v1'
        data['interpretation']=['Fixed configurations and original request inputs are unchanged; no selection search is rerun.',
            'Fresh synchronized A100 encoder and single-alpha decoder timing, three repetitions after three warmup calls.',
            'The depicted native encodings and PSNR are freshly reproduced; live-image attacks are redecoded.',
            'Existing exact-configuration external attack pixels are reused; missing standard attack images are regenerated.',
            'The latency gate uses measured mean encoder+decoder cost, maximized over the requested attacks.',
            'FPR remains an analytical bound under the declared null, not a fresh empirical FPR estimate.']
        data['source_sha256'].update({str(OUT/'plan.json'):plan_sha})
        (folder/'request_evidence.json').write_text(json.dumps(data,indent=2)+'\n')
    (OUT/'fresh_results_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'remeasurement_comparison.json').write_text(json.dumps(changes,indent=2)+'\n')
    (OUT/'audit.json').write_text(json.dumps({'passed':True,'configs':len(results),'configuration_attack_alpha_cells':verified,
        'methods_by_request':len(summary),'all_timings_fresh':True,'raw_array_aggregation_verified':True,'plan_sha256':plan_sha},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
