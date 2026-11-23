"""One isolated, checkpointed initial or corrected proposal; never uses final images."""
import math
import sys
import time
from common import *

def run(task):
    path = output(task)
    if path.exists():
        assert read(path)['task'] == task
        return
    started = time.time()
    v, arm, stage = task['variant'], task['arm'], task['stage']
    r = dict(task['request'],i=task['i'],cls=task['cls'],stress=None,stress_kind=None,protocol=CP.PROTOCOL)
    registered=read(RUN/'inputs/requests.json')[task['i']]
    assert registered['spec_id']==task['spec_id'] and registered['request']==task['request']
    rate, mean = margins(arm, stage.removesuffix('_patch'))
    evidence = None
    if v in VARIANTS:
        sg = install_variant(v, arm)
        import watermark_smt_topk as T
        if stage.endswith('_patch'):
            parent = dict(task,stage=stage.removesuffix('_patch'))
            proposal = read(output(parent))
            assert proposal['candidates'] and v in REFINE
            from composite_prediction import predict_composite, estimate_rate_offset
            prior = read(U/'fixed/inputs/prior_width.json')
            prior_sd = float(prior.get('prior_sd_p90') or prior.get('prior_sd_median') or .02)
            offsets, rate_offsets, evidence = {}, {}, []
            for c in proposal['candidates']:
                verdict = judge(r,c)
                assert not verdict['pending'] and not verdict['pass'], 'patch requires every initial candidate completely measured and failed'
                for a in r['attacks']:
                    cell = load_cell(c,a)['cell']; n = cell['n']
                    observed,_,_,_ = CP.selected_views(cell,c['order'],verdict['tau'])
                    prediction = predict_composite(sg,c,a,tau=verdict['tau'])
                    for fi,f in enumerate(c['order']):
                        ba = observed[fi]
                        p = prediction['fragments'][f]
                        se = float(ba.std(ddof=1)/math.sqrt(n)) if n>1 else 0.
                        d = sg.live_offset(f,a,c['s'][f],float(ba.mean()),se_live=se,prior_sd=prior_sd,
                                           k=2.,asymmetric=True,predicted=p['mean'])
                        key = tuple(p['curve_key'])
                        if d < -1e-12: offsets[key] = min(offsets.get(key,0.),d)
                        re = estimate_rate_offset(sg,f,a,next(iter(c['fe_on']),None),c['s'][f],verdict['tau'],
                                                  float((ba>=verdict['tau']).mean()),n,.9)
                        if re['key'] is not None and re['offset'] < -1e-12:
                            rk = tuple(re['key']); rate_offsets[rk] = min(rate_offsets.get(rk,0.),re['offset'])
                        evidence.append(dict(rank=c['rank'],attack=a,fragment=f,prediction=p,
                                             observed_mean=float(ba.mean()),mean_offset=min(0.,d),rate=re))
            sg = sg.with_composite_offsets(offsets,rate_offsets)
        cs = T.enumerate_topk(CP.solver_scenario(r,margin=mean),k=1 if stage.endswith('_patch') else 3,
                              enable_order=True,continuous_strength=True,surrogate=sg,rate_margin=rate)
        candidates = [candidate(c) for c in cs]
        for c in candidates:
            assert len(c['order'])==1 if v=='SingleOptimized' else all(c['s'][f]==1. for f in c['order'])
        if stage.endswith('_patch'):
            for c in candidates: c['rank'] = 4
        outcome = 'sat' if candidates else 'model_unsat'
    atomic(path,dict(task=task,request=r,rate_margin=rate,mean_margin=mean,candidates=candidates,
                     outcome=outcome,evidence=evidence,seconds=time.time()-started,complete=True))
    print('PROPOSAL_COMPLETE',v,arm,task['cls'],task['i'],stage,len(candidates),flush=True)

if __name__ == '__main__':
    verify_sources()
    task = read(sys.argv[1])
    run(task)
