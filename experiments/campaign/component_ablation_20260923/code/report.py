"""Report equal-request component comparisons; never label partial rates final."""
import collections
import statistics
from common import *

def publish(rows,status):
    out={}
    for v in VARIANTS:
        group=[r for r in rows if r['variant']==v]
        accepted=[r for r in group if r['status']=='accepted']
        classes={}
        for cls in C.CLASSES:
            subset=[r for r in group if r['cls']==cls];passed=[r for r in subset if r['status']=='accepted']
            classes[cls]=dict(n=len(subset),accepted=len(passed),percent=100*len(passed)/len(subset))
        out[v]=dict(n=len(group),accepted=len(accepted),percent=100*len(accepted)/len(group),
            psnr=statistics.mean(r['verdict']['psnr_live'] for r in accepted) if accepted else None,
            status_counts=dict(collections.Counter(r['status'] for r in group)),scenarios=classes,
            complete=all(not r['status'].startswith('pending') for r in group),
            macro_percent=statistics.mean(x['percent'] for x in classes.values()))
    atomic(RUN/'summary.json',dict(progress=status,variants=out,full_reference=read(RUN/'inputs/reference_full.json')))
    files=list((RUN/'proposals').glob('ablation/*/04/*/initial.json'))
    configs={};timings=[]
    for p in files:
        record=read(p);timings.append(record['seconds'])
        for c in record['candidates']:
            key=cfgkey(c)
            configs.setdefault(key,set()).update(record['request']['attacks'])
    missing=collections.Counter();total=0;cached=0
    for key,attacks in configs.items():
        c=dict(order=list(key[0]),fe_on=list(key[1]),s=dict(zip(key[0],key[2])))
        for a in attacks:
            total+=1
            if load_cell(c,a) is None:missing[a]+=1
            else:cached+=1
    atomic(RUN/'workload.json',dict(initial_proposals_complete=len(files),initial_proposals_total=2222,
        distinct_initial_configs=len(configs),initial_config_attack_pairs=total,cached_pairs=cached,
        missing_pairs_by_attack=dict(missing),
        cpu_seconds=sum(timings),median_proposal_seconds=statistics.median(timings) if timings else None,
        scope='all proposed initial candidates; actual execution stops at first passing candidate; correction and fallback may add work'))
