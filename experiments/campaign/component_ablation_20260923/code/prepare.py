"""Freeze two single-component restrictions on the existing retained ablation IDs."""
import collections
import datetime
from common import *

def prepare():
    assert not (RUN/'manifest.json').exists()
    baseline=read(PREVIOUS/'sources.json')
    for p,h in baseline['hashes'].items():assert sha(p)==h,('changed frozen source',p)
    source=U/'enumeration_geo_20260923/inputs/requests.json'
    absource=PAPER/'data/ablation_merged_requests_1124.json'
    wanted={r['spec_id'] for r in read(absource)}
    rows=[]
    for row in read(source):
        if row['spec_id'] in wanted:
            rows.append(dict(row,source_i=row['i'],i=len(rows)))
    assert len(rows)==len({r['spec_id'] for r in rows})==1111
    assert dict(collections.Counter(r['cls'] for r in rows))==dict(C1=195,C2=191,C3=279,C4=150,C5=296)
    atomic(RUN/'inputs/requests.json',rows)
    targets=[dict(cohort='ablation',variant=v,arm='04',cls=r['cls'],i=r['i'],
        spec_id=r['spec_id'],request=r['request']) for r in rows for v in VARIANTS]
    atomic(RUN/'inputs/reference_full.json',read(PAPER/'data/ablation_evaluation.json')['ablations']['04']['Full'])
    atomic(RUN/'inputs/provenance.json',dict(source_sha256={str(p):sha(p) for p in
        (source,absource,PAPER/'data/ablation_evaluation.json')}))
    atomic(RUN/'manifest.json',dict(policy=POLICY,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        variants=list(VARIANTS),targets=targets,priority_ablation_specs=sorted(r['spec_id'] for r in rows),
        requests=1111,evaluations=2222,arm='04',protocol=CP.PROTOCOL,
        restrictions=dict(SingleOptimized='selected fragment count <= 1; retain strength and compatible geometry optimization',
            UnitStrength='strength == 1 for each selected fragment; retain subset, order and compatible geometry search'),
        initial_rate_margin=.04,initial_mean_margin=.02,live_rate_floor=.9,
        schedule='three distinct-structure candidates, one request-local corrected candidate after complete failure; then margins (.02,.01) and (0,0); pending data never advances the search',
        live_n=100,live_offset=300,unmarker_n=30,final_required=False,
        cache='exact executed configuration, attack, image split, raw-BA protocol; unchanged native live measurement',
        cpu_workers=24,gpu_workers=4,scheduling='CPU proposals and aggregation on CPU; GPU jobs only for missing live measurements'))
    paths=[RUN/'manifest.json',*list((RUN/'inputs').glob('*.json')),
        *list((RUN/'code').glob('*.py')),*list((RUN/'code').glob('*.sbatch'))]
    hashes=dict(baseline['hashes']);hashes.update({str(p):sha(p) for p in paths})
    atomic(RUN/'sources.json',dict(hashes=hashes))
    from controller import sweep
    print(json.dumps(sweep()),flush=True)

if __name__=='__main__':prepare()
