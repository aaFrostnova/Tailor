import collections
from common import *

def prepare():
    assert not (RUN/'manifest.json').exists(), 'already registered'
    for d in ('logs','tasks','claims','errors','status','proposals','gpu_tasks'):
        (RUN/d).mkdir(parents=True,exist_ok=True)
    specs = read(PAPER/'data/request_specifications.json')['rows']
    subset = read(PAPER/'data/ablation_unique_reporting.json')['rows']
    lookup = {r['spec_id']:r for r in specs}
    ab = {r['spec_id']:r for r in subset}
    records = {c:read(U/'fixed/inputs'/f'{c}.json')['records'] for c in C.CLASSES}
    targets = []
    for variant in VARIANTS:
        rows = specs if variant == 'G_shared' else subset
        for arm in C.ARMS:
            for item in rows:
                # The subset's input-defined representative is shared by every
                # paired variant; Greedy aliases it back to the full manifest.
                rep = ab.get(item['spec_id'],item)
                cls,i = rep['cls'],rep['representative_i']
                r = records[cls][i]
                assert all((sorted(r[k]) if k=='attacks' else r[k]) == lookup[item['spec_id']]['inputs'][k] for k in CP.REQUEST_FIELDS)
                targets.append(dict(variant=variant,arm=arm,cls=cls,i=i,spec_id=item['spec_id'],
                                    request={k:r[k] for k in CP.REQUEST_FIELDS}))
    hashes = {str(p):sha(p) for p in (PAPER/'data/request_specifications.json',PAPER/'data/ablation_unique_reporting.json',
                 U/'source_hashes.json',U/'fixed/inputs/surrogate_canonical.json',NEW/'code/shared_greedy.py')}
    frozen = read(U/'source_hashes.json')
    # Verify and bind scientific Python sources and input manifests, without
    # repeatedly hashing multi-GB pretrained weights on every task.
    for p,h in frozen.items():
        if p.endswith('.py') or p.startswith(str(U/'fixed/inputs')):
            assert sha(p)==h, ('frozen source changed',p)
            hashes[p]=h
    for p in sorted((RUN/'code').glob('*')):
        if p.is_file(): hashes[str(p)]=sha(p)
    atomic(RUN/'sources.json',dict(hashes=hashes,source_freeze_sha256=sha(U/'source_hashes.json')))
    atomic(RUN/'manifest.json',dict(protocol=CP.PROTOCOL,variants=list(VARIANTS),targets=targets,
         arms=['04','06'],live_n=100,unmarker_n=30,
         live_offset=300,live_rate_floor=.9,latency_gate='same frozen modeled latency',
         final_required=False,expert_included=False,secondary_variants_deferred=['FixedOrder','MeanOnlyScreen'],
         ablation_representative_alias='Greedy uses the frozen subset representative for paired specs; all four request inputs identical',
         priority_ablation_specs=sorted(ab)))
    print('REGISTERED',len(targets),dict(collections.Counter(t['variant'] for t in targets)))

if __name__=='__main__': prepare()
