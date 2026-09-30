"""Freeze the same requests under the analytic unified-BA acceptance protocol."""
from pathlib import Path
from rigor_protocol import ROOT, SC, SOURCE, atomic, digest, read, validate_splits

PREVIOUS = SC / 'rigor_editing_only_20260911'
CF = Path('/data/tailor/project')
CLASSES = ('C1', 'C2', 'C3', 'C4', 'C5')
ARMS = ('04', '06')
NFPA = 'nfpa_sd21_xy40_s10_v1'


def verify_sources():
    paths = set((CF / 'src').rglob('*.py'))
    for folder in (CF/'scripts', CF/'scripts/defense', CF/'scripts/attack', SOURCE/'code', SC/'topk_capacity_output_20260910/code'):
        paths.update(folder.glob('*.py'))
    paths.add(SC/'topk_capacity_output_20260910/inputs/surrogate_canonical.json')
    for name in ('smt_inputs.json','smt_inputs_composite.json','smt_inputs_full.json'):
        paths.add(CF/'results/defense'/name)
    return {str(p): digest(p) for p in sorted(paths)}


def verify_scope():
    inputs = ROOT / 'fixed/inputs'
    table = read(inputs / 'surrogate_canonical.json')
    assert NFPA not in table['attacks']
    fields = ('i', 'attacks', 'fpr', 'min_psnr', 'max_ms')
    for cls in CLASSES:
        data = read(inputs/f'{cls}.json')
        old = read(PREVIOUS/'fixed/inputs'/f'{cls}.json')
        assert len(data['records']) == len(data['scenarios']) == len(old['records']) == 2000
        for index, (row, reference) in enumerate(zip(data['records'], old['records'])):
            assert row['i'] == index and NFPA not in row['attacks'] and 'min_bits' not in row
            assert all(row[field] == reference[field] for field in fields), 'request input changed'
    return table


def prepare():
    verify_scope()
    validate_splits(read(ROOT/'plans/image_splits.json'), verify_files=True)
    for folder in (ROOT/'fixed', ROOT/'fallback'):
        for name in ('inputs','logs','plans','checkpoints','stages','requests','summaries'):
            (folder/name).mkdir(parents=True, exist_ok=True)
    path = ROOT/'fixed/prepared.json'
    if path.exists():
        value = read(path)
        assert value['acceptance_protocol'] == 'unified_raw_ba_v1'
        expected={(cls,arm,index) for cls in CLASSES for arm in ARMS for index in range(2000)}
        actual=[tuple(row) for batch in read(ROOT/'fixed/plans/enumeration.json') for row in batch]
        assert len(actual)==20000 and set(actual)==expected, 'enumeration coverage changed'
        return value
    pending = [[cls, arm, index] for cls in CLASSES for arm in ARMS for index in range(2000)]
    atomic(ROOT/'fixed/plans/enumeration.json', [pending[i:i+10] for i in range(0,len(pending),10)])
    value = dict(protocol='four_inputs_capacity_output_unified_raw_ba_v1', acceptance_protocol='unified_raw_ba_v1',
        original_requests=10000, arm_request_evaluations=20000, reused_initial_enumerations=0,
        pending_enumerations=20000, calibration_passed=True,
        calibration_meaning='legacy offline measurements are a labelled prior; analytic threshold and mandatory new live checks',
        active_extension_attacks=['editing_ip2p_s20_v1'], excluded_attacks=[NFPA], previous_revision=str(PREVIOUS),
        final_test_feedback=False, old_detector_cells_reused=False, old_selection_choices_reused=False)
    atomic(path,value)
    atomic(ROOT/'fixed/followups.json', dict(runs=[dict(name='unified_ba_margin_fallback',run_dir=str(ROOT/'fallback'),required_audit='pipeline_audit.json')]))
    return value


def freeze_code():
    path=ROOT/'source_hashes.json'
    if path.exists():
        verify_frozen()
        sources=read(path)
        fallback=ROOT/'fallback/source_hashes.json'
        if fallback.exists():
            assert read(fallback)==sources, 'fallback source freeze differs'
        else:
            atomic(fallback,sources)
        return sources
    verify_scope()
    preflight=read(ROOT/'preflight_audit.json')
    assert preflight['passed'] is True and preflight['acceptance_protocol']=='unified_raw_ba_v1'
    sources=verify_sources()
    migration=read(ROOT/'plans/embedding_migration.json')
    assert migration['complete'] is True and migration['factory_equivalence_reviewed'] is True
    for name in ('source_hashes','measurement_files','model_weight_hashes','evidence'):
        sources.update(migration[name])
    for relative in ('code','fixed/code','fallback/code','fixed/inputs','tests','fixed/tests','fallback/tests','baselines/code','baselines/tests','single_baselines/code','single_baselines/tests'):
        for p in (ROOT/relative).glob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.sh','.sbatch'):
                sources[str(p)]=digest(p)
    for p in (ROOT/'plans').glob('*.json'):
        if p.name not in ('final_selection.json','final_tasks.json'):
            sources[str(p)]=digest(p)
    for base,methods in ((ROOT/'baselines',('H1_rule','H2_rule','smoke')),
                         (ROOT/'single_baselines',('S_VINE','S_TrustMark','S_VideoSeal','smoke'))):
        for p in (base/'policy_freeze.json',base/'protocol_sources.json'):
            assert p.exists(), ('baseline preregistration missing',str(p))
            sources[str(p)]=digest(p)
        for folder in [base/'inputs',*[base/m/'plans' for m in methods]]:
            for p in folder.glob('*.json'):
                if p.name not in ('final_selection.json','final_tasks.json'):
                    sources[str(p)]=digest(p)
    for p in (ROOT/'fixed/prepared.json',ROOT/'fixed/plans/enumeration.json',ROOT/'preflight_audit.json',ROOT/'combination_scope/plan.json'):
        assert p.exists(), ('required preregistration missing',str(p))
        sources[str(p)]=digest(p)
    for name, expected in preflight['evidence'].items():
        assert digest(name)==expected, ('preflight evidence changed',name)
        sources[name]=expected
    atomic(path,sources)
    atomic(ROOT/'fallback/source_hashes.json',sources)
    return sources


def verify_frozen():
    for name,expected in read(ROOT/'source_hashes.json').items():
        assert digest(name)==expected, ('frozen source changed',name)


if __name__=='__main__':
    import sys,json
    value=freeze_code() if sys.argv[1:]==['freeze'] else prepare()
    print(json.dumps(value if len(value)<30 else {'files':len(value)},indent=1))
