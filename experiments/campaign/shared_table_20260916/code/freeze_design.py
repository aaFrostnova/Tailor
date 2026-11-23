"""Bind the proposed experiment to the existing frozen scientific inputs."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MAIN = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = MAIN / 'source_hashes.json'
    old = json.loads(manifest_path.read_text())
    bindings = {}
    for directory, pattern in [('fixed/inputs', '*.json'), ('fixed/code', '*.py'), ('code', '*.py')]:
        for path in sorted((MAIN / directory).glob(pattern)):
            expected = old.get(str(path))
            if expected is None:
                raise RuntimeError('unbound main input: ' + str(path))
            actual = digest(path)
            if actual != expected:
                raise RuntimeError('changed frozen main input: ' + str(path))
            bindings[str(path)] = actual
    requests = {}
    for c in range(1, 6):
        path = MAIN / 'fixed/inputs' / ('C' + str(c) + '.json')
        records = json.loads(path.read_text())['records']
        assert len(records) == 2000
        requests['C' + str(c)] = len(records)
    split = MAIN / 'plans/image_splits.json'
    if not split.exists():
        raise RuntimeError('missing main image split: ' + str(split))
    own = {str(p.relative_to(HERE)): digest(p)
           for folder in ('code', 'tests') for p in sorted((HERE / folder).glob('*.py'))}
    own['ABLATION_PLAN_CN.md'] = digest(HERE / 'ABLATION_PLAN_CN.md')
    variants = [
        dict(id='G_shared', search='greedy_subset_exact_local', top_k=1, refinement=False, fallback=False),
        dict(id='S1', search='global', top_k=1, refinement=False, fallback=False),
        dict(id='S3', search='global', top_k=3, refinement=False, fallback=False),
        dict(id='S3_R', search='global', top_k=3, refinement=True, fallback=False),
        dict(id='Full', search='global', top_k=3, refinement=True, fallback=True),
        dict(id='Grid', reference='Full', change='strengths_on_frozen_base_curve_knots'),
        dict(id='NoRecoveryInteraction', reference='Full', change='remove_recovery_interactions_including_pair_frontend_responses'),
        dict(id='NoFrontend', reference='Full', change='all_frontends_off_and_recompute_threshold'),
        dict(id='FixedOrder', reference='Full', priority='secondary', change='pin_actual_order_keep_ordered_interference'),
        dict(id='MeanOnlyScreen', reference='Full', priority='secondary', change='disable_offline_rate_gate_only'),
    ]
    data = dict(
        design_version='shared_table_ablation_design_v1',
        status='design_and_greedy_cpu_smoke_only', formal_gpu_campaign_started=False,
        main_source_manifest=str(manifest_path), main_source_manifest_sha256=digest(manifest_path),
        frozen_main_files_verified=bindings, new_files_sha256=own,
        image_split=dict(path=str(split), sha256=digest(split)),
        unique_request_count=10000, requests_by_class=requests, margin_arms=['04', '06'],
        shared_table=str(MAIN / 'fixed/inputs/surrogate_canonical.json'),
        primary_search_comparison=['G_shared', 'S1'],
        incremental_process_comparisons=[['S1', 'S3'], ['S3', 'S3_R'], ['S3_R', 'Full']],
        common=dict(live_required=True, live_rate_floor=.90, maximum_frontends=1,
                    selection_latency='same_modeled_deployment_latency',
                    final_latency='measured_deployment_latency',
                    final_feedback_to_selection=False, capacity_is_input=False,
                    missing_or_unknown='unresolved'),
        variants=variants)
    target = HERE / 'experiment_design.json'
    if target.exists():
        existing = json.loads(target.read_text())
        if existing != data:
            raise RuntimeError('design changed; create an explicit new version instead of overwriting')
    else:
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(design=str(target), verified_main_files=len(bindings),
                          variants=len(variants), unique_requests=10000)))


if __name__ == '__main__':
    main()
