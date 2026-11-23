"""Six independent two-image single-fragment adapter checks, never formal data."""
import json

from rigor_protocol import B, ROOT, MAIN, digest, read
from final_holdout import freeze_selection, normalize_config, plan_tasks, require, write_once
from baseline_campaign import METHODS, FRAGMENT_BY_METHOD, freeze_sources, verify_sources


def prepare():
    require(ROOT.name == 'smoke', 'use BASELINE_RUN_ROOT=SB/smoke')
    freeze_sources()
    verify_sources()
    rows = []
    for method in METHODS:
        path = B / method / 'plans/proposals.json'
        proposal = read(path)
        require(proposal['method'] == method and len(proposal['rows']) == 20000, 'wrong smoke proposal population')
        cfg = normalize_config(proposal['rows'][0]['config'])
        fragment = FRAGMENT_BY_METHOD[method]
        require(cfg['order'] == [fragment] and cfg['s'] == {fragment: 1.0} and not cfg['fe_on'],
                'smoke uses a nonregistered single-fragment rule')
        source = dict(path=str(path.resolve()), sha256=digest(path))
        fprs = sorted({row['request']['fpr'] for row in proposal['rows']})
        require(len(fprs) == 6, 'expected all six frozen request FPR values in smoke')
        for attack in ('jpeg25', 'editing_ip2p_s20_v1'):
            for fpr in fprs:
                rows.append(dict(**{'class': 'C1'}, arm='04', request_id=len(rows),
                    inputs=dict(attacks=[attack], fpr=fpr, min_psnr_db=None, max_ms=1e9),
                    terminal_status='test_only_fixed_configuration', smoke_case=f'{method}/{attack}',
                    selected=dict(cfg=cfg, source=source, selection_metrics={'smoke_only': True})))
    write_once(ROOT / 'plans/image_splits.json', read(MAIN / 'plans/image_splits.json'))
    frozen_path = ROOT / 'plans/smoke_selection.json'
    frozen = freeze_selection(rows, ROOT / 'plans/image_splits.json', frozen_path, mode='smoke', smoke_n=2)
    plan = plan_tasks(frozen_path, ROOT / 'plans/smoke_tasks.json')
    cases = {str(row['request_id']): row['smoke_case'] for row in frozen['requests']}
    return dict(tasks=len(plan['tasks']), requests=len(rows), task_cases={i: sorted({cases[member.split('/')[-1]] for member in task['requests']})
                for i, task in enumerate(plan['tasks'])}, heavy_attack_executor='byte-identical to audited H1/H2 final_measurement.py')


if __name__ == '__main__':
    print(json.dumps(prepare()))
