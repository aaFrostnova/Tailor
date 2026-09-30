"""Freeze test-only examples on the smoke image role, never final images."""
from rigor_protocol import ROOT, atomic, digest
from final_holdout import freeze_selection, plan_tasks, write_once


def make(name, n, config, attacks):
    source = ROOT / 'plans' / f'smoke_{name}_fixture.json'
    write_once(source, dict(test_only=True, main_table_result=False, cfg=config, attacks=attacks))
    rows = []
    for i, fpr in enumerate((.01, 1e-6)):
        rows.append(dict(**{'class': 'C4'}, arm='04', request_id=i,
            inputs=dict(attacks=attacks, fpr=fpr, min_psnr_db=30., max_ms=8000.),
            terminal_status='test_only_fixed_configuration',
            selected=dict(cfg=config, source=dict(path=str(source), sha256=digest(source), rank=1,
                                                request_id=i), selection_metrics={})))
    frozen = ROOT / 'plans' / f'smoke_{name}_selection.json'
    freeze_selection(rows, ROOT / 'plans/image_splits.json', frozen, mode='smoke', smoke_n=n)
    return plan_tasks(frozen, ROOT / 'plans' / f'smoke_{name}_tasks.json')


if __name__ == '__main__':
    result = make('editing_revision', 2, dict(order=['VINE'], s={'VINE': .7}, fe_on=[]),
                  ['jpeg25', 'editing_ip2p_s20_v1'])
    print('editing_revision', 'tasks=', len(result['tasks']), 'images=', 2, flush=True)
