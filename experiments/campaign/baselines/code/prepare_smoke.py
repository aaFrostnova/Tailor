"""Build separate two-image adapter checks, never counted as formal baseline data."""
import json
from pathlib import Path

from rigor_protocol import B, ROOT, MAIN, digest, read
from final_holdout import freeze_selection, normalize_config, plan_tasks, require, write_once
from baseline_campaign import freeze_sources, verify_sources


def prepare():
    require(ROOT.name == 'smoke', 'use BASELINE_RUN_ROOT=B/smoke')
    freeze_sources()
    verify_sources()
    sources = [B / method / 'plans/proposals.json' for method in ('H1_rule', 'H2_rule')]
    configs = {}
    for path in sources:
        source = dict(path=str(path.resolve()), sha256=digest(path))
        for row in read(path)['rows']:
            cfg = normalize_config(row['config'])
            key = (tuple(cfg['order']), tuple(cfg['fe_on']))
            configs[key] = (cfg, source)
    def match(predicate):
        return next(value for value in configs.values() if predicate(value[0]))
    h1 = match(lambda c: c['order'] == ['VINE', 'TrustMark'] and c['fe_on'] == ['resync'])
    tests = [
        ('single_fragment', 'jpeg25', match(lambda c: len(c['order']) == 1 and not c['fe_on'])),
        ('resync', 'rot9', h1),
        ('scale', 'crop50', match(lambda c: c['fe_on'] == ['scale'])),
        ('dual_frontend', 'crop50', match(lambda c: c['fe_on'] == ['resync', 'scale'])),
        ('editing', 'editing_ip2p_s20_v1', h1),
        ('ctrlregen', 'ctrlregen_s03', h1),
        ('unmarker', 'unmarker', h1),
    ]
    rows = []
    for index, (name, attack, (cfg, source)) in enumerate(tests):
        rows.append(dict(**{'class': 'C1'}, arm='04', request_id=index,
            inputs=dict(attacks=[attack], fpr=.01, min_psnr_db=None, max_ms=1e9),
            terminal_status='test_only_fixed_configuration', smoke_case=name,
            selected=dict(cfg=cfg, source=source, selection_metrics={'smoke_only': True})))
    write_once(ROOT / 'plans/image_splits.json', read(MAIN / 'plans/image_splits.json'))
    frozen_path = ROOT / 'plans/smoke_selection.json'
    freeze_selection(rows, ROOT / 'plans/image_splits.json', frozen_path, mode='smoke', smoke_n=2)
    plan = plan_tasks(frozen_path, ROOT / 'plans/smoke_tasks.json')
    return dict(tasks=len(plan['tasks']), task_cases={
        i: [tests[int(member.split('/')[-1])][0] for member in task['requests']]
        for i, task in enumerate(plan['tasks'])})


if __name__ == '__main__':
    print(json.dumps(prepare()))
