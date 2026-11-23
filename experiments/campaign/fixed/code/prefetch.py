"""Pick one ready rank-1 configuration while full enumeration is still running."""
import hashlib
import json
from campaign import ROOT, RUN_ROOT, CLASSES, ARMS, read, atomic, task_status
from capacity_protocol import PROTOCOL, exact_cfg_key, exact_cfg_id, valid_cell


def next_task():
    for k in reversed(CLASSES):
        for arm in ARMS:
            configs = {}
            for path in sorted((ROOT / 'checkpoints' / f'{k}_{arm}').glob('q*.json')):
                row = read(path)
                if row.get('protocol') != PROTOCOL:
                    raise ValueError('legacy enumeration cannot enter unified prefetch')
                if not row['candidates']:
                    continue
                cand = row['candidates'][0]
                key = list(map(list, exact_cfg_key(cand['order'], cand['fe_on'], [cand['s'][f] for f in cand['order']])))
                cfgid = exact_cfg_id(key)
                entry = configs.setdefault(cfgid, {'key': key, 'cand': cand, 'attacks': set(), 'rows': []})
                entry['attacks'].update(row['attacks'])
                entry['rows'].append(row)
            for cfgid, c in configs.items():
                # Checking absence here is only a scheduling hint. The measurement
                # worker validates and reuses both cells and certification groups.
                attacks = sorted(c['attacks'])
                def complete(attack):
                    path = RUN_ROOT / 'live_topk/cells' / f'{cfgid}_{attack}_pool300_n100.json'
                    if not path.exists():
                        return False
                    return valid_cell(read(path).get('cell'), c['cand']['order'],
                                      30 if attack == 'unmarker' else 100,
                                      cfg=dict(order=c['cand']['order'], fe_on=c['cand']['fe_on']))
                if all(complete(a) for a in attacks):
                    continue
                digest = hashlib.sha256(json.dumps([cfgid, attacks]).encode()).hexdigest()[:16]
                base = ROOT / 'plans/prefetch' / f'{k}_{arm}_{digest}'
                plan, rows, manifest = [base.with_suffix(s) for s in ('.plan.json', '.rows.json', '.tasks.json')]
                if not manifest.exists():
                    atomic(plan, {'cls': k, 'arm': arm, 'rank': 1,
                                  'configs': [{'key': c['key'], 'cand': c['cand'], 'attacks': attacks}]})
                    atomic(rows, c['rows'])
                    atomic(manifest, [{'plan': str(plan), 'index': 0, 'prefetch_rows': str(rows)}])
                status = task_status(manifest, 0)
                if not status.exists() or not read(status).get('complete'):
                    return str(manifest)
    return None
