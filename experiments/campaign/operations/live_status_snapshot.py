"""Read current live cells without updating campaign decisions or measurements."""
import collections
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'fixed/code'), str(ROOT / 'code')]
import campaign
campaign.environment('04')
import capacity_protocol as CP


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


report = dict(started_at=now(), protocol=CP.PROTOCOL, live_rate_gate=0.9,
              source='read-only re-evaluation with frozen capacity_protocol.judge',
              independent_final_test=False, groups=[])
available = {p.name: p for p in (ROOT / 'live_topk/cells').glob('*.json')}
original_valid = CP.valid_cell
original_combine = CP.combine
valid_cache, combine_cache, loaded = {}, {}, {}


def memo_valid(cell, order, n, cfg=None, alphas=CP.REQUEST_ALPHAS):
    key = (id(cell), tuple(order), n, json.dumps(cfg, sort_keys=True), tuple(alphas))
    if key not in valid_cache:
        valid_cache[key] = original_valid(cell, order, n, cfg, alphas)
    return valid_cache[key]


def memo_combine(cell, order, tau, det_min=0.9):
    key = (id(cell), tuple(order), tau, det_min)
    if key not in combine_cache:
        combine_cache[key] = original_combine(cell, order, tau, det_min)
    return combine_cache[key]


CP.valid_cell, CP.combine = memo_valid, memo_combine

for folder in sorted((ROOT / 'live_topk').glob('C*_unified_ba_out*n2000_20260914')):
    if not (folder / 'requests_s00.json').exists():
        continue
    cls, tag = folder.name.split('_', 1)
    group_cells = {}
    for p in sorted((ROOT / ('certify_full_topk_' + folder.name) / cls).glob('g*.json')):
        record = json.loads(p.read_text())
        key = CP.exact_cfg_key(record['order'], record['fe'],
                               [record['s'][f] for f in record['order']])
        for attack, cell in record['attacks'].items():
            group_cells.setdefault((key, attack), []).append(
                dict(cell=cell, psnr_db=record.get('psnr_db'), source=str(p)))
    retained_groups = list(group_cells.values())

    def load_cell(key, attack):
        name = CP.exact_cfg_id(key) + '_' + attack + '_pool300_n100.json'
        n = 30 if attack == 'unmarker' else 100
        cfg = dict(order=list(key[0]), fe_on=list(key[1]))
        if name in available:
            if name not in loaded:
                loaded[name] = json.loads(available[name].read_text())
            record = loaded[name]
            if CP.valid_cell(record.get('cell'), list(key[0]), n, cfg=cfg):
                return record
        for record in group_cells.get((key, attack), []):
            if CP.valid_cell(record['cell'], list(key[0]), n, cfg=cfg):
                return record
        return None

    rows = []
    for p in sorted(folder.glob('requests_s*.json')):
        rows.extend(json.loads(p.read_text())['rows'])
    assert sorted(r['i'] for r in rows) == list(range(2000))
    patched = {}
    for p in sorted(folder.glob('patched_s*.json')):
        patched.update({int(k): v for k, v in json.loads(p.read_text())['candidates'].items()})
    counts, ranks, failures, pending_attacks = [collections.Counter() for _ in range(4)]
    examples = {}
    for row in rows:
        candidates = list(row['candidates'])
        if row['i'] in patched and not any(c['rank'] == 4 for c in candidates):
            candidates.append(patched[row['i']])
        if not candidates:
            counts['initial_model_unsat'] += 1
            continue
        for candidate in candidates:
            key = CP.exact_cfg_key(candidate['order'], candidate['fe_on'],
                                   [candidate['s'][f] for f in candidate['order']])
            cells = {a: load_cell(key, a) for a in row['attacks']}
            verdict = CP.judge(row, candidate, cells, 0.9)
            if verdict['pending']:
                counts['pending'] += 1
                pending_attacks.update(verdict['pending'])
                break
            counts['fully_measured_candidate_evaluations'] += 1
            if verdict['pass']:
                counts['accepted'] += 1
                ranks[candidate['rank']] += 1
                break
            counts['failed_candidate_evaluations'] += 1
            if not verdict['psnr_ok']:
                failures['PSNR'] += 1
            if not verdict['latency_ok']:
                failures['modeled_latency'] += 1
            for attack, stats in verdict['attacks'].items():
                if not stats['ok']:
                    failures[attack] += 1
                    examples.setdefault(attack, dict(request_index=row['i'], rank=candidate['rank'],
                        image_pass_rate=stats['rate'], mean_score=stats['mean'], tau=verdict['tau'],
                        passed_images=stats['passed_images'], n=stats['n']))
        else:
            counts['live_exhausted'] += 1
    assert sum(counts[k] for k in ('accepted', 'pending', 'initial_model_unsat', 'live_exhausted')) == 2000
    group = dict(group=folder.name, as_of=now(), counts=dict(counts), accepted_rank=dict(ranks),
                 failure_reasons=dict(failures), pending_attacks=dict(pending_attacks), examples=examples)
    report['groups'].append(group)
    print(json.dumps(group), flush=True)
    # Cached cells remain alive in loaded. Group-only cells must not leave stale id keys.
    valid_cache.clear()
    combine_cache.clear()

report['finished_at'] = now()
totals = collections.Counter()
for group in report['groups']:
    totals.update(group['counts'])
report['totals'] = dict(totals)
path = ROOT / 'operations' / ('live_snapshot_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S') + '.json')
path.write_text(json.dumps(report, indent=2) + '\n')
print('TOTALS', json.dumps(report['totals']), flush=True)
print('SAVED', path, flush=True)
