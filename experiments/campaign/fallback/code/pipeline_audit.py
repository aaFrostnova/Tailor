"""Main-table gate: independently replay verdicts from complete raw image cells."""
import copy
import functools
import hashlib
import json
from pathlib import Path

import capacity_protocol as P


@functools.lru_cache(maxsize=2048)
def read_cell(path):
    return json.loads(Path(path).read_text())


def cfg_id(candidate):
    key = P.exact_cfg_key(candidate['order'], candidate['fe_on'],
                          [candidate['s'][f] for f in candidate['order']])
    return P.exact_cfg_id(key)


def without_source(verdict):
    result = copy.deepcopy(verdict)
    for attack in result['attacks'].values():
        attack.pop('source', None)
    return result


def audit_row(row, path, cells_dir):
    """Reject skipped ranks, short samples, changed gates, stale stats or missing patch evidence."""
    assert row['protocol'] == P.PROTOCOL and 'min_bits' not in row
    assert not row['open'] and not any(v['pending'] for v in row['walk'])
    ranks = [c['rank'] for c in row['candidates']]
    assert ranks == sorted(set(ranks)) and all(c['certified'] for c in row['candidates'])
    assert all(1 <= rank <= 4 for rank in ranks)
    initial = [c for c in row['candidates'] if c['rank'] < 4]
    assert [c['rank'] for c in initial] == list(range(1, len(initial)+1)) and len(initial) <= 3
    assert all(a['psnr_db'] >= b['psnr_db'] - 1e-6 for a, b in zip(initial, initial[1:]))
    assert len({json.dumps(c['skeleton'], sort_keys=True) for c in initial}) == len(initial)
    visited, decisions = [], 0
    for v in row['walk']:
        assert not visited or not visited[-1]['pass'], 'continued after first live success'
        candidate = row['candidates'][len(visited)]
        assert v['rank'] == candidate['rank'] and v['cfg_id'] == cfg_id(candidate), 'skipped a candidate'
        assert set(v['attacks']) == set(row['attacks']), 'incomplete attack collection'
        cached = {}
        for attack in row['attacks']:
            n = 30 if attack == 'unmarker' else 100
            raw = read_cell(str(Path(cells_dir) / f"{v['cfg_id']}_{attack}_pool300_n100.json"))
            assert P.valid_cell(raw.get('cell'), candidate['order'], n,
                                cfg=dict(order=candidate['order'], fe_on=candidate['fe_on'])), (row['i'], attack, 'incomplete raw images')
            assert v['attacks'][attack]['n'] == n
            cached[attack] = raw
            decisions += n
        expected = P.judge(row, candidate, cached, det_min=.9)
        expected.update(rank=candidate['rank'], psnr_table=candidate['psnr_db'], cfg_id=cfg_id(candidate))
        assert without_source(v) == without_source(expected), (row['i'], v['rank'], 'raw live replay differs')
        visited.append(v)
    deployed = next((v['rank'] for v in visited if v['pass']), None)
    assert deployed == row['deployed_rank']
    assert deployed is not None or len(visited) == len(row['candidates']), 'unvisited candidate'
    rank4 = next((c for c in row['candidates'] if c['rank'] == 4), None)
    assert rank4 is None or initial, 'rank4 has no initial top-k search'
    if rank4 is not None or (row['candidates'] and deployed is None):
        p = Path(path) / f"patched_s{row['i']:02d}.json"
        patch = json.loads(p.read_text())
        key = str(row['i'])
        assert key in patch['evaluated'], 'rank4 solve missing'
        assert patch['evaluated'][key] == (rank4 is not None)
        assert patch['candidates'].get(key) == rank4, 'rank4 candidate differs from patch evidence'
    return dict(candidate_verdicts=len(visited), image_decisions=decisions,
                live_pass=int(deployed is not None), model_no_candidates=int(not row['candidates']))
