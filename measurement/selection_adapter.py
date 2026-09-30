"""Extract exact locked deployment configurations from complete fallback walks."""
from functools import lru_cache
from pathlib import Path
from rigor_protocol import ROOT, digest, read


def selection_rows(root=ROOT):
    root = Path(root)
    assert read(root / 'fallback/progress.json')['phase'] == 'complete'
    audit = read(root / 'fallback/pipeline_audit.json')
    assert audit['passed'] and audit['arm_request_evaluations'] == 20000
    @lru_cache(maxsize=None)
    def source_hash(path):
        return digest(path)
    @lru_cache(maxsize=None)
    def walk(path):
        rows = read(path)['rows']
        assert [x['i'] for x in rows] == list(range(2000))
        return rows
    output = []
    for cls in ('C1', 'C2', 'C3', 'C4', 'C5'):
        for arm in ('04', '06'):
            path = root / 'fallback/requests' / f'{cls}_{arm}.json'
            states = read(path)['rows']
            assert len(states) == 2000
            for state in states:
                assert state['status'] != 'pending_fallback'
                selected = None
                if state['selected'] is not None:
                    src = state['initial_source']
                    if state['status'] == 'fallback_live_pass':
                        assert state['attempts'][-1]['outcome'] == 'live_pass'
                        src = state['attempts'][-1]['source']
                    row = walk(src)[state['i']]
                    rank = state['selected']['rank']
                    assert row['deployed_rank'] == rank
                    candidate = next(c for c in row['candidates'] if c['rank'] == rank)
                    verdict = next(v for v in row['walk'] if v['rank'] == rank)
                    assert verdict['pass'] and not verdict['pending']
                    assert verdict['protocol'] == 'four_inputs_capacity_output_unified_raw_ba_v1'
                    selected = dict(cfg=dict(S=sorted(candidate['order']), order=candidate['order'],
                                   s=candidate['s'], fe_on=candidate['fe_on']),
                                   source=dict(path=src, sha256=source_hash(src), request_id=state['i'], rank=rank),
                                   selection_metrics=dict(psnr_db=verdict['psnr_live'],
                                     protocol=verdict['protocol'], threshold_count=verdict['threshold_count'],
                                     max_tests=verdict['max_tests'],
                                     capacity_bits_estimate=verdict['capacity_bits_estimate'],
                                     latency_model_ms=candidate['ms'], margins=state['selected_margins']))
                output.append(dict(**{'class': cls}, arm=arm, request_id=state['i'],
                    inputs=dict(attacks=state['attacks'], fpr=state['fpr'],
                                min_psnr_db=state['min_psnr'], max_ms=state['max_ms']),
                    terminal_status=state['status'], selected=selected,
                    source=dict(path=str(path), sha256=source_hash(str(path)))))
    assert len(output) == 20000
    return output


if __name__ == '__main__':
    from rigor_protocol import atomic
    out = selection_rows()
    atomic(ROOT / 'plans/selection_rows.json', out)
    print('Selection rows:', len(out), 'selected:', sum(x['selected'] is not None for x in out))
