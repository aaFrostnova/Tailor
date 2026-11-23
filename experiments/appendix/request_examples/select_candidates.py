"""Shortlist real TAILOR-F (m0 = 0.06) requests for request -> solution examples.

Read-only: reuses frozen live cells and the paper's judge; nothing is re-embedded,
re-attacked, or re-selected.  Output: candidates.json next to this script.
"""
import os
import collections
import json
import sys
from pathlib import Path

B = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs'))
U = B / 'results/rigor_editing_only_20260911/unified_ba_20260914'
P = Path(os.environ.get('TAILOR_PAPER', 'paper'))
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(U / 'shared_table_20260916/priority_20260917/code'))
sys.path.insert(0, str(P / 'scripts'))
import common as C
import ecc_id_guidance as E

current = json.loads((P / 'data/main_cohort.json').read_text())
keep = set(current['retained_spec_ids'])
specs = {s['spec_id']: s for s in json.loads((P / 'data/request_specifications.json').read_text())['rows']}
specs.update({s['spec_id']: s for s in json.loads((P / 'data/editing_requests.json').read_text())})
index = json.loads((B / 'ssim_preview_20260924/index/ssim_main_both_extended_index.json').read_text())
latency = json.loads((U / 'enumeration_geo_20260923/manifest.json').read_text())
catalog = json.loads((P / 'data/ecc_catalog.json').read_text())


def main_cfg(cfg_id, attacks):
    """Frozen selection-time latency model, as in extract_diverse_evidence.py."""
    cfg = dict(index['configs'][cfg_id]['cfg'])
    assert C.CP.exact_cfg_id(C.cfgkey(cfg)) == cfg_id
    ms = sum(latency['base_latency_ms'][f] for f in cfg['order'])
    for fe in cfg['fe_on']:
        values = [latency['frontend_latency_ms'][fe].get(a) for a in attacks]
        values = [max(0., float(v)) for v in values if isinstance(v, (int, float))]
        ms += max(values) if values else latency['frontend_latency_fallback_ms'][fe]
    cfg['ms'] = ms
    return cfg


cell_cache = {}
def cells_for(cfg, cfg_id, attacks):
    out = {}
    for a in attacks:
        if (cfg_id, a) not in cell_cache:
            cell_cache[(cfg_id, a)] = C.load_cell(cfg, a)
        out[a] = cell_cache[(cfg_id, a)]
    return out


def ecc_advice(sid, cfg_id, attacks, verdict):
    bundle = dict(schema='ecc_live_verdict_v1', request_id=sid, configuration_id=cfg_id,
                  codeword_length=100, requested_attacks=list(attacks),
                  expected_images={a: (30 if a == 'unmarker' else 100) for a in attacks},
                  verdict=dict(verdict, cfg_id=cfg_id))
    try:
        r = E.recommend_id_space(bundle, catalog)
        return dict(status=r['status'], radius=r['required_correction_radius'],
                    id_bits_max=r['id_bits_max'], worst=r.get('worst_attacks'),
                    per_attack={a: v['maximum_errors'] for a, v in r['per_attack'].items()})
    except Exception as exc:  # report, never hide
        return dict(status='error', error=repr(exc))


def group_of(scen, cfg):
    n, fe = len(cfg['order']), bool(cfg['fe_on'])
    if scen == 'S1' and n == 1 and not fe:
        return 'G1_single_S1'
    if scen in ('S3', 'S4') and n == 2 and set(cfg['order']) == {'VINE', 'VideoSeal'}:
        return f'G2_pair_{scen}'
    if scen in ('S2', 'S5') and n == 3:
        return f'G3_triple_{scen}' + ('_geo' if fe else '')
    return None


rows = [r for r in index['requestrows']['SMT06'] if r['spec_id'] in keep]
groups = collections.defaultdict(list)
for r in rows:
    spec = specs[r['spec_id']]
    scen = spec['cls'].replace('C', 'S')
    g = group_of(scen, index['configs'][r['cfg_id']]['cfg'])
    if g:
        groups[g].append((r, spec, scen))

result = {}
for g, members in sorted(groups.items()):
    out = []
    for r, spec, scen in members:
        req = spec['inputs']
        cfg = main_cfg(r['cfg_id'], req['attacks'])
        v = C.CP.judge(req, cfg, cells_for(cfg, r['cfg_id'], req['attacks']), .9)
        assert not v['pending'] and v['pass'], (g, r['spec_id'])
        assert abs(v['psnr_live'] - r['psnr_live']) < 1e-8
        rates = {a: av['rate'] for a, av in v['attacks'].items()}
        out.append(dict(
            spec_id=r['spec_id'], scenario=scen, cfg_id=r['cfg_id'],
            order=cfg['order'], strengths=cfg['s'], frontend=cfg['fe_on'],
            attacks=req['attacks'], n_attacks=len(req['attacks']), fpr=req['fpr'],
            psnr_min=req['min_psnr'], latency_budget_ms=req['max_ms'],
            psnr_live=v['psnr_live'], latency_model_ms=v['latency_model_ms'],
            tau=v['tau'], threshold_count=v['threshold_count'], max_tests=v['max_tests'],
            rates=rates, min_rate=min(rates.values()),
            psnr_slack=v['psnr_live'] - req['min_psnr'],
            ecc=ecc_advice(r['spec_id'], r['cfg_id'], req['attacks'], v)))
    result[g] = out
    print(f'{g}: {len(out)} requests judged', flush=True)

(OUT / 'candidates.json').write_text(json.dumps(result, indent=1) + '\n')
print('wrote', OUT / 'candidates.json')
