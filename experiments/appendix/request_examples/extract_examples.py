"""Freeze the three request -> solution examples (TAILOR-F, m0 = 0.06).

Read-only reproduction: the verdict is recomputed with the paper's judge from
frozen live cells and must match the main-table record.  Per-fragment pass rates
use each fragment's own readout at the deployed configuration's threshold.
"""
import os
import json
import math
import sys
from pathlib import Path

B = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs'))
U = B / 'results/rigor_editing_only_20260911/unified_ba_20260914'
P = Path(os.environ.get('TAILOR_PAPER', 'paper'))
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(U / 'shared_table_20260916/priority_20260917/code'))
import common as C

# Code catalog for the ECC-based ID guidance: BCH of length 127 shortened to a 100-bit
# transmitted codeword. Parity lengths are bchlib's generator degrees at m = 7, read off
# BCHCodec(m=7, t).ecc_bits; ID bits are the 100 transmitted bits minus the parity. Radius 18
# is the largest the family offers, since beyond it the parity exceeds the codeword.
# (radius, parity bits, ID bits); the deployed code is the radius-10 entry.
# Each radius is the designed distance of the generator whose roots are the cyclotomic
# cosets it contains. The last entry corrects 21, not the 18 that a library restricted to
# m*t < 2**m will construct for the same 98 parity bits.
CATALOG = [(1, 7, 93), (2, 14, 86), (3, 21, 79), (4, 28, 72), (5, 35, 65), (6, 42, 58),
           (7, 49, 51), (9, 56, 44), (10, 63, 37), (11, 70, 30), (13, 77, 23), (14, 84, 16),
           (15, 91, 9), (21, 98, 2)]
MAX_RADIUS = max(t for t, _, _ in CATALOG)


def eligible(radius):
    """The cheapest catalog entry covering `radius` as (radius, parity, ID bits), else None."""
    return next((e for e in CATALOG if e[0] >= radius), None)


def k_max(radius):
    """Largest advised ID length for `radius`; 0 when the family offers no covering entry."""
    e = eligible(radius)
    return e[2] if e else 0


# (group in candidates.json, spec-id prefix, display image index)
PICKS = [('G1_single_S1', 'a8b3c84677', 52),
         ('G2_pair_S3', '6b099d7e86', 25),
         ('G3_triple_S5_geo', '0827fa7c09', 98)]
USED_ELSEWHERE = {'a6674c50181e516b5e4f3108875d56a0078f745d7e72f7eb565700306cd0b0ea',
                  '26564eed072228370608de2716783f7e6117bd1e490fdea7e719d62a1d9712f5',
                  'e9a9b78bba7a6795ee96353af9cd573ddfd5b88d16feb1a6c89133c4c52c7ada'}

cands = json.loads((OUT / 'candidates.json').read_text())
index = json.loads((B / 'ssim_preview_20260924/index/ssim_main_both_extended_index.json').read_text())
smt06 = {r['spec_id']: r for r in index['requestrows']['SMT06']}
specs = {s['spec_id']: s for s in json.loads((P / 'data/request_specifications.json').read_text())['rows']}
specs.update({s['spec_id']: s for s in json.loads((P / 'data/editing_requests.json').read_text())})
retained = set(json.loads((P / 'data/main_cohort.json').read_text())['retained_spec_ids'])

examples = []
for number, (group, prefix, image_index) in enumerate(PICKS, 1):
    row = next(r for r in cands[group] if r['spec_id'].startswith(prefix))
    sid = row['spec_id']
    assert sid in retained and sid not in USED_ELSEWHERE
    spec = specs[sid]
    req = spec['inputs']
    cfg = dict(order=row['order'], fe_on=row['frontend'], s=row['strengths'], ms=row['latency_model_ms'])
    assert C.CP.exact_cfg_id(C.cfgkey(cfg)) == row['cfg_id'] == smt06[sid]['cfg_id']
    cells = {a: C.load_cell(cfg, a) for a in req['attacks']}
    v = C.CP.judge(req, cfg, cells, .9)
    assert v['pass'] and not v['pending']
    assert abs(v['psnr_live'] - smt06[sid]['psnr_live']) < 1e-8
    h, T = v['threshold_count'], v['max_tests']
    bound = T * sum(math.comb(100, k) for k in range(h, 101)) / 2 ** 100
    assert bound <= req['fpr']
    attacks = {}
    for a in req['attacks']:
        av = v['attacks'][a]
        bt = cells[a]['cell']['by_threshold'][str(h)]
        assert bt['threshold_count'] == h and len(bt['detected']) == av['n']
        assert sum(bt['detected']) == av['passed_images']
        per_frag = {f: sum(x is not None and x * 100 >= h - 1e-6 for x in bt['per_fragment_ba'][f]) / av['n']
                    for f in cfg['order']}
        accepted_errors = [int(round((1 - x) * 100))
                           for x, ok in zip(av['image_bit_accuracy'], av['image_pass']) if ok]
        attacks[a] = dict(n=av['n'], rate=av['rate'], mean_ba=av['mean'], passed=av['ok'],
                          capacity_bits=av['capacity_bits_estimate'],
                          ecc_radius=max(accepted_errors) if accepted_errors else 0,
                          fragment_rates=per_frag,
                          fused_rate=(sum(x is not None and x * 100 >= h - 1e-6 for x in bt['fused_ba']) / av['n']
                                      if len(cfg['order']) > 1 else None),
                          geometry_images=sum(bool(x) for x in bt['used_geometry']))
    radius_attack = max(attacks, key=lambda a: attacks[a]['ecc_radius'])
    radius = attacks[radius_attack]['ecc_radius']
    examples.append(dict(example=number, spec_id=sid, scenario=spec['cls'].replace('C', 'S'),
        multiplicity=spec.get('multiplicity'), display_image_index=image_index,
        request=dict(attacks=req['attacks'], fpr=req['fpr'], psnr_min=req['min_psnr'],
                     latency_max_ms=req['max_ms']),
        cfg_id=row['cfg_id'], cfg=cfg,
        verdict=dict(tau=v['tau'], threshold_count=h, max_tests=T, fpr_bound=bound,
                     psnr_live=v['psnr_live'], latency_model_ms=v['latency_model_ms'],
                     all_attacks_passed=v['cols_ok'],
                     ecc_radius=radius, ecc_k_max=k_max(radius), ecc_bottleneck=radius_attack,
                     ecc_parity=(eligible(radius)[1] if eligible(radius) else None),
                     ecc_max_radius=MAX_RADIUS,
                     capacity_bits=v['capacity_bits_estimate'],
                     capacity_bottleneck=v['capacity_bottleneck'],
                     capacity_basis=v['capacity_basis'],
                     attacks=attacks)))
    print(number, spec['cls'], sid[:12], cfg['order'], cfg['fe_on'], f"psnr={v['psnr_live']:.2f}",
          f"h={h} T={T} bound={bound:.3g}",
          f"cap={v['capacity_bits_estimate']:.1f}b @{v['capacity_bottleneck']}",
          f"| ECC r_u={radius} @{radius_attack} k_max={k_max(radius)}", flush=True)

(OUT / 'selected_examples.json').write_text(json.dumps(dict(
    schema='request_solution_examples_v1', method='SMT06 (TAILOR-F, initial margin 0.06)',
    live_rate_floor=.9, examples=examples,
    notes=['Verdicts reproduced from frozen live cells with capacity_protocol.judge; no re-selection.',
           'Fragment rates: each selected fragment readout at the configuration threshold count.',
           'FPR bound is analytical under the declared fresh-query null.',
           'capacity_bits: capacity_protocol estimate, min over requested attacks of '
           '100*(1-H2(1-mean best-path bit accuracy)); an estimate from the measured bit '
           'accuracy, not a test of a different payload length or ECC decoder.',
           'ecc_radius: max pre-ECC bit errors over the images the verifier ACCEPTS, per attack; '
           'ecc_k_max: largest catalog information dimension whose guaranteed hard-decision radius '
           'covers the worst accepted image over the requested attacks.']), indent=1) + '\n')
print('wrote selected_examples.json')
