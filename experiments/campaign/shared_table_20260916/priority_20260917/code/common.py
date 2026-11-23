"""Shared-model comparison; frozen scientific modules are imported read-only."""
import hashlib
import json
import os
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parents[1]
NEW = RUN.parent
U = NEW.parent
PAPER = Path(os.environ.get('TAILOR_PAPER', 'paper'))
PY = os.environ.get('TAILOR_PYTHON', 'python')
sys.path[:0] = [str(U/'fixed/code'), str(U/'code'), str(NEW/'code')]
import campaign as C
import capacity_protocol as CP

VARIANTS = ('G_shared', 'S1', 'S3', 'S3_R', 'Full', 'Grid', 'NoRecoveryInteraction', 'NoFrontend')
REFINE = {'S3_R', 'Full', 'Grid', 'NoRecoveryInteraction', 'NoFrontend'}
FALLBACK = {'Full', 'Grid', 'NoRecoveryInteraction', 'NoFrontend'}
XENV = {'ctrlregen_s03', 'ctrlregen_s05', 'ctrlregen_s07', 'unmarker', 'editing_ip2p_s20_v1'}
read, atomic = C.read, C.atomic

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def task_id(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def output(task):
    return RUN/'proposals'/task['variant']/task['arm']/task['spec_id']/(task['stage']+'.json')

def margins(arm, stage):
    return (int(arm)/100, .02) if stage == 'initial' else {'04':(.04,.02), '02':(.02,.01), '00':(0.,0.)}[stage]

def cfgkey(c):
    return CP.exact_cfg_key(c['order'], c['fe_on'], [c['s'][f] for f in c['order']])

def tag():
    return 'shared_priority_20260917'

def group_path(cls, key):
    return U/f'certify_full_topk_{cls}_{tag()}'/cls/f'g{int(CP.exact_cfg_id(key)[:12],16):04d}.json'

def load_cell(c, attack):
    key = cfgkey(c)
    detector_cfg = dict(order=c['order'], fe_on=c['fe_on'])
    p = U/'live_topk/cells'/f'{CP.exact_cfg_id(key)}_{attack}_pool300_n100.json'
    if p.exists():
        rec = read(p)
        if CP.valid_cell(rec.get('cell'), c['order'], 30 if attack=='unmarker' else 100, cfg=detector_cfg):
            # Standard native cache entries include their exact configuration or
            # point to an export group with that identity. Reject foreign keys.
            if 'key' in rec and tuple(map(tuple,rec['key'])) != key:
                raise ValueError('cache configuration identity mismatch')
            return rec
    # Cross-environment decoding publishes the group before the shared cell.
    for cls in C.CLASSES:
        p = group_path(cls, key)
        if not p.exists():
            continue
        g = read(p)
        assert CP.exact_cfg_key(g['order'], g['fe'], [g['s'][f] for f in g['order']]) == key
        cell = g.get('attacks',{}).get(attack)
        if CP.valid_cell(cell, c['order'], 30 if attack=='unmarker' else 100, cfg=detector_cfg):
            return dict(cell=cell, psnr_db=g['psnr_db'], source=str(p), n=cell['n'])
    return None

def judge(request, c):
    return CP.judge(request, c, {a:load_cell(c,a) for a in request['attacks']}, .9)

def verify_sources():
    binding = read(RUN/'sources.json')
    for p, expected in binding['hashes'].items():
        assert sha(p) == expected, ('source changed', p)

def candidate(c):
    assert c['certified'], 'uncertified solution cannot enter live validation'
    cfg = c['cfg']
    return dict(rank=c['rank'], skeleton=c['skeleton'].as_dict(),
                **{k:cfg[k] for k in ('order','fe','fe_on','s','ms','live_threshold','threshold_count','max_tests','surrogate_protocol')},
                psnr_db=c['psnr_db'], certified=True)

def install_variant(variant, arm):
    C.boot(arm)
    import watermark_smt_v2 as W
    import z3
    from surrogate_model import Surrogate, PWL
    sg = Surrogate.from_dict(read(U/'fixed/inputs/surrogate_canonical.json'))
    if variant == 'NoRecoveryInteraction':
        sg._delta = {k:PWL(v.xs, [0.]*len(v.xs)) for k,v in sg._delta.items()}
        sg._frontend = {k:v for k,v in sg._frontend.items() if '|with_' not in k}
    original = W.build
    def build(**kw):
        b = original(**kw)
        o, selected = b[:2]
        if variant == 'Grid':
            for f,s in o._svars.items():
                grid = sorted({x for a in sg.attacks for x in sg.base(f,a).xs})
                o.add(z3.Implies(selected[f], z3.Or(*[s == x for x in grid])))
        if variant == 'NoFrontend':
            o.add(*[z3.Not(v) for v in o._fevars.values()])
        return b
    W.build = build
    return sg
