"""Main-table controls on the 7,321-request cohort: A1 SingleOptimized and A3 AlwaysAllThree.

Frozen scientific modules are imported from the relocated root RUN/root (byte-identical copies of the
frozen runner, see relocate.py), so every new cell, exported group and configuration lock is written
under RUN/root. Old cells from the frozen cache are read only for attacks that passed the preregistered
bridging check (RUN/bridging/approval.json, written by bridge.py); without that file nothing frozen is
read. RUN comes from the environment variable TAILOR_MAIN_CONTROLS_RUN.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

RUN = Path(os.environ['TAILOR_MAIN_CONTROLS_RUN']).resolve()
ROOT = RUN / 'root'
FROZEN = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/'
              'rigor_editing_only_20260911/unified_ba_20260914')
POLICY = 'exp_main_controls_v1'
TAG = 'exp_main_controls_20261109'
PY = os.environ.get('TAILOR_PYTHON', 'python')
sys.path[:0] = [str(ROOT / 'fixed/code'), str(ROOT / 'code'), str(ROOT / 'shared_table_20260916/code')]
# This directory stays ahead of the frozen code; module names here never collide with frozen ones
# (the frozen fixed/code has a controller.py, hence mc_controller.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as C  # noqa: E402
import capacity_protocol as CP  # noqa: E402
import margin_fallback_policy as MFP  # noqa: E402

for _module in (C, CP, MFP):
    if Path(_module.__file__).resolve().parent != ROOT / 'fixed/code':
        raise ImportError(f'{_module.__name__} must come from the relocated root, not {_module.__file__}')
if C.RUN_ROOT != ROOT:
    raise ImportError(f'campaign writes under {C.RUN_ROOT}, expected {ROOT}')

VARIANTS = ('SingleOptimized', 'AlwaysAllThree')
FRAGMENTS = ('VINE', 'TrustMark', 'VideoSeal')
ARM = '06'
STAGES = ('initial',) + tuple(MFP.schedule(ARM))          # initial, 04, 02, 00
# A3 relaxes only the rule that every selected fragment must clear one requested attack.
NO_PASSENGERS = {'SingleOptimized': True, 'AlwaysAllThree': False}
XENV = {'ctrlregen_s03', 'ctrlregen_s05', 'ctrlregen_s07', 'unmarker', 'editing_ip2p_s20_v1'}
CELLS = ROOT / 'live_topk/cells'
FROZEN_CELLS = FROZEN / 'live_topk/cells'
APPROVAL = RUN / 'bridging/approval.json'
read, atomic = C.read, C.atomic


def transient(exc):
    """NFS errors a reader on another node can hit while a writer replaces a file; retry, never abort."""
    import errno
    if isinstance(exc, json.JSONDecodeError):
        return True
    return isinstance(exc, OSError) and exc.errno in (errno.ESTALE, errno.EIO)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_id(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def output(task):
    return RUN / 'proposals' / task['variant'] / task['arm'] / task['spec_id'] / (task['stage'] + '.json')


def margins(arm, stage):
    """(rate margin, mean margin): the initial margin of the arm, then the frozen backoff stages."""
    if stage == 'initial':
        return int(arm) / 100, .02
    return MFP.STAGES[stage]


def cfgkey(c):
    return CP.exact_cfg_key(c['order'], c['fe_on'], [c['s'][f] for f in c['order']])


def tag():
    return TAG


def group_path(cls, key):
    return ROOT / f'certify_full_topk_{cls}_{tag()}' / cls / f'g{int(CP.exact_cfg_id(key)[:12], 16):04d}.json'


def cell_name(key, attack):
    return f'{CP.exact_cfg_id(key)}_{attack}_pool300_n100.json'


def expected_n(attack):
    return 30 if attack == 'unmarker' else 100


_approval = dict(mtime=None, attacks=frozenset())


def approved_frozen_attacks():
    """Attacks whose frozen cells passed the bridging check; empty until bridge.py has decided."""
    try:
        mtime = APPROVAL.stat().st_mtime_ns
    except FileNotFoundError:
        return frozenset()
    if _approval['mtime'] != mtime:
        rec = read(APPROVAL)
        if rec.get('policy') != POLICY or not rec.get('complete'):
            raise ValueError('bridging approval is incomplete or belongs to another policy')
        _approval.update(mtime=mtime, attacks=frozenset(a for a, v in rec['attacks'].items() if v['approved']))
    return _approval['attacks']


LIGHT_APPROVAL = RUN / 'inputs/light_approval.json'
LIGHT_CHECK = RUN / 'inputs/light_check.json'


def light_attacks():
    """Attacks an L40S worker may measure: those a cross-GPU check approved (inputs/light_approval.json) and, in a
    cross-GPU check run only, the attacks under test (inputs/light_check.json); none without either."""
    allowed = set()
    if LIGHT_APPROVAL.exists():
        rec = read(LIGHT_APPROVAL)
        if not rec.get('complete'):
            raise ValueError('cross-GPU approval is incomplete')
        allowed |= {a for a, v in rec['attacks'].items() if v['approved']}
    if LIGHT_CHECK.exists():
        allowed |= set(read(LIGHT_CHECK)['attacks_under_test'])
    return frozenset(allowed)


def light_eligible(attacks):
    allowed = light_attacks()
    return bool(allowed) and set(attacks) <= allowed


def bridging_decided():
    return APPROVAL.exists()


def _valid_record(rec, key, attack, detector_cfg):
    if not CP.valid_cell(rec.get('cell'), list(key[0]), expected_n(attack), cfg=detector_cfg):
        return False
    if 'key' in rec and tuple(map(tuple, rec['key'])) != key:
        raise ValueError('cache configuration identity mismatch')
    return True


def load_cell(c, attack, frozen=True):
    """The run's own cell, else its exported group, else (bridged attacks only) the frozen cell."""
    key = cfgkey(c)
    detector_cfg = dict(order=c['order'], fe_on=c['fe_on'])
    p = CELLS / cell_name(key, attack)
    if p.exists():
        rec = read(p)
        if _valid_record(rec, key, attack, detector_cfg):
            return dict(rec, origin='run')
    # Cross-environment decoding publishes the group before the shared cell.
    for cls in C.CLASSES:
        g_path = group_path(cls, key)
        if not g_path.exists():
            continue
        g = read(g_path)
        assert CP.exact_cfg_key(g['order'], g['fe'], [g['s'][f] for f in g['order']]) == key
        cell = g.get('attacks', {}).get(attack)
        if CP.valid_cell(cell, c['order'], expected_n(attack), cfg=detector_cfg):
            return dict(cell=cell, psnr_db=g['psnr_db'], source=str(g_path), n=cell['n'], origin='run')
    if frozen and attack in approved_frozen_attacks():
        p = FROZEN_CELLS / cell_name(key, attack)
        if p.exists():
            rec = read(p)
            if _valid_record(rec, key, attack, detector_cfg):
                return dict(rec, origin='frozen', origin_path=str(p))
    return None


def judge(request, c):
    return CP.judge(request, c, {a: load_cell(c, a) for a in request['attacks']}, .9)


def verify_sources():
    binding = read(RUN / 'sources.json')
    for p, expected in binding['hashes'].items():
        assert sha(p) == expected, ('source changed', p)


def candidate(c):
    assert c['certified'], 'uncertified solution cannot enter live validation'
    cfg = c['cfg']
    return dict(rank=c['rank'], skeleton=c['skeleton'].as_dict(),
                **{k: cfg[k] for k in ('order', 'fe', 'fe_on', 's', 'ms', 'live_threshold', 'threshold_count',
                                       'max_tests', 'surrogate_protocol')},
                psnr_db=c['psnr_db'], certified=True)


def check_candidate(variant, c):
    if variant == 'SingleOptimized':
        assert len(c['order']) == 1, ('SingleOptimized returned a composition', c['order'])
    elif variant == 'AlwaysAllThree':
        assert sorted(c['order']) == sorted(FRAGMENTS), ('AlwaysAllThree returned a subset', c['order'])
    else:
        raise ValueError(variant)


def add_restriction(variant, built):
    """A1: at most one selected fragment. A3: every fragment selected. Nothing else changes."""
    import z3
    o, selected = built[0], built[1]
    if variant == 'SingleOptimized':
        o.add(z3.Sum([z3.If(v, 1, 0) for v in selected.values()]) <= 1)
    elif variant == 'AlwaysAllThree':
        assert sorted(selected) == sorted(FRAGMENTS), sorted(selected)
        for v in selected.values():
            o.add(v)
    else:
        raise ValueError(variant)
    return built


def install_variant(variant, arm):
    assert variant in VARIANTS and arm == ARM
    C.boot(arm)
    import watermark_smt_v2 as W
    from surrogate_model import Surrogate
    if Path(W.__file__).resolve().parent != ROOT / 'fixed/code':
        raise ImportError(f'solver loaded from {W.__file__}')
    sg = Surrogate.from_dict(read(ROOT / 'fixed/inputs/surrogate_canonical.json'))
    original = getattr(W, '_exp_original_build', None) or W.build
    W._exp_original_build = original
    W.build = lambda **kw: add_restriction(variant, original(**kw))
    return sg
