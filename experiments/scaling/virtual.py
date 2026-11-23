"""Virtual watermark fragments for the solver-scaling experiment (CPU only; nothing is embedded or decoded).

Question (user, 2026-11-11): with how many watermarks does the grid baseline fall behind the SMT solver? Only
three watermarks are measured, so fragments 4..7 are VIRTUAL: perturbed copies of a measured fragment's
surrogate curves. The instance then has the size and the structure a real n-watermark instance would have
(strength curves, ordered overwrite interference, pair cascades, stages, per-image rates, thresholds); the
numbers are borrowed, so the experiment measures search effort, not watermark quality.

Virtual fragment W<k> (k = 4..7; source VINE, TrustMark, VideoSeal, VINE):
  range, clean floors, latency                  the source's
  mean BA curves (plain, resync, angle) and     the source's, shifted by a per-attack offset u ~ U[-SHIFT, SHIFT]
  their per-image BA records                    (clipped to [0, 1]); curve and records get the same offset
  single-fragment distortion d                  the source's, scaled by c ~ U[DSCALE]
  scale and tile stages                         not hosted: the ring re-embeds VINE and the tiled grid TrustMark
                                                (watermark_smt_v2 FE_AFFECTS), so a virtual fragment reads its
                                                plain curve under them, as VideoSeal does today
  overwrite delta, pair curves, excess e        cross-source pair: the source pair's curve. Same-source pair (no
                                                measurement exists): the pointwise maximum of the host source's
                                                curves with the other two sources, x rescaled linearly to the
                                                source's strength range (delta) or summed range (e); no pair
                                                replacement curve
Offsets and scales come from random.Random(f'{SEED}:{name}'): fixed, reproducible and the same for every n.

The solver is the booted TAILOR solver with data-only extensions that leave n = 3 unchanged (check_n3.py):
  * W.FR, W.D['fragments'] and the clean-floor table list the virtual fragments;
  * the threshold T(c) for c selected fragments depends only on c and the stage
    (unified_detector.budget_for_config), and the frozen add_strength_order caps c at 3 because only three
    fragments existed: its representative names now come from all fragments and its count chains run to n;
  * unified_detector accepts the virtual names.
"""
import copy
import inspect
import json
import random
from pathlib import Path

import numpy as np

SOURCES = ('VINE', 'TrustMark', 'VideoSeal')
VIRTUAL = ('W4', 'W5', 'W6', 'W7')
SEED = 20261111
SHIFT = 0.03
DSCALE = (0.85, 1.15)
VIRTUAL_STAGES = ('resync', 'angle')          # stages that change no fragment-specific embed


def source(f):
    return f if f in SOURCES else SOURCES[VIRTUAL.index(f) % len(SOURCES)]


def fragments(n):
    if not len(SOURCES) <= n <= len(SOURCES) + len(VIRTUAL):
        raise ValueError(f'n must be in 3..{len(SOURCES) + len(VIRTUAL)}')
    return list(SOURCES) + list(VIRTUAL[:n - len(SOURCES)])


def params(name, attacks):
    rng = random.Random(f'{SEED}:{name}')
    shift = {a: rng.uniform(-SHIFT, SHIFT) for a in sorted(attacks)}
    return dict(shift=shift, dscale=rng.uniform(*DSCALE))


def _clip(y):
    return min(1.0, max(0.0, float(y)))


def _shifted(curve, u):
    return {'xs': list(curve['xs']), 'ys': [_clip(y + u) for y in curve['ys']]}


def _scaled(curve, c):
    return {'xs': list(curve['xs']), 'ys': [float(y) * c for y in curve['ys']]}


def _rescaled(curve, frm, to):
    (a, b), (c, d) = frm, to
    return {'xs': [c + (float(x) - a) * (d - c) / (b - a) for x in curve['xs']], 'ys': list(curve['ys'])}


def _pointwise_max(curves):
    xs = sorted({float(x) for cv in curves for x in cv['xs']})
    ys = [max(float(np.interp(x, cv['xs'], cv['ys'])) for cv in curves) for x in xs]
    return {'xs': xs, 'ys': ys}


def _shift_record(record, u, src, dst):
    out = copy.deepcopy(record)
    out['ba'] = [[_clip(b + u) for b in row] for row in record['ba']]
    if 'raw_views' in out:
        for cell in out['raw_views']:
            for name in ('ba', 'cas_ba'):
                if name in cell and src in cell[name]:
                    cell[name][dst] = [None if b is None else _clip(b + u) for b in cell[name].pop(src)]
        out['ba'] = [list(cell['ba'][dst]) for cell in out['raw_views']]
    if out.get('ver') is not None:
        out['ver'] = [[False] * len(row) for row in out['ba']]
    return out


def virtual_surrogate_dict(base, n):
    """The frozen surrogate document extended to the first n fragments (n = 3 returns it unchanged)."""
    frs = fragments(n)
    out = copy.deepcopy(base)
    if n == len(SOURCES):
        return out
    out['fragments'] = frs
    attacks = list(base['attacks'])
    rng = {f: tuple(base['ranges'][source(f)]) for f in frs}
    P = {f: params(f, attacks) for f in frs if f not in SOURCES}
    for v, p in P.items():
        s = source(v)
        out['ranges'][v] = list(rng[v])
        out['d'][v] = _scaled(base['d'][s], p['dscale'])
        for a in attacks:
            u = p['shift'][a]
            if f'{s}|{a}' in base['base']:
                out['base'][f'{v}|{a}'] = _shifted(base['base'][f'{s}|{a}'], u)
            if f'{s}|{a}' in base.get('cap', {}):
                out['cap'][f'{v}|{a}'] = copy.deepcopy(base['cap'][f'{s}|{a}'])
            for st in VIRTUAL_STAGES:
                key = f'base_fe_{st}_{s}|{a}'
                if key in base['frontend']:
                    out['frontend'][f'base_fe_{st}_{v}|{a}'] = _shifted(base['frontend'][key], u)
            for stage in (None,) + VIRTUAL_STAGES:
                pk = f'{stage}:{s}|{a}' if stage else f'{s}|{a}'
                if pk in base['perimage']:
                    out['perimage'][f'{stage}:{v}|{a}' if stage else f'{v}|{a}'] = \
                        _shift_record(base['perimage'][pk], u, s, v)
    # ordered overwrite deltas and pair curves, unordered excess distortion, for every pair with a virtual member
    for g in frs:
        for f in frs:
            if g == f or (g in SOURCES and f in SOURCES):
                continue
            sg_, sf_ = source(g), source(f)
            for a in attacks:
                if sg_ != sf_:
                    curve = copy.deepcopy(base['delta'][f'{sg_}|{sf_}|{a}'])
                else:
                    others = [o for o in SOURCES if o != sf_]
                    curve = _pointwise_max([_rescaled(base['delta'][f'{o}|{sf_}|{a}'], base['ranges'][o],
                                                      base['ranges'][sf_]) for o in others])
                out['delta'][f'{g}|{f}|{a}'] = curve
    for host in frs:
        for partner in frs:
            if host == partner or (host in SOURCES and partner in SOURCES) or source(host) == source(partner):
                continue
            sh, sp = source(host), source(partner)
            u = P[host]['shift'] if host in P else None
            for a in attacks:
                stages = VIRTUAL_STAGES if host in P else ('resync', 'scale', 'angle', 'tile')
                for st in stages:
                    key = f'base_fe_{st}_{sh}|{a}|with_{sp}'
                    if key in base['frontend']:
                        cv = base['frontend'][key]
                        out['frontend'][f'base_fe_{st}_{host}|{a}|with_{partner}'] = \
                            _shifted(cv, u[a]) if u else copy.deepcopy(cv)
                key = f'ctrl_pair_{sh}|{a}|with_{sp}'
                if key in base['frontend']:
                    cv = base['frontend'][key]
                    out['frontend'][f'ctrl_pair_{host}|{a}|with_{partner}'] = _shifted(cv, u[a]) if u else copy.deepcopy(cv)
    for i, f in enumerate(frs):
        for g in frs[i + 1:]:
            if f in SOURCES and g in SOURCES:
                continue
            key = '|'.join(sorted((f, g)))
            sf_, sg_ = source(f), source(g)
            if sf_ != sg_:
                curve = copy.deepcopy(base['e']['|'.join(sorted((sf_, sg_)))])
            else:
                lo, hi = base['ranges'][sf_]
                curves = []
                for o in SOURCES:
                    if o == sf_:
                        continue
                    olo, ohi = base['ranges'][o]
                    curves.append(_rescaled(base['e']['|'.join(sorted((sf_, o)))], (lo + olo, hi + ohi),
                                            (2 * lo, 2 * hi)))
                curve = _pointwise_max(curves)
            out['e'][key] = curve
    out.setdefault('source', {})['virtual_fragments'] = dict(
        n=n, seed=SEED, shift=SHIFT, dscale=list(DSCALE), sources={v: source(v) for v in P},
        params={v: dict(dscale=p['dscale'], shift=p['shift']) for v, p in P.items()})
    return out


# ---- solver extensions -------------------------------------------------------------------------------------------
_CHAINS = (
    ("        names = [host] + [g for g in ('VINE', 'TrustMark', 'VideoSeal') if g != host]",
     "        names = [host] + [g for g in tuple(FRs) if g != host]"),
    ("        return z3.If(nsel <= 1, at(1), z3.If(nsel == 2, at(2), at(3)))",
     "        return _scaling_count_chain(nsel, at, len(FRs))"),
    ("    thr = z3.If(nsel <= 1, stage_threshold(1, margin),\n"
     "                z3.If(nsel == 2, stage_threshold(2, margin), stage_threshold(3, margin)))",
     "    thr = _scaling_count_chain(nsel, lambda k: stage_threshold(k, margin), len(FRs))"),
    ("    opt._threshold_expr = z3.If(nsel <= 1, stage_threshold(1),\n"
     "                               z3.If(nsel == 2, stage_threshold(2), stage_threshold(3)))",
     "    opt._threshold_expr = _scaling_count_chain(nsel, stage_threshold, len(FRs))"),
)


def _scaling_count_chain(nsel, value, n):
    """If(nsel <= 1, value(1), If(nsel == 2, value(2), ... value(n))): for n = 3 the frozen chain, built in the
    same order (value(1), value(2), value(3))."""
    import z3

    def tail(k):
        if k >= n:
            return value(n)
        return z3.If(nsel == k, value(k), tail(k + 1))
    return z3.If(nsel <= 1, value(1), tail(2))


def install(ns, n, clean_floor_dir=None):
    """Extend the booted solver (namespace from grid.frozen.boot) to the first n fragments; idempotent per n.
    The extended clean-floor table is written to a private directory (a fresh temporary one by default)."""
    W, UD = ns.W, ns.UD
    frs = fragments(n)
    if getattr(W, '_scaling_n', None) is not None:
        if W._scaling_n != n:
            raise RuntimeError('solver already extended for another n')
        return
    W.FR = list(frs)
    for v in frs[len(SOURCES):]:
        W.D['fragments'][v] = copy.deepcopy(W.D['fragments'][source(v)])
        UD._FRAGMENTS[v.lower()] = v.lower()
    clean = json.loads(Path(W.CLEAN_FLOOR_JSON).read_text())
    for v in frs[len(SOURCES):]:
        clean['rows'][v] = copy.deepcopy(clean['rows'][source(v)])
    if clean_floor_dir is None:
        import tempfile
        clean_floor_dir = tempfile.mkdtemp(prefix='scaling_floors_')
    path = Path(clean_floor_dir) / f'clean_minimum_strength_n{n}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean))
    W.CLEAN_FLOOR_JSON = str(path)
    src = inspect.getsource(W.add_strength_order)
    for old, new in _CHAINS:
        if src.count(old) != 1:
            raise AssertionError(f'add_strength_order changed; review the scaling extension: {old[:60]!r}')
        src = src.replace(old, new)
    W._scaling_count_chain = _scaling_count_chain
    exec(compile(src, W.__file__, 'exec'), W.__dict__)
    W._scaling_n = n


def load(ns, n):
    """(virtual Surrogate, document) for n fragments from the frozen table."""
    import hashlib
    from grid import frozen
    data = frozen.surrogate_path(ns).read_bytes()
    if hashlib.sha256(data).hexdigest() != frozen.SURROGATE_SHA256:
        raise ValueError('surrogate_canonical.json changed')
    base = json.loads(data)
    doc = virtual_surrogate_dict(base, n)
    return ns.SM.Surrogate.from_dict(doc), doc
