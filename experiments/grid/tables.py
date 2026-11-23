"""Precomputed on-knot lookup tables for C2, built once per process from the frozen surrogate.

Two modes.
  raw     (C2)  A lone fragment's mean BA at a knot is the mean of the raw per-image BA of the record behind the
                curve the solver reads (perimage 'F|a', or 'stage:F|a' when the stage's replacement curve is read);
                the per-image pass rate is the fraction of raw per-image BA >= tau, without isotonic smoothing.
                Where that record or that knot does not exist, the stored fitted curve at the knot gives the mean
                and the solver's isotonic floor rule gives the rate decision, and the cell is flagged.
  fitted        Every quantity from the stored curves and the solver's own rate rule, i.e. what
                composite_prediction.predict_composite returns at an on-knot configuration (correctness check).
In both modes the composition corrections (ordered-pair delta curves at the overwriter's knot, pair replacement
curves at the host's knot) and the pairwise excess distortion e (at the knot sum) are the stored fitted curves.
Single-fragment distortion is the fitted d_f plus stage penalty, unless a raw table measured at the knots
(experiments/grid_measure/measure_distortion.py) is supplied, which then replaces those single-fragment terms.

Strengths are only ever knot values: every table is indexed by knot position, nothing is evaluated between knots.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import frozen
from . import knots as K

FRAGMENTS = K.FRAGMENTS
STAGES = K.STAGES
N_BITS = 100                                       # unified_detector.N_BITS; tau = count / 100
# Embed-side fidelity charges, watermark_smt_v2.add_strength_order:894-936. The ring is charged to VINE's
# strength (nested_penalty), the tiled grid and the sync mark to TrustMark's; the sync mark is paid whatever
# the fragment set, at the curve's mid-range value when TrustMark is absent (FE_MARK_INDEPENDENT).
FE_HOST = {'resync': 'TrustMark', 'scale': 'VINE', 'tile': 'TrustMark'}
FE_MARK_INDEPENDENT = {'resync'}
# Which fragment's embed a stage changes (watermark_smt_v2.py:766), for the admissibility rule (833-846).
FE_AFFECTS = {'resync': FRAGMENTS, 'scale': ('VINE',), 'tile': ('TrustMark',)}
RATE_TOL = 1e-12                                    # composite_prediction.floor_strength compares y >= level - 1e-12
MODES = ('raw', 'fitted')


def load_raw_distortion(path, knots):
    """{(fragment, stage or 'none'): array over the fragment's knots (nan where unmeasured)} and the file digest.

    Accepts {f: {stage_or_none: {knot: mean_mse_255}}} or the measure_distortion.py document
    ({'schema': 'tailor_exp_raw_distortion_v1', 'table': {f: {stage: {repr(knot): {'mean_mse': ...}}}}})."""
    data = Path(path).read_bytes()
    doc = json.loads(data)
    if 'table' in doc and 'schema' in doc:
        if doc.get('distortion_domain', 'mse_255') != 'mse_255':
            raise ValueError('raw distortion must be MSE on the 0..255 scale')
        doc = doc['table']
    out = {}
    for fragment, by_stage in doc.items():
        if fragment not in knots:
            raise ValueError(f'unknown fragment in raw distortion table: {fragment}')
        for stage, by_knot in by_stage.items():
            if stage not in ('none',) + STAGES:
                raise ValueError(f'unknown stage in raw distortion table: {stage}')
            values = np.full(len(knots[fragment]), np.nan)
            for knot, value in by_knot.items():
                i = K.knot_index(knots[fragment], float(knot))
                if i is None:
                    raise ValueError(f'raw distortion at {fragment}={knot} is not a measured knot')
                v = float(value['mean_mse'] if isinstance(value, dict) else value)
                if not math.isfinite(v) or v < 0:
                    raise ValueError(f'invalid raw distortion {fragment}/{stage}/{knot}: {v}')
                values[i] = v
            out[(fragment, stage)] = values
    return out, hashlib.sha256(data).hexdigest()


class Record:
    """One per-image record mapped onto the knots of its fragment."""
    __slots__ = ('key', 'present', 'n', 'mean', 'surv', 'rows')

    def __init__(self, key, record, fragment, knots):
        size = len(knots)
        self.key = key
        self.present = np.zeros(size, bool)
        self.n = np.zeros(size, int)
        self.mean = np.full(size, np.nan)
        # surv[k, c] = number of images at knot k with BA >= c / 100, for every attainable count c = 0..101;
        # BA >= tau is evaluated exactly as Surrogate.rate_curve does (surrogate_model.py:92).
        self.surv = np.zeros((size, N_BITS + 2), int)
        self.rows = [None] * size
        taus = np.arange(N_BITS + 2) / N_BITS
        for x, row in zip(record['xs'], K.record_rows(record, fragment)):
            i = K.knot_index(knots, x)
            if i is None:
                continue
            row = np.asarray(row, float)
            if row.ndim != 1 or not len(row) or not np.all(np.isfinite(row)):
                raise ValueError(f'invalid per-image row in {key} at {x}')
            self.present[i] = True
            self.n[i] = len(row)
            self.mean[i] = math.fsum(row) / len(row)
            self.surv[i] = (row[None, :] >= taus[:, None]).sum(axis=1)
            self.rows[i] = row


class GridTables:
    def __init__(self, sg=None, *, mode='raw', raw_distortion=None, pair_min_cascade=0.02, ns=None, sha=None):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.ns = ns or frozen.boot()
        if sg is None:
            sg, sha = frozen.load_surrogate(self.ns)
        self.sg, self.surrogate_sha256, self.mode = sg, sha, mode
        self.pair_min_cascade = float(pair_min_cascade)
        if tuple(sg.fragments) != FRAGMENTS:
            raise ValueError(f'unexpected fragment order {sg.fragments}')
        self.F = FRAGMENTS
        self.attacks = tuple(sg.attacks)
        self.knots = K.knot_sets(sg._perimage)
        self.kx = {f: np.array(self.knots[f], float) for f in self.F}
        self.records = {}
        for key, record in sg._perimage.items():
            stage, fragment, attack = K.parse_key(key)
            if fragment in self.knots:
                self.records[(stage, fragment, attack)] = Record(key, record, fragment, self.knots[fragment])
        self.level = {(st, f, a): self._level(st, f, a) for st in (None,) + STAGES for f in self.F for a in self.attacks}
        self.pair = self._pairs()
        self.delta = {(g, f, a): np.array([sg.delta(g, f, a).eval(x) for x in self.knots[g]])
                      for g in self.F for f in self.F if g != f for a in self.attacks}
        self.gains = self._gains()
        self._distortion()
        self.raw_distortion, self.raw_distortion_sha256 = {}, None
        if raw_distortion is not None and mode == 'raw':
            self.raw_distortion, self.raw_distortion_sha256 = load_raw_distortion(raw_distortion, self.knots)

    # ---- mean BA of a lone fragment at the knots ---------------------------------------------------------------
    def has_stage_curve(self, stage, f, a):
        return stage is not None and self.sg.frontend(f'base_fe_{stage}_{f}|{a}') is not None

    def _level(self, stage, f, a):
        """composite_expression (composite_prediction.py:85-157) for fragment f alone under `stage`."""
        sg = self.sg
        name, key = K.solver_curve(sg, f, a, stage)
        curve = sg.base(f, a) if name.startswith('base:') else sg.frontend(name)
        fitted = np.array([curve.eval(x) for x in self.knots[f]])
        has_levels = any(sg.frontend(f'base_fe_{st}_{f}|{a}') is not None for st in STAGES)
        legacy = not has_levels and not name.startswith('base:')
        if has_levels or legacy:                                          # took_effect -> b.cap(base)
            fitted = np.minimum(1.0, fitted)
        record = self.records.get(K.parse_key(key)) if key else None
        raw_ok = record.present if (record is not None and self.mode == 'raw') else np.zeros(len(fitted), bool)
        values = fitted.copy()
        if raw_ok.any():
            values[raw_ok] = np.minimum(1.0, record.mean[raw_ok])
        return dict(values=values, fitted=fitted, raw=raw_ok, curve=name, record=key)

    def _pairs(self):
        """Pair replacement curves usable under the cascade rule (composite_prediction.py:106-128)."""
        sg, out = self.sg, {}
        for stage in STAGES:
            for f in self.F:
                for a in self.attacks:
                    if not self.has_stage_curve(stage, f, a):
                        continue
                    for g in self.F:
                        if g == f:
                            continue
                        name = f'base_fe_{stage}_{f}|{a}|with_{g}'
                        pair = sg.frontend(name)
                        if pair is None:
                            continue
                        control = sg.frontend(f'ctrl_pair_{f}|{a}|with_{g}')
                        if control is not None:
                            grid = sorted(set(pair.xs) | set(control.xs))
                            if max(pair.eval(x) - control.eval(x) for x in grid) < self.pair_min_cascade:
                                continue
                        out[(stage, f, a, g)] = dict(
                            name=name, values=np.minimum(1.0, np.array([pair.eval(x) for x in self.knots[f]])))
        return out

    def _gains(self):
        """Front-end gains exactly as the solver computes them (composite_expression 'frontend_gains')."""
        CPD = self.ns.CPD
        out = {}
        strengths = {f: self.knots[f][0] for f in self.F}
        frontends = {st: False for st in CPD.FRONTEND_ORDER}
        for f in self.F:
            for a in self.attacks:
                res = CPD.composite_expression(self.sg, f, a, strengths, lambda h, g: False, frontends,
                                               CPD.NumericBackend(), pair_min_cascade=self.pair_min_cascade)
                for stage, gain in res['frontend_gains'].items():
                    out[(stage, f, a)] = float(gain)
        return out

    # ---- per-image pass rate at the knots ----------------------------------------------------------------------
    def rate(self, stage, f, a, count, level, rule):
        """Rate decision of fragment f at every knot: the solver's record choice (composite_prediction.rate_source)
        and, under rule 'raw', the raw fraction >= tau at knots the record has (solver floor rule elsewhere)."""
        CPD = self.ns.CPD
        tau = count / N_BITS
        size = len(self.knots[f])
        curve, actual, missing = CPD.rate_source(self.sg, f, a, tau, stage)
        info = dict(source=None, actual_stage=actual, missing_changed_embed=bool(missing), floor=None,
                    raw_rate=np.full(size, np.nan), solver=np.zeros(size, bool), fallback=np.zeros(size, bool))
        if missing:
            return np.zeros(size, bool), info
        if curve is None:                                   # no per-image record: det_cond bound(None) is True
            info['solver'][:] = True
            return np.ones(size, bool), info
        record = self.records[(actual, f, a)]
        info['source'] = record.key
        floor = CPD.floor_strength(curve, level)
        info['floor'] = floor
        info['solver'] = (self.kx[f] >= floor) if floor is not None else np.zeros(size, bool)
        if rule == 'solver':
            return info['solver'].copy(), info
        present = record.present
        rate = np.full(size, np.nan)
        rate[present] = record.surv[present, count] / record.n[present]
        info['raw_rate'] = rate
        info['fallback'] = ~present
        used = info['solver'].copy()
        used[present] = rate[present] >= level - RATE_TOL
        return used, info

    # ---- distortion --------------------------------------------------------------------------------------------
    def _distortion(self):
        sg = self.sg
        self.d = {f: np.array([sg.d(f).eval(x) for x in self.knots[f]]) for f in self.F}
        self.e = {}
        for i, f in enumerate(self.F):
            for g in self.F[i + 1:]:
                curve = sg.e(f, g)
                self.e[(f, g)] = np.array([[curve.eval(x + y) for y in self.knots[g]] for x in self.knots[f]])
        self.penalty = {}
        npc = sg.frontend('nested_penalty')
        if npc is not None:
            self.penalty[('scale', 'VINE')] = np.array([npc.eval(x) for x in self.knots['VINE']])
        for stage, host in FE_HOST.items():
            if stage == 'scale':
                continue
            pc = sg.frontend(f'penalty_fe_{stage}')
            if pc is None:
                continue
            self.penalty[(stage, host)] = np.maximum(0.0, np.array([pc.eval(x) for x in self.knots[host]]))
            if stage in FE_MARK_INDEPENDENT:
                self.penalty[(stage, None)] = max(0.0, float(pc.eval(0.5 * (pc.xs[0] + pc.xs[-1]))))

    def penalty_host(self, order, stage):
        """The fragment whose single-fragment term carries the stage's embed-side cost."""
        if stage in ('scale', 'tile'):
            return FE_HOST[stage]
        if stage == 'resync':
            return 'TrustMark' if 'TrustMark' in order else next(f for f in self.F if f in order)
        return None

    def fitted_penalty(self, stage, f, order):
        """watermark_smt_v2.py:894-936 for the host fragment f, as an array over f's knots (zeros if none)."""
        size = len(self.knots[f])
        if stage == 'scale':
            return self.penalty.get(('scale', 'VINE'), np.zeros(size))
        if stage == 'tile':
            return self.penalty.get(('tile', 'TrustMark'), np.zeros(size))
        if stage == 'resync':
            if f == 'TrustMark' and ('resync', 'TrustMark') in self.penalty:
                return self.penalty[('resync', 'TrustMark')]
            return np.full(size, self.penalty.get(('resync', None), 0.0))
        return np.zeros(size)

    def single_terms(self, order, stage):
        """{f: (values over f's knots, raw-used mask, fitted-penalty mask)} for the single-fragment part of D."""
        host = self.penalty_host(order, stage)
        out = {}
        for f in self.F:
            if f not in order:
                continue
            raw_none = self.raw_distortion.get((f, 'none'))
            d = self.d[f].copy()
            raw_used = np.zeros(len(d), bool)
            if raw_none is not None:
                ok = np.isfinite(raw_none)
                d[ok], raw_used[ok] = raw_none[ok], True
            pen_fitted = np.zeros(len(d), bool)
            if f == host:
                pen = self.fitted_penalty(stage, f, order)
                whole = self.raw_distortion.get((f, stage))
                ok = np.isfinite(whole) if whole is not None else np.zeros(len(d), bool)
                pen_fitted = ~ok & (pen != 0)
                d = np.where(ok, whole if whole is not None else 0.0, d + pen)
                raw_used = np.where(ok, True, raw_used)
            out[f] = (d, raw_used, pen_fitted)
        return out
