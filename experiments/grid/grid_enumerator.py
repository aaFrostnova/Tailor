"""C2: budget-matched enumeration of the measured strength knots (offline part).

Search space. Every nonempty fragment subset in every embedding order (3 singles, 6 ordered pairs, 6 ordered
triples) x every compatible geometric stage (none, resync, angle; scale only with VINE, tile only with
TrustMark; at most one stage, watermark_smt_v2.build:502-509) x the Cartesian product of the selected
fragments' knot sets: 67 ordered fragment-and-stage structures and 58,433 configurations before any floor.
No strength between knots is ever formed.

Per request, everything below runs inside the wall-clock budget B (only GridTables, built once per process
like the method's surrogate load, is outside it):
  1. cheap validity pruning, fixed before any coverage outcome is seen: unmeasured attack (watermark_smt_v2.py
     848-856), clean floor and range (707-715), a stage only if responsible for a requested attack (757-760,
     937-942), embed-changing stage without replacement curves (833-846), stage cost measured (946-954),
     latency (526-542), impossible threshold (stage_threshold, 781-790), fidelity floor D <= 255^2/10^(q/10)
     (955-963);
  2. ranking by predicted distortion D, then fewer stages, then structure (order names, stage name), then knot
     tuple: the visiting order is fixed before any candidate is evaluated;
  3. visiting candidates in that order in vectorized chunks: attack a is covered when some selected fragment
     has mean >= tau + mean margin and per-image rate >= det_min + rate margin (873-881, det_cond 792-818,
     stage_threshold 781-790), every attack must be covered (881), every selected fragment must cover one
     attack (watermark_smt_topk.no_passenger_clause);
  4. the first feasible candidate of a structure is its optimum; the search keeps the 3 best distinct
     structures and stops when they are certified (sorted order) or when B runs out.
A structure is (order, stage); the order fixes the fragment set, and a stage responsible for no requested
attack is never enumerated (skeleton_clause leaves such a stage out of the identity, and its configurations are
dominated by the same embed without it), so distinct structures are distinct skeletons of enumerate_topk.
"""
import collections
import itertools
import math
import time

import numpy as np

from . import knots as K
from .tables import FE_AFFECTS, FE_HOST, N_BITS

FRAGMENTS = K.FRAGMENTS
STAGES = K.STAGES
STAGE_HOST = {'scale': 'VINE', 'tile': 'TrustMark'}  # build():507-509; unified_detector.normalize_config:105-108
INF = float('inf')
MARGIN_STAGES = ('initial', '04', '02', '00')
SCHEMA = 'tailor_exp_c2_grid_v1'
FIRST_CHUNK, MAX_CHUNK = 64, 4096


class Structure(tuple):
    """(order, stage): the discrete part of a C2 configuration."""
    __slots__ = ()

    def __new__(cls, order, stage=None):
        return super().__new__(cls, (tuple(order), stage))

    order = property(lambda s: tuple.__getitem__(s, 0))
    stage = property(lambda s: tuple.__getitem__(s, 1))
    m = property(lambda s: len(s.order))
    fe_on = property(lambda s: [s.stage] if s.stage else [])

    def detector_cfg(self):
        return dict(order=list(self.order), fe_on=self.fe_on)

    def lex_key(self):
        return (self.order, self.stage or '')

    def as_dict(self):
        return dict(frags=sorted(self.order), order=list(self.order), fe=self.fe_on)

    def __repr__(self):
        return '+'.join(self.order) + (f'|{self.stage}' if self.stage else '')


def structures(fragments=FRAGMENTS, geometry=True, hosts=STAGE_HOST):
    out = []
    for m in range(1, len(fragments) + 1):
        for subset in itertools.combinations(fragments, m):
            for order in itertools.permutations(subset):
                for stage in (None,) + (STAGES if geometry else ()):
                    host = hosts.get(stage)
                    if host is None or host in order:
                        out.append(Structure(order, stage))
    return out


def iter_space(knot_sets, fragments=FRAGMENTS, geometry=True):
    """Every (structure, strength tuple in embed order) of the unpruned space."""
    for s in structures(fragments, geometry):
        for combo in itertools.product(*(knot_sets[f] for f in s.order)):
            yield s, combo


def margins(ns, stage):
    """(rate margin, mean margin) of a margin stage: arm 06 initial, then the frozen fallback ladder
    (experiments/main_controls/common.py:69-73, margin_fallback_policy.STAGES)."""
    if stage not in MARGIN_STAGES:
        raise ValueError(f'margin stage must be one of {MARGIN_STAGES}')
    return tuple(float(x) for x in ns.common.margins(ns.arm, stage))


def psnr_of(D):
    return 10.0 * math.log10(255.0 ** 2 / D) if D > 0 else INF          # watermark_smt_topk.db


class GridSearch:
    """One request. Construction is the cheap pruning and ranking; run() is the budgeted visit."""

    def __init__(self, tables, request, *, rate_margin, mean_margin, det_min=None, fe_gain_min=0.05,
                 no_passengers=True, rate_rule=None, t0=None):
        self.t0 = time.perf_counter() if t0 is None else t0
        self.tables, self.ns = tables, tables.ns
        W = self.ns.W
        self.request = dict(attacks=list(request['attacks']), fpr=float(request['fpr']),
                            min_psnr=request['min_psnr'], max_ms=float(request['max_ms']))
        self.attacks = self.request['attacks']
        self.rate_margin, self.mean_margin = float(rate_margin), float(mean_margin)
        self.det_min = W.DEFAULT_DET_MIN if det_min is None else float(det_min)
        self.fe_gain_min, self.no_passengers = float(fe_gain_min), bool(no_passengers)
        self.rate_rule = rate_rule or ('raw' if tables.mode == 'raw' else 'solver')
        if self.rate_rule not in ('raw', 'solver') or (tables.mode == 'fitted' and self.rate_rule != 'solver'):
            raise ValueError('fitted mode uses the solver rate rule; raw mode uses raw or solver')
        self._rates = {}
        self._rules()
        self._candidates()
        self._wiring()
        self.setup_s = time.perf_counter() - self.t0

    # ---- request-level rules ------------------------------------------------------------------------------------
    def _rules(self):
        T, ns, W, UD = self.tables, self.ns, self.ns.W, self.ns.UD
        sg, attacks, fpr = T.sg, self.attacks, self.request['fpr']
        if not attacks or len(set(attacks)) != len(attacks):
            raise ValueError('attacks must be a nonempty list of distinct names')  # capacity_protocol.py:38
        if not 0 < fpr < 1 or self.request['max_ms'] <= 0:
            raise ValueError('fpr must be in (0, 1) and max_ms positive')         # capacity_protocol.py:39
        missing = [a for a in attacks if a not in sg.attacks]
        if missing:
            raise ValueError(f'no measured surrogate for {sorted(set(missing))}')  # watermark_smt_v2.py:848-856
        # Clean floor for the request's level: min_ba = beta_from_fpr(fpr) (capacity_protocol.solver_scenario),
        # floors = clean_floors(min_ba, 0) (watermark_smt_v2.py:707), range [max(lo, floor), hi] (710-715).
        self.min_ba = W.beta_from_fpr(fpr)
        floors = W.clean_floors(self.min_ba, 0)
        self.lo, self.hi = {}, {}
        for f in FRAGMENTS:
            lo, hi = sg.range(f)
            self.lo[f], self.hi[f] = max(lo, floors.get(f, lo)), hi
            if self.lo[f] > self.hi[f]:
                raise ValueError(f'clean floor above the measured top for {f}')
        # Stage responsibility (composite_expression gains, watermark_smt_v2.py:757-760 and 937-942).
        gain, resp = set(), collections.defaultdict(set)
        for a in attacks:
            for f in FRAGMENTS:
                for st in STAGES:
                    g = T.gains.get((st, f, a))
                    if g is None:
                        continue
                    gain.add(st)
                    if g >= self.fe_gain_min:
                        resp[st].add(a)
        self.fe_gain = sorted(gain, key=STAGES.index)
        self.fe_resp = {st: sorted(resp[st]) for st in STAGES if resp.get(st)}
        for st, host in FE_HOST.items():                      # watermark_smt_v2.py:946-954
            if st != 'scale' and st in gain and sg.frontend(f'penalty_fe_{st}') is None:
                raise ValueError(f"front-end '{st}' has a measured effect but no measured fidelity cost")
        # An embed-changing stage may not be on with a fragment it changes when a requested non-adversarial
        # attack has no replacement curve for that fragment (watermark_smt_v2.py:833-846).
        self.forbidden = set()
        if any(k.startswith('base_fe_') for k in sg._frontend):
            for st, affected in FE_AFFECTS.items():
                for f in affected:
                    if any(a not in W.ADVERSARIAL and sg.frontend(f'base_fe_{st}_{f}|{a}') is None for a in attacks):
                        self.forbidden.add((st, f))
        # Latency (watermark_smt_v2.py:526-542): fragment costs + the stage's worst requested column, clipped at 0.
        defaults = {'resync': W.FE_MS['resync'], 'scale': W.FE_MS['nested'], 'angle': 0.0, 'tile': 0.0}
        self.stage_ms = {None: 0.0}
        for st in STAGES:
            v = [sg.latency(f'latency_ms_{st}|{a}') for a in attacks]
            v = [max(0.0, float(x)) for x in v if isinstance(x, (int, float))]
            self.stage_ms[st] = max(v) if v else defaults[st]
        self.cost_ms = {f: W.cost_ms(f) for f in FRAGMENTS}
        q = self.request['min_psnr']
        self.d_max = (255.0 ** 2) / (10.0 ** (float(q) / 10.0)) if q is not None and q > 0 else INF
        self.rate_level = min(1.0, self.det_min + self.rate_margin)          # det_cond:801
        self._fpr_thresholds = {}

    def threshold(self, s):
        """(count, tau, max_tests, mean threshold) for the structure's actual T(c) (unified_detector)."""
        key = (s.m, s.stage)
        if key not in self._fpr_thresholds:
            UD = self.ns.UD
            cfg = s.detector_cfg()
            count = UD.threshold_count(cfg, self.request['fpr'])
            tau = count / N_BITS
            mean = tau if tau > 1 else min(1.0, tau + self.mean_margin)       # stage_threshold:783-785
            self._fpr_thresholds[key] = (count, tau, UD.budget_for_config(cfg)['max_tests'], mean)
        return self._fpr_thresholds[key]

    def latency(self, s):
        return sum(self.cost_ms[f] for f in FRAGMENTS if f in s.order) + self.stage_ms[s.stage]

    def structure_rule(self, s):
        """Reason a whole structure is pruned, or None."""
        if s.stage is not None and not self.fe_resp.get(s.stage):
            return 'stage_without_responsible_attack'
        if any((s.stage, f) in self.forbidden for f in s.order):
            return 'stage_inadmissible'
        if self.latency(s) > self.request['max_ms']:
            return 'latency'
        if self.threshold(s)[0] > N_BITS:
            return 'impossible_threshold'
        return None

    # ---- candidates: pruning and ranking --------------------------------------------------------------------------
    def distortion(self, s, kn):
        """Predicted D for candidates of one structure; kn holds knot indices in FRAGMENTS order (-1 = absent)."""
        T, fi = self.tables, {f: i for i, f in enumerate(FRAGMENTS)}
        terms = T.single_terms(s.order, s.stage)
        D = np.zeros(len(kn))
        for f in FRAGMENTS:
            if f in s.order:
                D = D + terms[f][0][kn[:, fi[f]]]
        for (f, g), E in T.e.items():
            if f in s.order and g in s.order:
                D = D + E[kn[:, fi[f]], kn[:, fi[g]]]
        return D

    def _candidates(self):
        T = self.tables
        self.all_structures = structures()
        self.n_space = K.count_space(T.knots, self.all_structures)
        self.pruned = collections.Counter()
        kept, sids, kns, Ds = [], [], [], []
        for s in self.all_structures:
            size = math.prod(len(T.knots[f]) for f in s.order)
            reason = self.structure_rule(s)
            if reason:
                self.pruned[reason] += size
                continue
            idx = [np.flatnonzero((T.kx[f] >= self.lo[f]) & (T.kx[f] <= self.hi[f])) for f in s.order]
            inside = math.prod(len(i) for i in idx)
            self.pruned['clean_floor_or_range'] += size - inside
            if not inside:
                continue
            kn = np.full((inside, 3), -1, np.int64)
            for f, col in zip(s.order, np.meshgrid(*idx, indexing='ij')):
                kn[:, FRAGMENTS.index(f)] = col.ravel()
            D = self.distortion(s, kn)
            ok = D <= self.d_max
            self.pruned['fidelity'] += int((~ok).sum())
            if ok.any():
                sids.append(np.full(int(ok.sum()), len(kept)))
                kns.append(kn[ok])
                Ds.append(D[ok])
                kept.append(s)
        self.structures = kept
        if kept:
            sid, kn, D = np.concatenate(sids), np.concatenate(kns), np.concatenate(Ds)
        else:
            sid, kn, D = np.zeros(0, int), np.zeros((0, 3), np.int64), np.zeros(0)
        lex = sorted(range(len(kept)), key=lambda i: kept[i].lex_key())
        rank = np.empty(len(kept), int)
        rank[lex] = np.arange(len(kept))
        nstage = np.array([s.stage is not None for s in kept], int)
        order = np.lexsort((kn[:, 2], kn[:, 1], kn[:, 0], rank[sid] if len(kept) else sid,
                            nstage[sid] if len(kept) else sid, D))
        self.c_sid, self.c_kn, self.c_D = sid[order], kn[order], D[order]
        self.n_ranked = len(self.c_D)
        assert self.n_space == self.n_ranked + sum(self.pruned.values())

    # ---- coverage wiring (flat tables, one row per structure) --------------------------------------------------
    def rate(self, stage, f, a, count):
        key = (stage, f, a, count)
        if key not in self._rates:
            self._rates[key] = self.tables.rate(stage, f, a, count, self.rate_level, self.rate_rule)
        return self._rates[key]

    def wiring(self, s, f, a):
        """(level key and values, applied overwriters, pair partner) of fragment f under attack a in s, as
        composite_expression selects them (pair precedence: last selected partner in fragment order)."""
        T = self.tables
        pos = {g: i for i, g in enumerate(s.order)}
        after = [g for g in FRAGMENTS if g != f and g in pos and pos[g] > pos[f]]
        pair_g = None
        if T.has_stage_curve(s.stage, f, a):
            for g in FRAGMENTS:
                if g in after and (s.stage, f, a, g) in T.pair:
                    pair_g = g
        if pair_g is not None:
            key, values = ('pair', s.stage, f, a, pair_g), T.pair[(s.stage, f, a, pair_g)]['values']
        else:
            key, values = ('lvl', s.stage, f, a), T.level[(s.stage, f, a)]['values']
        return key, values, [g for g in after if g != pair_g], pair_g

    def _wiring(self):
        T, A, S = self.tables, len(self.attacks), len(self.structures)
        flat = {'lv': [[], {}, 0], 'dl': [[], {}, 0], 'rp': [[], {}, 0]}

        def offset(kind, key, values):
            arrays, index, _ = flat[kind]
            if key not in index:
                index[key] = flat[kind][2]
                arrays.append(values)
                flat[kind][2] += len(values)
            return index[key]

        self.LVO = np.zeros((S, 3, A), np.int64)
        self.RPO = np.zeros((S, 3, A), np.int64)
        self.DLO = np.full((S, 3, A, 3), -1, np.int64)
        self.SEL = np.zeros((S, 3), bool)
        self.THR = np.zeros(S)
        for si, s in enumerate(self.structures):
            count, _, _, mean_thr = self.threshold(s)
            self.THR[si] = mean_thr
            for fi, f in enumerate(FRAGMENTS):
                if f not in s.order:
                    continue
                self.SEL[si, fi] = True
                for j, a in enumerate(self.attacks):
                    key, values, overwriters, _ = self.wiring(s, f, a)
                    self.LVO[si, fi, j] = offset('lv', key, values)
                    for g in overwriters:
                        self.DLO[si, fi, j, FRAGMENTS.index(g)] = offset('dl', ('delta', g, f, a), T.delta[(g, f, a)])
                    self.RPO[si, fi, j] = offset('rp', (s.stage, f, a, count), self.rate(s.stage, f, a, count)[0])
        cat = lambda kind, dtype: (np.concatenate(flat[kind][0]).astype(dtype) if flat[kind][0]
                                   else np.zeros(1, dtype))
        self.LVF, self.DLF, self.RPF = cat('lv', float), cat('dl', float), cat('rp', bool)

    def evaluate_positions(self, lo, hi, return_cover=False):
        """Feasibility of the ranked candidates lo..hi-1, vectorized over candidates and attacks."""
        sid, kn = self.c_sid[lo:hi], self.c_kn[lo:hi]
        n, A = len(sid), len(self.attacks)
        cover = np.zeros((n, 3, A), bool)
        sel = self.SEL[sid]
        safe = np.where(kn >= 0, kn, 0)
        thr = self.THR[sid][:, None]
        for fi in range(3):
            if not sel[:, fi].any():
                continue
            k = safe[:, fi][:, None]
            level = self.LVF[self.LVO[sid, fi, :] + k]
            drop = np.zeros((n, A))
            for gi in range(3):                       # composite_expression sums drops in fragment order
                off = self.DLO[sid, fi, :, gi]
                applied = off >= 0
                if applied.any():
                    drop = drop + np.where(applied, self.DLF[np.where(applied, off + safe[:, gi][:, None], 0)], 0.0)
            mean = level - drop
            ok = (mean >= thr) & self.RPF[self.RPO[sid, fi, :] + k]
            cover[:, fi, :] = ok & sel[:, fi][:, None]
        feasible = cover.any(axis=1).all(axis=1)
        if self.no_passengers:
            feasible &= (cover.any(axis=2) | ~sel).all(axis=1)
        return (feasible, cover) if return_cover else feasible

    # ---- budgeted visit ----------------------------------------------------------------------------------------
    def run(self, budget_s=INF, stop_when_complete=True, k=3):
        budget_s = INF if budget_s is None else float(budget_s)
        found, examined, evaluated, first, stop = {}, 0, 0, None, None
        pos, chunk, N = 0, FIRST_CHUNK, self.n_ranked
        while pos < N:
            if time.perf_counter() - self.t0 >= budget_s:
                stop = 'budget'
                break
            hi = min(N, pos + chunk)
            feasible = self.evaluate_positions(pos, hi)
            evaluated += hi - pos
            end = hi
            for h in pos + np.flatnonzero(feasible):
                if first is None:
                    first = time.perf_counter() - self.t0
                sid = int(self.c_sid[h])
                if sid not in found:
                    found[sid] = int(h)
                    if stop_when_complete and len(found) == k:
                        end, stop = int(h) + 1, 'complete'
                        break
            examined += end - pos
            pos = end
            chunk = min(MAX_CHUNK, 2 * chunk)
            if stop == 'complete':
                break
        total = time.perf_counter() - self.t0
        exhausted = pos >= N
        best = sorted(found.values())[:k]
        status = 'sat' if best else ('model_unsat' if exhausted else 'budget_exhausted')
        return dict(
            status=status, complete=bool(exhausted or stop == 'complete'), exhausted=bool(exhausted),
            stopped_by=stop or 'exhausted', budget_s=None if budget_s == INF else budget_s,
            over_budget=bool(total > budget_s), n_space=self.n_space, n_ranked=self.n_ranked,
            pruned=dict(sorted(self.pruned.items())), examined=examined, evaluated=evaluated,
            setup_s=self.setup_s, search_s=total - self.setup_s, total_s=total, first_feasible_s=first,
            mode=self.tables.mode, rate_rule=self.rate_rule, rate_margin=self.rate_margin,
            mean_margin=self.mean_margin, rate_level=self.rate_level, det_min=self.det_min,
            min_ba=self.min_ba, strength_floor=dict(self.lo), fe_responsible=self.fe_resp,
            stage_ms={str(k_): v for k_, v in self.stage_ms.items()},
            candidates=[self.describe(h, rank) for rank, h in enumerate(best, 1)])

    # ---- one configuration, scalar path (reporting and cross-checks) ------------------------------------------------
    def evaluate(self, order, stage, strengths, details=True):
        """Every rule for one on-knot configuration, by a scalar path independent of the vectorized one.

        strengths: {fragment: knot value}. Raises if a strength is not one of the fragment's knots."""
        T = self.tables
        s = Structure(order, stage)
        kidx = {}
        for f in s.order:
            i = K.knot_index(T.knots[f], strengths[f])
            if i is None:
                raise ValueError(f'{f}={strengths[f]} is not a measured knot')
            kidx[f] = i
        count, tau, max_tests, mean_thr = self.threshold(s)
        violations = []
        rule = self.structure_rule(s)
        if rule:
            violations.append(rule)
        if any(not (self.lo[f] <= T.knots[f][kidx[f]] <= self.hi[f]) for f in s.order):
            violations.append('clean_floor_or_range')
        terms = T.single_terms(s.order, s.stage)
        D = 0.0
        d_flags = []
        for f in FRAGMENTS:
            if f in s.order:
                value, raw_used, pen_fitted = terms[f]
                D += value[kidx[f]]
                if not raw_used[kidx[f]]:
                    d_flags.append(f'distortion_fitted:{f}')
                if pen_fitted[kidx[f]]:
                    d_flags.append(f'stage_penalty_fitted:{s.stage}')
        for (f, g), E in T.e.items():
            if f in s.order and g in s.order:
                D += E[kidx[f], kidx[g]]
        if D > self.d_max:
            violations.append('fidelity')
        cells, cover, flags = {}, {}, set(d_flags)
        solver_cover = {}
        for a in self.attacks:
            cells[a] = {}
            for f in s.order:
                key, values, overwriters, pair_g = self.wiring(s, f, a)
                level = float(values[kidx[f]])
                drops = [float(T.delta[(g, f, a)][kidx[g]]) if g in overwriters else 0.0
                         for g in FRAGMENTS if g != f]
                mean = level - sum(drops)
                used, info = self.rate(s.stage, f, a, count)
                rate_ok, solver_ok = bool(used[kidx[f]]), bool(info['solver'][kidx[f]])
                if key[0] == 'pair':
                    source = 'pair_fitted:' + T.pair[key[1:]]['name']
                    flags.add(source)
                else:
                    lv = T.level[key[1:]]
                    if lv['raw'][kidx[f]]:
                        source = 'raw:' + lv['record']
                    else:
                        source = 'fitted:' + lv['curve']
                        if T.mode == 'raw':
                            flags.add(f'mean_fitted:{lv["curve"]}')
                if info['missing_changed_embed']:
                    flags.add(f'rate_missing_changed_embed:{s.stage}:{f}|{a}')
                elif self.rate_rule == 'raw' and info['source'] and info['fallback'][kidx[f]]:
                    flags.add(f'rate_fitted:{info["source"]}')
                cov = bool(mean >= mean_thr and rate_ok)
                cover[(f, a)] = cov
                solver_cover[(f, a)] = bool(mean >= mean_thr and solver_ok)
                if details:
                    raw_rate = info['raw_rate'][kidx[f]]
                    cells[a][f] = dict(mean=mean, mean_source=source, level=level,
                                       drops={g: d for g, d in zip([g for g in FRAGMENTS if g != f], drops) if d},
                                       rate_source=info['source'], rate_stage=info['actual_stage'],
                                       raw_rate=None if not np.isfinite(raw_rate) else float(raw_rate),
                                       rate_floor_strength=info['floor'], rate_pass=rate_ok,
                                       rate_pass_solver=solver_ok, mean_pass=bool(mean >= mean_thr), cover=cov)
        covered = {a: [f for f in s.order if cover[(f, a)]] for a in self.attacks}
        useful = {f: [a for a in self.attacks if cover[(f, a)]] for f in s.order}
        if not all(covered.values()):
            violations.append('coverage')
        if self.no_passengers and not all(useful.values()):
            violations.append('passenger')
        solver_ok = (all(any(solver_cover[(f, a)] for f in s.order) for a in self.attacks) and
                     (not self.no_passengers or all(any(solver_cover[(f, a)] for a in self.attacks) for f in s.order)))
        label = 'grid_lookup_surrogate' if s.m > 1 else 'onknot_lookup'
        if self.tables.mode == 'fitted':
            label = 'fitted_surrogate'
        # Exact on-knot: a lone fragment whose every cell read a raw record at its knot (no curve anywhere).
        exact_response = (self.tables.mode == 'raw' and s.m == 1 and
                          not any(fl.startswith(('mean_fitted:', 'rate_fitted:', 'pair_fitted:')) for fl in flags))
        exact_distortion = not any(fl.startswith(('distortion_fitted:', 'stage_penalty_fitted:')) for fl in flags)
        out = dict(structure=s.as_dict(), order=list(s.order), fe_on=s.fe_on,
                   fe={st: st == s.stage for st in STAGES}, s={f: T.knots[f][kidx[f]] for f in s.order},
                   knot_index=kidx, D_pred=D, psnr_pred=psnr_of(D), psnr_db=psnr_of(D), ms=self.latency(s),
                   live_threshold=tau, threshold_count=count, max_tests=max_tests, mean_threshold=mean_thr,
                   rate_level=self.rate_level, label=label, surrogate_protocol=label,
                   exact_onknot_response=exact_response, exact_onknot_distortion=exact_distortion,
                   feasible=not violations, violations=violations, coverage=covered, useful=useful,
                   feasible_under_solver_rate=bool(solver_ok and not [v for v in violations
                                                                       if v not in ('coverage', 'passenger')]),
                   rate_disagreements=sorted(f'{f}|{a}' for (f, a) in cover
                                             if cells.get(a, {}).get(f, {}).get('rate_pass') !=
                                             cells.get(a, {}).get(f, {}).get('rate_pass_solver')) if details else None,
                   flags=sorted(flags))
        if details:
            out['cells'] = cells
        return out

    def describe(self, position, rank):
        s = self.structures[int(self.c_sid[position])]
        kn = self.c_kn[position]
        strengths = {f: self.tables.knots[f][int(kn[FRAGMENTS.index(f)])] for f in s.order}
        out = self.evaluate(s.order, s.stage, strengths)
        if not out['feasible'] or abs(out['D_pred'] - float(self.c_D[position])) > 1e-9 * max(1.0, out['D_pred']):
            raise AssertionError(f'scalar and vectorized evaluations disagree on {s} {strengths}')
        out['rank'] = rank
        out['position'] = int(position)
        return out


def search(tables, request, *, margin_stage='initial', budget_s=INF, stop_when_complete=True, k=3, **kw):
    """C2 for one request at one margin stage; the clock starts before any request-specific work."""
    t0 = time.perf_counter()
    rate_margin, mean_margin = margins(tables.ns, margin_stage)
    gs = GridSearch(tables, request, rate_margin=rate_margin, mean_margin=mean_margin, t0=t0, **kw)
    out = gs.run(budget_s=budget_s, stop_when_complete=stop_when_complete, k=k)
    out['margin_stage'] = margin_stage
    return out
