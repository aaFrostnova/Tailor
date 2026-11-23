"""The C2 grid baseline (experiments/grid) for n fragments, for the solver-scaling experiment.

The grid code is written for the three measured fragments: module-level FRAGMENTS tuples and arrays with three
fragment columns. install(n) points the module-level fragment lists at the n fragments (they are read at call
time) and GridSearchN overrides the three methods that hard-code three columns, with the same algorithm:
  _candidates          every structure of the n fragments, the same pruning, the same ranking key (D, fewer
                       stages, structure, knot tuple), knot indices stored as int8 and structure ids as int32
                       (the space reaches 1.7e8 configurations at n = 5);
  _wiring              flat coverage tables with n fragment columns;
  evaluate_positions   the same vectorized coverage rule over n fragment columns.
For n = 3 the results equal grid.grid_enumerator.GridSearch (check_n3.py). Knots of a virtual fragment are its
source's (its per-image records are copies on the source's knots).
"""
import collections
import math

import numpy as np

from grid import grid_enumerator as GE
from grid import knots as K
from grid import tables as TB

from . import virtual as V

KDTYPE = np.int8
_STATE = {}


def install(n):
    frs = tuple(V.fragments(n))
    if 'n' in _STATE:
        if _STATE['n'] != n:
            raise RuntimeError('grid already extended for another n')
        return frs
    K.FRAGMENTS = TB.FRAGMENTS = GE.FRAGMENTS = frs
    TB.FE_AFFECTS['resync'] = frs                  # the sync mark changes every fragment's embed (same dict in GE)
    original = K.knot_sets

    def knot_sets(perimage, check=True):
        knots = K.derive_knots(perimage, fragments=frs)
        real = {f: knots[f] for f in V.SOURCES}
        if check and real != K.EXPECTED_KNOTS:
            raise AssertionError(f'measured knots differ from the user-approved sets: {real}')
        for f in frs[len(V.SOURCES):]:
            if knots[f] != knots[V.source(f)]:
                raise AssertionError(f'virtual {f} is not on its source knots')
        return knots
    K.knot_sets = knot_sets
    _STATE.update(n=n, original_knot_sets=original)
    return frs


class GridSearchN(GE.GridSearch):
    def _candidates(self):
        T, F = self.tables, GE.FRAGMENTS
        NF = len(F)
        self.all_structures = GE.structures(fragments=F)
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
            kn = np.full((inside, NF), -1, KDTYPE)
            for f, col in zip(s.order, np.meshgrid(*idx, indexing='ij')):
                kn[:, F.index(f)] = col.ravel()
            D = self.distortion(s, kn)
            ok = D <= self.d_max
            self.pruned['fidelity'] += int((~ok).sum())
            if ok.any():
                sids.append(np.full(int(ok.sum()), len(kept), np.int32))
                kns.append(kn[ok])
                Ds.append(D[ok])
                kept.append(s)
        self.structures = kept
        if kept:
            sid, kn, D = np.concatenate(sids), np.concatenate(kns), np.concatenate(Ds)
        else:
            sid, kn, D = np.zeros(0, np.int32), np.zeros((0, NF), KDTYPE), np.zeros(0)
        del sids, kns, Ds
        lex = sorted(range(len(kept)), key=lambda i: kept[i].lex_key())
        rank = np.empty(len(kept), np.int32)
        rank[lex] = np.arange(len(kept))
        nstage = np.array([s.stage is not None for s in kept], np.int8)
        keys = tuple(kn[:, j] for j in reversed(range(NF)))
        order = np.lexsort(keys + (rank[sid] if len(kept) else sid, nstage[sid] if len(kept) else sid, D))
        self.c_sid, self.c_kn, self.c_D = sid[order], kn[order], D[order]
        del order
        self.n_ranked = len(self.c_D)
        assert self.n_space == self.n_ranked + sum(self.pruned.values())

    def _wiring(self):
        T, A, S, F = self.tables, len(self.attacks), len(self.structures), GE.FRAGMENTS
        NF = len(F)
        flat = {'lv': [[], {}, 0], 'dl': [[], {}, 0], 'rp': [[], {}, 0]}

        def offset(kind, key, values):
            arrays, index, _ = flat[kind]
            if key not in index:
                index[key] = flat[kind][2]
                arrays.append(values)
                flat[kind][2] += len(values)
            return index[key]

        self.LVO = np.zeros((S, NF, A), np.int64)
        self.RPO = np.zeros((S, NF, A), np.int64)
        self.DLO = np.full((S, NF, A, NF), -1, np.int64)
        self.SEL = np.zeros((S, NF), bool)
        self.THR = np.zeros(S)
        for si, s in enumerate(self.structures):
            count, _, _, mean_thr = self.threshold(s)
            self.THR[si] = mean_thr
            for fi, f in enumerate(F):
                if f not in s.order:
                    continue
                self.SEL[si, fi] = True
                for j, a in enumerate(self.attacks):
                    key, values, overwriters, _ = self.wiring(s, f, a)
                    self.LVO[si, fi, j] = offset('lv', key, values)
                    for g in overwriters:
                        self.DLO[si, fi, j, F.index(g)] = offset('dl', ('delta', g, f, a), T.delta[(g, f, a)])
                    self.RPO[si, fi, j] = offset('rp', (s.stage, f, a, count), self.rate(s.stage, f, a, count)[0])
        cat = lambda kind, dtype: (np.concatenate(flat[kind][0]).astype(dtype) if flat[kind][0]
                                   else np.zeros(1, dtype))
        self.LVF, self.DLF, self.RPF = cat('lv', float), cat('dl', float), cat('rp', bool)

    def evaluate_positions(self, lo, hi, return_cover=False):
        NF = len(GE.FRAGMENTS)
        sid, kn = self.c_sid[lo:hi], self.c_kn[lo:hi].astype(np.int64)
        n, A = len(sid), len(self.attacks)
        cover = np.zeros((n, NF, A), bool)
        sel = self.SEL[sid]
        safe = np.where(kn >= 0, kn, 0)
        thr = self.THR[sid][:, None]
        for fi in range(NF):
            if not sel[:, fi].any():
                continue
            k = safe[:, fi][:, None]
            level = self.LVF[self.LVO[sid, fi, :] + k]
            drop = np.zeros((n, A))
            for gi in range(NF):                      # composite_expression sums drops in fragment order
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
