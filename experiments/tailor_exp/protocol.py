"""Verification arithmetic of the frozen unified raw-BA protocol (unified_raw_ba_v1).

Re-implemented from the written rule (plans/preregistration.json of the run root,
src/method.tex) and cross-checked against the frozen decoder
(unified_ba_20260914/code/unified_detector.py) by tests/test_protocol.py:

* a bit matches only when its finite aligned LLR has the strictly correct sign; 0 is a miss;
* T(c) = K0(m) + 5m [resync] + 133 [scale] + 81 [tile] + 65m [angle], K0(1) = 1, K0(m) = m + 1;
* theta(c, alpha) = smallest integer h in 0..101 with T(c) * P(Bin(100, 1/2) >= h) <= alpha,
  compared exactly against the rational value of the float alpha; h = 101 means no
  number of matches can pass (impossible threshold).
"""
from fractions import Fraction
from functools import lru_cache
from math import comb, isfinite
from numbers import Real

import numpy as np

N_BITS = 100
PROTOCOL = 'unified_raw_ba_v1'
FRAGMENTS = ('vine', 'trustmark', 'videoseal')
STAGES = ('resync', 'scale', 'tile', 'angle')
STAGE_HOST = {'scale': 'vine', 'tile': 'trustmark'}
IMPOSSIBLE = N_BITS + 1
_TAIL = tuple(sum(comb(N_BITS, j) for j in range(t, N_BITS + 1)) for t in range(N_BITS + 2))


class ProtocolError(ValueError):
    """A configuration or input outside the frozen verification protocol."""


def _normalize(order, stages):
    order = [str(f).lower() for f in order]
    stages = sorted(str(s) for s in stages)
    if not order or len(order) != len(set(order)) or any(f not in FRAGMENTS for f in order):
        raise ProtocolError(f'invalid fragment order {order}')
    if len(stages) != len(set(stages)) or any(s not in STAGES for s in stages):
        raise ProtocolError(f'invalid geometric stages {stages}')
    for stage, host in STAGE_HOST.items():
        if stage in stages and host not in order:
            raise ProtocolError(f'{stage} recovery requires {host}')
    return order, stages


def max_tests(order, stages):
    order, stages = _normalize(order, stages)
    m = len(order)
    total = 1 if m == 1 else m + 1
    total += 5 * m if 'resync' in stages else 0
    total += 133 if 'scale' in stages else 0
    total += 81 if 'tile' in stages else 0
    total += 65 * m if 'angle' in stages else 0
    return total


def check_deployable(order, stages):
    """The deployed method selects at most one geometric stage."""
    order, stages = _normalize(order, stages)
    if len(stages) > 1:
        raise ProtocolError(f'at most one geometric stage may be selected, got {stages}')
    return order, stages


def _alpha(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ProtocolError('alpha must be a real number')
    alpha = float(alpha)
    if not isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ProtocolError('alpha must lie strictly between 0 and 1')
    return alpha


@lru_cache(maxsize=8192)
def _threshold(t_c, alpha):
    budget = Fraction(alpha) / t_c
    for h, numerator in enumerate(_TAIL):
        if Fraction(numerator, 1 << N_BITS) <= budget:
            return h
    raise AssertionError('the tail at 101 is zero and always meets the budget')


def threshold_for_tests(t_c, alpha):
    if isinstance(t_c, bool) or not isinstance(t_c, int) or t_c < 1:
        raise ProtocolError('T(c) must be a positive integer')
    return _threshold(t_c, _alpha(alpha))


def threshold_count(order, stages, alpha):
    return threshold_for_tests(max_tests(order, stages), alpha)


def is_impossible(count):
    return count >= IMPOSSIBLE


def strict_matches(llrs, word):
    llrs = np.asarray(llrs, dtype=np.float64)
    word = np.asarray(word)
    if llrs.shape != (N_BITS,) or word.shape != (N_BITS,):
        raise ProtocolError('expected 100 LLRs and a 100-bit word')
    if not np.all(np.isfinite(llrs)) or not np.all((word == 0) | (word == 1)):
        raise ProtocolError('LLRs must be finite and the word binary')
    return int(np.count_nonzero(((llrs > 0) & (word == 1)) | ((llrs < 0) & (word == 0))))
