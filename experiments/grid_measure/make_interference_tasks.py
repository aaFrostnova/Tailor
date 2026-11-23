"""Predeclared configurations for the ordered interference matrix (PR3) and the triple residuals (PR7).

Strength knots per fragment: the measured knots nearest to the 10th, 50th and 90th percentile of the strengths
TAILOR-F deployed for its accepted requests (VINE 0.3/0.5/0.9, TrustMark 0.55/0.7/1.0, VideoSeal
0.625/0.875/1.375, computed 2026-11-09 from the held-out manifest), so the matrix covers the strengths the
optimizer actually selects. No stage. Attacks: clean, jpeg25 (signal), crop75 and rot9 (geometric), regen
(regeneration). PR3: 9 single-fragment and 54 ordered-pair configurations. PR7: the 6 orders of the triple at
low, mid, high and mixed strengths; their pairwise prediction comes from the PR3 measurements at the same knots.

Usage: python make_interference_tasks.py OUT_DIR
"""
import itertools
import json
import sys
from pathlib import Path

KNOTS = {'VINE': (0.3, 0.5, 0.9), 'TrustMark': (0.55, 0.7, 1.0), 'VideoSeal': (0.625, 0.875, 1.375)}
LEVELS = ('low', 'mid', 'high')
MATRIX_ATTACKS = ['clean', 'jpeg25', 'crop75', 'rot9', 'regen']
TRIPLE_ATTACKS = ['clean', 'jpeg25', 'regen']
MIXED = {'VINE': 'high', 'TrustMark': 'low', 'VideoSeal': 'mid'}


def name(order, levels):
    return '__'.join(f'{f}-{lv}' for f, lv in zip(order, levels))


def tasks():
    frags = list(KNOTS)
    matrix = []
    for f in frags:
        for i, lv in enumerate(LEVELS):
            matrix.append(dict(name=name([f], [lv]), order=[f], fe_on=[], s={f: KNOTS[f][i]}, attacks=MATRIX_ATTACKS,
                               kind='single', levels={f: lv}))
    for f, g in itertools.permutations(frags, 2):
        for i, j in itertools.product(range(3), range(3)):
            matrix.append(dict(name=name([f, g], [LEVELS[i], LEVELS[j]]), order=[f, g], fe_on=[],
                               s={f: KNOTS[f][i], g: KNOTS[g][j]}, attacks=MATRIX_ATTACKS, kind='pair',
                               levels={f: LEVELS[i], g: LEVELS[j]}))
    triples = []
    combos = {'low': {f: 'low' for f in frags}, 'mid': {f: 'mid' for f in frags},
              'high': {f: 'high' for f in frags}, 'mixed': MIXED}
    for label, lv in combos.items():
        for order in itertools.permutations(frags):
            triples.append(dict(name=f'triple_{label}__' + name(order, [lv[f] for f in order]), order=list(order),
                                fe_on=[], s={f: KNOTS[f][LEVELS.index(lv[f])] for f in order},
                                attacks=TRIPLE_ATTACKS, kind='triple', levels=lv, combo=label))
    return matrix, triples


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    matrix, triples = tasks()
    assert len(matrix) == 63 and len(triples) == 24
    (out / 'interference_tasks.json').write_text(json.dumps(matrix, indent=1) + '\n')
    (out / 'triple_tasks.json').write_text(json.dumps(triples, indent=1) + '\n')
    print(len(matrix), len(triples))
