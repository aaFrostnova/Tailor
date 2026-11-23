"""Measured strength knots per fragment for C2, and which raw per-image cells exist at them.

User decision 2026-11-09 (experiments/protocol.md section 8 item 6): the knot set of a fragment is exactly the set of
strengths from which the method's model was built, i.e. the raw per-image knots of the frozen surrogate's
perimage block (the records Surrogate.rate_curve reads, surrogate_model.py:78-92). derive_knots() reads them
from the table (the knot grid shared by most of the fragment's records) and knot_sets() asserts the result
against EXPECTED_KNOTS. Records on other grids (editing, UnMarker, CtrlRegen+ extensions) are reported by
availability(), never merged into the knot set.
"""
import collections
import math

FRAGMENTS = ('VINE', 'TrustMark', 'VideoSeal')
STAGES = ('resync', 'scale', 'angle', 'tile')              # composite_prediction.FRONTEND_ORDER
EXPECTED_KNOTS = {
    'VINE': (0.14, 0.17, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    'TrustMark': (0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0),
    'VideoSeal': (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.75, 2.0, 2.25, 2.5, 2.8, 3.2),
}
TOL = 1e-9          # a record abscissa within TOL of a knot is that knot (JSON floats such as 0.6000000000000001)


def parse_key(key):
    """'F|a' -> (None, F, a); 'stage:F|a' -> (stage, F, a) (Surrogate.perimage_key, surrogate_model.py:72-73)."""
    stage, rest = key.split(':', 1) if ':' in key else (None, key)
    fragment, attack = rest.split('|', 1)
    return stage, fragment, attack


def perimage_key(fragment, attack, stage=None):
    return f'{stage}:{fragment}|{attack}' if stage else f'{fragment}|{attack}'


def record_rows(record, fragment):
    """Per-knot rows of per-image BA exactly as Surrogate.rate_curve reads them (surrogate_model.py:90-91):
    the primary view of a two-view record, else the stored BA."""
    if 'raw_views' in record:
        return [cell['ba'][fragment] for cell in record['raw_views']]
    return record['ba']


def derive_knots(perimage, fragments=FRAGMENTS):
    """{fragment: knot tuple}: the grid shared by the largest number of the fragment's per-image records."""
    grids = {f: collections.Counter() for f in fragments}
    for key, record in perimage.items():
        _, fragment, _ = parse_key(key)
        if fragment in grids:
            grids[fragment][tuple(float(x) for x in record['xs'])] += 1
    out = {}
    for fragment, counter in grids.items():
        ranked = counter.most_common()
        if not ranked or (len(ranked) > 1 and ranked[1][1] == ranked[0][1]):
            raise ValueError(f'no unique modal knot grid for {fragment}: {ranked[:2]}')
        out[fragment] = ranked[0][0]
    return out


def knot_sets(perimage, check=True):
    knots = derive_knots(perimage)
    if check and knots != EXPECTED_KNOTS:
        raise AssertionError(f'measured knots differ from the user-approved sets: {knots}')
    return knots


def knot_index(knots, x):
    """Index of the knot equal to x (within TOL), else None. Never rounds to a neighbouring knot."""
    for i, k in enumerate(knots):
        if abs(float(k) - float(x)) <= TOL:
            return i
    return None


def count_space(knots, structures):
    """Number of (structure, knot tuple) configurations before any floor or constraint."""
    return sum(math.prod(len(knots[f]) for f in s.order) for s in structures)


def solver_curve(sg, fragment, attack, stage):
    """The fitted mean curve composite_expression reads for a lone fragment with `stage` on (or None):
    the stage's replacement curve when it exists, else the legacy nested/resync curve when no stage has a
    replacement curve for the cell, else the plain curve (composite_prediction.py:88-157)."""
    if stage is not None:
        name = f'base_fe_{stage}_{fragment}|{attack}'
        if sg.frontend(name) is not None:
            return name, perimage_key(fragment, attack, stage)
        if not any(sg.frontend(f'base_fe_{st}_{fragment}|{attack}') is not None for st in STAGES) \
                and stage in ('scale', 'resync'):
            legacy = f"base_{'nested' if stage == 'scale' else stage}_{fragment}|{attack}"
            if sg.frontend(legacy) is not None:
                return legacy, None
    return f'base:{fragment}|{attack}', perimage_key(fragment, attack)


def availability(sg, knots, attacks=None):
    """Which C2 cells have an exact on-knot raw value.

    For every (stage or none, fragment, attack): the fitted curve the solver reads for the lone fragment, the
    per-image record that backs it, the knots the record has and lacks, and its image counts. A cell whose
    record is absent, or that lacks a knot, falls back to the fitted curve at that knot in C2 (flagged)."""
    attacks = list(attacks or sg.attacks)
    perimage = sg._perimage
    cells = []
    for stage in (None,) + STAGES:
        for fragment in FRAGMENTS:
            if fragment not in knots:
                continue
            for attack in attacks:
                curve, key = solver_curve(sg, fragment, attack, stage)
                record = perimage.get(key) if key else None
                row = dict(stage=stage, fragment=fragment, attack=attack, curve=curve, record=key,
                           present=record is not None)
                if record is not None:
                    xs = [float(x) for x in record['xs']]
                    rows = record_rows(record, fragment)
                    on = {knot_index(knots[fragment], x): len(r) for x, r in zip(xs, rows)}
                    row['n'] = {repr(knots[fragment][i]): n for i, n in sorted((i, n) for i, n in on.items()
                                                                               if i is not None)}
                    row['missing_knots'] = [k for i, k in enumerate(knots[fragment]) if i not in on]
                    row['off_grid_xs'] = [x for x in xs if knot_index(knots[fragment], x) is None]
                    row['two_view'] = 'raw_views' in record
                else:
                    row['missing_knots'] = list(knots[fragment])
                    row['off_grid_xs'], row['n'], row['two_view'] = [], {}, False
                row['status'] = ('exact' if record is not None and not row['missing_knots'] else
                                 'partial' if record is not None else 'fitted_only')
                cells.append(row)
    summary = collections.Counter((c['stage'] is not None, c['status']) for c in cells)
    stage_cells = [c for c in cells if c['stage'] is not None and c['curve'].startswith('base_fe_')]
    return dict(
        knots={f: list(v) for f, v in knots.items()}, cells=cells,
        summary={f"{'stage' if s else 'plain'}:{status}": n for (s, status), n in sorted(summary.items())},
        fitted_only_stage_curves=sorted({c['curve'] for c in stage_cells if c['status'] == 'fitted_only'}),
        partial=sorted({(c['record'], tuple(c['missing_knots'])) for c in cells if c['status'] == 'partial'}),
        reduced_n=sorted({c['record'] for c in cells if c['present'] and any(n != 100 for n in c['n'].values())}),
        records_off_grid=sorted({c['record'] for c in cells if c['present'] and c['off_grid_xs']}))


def main(argv=None):
    """Print the knot sets and the availability summary; optionally write the full report as JSON."""
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from grid import frozen
    argv = sys.argv[1:] if argv is None else argv
    sg, digest = frozen.load_surrogate()
    knots = knot_sets(sg._perimage)
    report = availability(sg, knots)
    report['surrogate_sha256'] = digest
    print('knots', {f: list(v) for f, v in knots.items()})
    print('summary', report['summary'])
    print('fitted-only stage curves', len(report['fitted_only_stage_curves']))
    for record, missing in report['partial']:
        print('partial', record, 'missing', list(missing))
    print('reduced n', report['reduced_n'])
    print('off-grid records', report['records_off_grid'])
    if argv:
        Path(argv[0]).write_text(json.dumps(report, indent=1, default=list) + '\n')
    return report


if __name__ == '__main__':
    main()
