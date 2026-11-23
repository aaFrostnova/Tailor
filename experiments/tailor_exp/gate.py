"""PR0 gate: reproduce the frozen main and screening-margin tables from hash-checked records.

The gate reads, never writes, except one report under --output (which must not exist yet).
Every input is addressed by a path relative to the paper repository ('paper:') or to the run
root ('run:'), and must match a pinned sha256 before it is parsed:

* the cohort file data/main_cohort.json is pinned here; the sha256 values it
  records for the nine run-root result files then pin those files by relative path, so a moved
  run root still verifies and an altered file always aborts;
* the frozen tables are read from git at REFERENCE_COMMIT and pinned too.

From the per-request records the gate recomputes, for the 13 methods of the frozen cohort file,
the per-scenario satisfaction, the macro average, the own-success PSNR and the paired PSNR, and
checks (a) equality with the summaries stored in the cohort file, (b) that the stored SSIM values
were computed on exactly the recomputed accepted sets, and (c) byte equality of every row of
tab/main_results.tex and tab/screening_margin.tex at REFERENCE_COMMIT.
"""
import os
import argparse
import collections
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import mean

from . import cohort, protocol, schema

FROZEN_RUN_ROOT = (os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/'
                   'rigor_editing_only_20260911/unified_ba_20260914')
OLD_PAPER_PREFIX = os.environ.get('TAILOR_PAPER', 'paper') + '/'
REFERENCE_COMMIT = os.environ.get('TAILOR_PAPER_COMMIT', 'HEAD')
COHORT_FILE = 'data/main_cohort.json'
SSIM_FILE = 'data/ssim_main_20260924.json'
PAPER_PINS = {
    COHORT_FILE: 'ec437a9ad71eacdba36a4dfaa31de87b649ec9a62233cc927c359e38fcd297c6',
    SSIM_FILE: '79a53747189a6a9fec8297fc41e85c7a1aacc2567499ce38732211e5561bef08',
    'data/request_specifications.json': '5245e56eacfa959da03bf2e38d16328e9ce302b562b40abf05876afe01664e4d',
    'data/editing_requests.json': '05a49611ff9b596a9179f539304025477ca79eed4ff3773d22dcb7ad61d6814c',
}
TABLE_PINS = {
    'tab/main_results.tex': '830e9260aca08d1d809fa169efae013e68c0379b84db7e9095cda39416d40794',
    'tab/screening_margin.tex': 'd4b22f7a51ca55b55e01fb316de063c7968ac6882adccb1312a3cf30b5935137',
}
RUN_INPUTS = (
    'quality_paired_20260921/requests.json', 'fixed1_20260918/results.json', 'single_geo_20260921/results.json',
    'live_failure_fallback_20260921/amended_results.json', 'editing_request_extension_20260921/results.json',
    'single_extra_20260923/results.json', 'single_extra_20260923/audit.json',
    'enumeration_geo_20260923/results.json', 'enumeration_geo_20260923/audit.json',
)
METHODS = ('StegaStamp', 'MaskWM', 'VINE', 'TrustMark', 'VideoSeal', 'VINEGeo', 'TrustMarkGeo', 'Fixed1',
           'EnumGeo', 'Greedy04', 'SMT04', 'Greedy06', 'SMT06')
MAIN_ORDER = ('StegaStamp', 'MaskWM', 'VINE', 'TrustMark', 'VideoSeal', 'VINEGeo', 'TrustMarkGeo', 'Fixed1',
              'EnumGeo', 'Greedy06', 'SMT06')
MARGIN_ORDER = ('Greedy04', 'SMT04', 'Greedy06', 'SMT06')
EDITING_VARIANT = {
    'SMT04': ('Full', '04'), 'SMT06': ('Full', '06'), 'Greedy04': ('G_shared', '04'),
    'Greedy06': ('G_shared', '06'), 'Fixed1': ('Fixed1', 'na'), 'VINE': ('Single_VINE', 'na'),
    'TrustMark': ('Single_TrustMark', 'na'), 'VideoSeal': ('Single_VideoSeal', 'na'),
    'VINEGeo': ('SingleGeo_VINE', 'na'), 'TrustMarkGeo': ('SingleGeo_TrustMark', 'na'),
}
PAIR_BASELINES = ('Fixed1', 'EnumGeo', 'VINE', 'TrustMark', 'VideoSeal', 'MaskWM', 'StegaStamp', 'VINEGeo',
                  'TrustMarkGeo')
LABELS = dict(VINEGeo='VINE + Geo.', TrustMarkGeo='TrustMark + Geo.', Fixed1='Enumeration',
              EnumGeo='Enumeration + Geo.', Greedy04=r'\sysgreedy{}', Greedy06=r'\sysgreedy{}',
              SMT04=r'\sysfull{}', SMT06=r'\sysfull{}')
SCENARIOS = ('S1', 'S2', 'S3', 'S4', 'S5')
TOLERANCE = 1e-9


class GateError(RuntimeError):
    """An input is missing, unpinned or altered, or the records break the cohort contract."""


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _ids_hash(ids):
    return _sha256(json.dumps(sorted(ids)).encode())


class _Inputs:
    """Reads pinned inputs by relative path and records what it read."""

    def __init__(self, paper, run_root):
        self.roots = dict(paper=Path(paper), run=Path(run_root))
        self.pins = {f'paper:{rel}': h for rel, h in PAPER_PINS.items()}
        self.read_hashes = {}

    def read(self, space, rel):
        key = f'{space}:{rel}'
        if key not in self.pins:
            raise GateError(f'{key} is not pinned')
        path = self.roots[space] / rel
        if not path.is_file():
            raise GateError(f'{key} is missing at {path}')
        raw = path.read_bytes()
        digest = _sha256(raw)
        if digest != self.pins[key]:
            raise GateError(f'{key} has sha256 {digest}, pinned {self.pins[key]}')
        self.read_hashes[key] = digest
        return json.loads(raw)

    def pin_run_inputs(self, recorded):
        old_run = FROZEN_RUN_ROOT + '/'
        for path, digest in recorded.items():
            if path.startswith(old_run):
                self.pins['run:' + path[len(old_run):]] = digest
            elif path.startswith(OLD_PAPER_PREFIX):
                rel = path[len(OLD_PAPER_PREFIX):]
                if rel in PAPER_PINS and PAPER_PINS[rel] != digest:
                    raise GateError(f'paper:{rel} pin disagrees with the cohort file record')
                self.pins.setdefault(f'paper:{rel}', digest)
            else:
                raise GateError(f'cohort file records an input outside both roots: {path}')
        unpinned = [rel for rel in RUN_INPUTS if f'run:{rel}' not in self.pins]
        if unpinned:
            raise GateError(f'run inputs without a recorded sha256: {unpinned}')


def _assemble(inputs):
    base = inputs.read('run', 'quality_paired_20260921/requests.json')['methods']
    methods = {k: list(v.values()) for k, v in base.items()}
    methods['Fixed1'] = inputs.read('run', 'fixed1_20260918/results.json')['rows']
    geo = inputs.read('run', 'single_geo_20260921/results.json')['rows']
    for fragment in ('VINE', 'TrustMark'):
        methods[fragment + 'Geo'] = [r for r in geo if r['fragment'] == fragment]
    amended = inputs.read('run', 'live_failure_fallback_20260921/amended_results.json')['rows']
    for arm in ('04', '06'):
        methods['SMT' + arm] = [r for r in amended if r['cohort'] == 'main' and r['arm'] == arm]
    editing = inputs.read('run', 'editing_request_extension_20260921/results.json')['rows']
    methods = {k: methods[k] + [r for r in editing if (r['variant'], r['arm']) == v]
               for k, v in EDITING_VARIANT.items()}
    extra = inputs.read('run', 'single_extra_20260923/results.json')
    extra_audit = inputs.read('run', 'single_extra_20260923/audit.json')
    if not extra_audit.get('passed') or (
            extra_audit.get('result_sha256') != inputs.read_hashes['run:single_extra_20260923/results.json']):
        raise GateError('single_extra_20260923/audit.json does not certify its results file')
    for name in ('MaskWM', 'StegaStamp'):
        methods[name] = [r for r in extra['rows'] if r['method'] == name]
    enum = inputs.read('run', 'enumeration_geo_20260923/results.json')
    enum_audit = inputs.read('run', 'enumeration_geo_20260923/audit.json')
    if not enum_audit.get('passed') or (
            enum_audit.get('results_sha256') != inputs.read_hashes['run:enumeration_geo_20260923/results.json']):
        raise GateError('enumeration_geo_20260923/audit.json does not certify its results file')
    methods['EnumGeo'] = enum['rows']
    return methods


def _psnr(record):
    value = record.get('psnr_live')
    if value is None:
        value = record['verdict']['psnr_live']
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GateError(f"non-finite PSNR for accepted request {record['spec_id']}")
    return value


def _index(method, rows, manifest):
    index = {}
    for r in rows:
        if r['spec_id'] not in manifest:
            continue
        if r['spec_id'] in index:
            raise GateError(f"{method}: two records for {r['spec_id']}")
        if r['cls'] != manifest[r['spec_id']]['cls']:
            raise GateError(f"{method}: record class {r['cls']} disagrees with the request specification")
        try:
            r['_status'] = schema.canonical_status(r['status'])
        except schema.SchemaError as err:
            raise GateError(f'{method}: {err}') from err
        index[r['spec_id']] = r
    missing = len(manifest) - len(index)
    if missing:
        raise GateError(f'{method}: {missing} cohort requests have no record')
    return index


def _summary(index):
    scenarios = {}
    for s in SCENARIOS:
        part = [r for r in index.values() if r['cls'] == 'C' + s[1:]]
        ok = [r for r in part if r['_status'] == 'accepted']
        scenarios[s] = dict(n=len(part), accepted=len(ok), percent=100 * len(ok) / len(part),
                            psnr=mean(map(_psnr, ok)) if ok else None)
    ok = [r for r in index.values() if r['_status'] == 'accepted']
    return dict(n=len(index), accepted=len(ok), percent=100 * len(ok) / len(index),
                psnr=mean(map(_psnr, ok)) if ok else None,
                macro_percent=mean(v['percent'] for v in scenarios.values()), scenarios=scenarios,
                status_counts=dict(collections.Counter(r['_status'] for r in index.values())),
                raw_status_counts=dict(collections.Counter(r['status'] for r in index.values())),
                accepted_spec_ids_sha256=_ids_hash(r['spec_id'] for r in ok))


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOLERANCE


def _compare_summary(method, mine, frozen):
    out = []
    for key in ('n', 'accepted'):
        if mine[key] != frozen[key]:
            out.append(f'{method}.{key}: {mine[key]} != {frozen[key]}')
    for key in ('percent', 'macro_percent', 'psnr'):
        if not _close(mine[key], frozen[key]):
            out.append(f'{method}.{key}: {mine[key]} != {frozen[key]}')
    for s in SCENARIOS:
        a, b = mine['scenarios'][s], frozen['scenarios'][s]
        for key in ('n', 'accepted'):
            if a[key] != b[key]:
                out.append(f'{method}.{s}.{key}: {a[key]} != {b[key]}')
        for key in ('percent', 'psnr'):
            if not _close(a[key], b[key]):
                out.append(f'{method}.{s}.{key}: {a[key]} != {b[key]}')
    return out


def _pairs(indexes, ids):
    pairs = {}
    for arm in ('04', '06'):
        ours = indexes['SMT' + arm]
        for base in ('Greedy' + arm, *PAIR_BASELINES):
            joint = sorted(i for i in ids if ours[i]['_status'] == indexes[base][i]['_status'] == 'accepted')
            pairs[(arm, base)] = dict(
                arm=arm, baseline=base, n=len(joint), joint_spec_ids_sha256=_ids_hash(joint),
                psnr_baseline=mean(_psnr(indexes[base][i]) for i in joint) if joint else None,
                psnr_ours=mean(_psnr(ours[i]) for i in joint) if joint else None)
    return pairs


UNIFIED_TAG = 'unified_raw_ba_v1'


def _verdicts(record):
    """(candidate, verdict, where) for the verdict stored at the top of a record and in each walk step."""
    out = []
    if isinstance(record.get('verdict'), dict):
        out.append((record.get('candidate'), record['verdict'], 'top'))
    for step in record.get('walk') or []:
        if isinstance(step, dict) and isinstance(step.get('verdict'), dict):
            out.append((step.get('candidate', step), step['verdict'], 'walk'))
    return out


def _verdict_check(indexes, manifest):
    """Stored verdicts must carry the unified protocol, the T(c) of their candidate and theta(T(c), alpha)
    of the request; an accepted record's own verdict must have passed. Records without a stored verdict
    are counted, not judged (coverage is reported per method)."""
    methods, mismatches = {}, []
    for name, index in indexes.items():
        counts = dict(with_verdict=0, without_verdict=0, verdicts=0)
        for spec_id, record in index.items():
            found = _verdicts(record)
            counts['with_verdict' if found else 'without_verdict'] += 1
            alpha = manifest[spec_id]['alpha']
            for candidate, verdict, where in found:
                counts['verdicts'] += 1

                def bad(kind, detail):
                    mismatches.append(dict(method=name, spec_id=spec_id, where=where, kind=kind, detail=detail))

                tag = verdict.get('protocol')
                if tag is not None and UNIFIED_TAG not in str(tag):
                    bad('protocol', tag)
                    continue
                max_tests, threshold = verdict.get('max_tests'), verdict.get('threshold_count')
                if not isinstance(max_tests, int) or not isinstance(threshold, int):
                    bad('incomplete_verdict', dict(max_tests=max_tests, threshold_count=threshold))
                    continue
                if threshold != protocol.threshold_for_tests(max_tests, alpha):
                    bad('threshold', dict(max_tests=max_tests, stored=threshold, alpha=alpha))
                order = candidate.get('order') if isinstance(candidate, dict) else None
                if order:
                    names = [str(f).lower() for f in order]
                    if all(f in protocol.FRAGMENTS for f in names):
                        try:
                            expected = protocol.max_tests(names, candidate.get('fe_on') or [])
                        except protocol.ProtocolError as err:
                            bad('candidate', str(err))
                            expected = max_tests
                    else:
                        expected = 1  # a non-TAILOR watermark is one readout
                    if expected != max_tests:
                        bad('max_tests', dict(order=names, expected=expected, stored=max_tests))
                if where == 'top' and record['_status'] == 'accepted' and verdict.get('pass') is False:
                    bad('accepted_but_failed', None)
        methods[name] = counts
    return dict(methods=methods, mismatches=mismatches)


def rank(value, values, digits=2):
    if value is None:
        return '-'
    number = round(value, digits)
    levels = sorted({round(v, digits) for v in values if v is not None}, reverse=True)
    text = f'{value:.{digits}f}'
    if number == levels[0]:
        return r'\textbf{' + text + '}'
    if len(levels) > 1 and number == levels[1]:
        return r'\underline{' + text + '}'
    return text


def _main_rows(summaries, ssim, pairs):
    metrics = {k: [*[summaries[k]['scenarios'][s]['percent'] for s in SCENARIOS],
                   summaries[k]['macro_percent'], summaries[k]['psnr']] for k in MAIN_ORDER}
    quality = {k: ssim[k] for k in MAIN_ORDER}
    rows = []
    for k in MAIN_ORDER:
        values = [rank(v, [metrics[o][j] for o in MAIN_ORDER]) for j, v in enumerate(metrics[k])]
        values[5] = r'\cellcolor{orange!8}' + values[5]
        pair = pairs.get(('06', k))
        psnr_p = f"{pair['psnr_baseline']:.2f} / {pair['psnr_ours']:.2f}" if pair and pair['n'] else '-'
        rows.append(' & '.join([LABELS.get(k, k), *values, rank(quality[k], quality.values(), 4), psnr_p]) + r' \\')
    return rows


def _margin_rows(summaries, ssim):
    metrics = {k: [*[summaries[k]['scenarios'][s]['percent'] for s in SCENARIOS],
                   summaries[k]['macro_percent'], summaries[k]['psnr'], ssim[k]] for k in MARGIN_ORDER}
    rows = []
    for k in MARGIN_ORDER:
        values = [rank(v, [metrics[o][j] for o in MARGIN_ORDER], 4 if j == 7 else 2)
                  for j, v in enumerate(metrics[k])]
        values[5] = r'\cellcolor{orange!8}' + values[5]
        rows.append(' & '.join([LABELS[k], '0.' + k[-2:], *values]) + r' \\')
    return rows


def _frozen_table_rows(paper, rel):
    try:
        raw = subprocess.run(['git', '-C', str(paper), 'show', f'{REFERENCE_COMMIT}:{rel}'],
                             check=True, capture_output=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise GateError(f'cannot read {rel} at {REFERENCE_COMMIT}: {err}') from err
    digest = _sha256(raw)
    if digest != TABLE_PINS[rel]:
        raise GateError(f'{rel}@{REFERENCE_COMMIT[:7]} has sha256 {digest}, pinned {TABLE_PINS[rel]}')
    lines = raw.decode().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('Method &'))
    rows = []
    for line in lines[start + 1:]:
        if line.startswith(r'\bottomrule'):
            break
        if line.strip() != r'\midrule':
            rows.append(line.rstrip())
    return rows, digest


def _compare_rows(mine, frozen):
    mismatches = [dict(row=i, expected=f, recomputed=m) for i, (m, f) in enumerate(zip(mine, frozen)) if m != f]
    if len(mine) != len(frozen):
        mismatches.append(dict(row='count', expected=len(frozen), recomputed=len(mine)))
    return dict(rows_compared=min(len(mine), len(frozen)), mismatches=mismatches)


def run_gate(paper, run_root):
    paper, run_root = Path(paper), Path(run_root)
    inputs = _Inputs(paper, run_root)
    frozen = inputs.read('paper', COHORT_FILE)
    inputs.pin_run_inputs(frozen['source_sha256'])
    ssim_doc = inputs.read('paper', SSIM_FILE)
    for rel in cohort.SPEC_FILES:
        inputs.read('paper', rel)
    manifest = {r['spec_id']: r for r in cohort.main_cohort_rows(paper)}
    if set(manifest) != set(frozen['retained_spec_ids']):
        raise GateError('cohort manifest differs from the retained IDs of the cohort file')
    methods = _assemble(inputs)
    indexes = {k: _index(k, methods[k], manifest) for k in METHODS}
    summaries = {k: _summary(indexes[k]) for k in METHODS}
    ids = sorted(manifest)
    pairs = _pairs(indexes, ids)
    verdicts = _verdict_check(indexes, manifest)

    summary_mismatches = [m for k in METHODS for m in _compare_summary(k, summaries[k], frozen['summary'][k])]
    paired_mismatches = []
    for p in frozen['paired']:
        mine = pairs.get((p['arm'], p['baseline']))
        if mine is None:
            paired_mismatches.append(f"no recomputed pair for {p['arm']}/{p['baseline']}")
            continue
        if mine['n'] != p['n'] or not _close(mine['psnr_baseline'], p['psnr_baseline']) or not _close(
                mine['psnr_ours'], p['psnr_ours']):
            paired_mismatches.append(f"{p['arm']}/{p['baseline']}: {mine} != {p}")
    ssim_set_mismatches, ssim = [], {}
    own = {o['method']: o for o in ssim_doc['own']}
    for k in METHODS:
        if own[k]['accepted_spec_ids_sha256'] != summaries[k]['accepted_spec_ids_sha256'] or own[k]['n'] != summaries[k]['accepted']:
            ssim_set_mismatches.append(f'{k}: SSIM accepted set differs from the recomputed accepted set')
        ssim[k] = own[k]['ssim']
    for p in ssim_doc['paired']:
        mine = pairs.get((p['arm'], p['baseline']))
        if mine is None or mine['joint_spec_ids_sha256'] != p['joint_spec_ids_sha256'] or mine['n'] != p['n']:
            ssim_set_mismatches.append(f"{p['arm']}/{p['baseline']}: SSIM joint set differs")
    if ssim_doc.get('retained_spec_ids_sha256') != _ids_hash(ids):
        ssim_set_mismatches.append('SSIM file was computed for another request set')

    main_frozen, main_digest = _frozen_table_rows(paper, 'tab/main_results.tex')
    margin_frozen, margin_digest = _frozen_table_rows(paper, 'tab/screening_margin.tex')
    main_table = _compare_rows(_main_rows(summaries, ssim, pairs), main_frozen)
    margin_table = _compare_rows(_margin_rows(summaries, ssim), margin_frozen)

    checks = dict(main_table=main_table['mismatches'], screening_margin=margin_table['mismatches'],
                  summary=summary_mismatches, paired=paired_mismatches, ssim_sets=ssim_set_mismatches,
                  verdict_check=verdicts['mismatches'])
    failures = [name for name, items in checks.items() if items]
    return dict(
        gate='pr0_main_table_reproduction_v1', passed=not failures, failures=failures,
        reference_commit=REFERENCE_COMMIT, paper=str(paper), run_root=str(run_root),
        inputs=dict(sorted(inputs.read_hashes.items())),
        tables={f'tab/main_results.tex@{REFERENCE_COMMIT[:7]}': main_digest,
                f'tab/screening_margin.tex@{REFERENCE_COMMIT[:7]}': margin_digest},
        cohort=dict(n=len(ids), retained_spec_ids_sha256=_ids_hash(ids), cohort_hash=cohort.cohort_hash(
            list(manifest.values()))),
        methods=summaries, pairs=[pairs[k] for k in sorted(pairs)],
        main_table=main_table, screening_margin=margin_table, summary_mismatches=summary_mismatches,
        paired_mismatches=paired_mismatches, ssim_set_mismatches=ssim_set_mismatches, verdict_check=verdicts,
        notes=['SSIM values are taken from data/ssim_main_20260924.json; only their accepted sets are recomputed.',
               'Inputs are hashed by relative path; the 7,321 cohort file is the sole source of request IDs.',
               'Decisions are recounted from stored statuses and stored verdicts are checked where present; '
               'no decision is re-judged from live cells (see experiments/protocol.md, section 9).'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--paper', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists():
        raise GateError(f'{args.output} already exists; the gate writes only into a fresh directory')
    report = run_gate(args.paper, args.run_root)
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(passed=report['passed'], failures=report['failures'],
                          main_rows=report['main_table']['rows_compared'],
                          margin_rows=report['screening_margin']['rows_compared'])))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
