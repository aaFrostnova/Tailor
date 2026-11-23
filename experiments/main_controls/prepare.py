"""Freeze the main-control manifest: 7,321 cohort requests x {SingleOptimized, AlwaysAllThree} at arm 06.

Run once, from the repository, after relocate.py and with experiments/main_controls committed:
  TAILOR_MAIN_CONTROLS_RUN=$RUN python experiments/main_controls/prepare.py
It snapshots the orchestration code into RUN/code (the jobs run that copy), writes RUN/inputs/requests.json
and RUN/manifest.json, and binds every executed file and input by sha256 in RUN/sources.json.
"""
import collections
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'experiments'))
from tailor_exp import cohort  # noqa: E402
import relocate as RL  # noqa: E402
from common import (ARM, CP, FROZEN, NO_PASSENGERS, POLICY, ROOT, RUN, STAGES, TAG, VARIANTS, atomic,  # noqa: E402
                    margins, read, sha)

REQUEST_SOURCE = FROZEN / 'enumeration_geo_20260923/inputs/requests.json'
CODE_FILES = ('common.py', 'propose.py', 'mc_controller.py', 'worker.py', 'gpu.py', 'report.py', 'bridge.py',
              'worker.sbatch', 'controller.sbatch')


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def cohort_requests():
    """Frozen request rows of the 7,321 cohort, checked field by field against the cohort manifest."""
    rows = cohort.main_cohort_rows(REPO)
    by_id = {r['spec_id']: r for r in rows}
    source = read(REQUEST_SOURCE)
    seen, out = set(), []
    for row in source:
        sid = row['spec_id']
        if sid not in by_id:
            continue
        if sid in seen:
            raise ValueError(f'duplicate request row for {sid}')
        seen.add(sid)
        m, q = by_id[sid], row['request']
        if (row['cls'] != m['cls'] or sorted(q['attacks']) != m['attacks'] or q['fpr'] != m['alpha']
                or q['min_psnr'] != m['psnr_floor_db'] or q['max_ms'] != m['latency_ceiling_ms']):
            raise ValueError(f'request row {sid} disagrees with the cohort manifest')
        out.append(dict(spec_id=sid, cls=row['cls'], source_i=row['i'], source_cohort=row['cohort'],
                        request=q, i=len(out)))
    if seen != set(by_id):
        raise ValueError(f'{len(set(by_id) - seen)} cohort requests have no frozen request row')
    counts = dict(collections.Counter(r['cls'] for r in out))
    expected = {'C' + s[1:]: n for s, n in cohort.EXPECTED_MAIN['scenarios'].items()}
    if counts != expected:
        raise ValueError(f'scenario counts {counts} != {expected}')
    return rows, out


def prepare():
    if (RUN / 'manifest.json').exists():
        raise SystemExit('manifest exists; a prepared run is never re-prepared in place')
    if git('status', '--porcelain', '--', 'experiments/main_controls'):
        raise SystemExit('commit experiments/main_controls before preparing a run')
    relocation = RL.verify(RUN)
    cohort_rows, rows = cohort_requests()
    atomic(RUN / 'inputs/requests.json', rows)
    code = RUN / 'code'
    code.mkdir(parents=True, exist_ok=False)
    for name in CODE_FILES:
        shutil.copy2(HERE / name, code / name)
    targets = [dict(variant=v, arm=ARM, cls=r['cls'], i=r['i'], spec_id=r['spec_id'], request=r['request'])
               for r in rows for v in VARIANTS]
    manifest = dict(
        policy=POLICY, tag=TAG, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        repository=dict(commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD')),
        cohort=dict(n=len(rows), cohort_hash=cohort.cohort_hash(cohort_rows),
                    scenarios=dict(collections.Counter(r['cls'] for r in rows)),
                    request_source=str(REQUEST_SOURCE), request_source_sha256=sha(REQUEST_SOURCE)),
        variants=list(VARIANTS), arm=ARM, stages=list(STAGES),
        margins={s: list(margins(ARM, s)) for s in STAGES}, no_passengers=NO_PASSENGERS,
        restrictions=dict(
            SingleOptimized='at most one selected fragment; strength, compatible stages, threshold, candidate '
                            'search, correction and backoff as in TAILOR-F',
            AlwaysAllThree='VINE, TrustMark and VideoSeal all selected; strength, order and compatible stages '
                           'optimized; only the no-passenger rule (every selected fragment clears one requested '
                           'attack) is relaxed; all attack, FPR, fidelity and latency constraints kept'),
        live_gate=dict(rate_floor=.9, mean='>= tau(T(c), alpha)', psnr='measured embed PSNR >= floor',
                       latency='modeled', images='pool offset 300, n=100 (UnMarker first 30)'),
        cache=dict(run_cells=str(ROOT / 'live_topk/cells'),
                   frozen_reuse='only attacks approved in bridging/approval.json'),
        targets=targets, requests=len(rows), evaluations=len(targets),
        protocol=CP.PROTOCOL, relocation_sha256=sha(RUN / 'relocation.json'))
    atomic(RUN / 'manifest.json', manifest)
    paths = [RUN / 'manifest.json', RUN / 'relocation.json', RUN / 'inputs/requests.json',
             *(code / name for name in CODE_FILES),
             *(ROOT / rec['path'] for rec in relocation['files'])]
    atomic(RUN / 'sources.json', dict(hashes={str(p): sha(p) for p in paths}))
    print('PREPARED', len(rows), 'requests', len(targets), 'evaluations', manifest['repository'], flush=True)


def rebind():
    """Refresh RUN/code from the committed repository and re-bind the hashes; requests and targets stay.

    Only for operational fixes (scheduling, partitions); every rebind is appended to RUN/rebinds.jsonl with
    the old and new commit and the changed files."""
    if git('status', '--porcelain', '--', 'experiments/main_controls'):
        raise SystemExit('commit experiments/main_controls before rebinding')
    RL.verify(RUN)
    manifest = read(RUN / 'manifest.json')
    old = read(RUN / 'sources.json')['hashes']
    code = RUN / 'code'
    for name in CODE_FILES:
        shutil.copy2(HERE / name, code / name)
    paths = [Path(p) for p in old]
    new = {str(p): sha(p) for p in paths}
    changed = sorted(p for p in new if new[p] != old[p])
    if any(not p.startswith(str(code) + '/') for p in changed):
        raise SystemExit(f'rebind may change only RUN/code files, not {changed}')
    atomic(RUN / 'sources.json', dict(hashes=new))
    entry = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), prepared_commit=manifest['repository']['commit'],
                 commit=git('rev-parse', 'HEAD'), changed=changed)
    with (RUN / 'rebinds.jsonl').open('a') as f:
        f.write(json.dumps(entry) + '\n')
    print('REBOUND', json.dumps(entry), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['rebind']:
        rebind()
    else:
        prepare()
