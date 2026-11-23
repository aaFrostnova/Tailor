"""CPU re-judge of the frozen TAILOR-F (SMT06) records of the 7,321 cohort from the frozen live cells.

Entry condition for the new main-table arms (experiments/protocol.md, section 9): every accepted TAILOR-F record
is judged again with the frozen judge on its selected configuration, read from the frozen cell cache and
the frozen exported groups; it must pass with the stored live PSNR. Records that did not pass are checked
for a complete chain: a live-exhausted record must have walked candidates and none of them may pass or be
pending when judged again; a model-UNSAT record must have no candidate at any margin stage that passes.

Usage: TAILOR_MAIN_CONTROLS_RUN=$RUN python rejudge.py OUTPUT_JSON [LIMIT]
(LIMIT, for a smoke run only: the first LIMIT records of each stored status.)
"""
import os
import collections
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CP, FROZEN, cfgkey, read  # noqa: E402

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'experiments'))
from tailor_exp import gate  # noqa: E402

TAGS06 = ('unified_ba_out06n2000_20260914', 'unified_ba_out04n2000_20260914',
          'unified_ba_fallback02_from06n2000_20260914', 'unified_ba_fallback00_from06n2000_20260914',
          'live_failure_fallback_20260921', 'editing_request_extension_20260921')
CLASSES = ('C1', 'C2', 'C3', 'C4', 'C5')
OLD_ROOT = os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/rigor_editing_only_20260911/unified_ba_20260914'


def frozen_path(path):
    return Path(str(path).replace(OLD_ROOT, str(FROZEN)))


_files = {}


def cached(path):
    path = frozen_path(path)
    if path not in _files:
        _files[path] = read(path)
    return _files[path]


def request_candidates(rec, fb):
    """Every candidate any margin stage proposed for this request in the TAILOR-F chain."""
    out = []
    if fb is not None:
        sources = [fb.get('initial_source')] + [a.get('source') for a in fb.get('attempts') or []]
        for src in filter(None, sources):
            folder = frozen_path(src).parent
            rows = cached(src)['rows'] if frozen_path(src).name == 'walk.json' else []
            out += [c for r in rows if r['i'] == fb['i'] for c in r['candidates']]
            for patch in sorted(folder.glob('patched_s*.json')):
                c = cached(patch)['candidates'].get(str(fb['i']))
                if c:
                    out.append(c)
        folder = FROZEN / 'live_failure_fallback_20260921/proposals/main/Full/06' / rec['spec_id']
    else:
        folder = FROZEN / 'editing_request_extension_20260921/proposals/editing/Full/06' / rec['spec_id']
    for p in sorted(folder.glob('*.json')):
        out += read(p).get('candidates') or []
    return out


class FrozenCells:
    """The frozen judge's view: shared cells first, then the exported groups of the TAILOR-F tags."""

    def __init__(self):
        self.groups = {}
        for tag in TAGS06:
            for cls in CLASSES:
                for p in glob.glob(str(FROZEN / f'certify_full_topk_{cls}_{tag}' / cls / 'g*.json')):
                    self.groups.setdefault(Path(p).name, []).append(p)
        self.cache = {}

    def load(self, c, attack):
        key = cfgkey(c)
        ck = (key, attack)
        if ck in self.cache:
            return self.cache[ck]
        n = 30 if attack == 'unmarker' else 100
        detector_cfg = dict(order=c['order'], fe_on=c['fe_on'])
        cfg_id = CP.exact_cfg_id(key)
        rec = None
        p = FROZEN / 'live_topk/cells' / f'{cfg_id}_{attack}_pool300_n100.json'
        if p.exists():
            r = read(p)
            if CP.valid_cell(r.get('cell'), c['order'], n, cfg=detector_cfg):
                rec = r
        if rec is None:
            for gp in self.groups.get(f'g{int(cfg_id[:12], 16):04d}.json', []):
                g = read(gp)
                if CP.exact_cfg_key(g['order'], g['fe'], [g['s'][f] for f in g['order']]) != key:
                    continue
                cell = g.get('attacks', {}).get(attack)
                if CP.valid_cell(cell, c['order'], n, cfg=detector_cfg):
                    rec = dict(cell=cell, psnr_db=g.get('psnr_db'), source=gp, n=cell['n'])
                    break
        self.cache[ck] = rec
        return rec


def candidate_index():
    """cfg_id -> candidate over every candidate list the TAILOR-F record chain can point to."""
    index = {}

    def add(c):
        if isinstance(c, dict) and c.get('order') and isinstance(c.get('s'), dict):
            index.setdefault(CP.exact_cfg_id(cfgkey(c)), c)
    for tag in TAGS06[:4]:
        for cls in CLASSES:
            folder = FROZEN / 'live_topk' / f'{cls}_{tag}'
            for p in sorted(folder.glob('requests_s*.json')):
                for r in read(p)['rows']:
                    for c in r['candidates']:
                        add(c)
            for p in sorted(folder.glob('patched_s*.json')):
                for c in read(p)['candidates'].values():
                    add(c)
    for p in glob.glob(str(FROZEN / 'live_failure_fallback_20260921/proposals/**/*.json'), recursive=True):
        for c in read(p).get('candidates') or []:
            add(c)
    for p in glob.glob(str(FROZEN / 'editing_request_extension_20260921/proposals/**/*.json'), recursive=True):
        for c in read(p).get('candidates') or []:
            add(c)
    return index


def main(out_path, limit=None):
    paper = REPO
    inputs = gate._Inputs(paper, FROZEN)
    frozen = inputs.read('paper', gate.COHORT_FILE)
    inputs.pin_run_inputs(frozen['source_sha256'])
    manifest = {r['spec_id']: r for r in gate.cohort.main_cohort_rows(paper)}
    methods = gate._assemble(inputs)
    index = gate._index('SMT06', methods['SMT06'], manifest)
    requests = {r['spec_id']: r['request'] for r in read(
        FROZEN / 'enumeration_geo_20260923/inputs/requests.json') if r['spec_id'] in manifest}
    fallback_rows = {}
    for cls in CLASSES:
        p = FROZEN / 'fallback/requests' / f'{cls}_06.json'
        if p.exists():
            for r in read(p)['rows']:
                fallback_rows[(cls, r['i'])] = r
    cands = candidate_index()
    cells = FrozenCells()
    results, problems = collections.Counter(), []
    psnr_dev = []
    seen = collections.Counter()
    for sid, rec in sorted(index.items()):
        status = rec['_status']
        seen[status] += 1
        if limit is not None and seen[status] > limit:
            continue
        fb = fallback_rows.get((rec['cls'], rec.get('i'))) if rec.get('cohort') == 'main' else None
        request = dict(requests[sid])
        if fb is not None:
            if sorted(fb['attacks']) != sorted(request['attacks']) or fb['fpr'] != request['fpr']:
                problems.append(dict(spec_id=sid, kind='request_mismatch'))
                continue
            request['attacks'] = list(fb['attacks'])       # the order the frozen judge used
        if status == 'accepted':
            c = rec.get('candidate') or cands.get(rec.get('cfg_id'))
            if c is None:
                problems.append(dict(spec_id=sid, kind='accepted_candidate_not_found', cfg_id=rec.get('cfg_id')))
                continue
            if rec.get('cfg_id') and CP.exact_cfg_id(cfgkey(c)) != rec['cfg_id']:
                problems.append(dict(spec_id=sid, kind='cfg_id_mismatch'))
                continue
            v = CP.judge(request, c, {a: cells.load(c, a) for a in request['attacks']}, .9)
            stored = rec.get('psnr_live')
            if stored is None and isinstance(rec.get('verdict'), dict):
                stored = rec['verdict'].get('psnr_live')
            if v['pending']:
                problems.append(dict(spec_id=sid, kind='accepted_cells_missing', pending=v['pending']))
            elif not v['pass']:
                problems.append(dict(spec_id=sid, kind='accepted_fails_rejudge',
                                     failing=[a for a, s in v['attacks'].items() if not s['ok']],
                                     psnr_ok=v['psnr_ok'], latency_ok=v['latency_ok']))
            else:
                psnr_dev.append(abs(v['psnr_live'] - stored) if stored is not None else None)
                results['accepted_rejudged_pass'] += 1
        else:
            results[f'not_accepted_{status}'] += 1
            proposed = request_candidates(rec, fb)
            results[f'not_accepted_{status}_candidates_rejudged'] += len(proposed)
            if status == 'live_exhausted' and not proposed:
                problems.append(dict(spec_id=sid, kind='live_exhausted_without_candidates'))
            for c in proposed:
                v = CP.judge(request, c, {a: cells.load(c, a) for a in request['attacks']}, .9)
                if v['pass'] or v['pending']:
                    problems.append(dict(spec_id=sid, kind='non_accepted_candidate_passes_or_pending',
                                         status=status, cfg_id=CP.exact_cfg_id(cfgkey(c)), pending=v['pending']))
    dev = [d for d in psnr_dev if d is not None]
    report = dict(method='SMT06', cohort=len(manifest), records=len(index), limit=limit,
                  status_counts=dict(collections.Counter(r['_status'] for r in index.values())),
                  results=dict(results), problems=problems, problem_counts=dict(collections.Counter(
                      p['kind'] for p in problems)),
                  psnr_abs_deviation=dict(n=len(dev), max=max(dev) if dev else None,
                                          over_1e_6=sum(d > 1e-6 for d in dev), unknown=psnr_dev.count(None)),
                  candidates_indexed=len(cands), passed=not problems)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'problems'}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
