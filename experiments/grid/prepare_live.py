"""Live validation of the grid baseline C2 on the 7,321 cohort, with the TAILOR-F live walk.

Proposals come from the exhaustive offline C2 rows (run_budgeted_search.py, budget inf) for the margin stages
initial, 04, 02 and 00: the top-3 distinct structures at each stage, walked in rank order on the same 100
selection images with the same gate, then the next margin stage. The request-local correction is a model
refinement of the SMT surrogate; C2 is a lookup baseline, so its correction stages are empty (disclosed).
Measurement, cell cache, bridging approval and GPU routing are those of experiments/main_controls (separate run root,
job prefix exp_c2).

Usage:
  python experiments/main_controls/relocate.py RUNL
  TAILOR_MAIN_CONTROLS_RUN=RUNL python experiments/grid/prepare_live.py OFFLINE_DIR MAIN_CONTROLS_RUN
  TAILOR_MAIN_CONTROLS_RUN=RUNL TAILOR_JOB_PREFIX=exp_c2 sbatch ... RUNL/code/controller.sbatch
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
sys.path.insert(0, str(REPO / 'experiments/main_controls'))
import relocate as RL  # noqa: E402
from common import ARM, CP, RUN, STAGES, atomic, output, read, sha  # noqa: E402
from gpu import preflight  # noqa: E402
from prepare import CODE_FILES  # noqa: E402

VARIANT = 'GridC2'
POLICY = 'exp_c2_live_v1'
KEEP = ('order', 'fe_on', 'fe', 's', 'ms', 'live_threshold', 'threshold_count', 'max_tests', 'psnr_db', 'label',
        'exact_onknot_response', 'exact_onknot_distortion', 'flags')


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def offline_rows(offline_dir):
    """(spec_id, margin stage) -> exhaustive offline row."""
    rows = {}
    for p in sorted(Path(offline_dir).glob('*.jsonl')):
        for line in open(p):
            r = json.loads(line)
            if r.get('budget_s') is not None:
                continue
            key = (r['spec_id'], r['margin_stage'])
            if key in rows:
                raise SystemExit(f'duplicate exhaustive row {key}')
            rows[key] = r
    return rows


def candidate(c, rank):
    out = {k: c[k] for k in KEEP if k in c}
    fe = c.get('fe')
    if isinstance(fe, dict):
        out['fe'] = fe
    out.update(rank=rank, certified=True, surrogate_protocol=c.get('label', 'grid_lookup_surrogate'), source='c2_offline')
    return out


def prepare(offline_dir, main_run):
    if (RUN / 'manifest.json').exists():
        raise SystemExit('manifest exists; a prepared run is never re-prepared in place')
    if git('status', '--porcelain', '--', 'experiments/main_controls', 'experiments/grid'):
        raise SystemExit('commit experiments/main_controls and experiments/grid first')
    relocation = RL.verify(RUN)
    main_run = Path(main_run)
    (RUN / 'inputs').mkdir(exist_ok=True)
    shutil.copy2(main_run / 'inputs/requests.json', RUN / 'inputs/requests.json')
    for name in ('bridging/approval.json', 'bridging/report.json'):
        (RUN / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(main_run / name, RUN / name)
    shutil.copy2(main_run / 'inputs/light_approval.json', RUN / 'inputs/light_approval.json')
    requests = read(RUN / 'inputs/requests.json')
    rows = offline_rows(offline_dir)
    targets, counts = [], collections.Counter()
    for r in requests:
        t = dict(variant=VARIANT, arm=ARM, cls=r['cls'], i=r['i'], spec_id=r['spec_id'], request=r['request'])
        targets.append(t)
        for stage in STAGES:
            row = rows.get((r['spec_id'], stage))
            if row is None:
                raise SystemExit(f"no exhaustive offline row for {r['spec_id']} at {stage}")
            cs = [candidate(c, k) for k, c in enumerate(row['candidates'], 1)]
            for c in cs:
                preflight(dict(candidate=c, attacks=r['request']['attacks']))
            atomic(output(dict(t, stage=stage)), dict(task=dict(t, stage=stage), candidates=cs, outcome=row['status'],
                                                       source='c2_offline', rate_margin=row['rate_margin'],
                                                       mean_margin=row['mean_margin'], complete=True))
            atomic(output(dict(t, stage=stage + '_patch')), dict(task=dict(t, stage=stage + '_patch'), candidates=[],
                                                                  outcome='not_applicable', complete=True,
                                                                  note='C2 has no request-local model correction'))
            counts[(stage, row['status'])] += 1
    code = RUN / 'code'
    code.mkdir(parents=True, exist_ok=False)
    for name in CODE_FILES:
        shutil.copy2(REPO / 'experiments/main_controls' / name, code / name)
    manifest = dict(policy=POLICY, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    repository=dict(commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD')),
                    variant=VARIANT, arm=ARM, stages=list(STAGES), targets=targets, requests=len(targets),
                    offline=dict(dir=str(offline_dir), stage_status={f'{s}|{st}': n for (s, st), n in sorted(counts.items())}),
                    correction='none (lookup baseline)', protocol=CP.PROTOCOL, relocation_sha256=sha(RUN / 'relocation.json'),
                    bridging_from=str(main_run))
    atomic(RUN / 'manifest.json', manifest)
    atomic(RUN / 'capacity.json', dict(cpu=0, gpu=4, gpu_light=4))
    paths = [RUN / 'manifest.json', RUN / 'relocation.json', RUN / 'inputs/requests.json',
             *(code / n for n in CODE_FILES), *(RUN / 'root' / rec['path'] for rec in relocation['files'])]
    atomic(RUN / 'sources.json', dict(hashes={str(p): sha(p) for p in paths}))
    print('PREPARED', len(targets), 'targets', dict(counts), flush=True)


if __name__ == '__main__':
    prepare(sys.argv[1], sys.argv[2])
