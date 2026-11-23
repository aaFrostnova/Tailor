"""Cross-GPU check for the cross-environment attacks: do L40S measurements agree with the November A100 cells?

The 2026-11-09 L40S check (crossgpu.py) covered the in-process attacks only. The user approved UnMarker on
L40S on 2026-11-10 subject to this check; CtrlRegen+ (s03, s05) and editing are measured in the same run so the
user can decide on them with evidence. An attack becomes usable by L40S workers only through `approve`, which
names the attacks explicitly.

Configurations (fixed before any L40S measurement): the bridge_v1 configurations of SOURCE_RUN that carry the
attacks under test (their November A100 bridging cells are the reference; their September frozen cells are
reported too), plus the first three configurations by configuration id whose UnMarker cell SOURCE_RUN measured on
an A100 (UnMarker only, for more power). Same selection images, keys, classes and code as the reference.

Decision rule (registered in the manifest before measuring; per attack, pooled over configuration x threshold
buckets, L40S minus November A100):
  ctrlregen_s03, ctrlregen_s05, editing_ip2p_s20_v1: bridge_v1 stochastic tolerances, which the A100 re-run
    met: |shift_rate| <= 0.03, |shift_mean| <= 0.015, max |d_rate| <= 0.10, max |d_mean| <= 0.04;
  unmarker: |shift_rate| <= 0.03, |shift_mean| <= 0.015, max |d_mean| <= 0.04, and max |d_rate| <= 0.167, the
    largest difference between the two A100 runs of the same configurations (bridge_v1's 0.10 measures
    UnMarker's own randomness, which an A100 re-run already exceeds).

Usage:
  python experiments/main_controls/relocate.py RUNU
  TAILOR_MAIN_CONTROLS_RUN=RUNU python experiments/main_controls/crossgpu_xenv.py prepare SOURCE_RUN
  TAILOR_MAIN_CONTROLS_RUN=RUNU python experiments/main_controls/crossgpu_xenv.py submit [N]
  TAILOR_MAIN_CONTROLS_RUN=RUNU python experiments/main_controls/crossgpu_xenv.py decide
  TAILOR_MAIN_CONTROLS_RUN=RUNU python experiments/main_controls/crossgpu_xenv.py approve ATTACK[,ATTACK] TARGET_RUN...
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
sys.path.insert(0, str(HERE))
import relocate as RL  # noqa: E402
from bridge import BRIDGE_CLS, TOL_STOCHASTIC, compare_cells  # noqa: E402
from common import CP, FROZEN_CELLS, RUN, atomic, cell_name, cfgkey, read, sha, task_id  # noqa: E402
from prepare import CODE_FILES  # noqa: E402

POLICY = 'exp_crossgpu_l40s_xenv_v1'
UNDER_TEST = ('unmarker', 'ctrlregen_s03', 'ctrlregen_s05', 'editing_ip2p_s20_v1')
EXTRA_UNMARKER = 3
UNMARKER_MAX_RATE = 0.167          # A100 re-run maximum, SOURCE_RUN/bridging/report.json (unmarker)
RULES = {a: dict(TOL_STOCHASTIC) for a in UNDER_TEST}
RULES['unmarker'] = dict(TOL_STOCHASTIC, max_rate=UNMARKER_MAX_RATE)
JOB = 'exp_xg_gpul'


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def source_cell(source_run, key, attack):
    return Path(source_run) / 'root/live_topk/cells' / cell_name(key, attack)


def prepare(source_run):
    source_run = Path(source_run)
    if (RUN / 'manifest.json').exists():
        raise SystemExit('manifest exists')
    if git('status', '--porcelain', '--', 'experiments/main_controls'):
        raise SystemExit('commit experiments/main_controls first')
    relocation = RL.verify(RUN)
    report = read(source_run / 'bridging/report.json')
    observed = report['attacks']['unmarker']['max_abs_d_rate']
    if abs(observed - UNMARKER_MAX_RATE) > 5e-4:
        raise SystemExit(f'A100 re-run maximum for UnMarker is {observed}, the rule says {UNMARKER_MAX_RATE}')
    from gpu import preflight
    items = []
    for it in read(source_run / 'bridging/selection.json')['items']:
        attacks = [a for a in it['attacks'] if a in UNDER_TEST]
        if attacks:
            items.append(dict(source='bridging', cls=BRIDGE_CLS, candidate=it['candidate'], attacks=attacks))
    bridged = {CP.exact_cfg_id(cfgkey(it['candidate'])) for it in items}
    extra = []
    for p in sorted((source_run / 'gpu_tasks').glob('*/complete.json')):
        rec = read(p)
        if 'A100' not in rec.get('device', '') or 'unmarker' not in rec.get('measured', []):
            continue
        t = rec['task']
        digest = CP.exact_cfg_id(cfgkey(t['candidate']))
        if digest not in bridged and digest not in {d for d, _ in extra}:
            extra.append((digest, dict(source='a100_run_cell', cls=t['cls'], candidate=t['candidate'],
                                       attacks=['unmarker'])))
    items += [it for _, it in sorted(extra, key=lambda x: x[0])[:EXTRA_UNMARKER]]
    references, paths = [], []
    for it in items:
        c = it['candidate']
        key = cfgkey(c)
        for a in it['attacks']:
            p = source_cell(source_run, key, a)
            if not p.exists():
                raise SystemExit(f'no November A100 reference cell {p.name}')
            f = FROZEN_CELLS / cell_name(key, a)
            references.append(dict(cfg_id=CP.exact_cfg_id(key), attack=a, a100_nov=str(p), a100_nov_sha256=sha(p),
                                   a100_sep=str(f) if f.exists() else None,
                                   a100_sep_sha256=sha(f) if f.exists() else None))
        ref = read(source_cell(source_run, key, it['attacks'][0]))
        cand = dict(c, rank=1, ms=c.get('ms', 0.0), psnr_db=c.get('psnr_db', ref.get('psnr_db') or 0.0))
        task = dict(candidate=cand, attacks=it['attacks'], cls=it['cls'], run_cells_only=True)
        preflight(task)
        p = RUN / 'tasks' / ('gpu_' + task_id(task) + '.json')
        p.parent.mkdir(exist_ok=True)
        atomic(p, task)
        paths.append(str(p))
    (RUN / 'inputs').mkdir(exist_ok=True)
    atomic(RUN / 'inputs/light_check.json', dict(attacks_under_test=list(UNDER_TEST), policy=POLICY,
                                                 purpose='cross-GPU check run only; never copied to a result run'))
    atomic(RUN / 'queue_gpu.json', paths)
    code = RUN / 'code'
    code.mkdir(parents=True, exist_ok=False)
    for name in CODE_FILES:
        shutil.copy2(HERE / name, code / name)
    atomic(RUN / 'manifest.json', dict(
        policy=POLICY, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        repository=dict(commit=git('rev-parse', 'HEAD')), device='L40S (gpu, --gres=gpu:l40s:1)',
        source_run=str(source_run), reference='November A100 cells of SOURCE_RUN (September frozen cells reported)',
        rules=RULES, attacks_under_test=list(UNDER_TEST), tasks=paths, references=references,
        selection_rule=('bridge_v1 configurations carrying the attacks under test, plus the first '
                        f'{EXTRA_UNMARKER} configurations by id with an A100-measured UnMarker cell in SOURCE_RUN')))
    hashes = [RUN / 'manifest.json', RUN / 'relocation.json', *(code / n for n in CODE_FILES),
              *(RUN / 'root' / rec['path'] for rec in relocation['files'])]
    atomic(RUN / 'sources.json', dict(hashes={str(p): sha(p) for p in hashes}))
    print('PREPARED', len(paths), 'tasks', collections.Counter(a for it in items for a in it['attacks']), flush=True)


def submit(workers=6):
    paths = read(RUN / 'queue_gpu.json')
    n = min(int(workers), len(paths))
    (RUN / 'logs').mkdir(exist_ok=True)
    cmd = ['sbatch', '--parsable', '-p', 'gpu', '--gres=gpu:l40s:1', '-c', '6', '--mem=64G',
           '-t', '08:00:00', f'--array=0-{n - 1}%{n}', '-J', JOB, '-o', str(RUN / 'logs/gpu_light_%A_%a.log'),
           '--chdir=' + str(RUN), '--export=ALL', str(RUN / 'code/worker.sbatch'), 'gpu_light']
    job = subprocess.check_output(cmd, text=True, timeout=60).strip().split(';')[0]
    with (RUN / 'jobs.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind='gpu_light', job=job, workers=n)) + '\n')
    print('XENV_CHECK_SUBMITTED', job, n, flush=True)


def judge(attack, rows):
    if not rows:
        return dict(approved=False, reason='no complete comparison')
    d_rate = [r['d_rate'] for r in rows]
    d_mean = [r['d_mean'] for r in rows]
    stats = dict(pairs_buckets=len(rows), max_abs_d_rate=max(map(abs, d_rate)), max_abs_d_mean=max(map(abs, d_mean)),
                 shift_rate=sum(d_rate) / len(d_rate), shift_mean=sum(d_mean) / len(d_mean),
                 min_agreement=min(r['agreement'] for r in rows))
    t = RULES[attack]
    ok = (abs(stats['shift_rate']) <= t['shift_rate'] and abs(stats['shift_mean']) <= t['shift_mean']
          and stats['max_abs_d_rate'] <= t['max_rate'] and stats['max_abs_d_mean'] <= t['max_mean'])
    return dict(approved=bool(ok), tolerances=t, **stats)


def decide():
    from common import load_cell
    from unified_detector import threshold_count
    manifest = read(RUN / 'manifest.json')
    refs = {(r['cfg_id'], r['attack']): r for r in manifest['references']}
    vs_nov, vs_sep, pairs, missing, devices = (collections.defaultdict(list), collections.defaultdict(list), [], [],
                                               collections.Counter())
    for p in manifest['tasks']:
        task = read(p)
        c = task['candidate']
        key = cfgkey(c)
        cfg_id = CP.exact_cfg_id(key)
        done = RUN / 'gpu_tasks' / task_id(task) / 'complete.json'
        if not done.exists():
            missing.append([cfg_id[:12], 'task'])
            continue
        devices[read(done)['device']] += 1
        thresholds = sorted({str(threshold_count(dict(order=c['order'], fe_on=c['fe_on']), a))
                             for a in CP.REQUEST_ALPHAS}, key=int)
        for a in task['attacks']:
            ref = refs[(cfg_id, a)]
            if sha(ref['a100_nov']) != ref['a100_nov_sha256']:
                raise SystemExit(f'reference cell changed since prepare: {ref["a100_nov"]}')
            new = load_cell(c, a, frozen=False)
            if new is None:
                missing.append([cfg_id[:12], a])
                continue
            rows = compare_cells(read(ref['a100_nov'])['cell'], new['cell'], thresholds)
            vs_nov[a] += rows
            sep_rows = compare_cells(read(ref['a100_sep'])['cell'], new['cell'], thresholds) if ref['a100_sep'] else None
            if sep_rows:
                vs_sep[a] += sep_rows
            pairs.append(dict(cfg_id=cfg_id, attack=a, psnr_new=new.get('psnr_db'), vs_a100_nov=rows,
                              vs_a100_sep=sep_rows))
    if missing:
        raise SystemExit(f'check incomplete: {missing}')
    if set(devices) != {'NVIDIA L40S'}:
        raise SystemExit(f'check tasks ran on {dict(devices)}, expected L40S only')
    verdicts = {a: judge(a, vs_nov[a]) for a in UNDER_TEST}
    informational = {a: judge(a, vs_sep[a]) for a in UNDER_TEST if vs_sep[a]}
    atomic(RUN / 'report.json', dict(policy=POLICY, manifest_sha256=sha(RUN / 'manifest.json'), devices=dict(devices),
                                     reference='November A100', attacks=verdicts,
                                     versus_september_frozen=informational, pairs=pairs))
    for a, v in verdicts.items():
        print(a, 'APPROVED' if v['approved'] else 'NOT APPROVED',
              {k: round(x, 4) for k, x in v.items() if isinstance(x, float)}, flush=True)


def approve(attacks, targets):
    """Add attacks this check approved to the light approval of result runs (explicit list, per user decision)."""
    report = read(RUN / 'report.json')
    rsha = sha(RUN / 'report.json')
    for a in attacks:
        if not report['attacks'].get(a, {}).get('approved'):
            raise SystemExit(f'{a} did not pass this check')
    for target in map(Path, targets):
        p = target / 'inputs/light_approval.json'
        rec = read(p)
        if not rec.get('complete'):
            raise SystemExit(f'{p} is incomplete')
        for a in attacks:
            rec['attacks'][a] = dict(approved=True, kind='stochastic', check=POLICY, check_run=str(RUN),
                                     check_report_sha256=rsha)
        rec.setdefault('amendments', []).append(dict(
            at=datetime.datetime.now(datetime.timezone.utc).isoformat(), attacks=list(attacks), check_run=str(RUN),
            check_report_sha256=rsha, note='approved for L40S workers after the cross-environment check'))
        atomic(p, rec)
        print('APPROVED_FOR_L40S', target.name, attacks, flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prepare':
        prepare(sys.argv[2])
    elif cmd == 'submit':
        submit(sys.argv[2] if len(sys.argv) > 2 else 6)
    elif cmd == 'decide':
        decide()
    elif cmd == 'approve':
        approve(sys.argv[2].split(','), sys.argv[3:])
    else:
        raise SystemExit(__doc__)
