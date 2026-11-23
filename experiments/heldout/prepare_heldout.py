"""Held-out transfer (PR5): measure every TAILOR-F deployed configuration once on frozen new images.

Each accepted TAILOR-F (SMT06) record of the 7,321 cohort keeps its frozen configuration (fragments,
order, strengths, stage, threshold rule, key). Configurations shared by several requests are measured
once on the union of their requests' attacks. The held-out images (inputs/image_set.json, frozen by
freeze_images.py) never entered selection; no held-out outcome is fed back to any search.

Steps (RUNH = the held-out run directory):
  python experiments/main_controls/relocate.py RUNH
  cp <frozen image set> RUNH/inputs/image_set.json
  TAILOR_MAIN_CONTROLS_RUN=RUNH python experiments/heldout/prepare_heldout.py prepare
  TAILOR_MAIN_CONTROLS_RUN=RUNH python experiments/heldout/prepare_heldout.py submit WORKERS [--light] [--after=JOBID]
"""
import os
import collections
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'experiments/main_controls'), str(REPO / 'experiments')]
import relocate as RL  # noqa: E402
from common import CP, RUN, atomic, cfgkey, read, sha, task_id  # noqa: E402
from gpu import preflight  # noqa: E402
import rejudge  # noqa: E402
from tailor_exp import gate  # noqa: E402

POLICY = 'exp_heldout_v1'
JOB = 'exp_ho_gpu'
CODE_FILES = ('common.py', 'propose.py', 'mc_controller.py', 'worker.py', 'gpu.py', 'report.py', 'bridge.py',
              'worker.sbatch', 'controller.sbatch')


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def accepted_assignments():
    """(spec_id, cls, request, candidate, calibration PSNR) for every accepted TAILOR-F record."""
    inputs = gate._Inputs(REPO, rejudge.FROZEN)
    frozen = inputs.read('paper', gate.COHORT_FILE)
    inputs.pin_run_inputs(frozen['source_sha256'])
    manifest = {r['spec_id']: r for r in gate.cohort.main_cohort_rows(REPO)}
    index = gate._index('SMT06', gate._assemble(inputs)['SMT06'], manifest)
    requests = {r['spec_id']: r for r in read(RUN / 'inputs/requests.json')}
    cands = rejudge.candidate_index()
    rows = []
    for sid, rec in sorted(index.items()):
        if rec['_status'] != 'accepted':
            continue
        c = rec.get('candidate') or cands.get(rec.get('cfg_id'))
        if c is None:
            raise SystemExit(f'accepted record {sid} has no recoverable candidate')
        cfg_id = CP.exact_cfg_id(cfgkey(c))
        if rec.get('cfg_id') and rec['cfg_id'] != cfg_id:
            raise SystemExit(f'accepted record {sid}: candidate does not match its cfg_id')
        psnr = rec.get('psnr_live')
        if psnr is None and isinstance(rec.get('verdict'), dict):
            psnr = rec['verdict'].get('psnr_live')
        rows.append(dict(spec_id=sid, cls=rec['cls'], request=requests[sid]['request'], cfg_id=cfg_id,
                         candidate={k: c[k] for k in ('order', 'fe_on', 'fe', 's', 'ms', 'psnr_db', 'live_threshold',
                                                      'threshold_count', 'max_tests') if k in c},
                         calibration_psnr=psnr))
    return manifest, rows


def prepare():
    if (RUN / 'manifest.json').exists():
        raise SystemExit('manifest exists; a prepared run is never re-prepared in place')
    if git('status', '--porcelain', '--', 'experiments/main_controls', 'experiments/heldout'):
        raise SystemExit('commit experiments/main_controls and experiments/heldout before preparing')
    relocation = RL.verify(RUN)
    image_set = read(RUN / 'inputs/image_set.json')
    if image_set.get('schema') != 'tailor_exp_image_set_v1' or image_set['n'] != 100:
        raise SystemExit('inputs/image_set.json is not a frozen 100-image set')
    # the main-control request rows are the cohort's frozen request inputs
    shutil.copy2(Path(os.environ.get('TAILOR_RUNS', 'runs') + '/'
                      'main_controls_20261109/inputs/requests.json'), RUN / 'inputs/requests.json')
    cohort_manifest, rows = accepted_assignments()
    groups = collections.OrderedDict()
    for r in rows:
        g = groups.setdefault(r['cfg_id'], dict(candidate=r['candidate'], attacks=set(), cls=r['cls'], specs=[]))
        g['attacks'].update(r['request']['attacks'])
        g['specs'].append(r['spec_id'])
    tasks = []
    for cfg_id, g in groups.items():
        task = dict(candidate=dict(g['candidate'], rank=1), attacks=sorted(g['attacks']), cls=g['cls'],
                    run_cells_only=True)
        preflight(task)
        p = RUN / 'tasks' / ('gpu_' + task_id(task) + '.json')
        atomic(p, task)
        tasks.append(str(p))
    atomic(RUN / 'queue_gpu.json', tasks)
    code = RUN / 'code'
    code.mkdir(parents=True, exist_ok=False)
    for name in CODE_FILES:
        shutil.copy2(REPO / 'experiments/main_controls' / name, code / name)
    manifest = dict(
        policy=POLICY, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        repository=dict(commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD')),
        method='TAILOR-F (SMT06, frozen deployed configurations)', image_role='heldout',
        image_set_sha256=image_set['images_sha256'], image_set_file_sha256=sha(RUN / 'inputs/image_set.json'),
        cohort=dict(n=len(cohort_manifest), accepted=len(rows),
                    scenarios=dict(collections.Counter(r['cls'] for r in rows))),
        configurations=len(groups), tasks=len(tasks),
        attacks_per_configuration=dict(collections.Counter(len(g['attacks']) for g in groups.values())),
        assignments=rows, no_feedback='held-out cells are read only by the held-out report; no search reads them',
        relocation_sha256=sha(RUN / 'relocation.json'))
    hook = RUN / 'identity_hook'
    if (hook / 'identity.json').exists():
        manifest['identity'] = dict(read(hook / 'identity.json'), hook=str(hook),
                                    rule='cover i is embedded and verified under img_{offset+i}')
    atomic(RUN / 'manifest.json', manifest)
    paths = [RUN / 'manifest.json', RUN / 'relocation.json', RUN / 'inputs/image_set.json',
             RUN / 'inputs/requests.json', *(code / n for n in CODE_FILES),
             *(RUN / 'root' / rec['path'] for rec in relocation['files'])]
    if (hook / 'identity.json').exists():
        paths += [hook / 'identity.json', hook / 'sitecustomize.py']
    atomic(RUN / 'sources.json', dict(hashes={str(p): sha(p) for p in paths}))
    print('PREPARED', len(rows), 'accepted requests', len(groups), 'configurations', flush=True)


def submit(workers, light=False, after=None):
    """A100 workers take every task; light (L40S) workers only tasks without cross-environment attacks.

    after: a job id; the new workers start once that job has ended (a relay past the 8 h walltime). Workers
    with nothing left to run exit at once, so a relay that finds the queue empty costs no GPU time.
    """
    n = int(workers)
    (RUN / 'logs').mkdir(exist_ok=True)
    gpu = ['-p', 'gpu', '--gres=gpu:l40s:1'] if light else \
        ['-p', 'gpu', '--gres=gpu:1', '--constraint=a100-80g']
    kind = 'gpu_light' if light else 'gpu'
    dep = [f'--dependency=afterany:{int(after)}'] if after else []
    cmd = ['sbatch', '--parsable', *gpu, *dep, '-c', '6', '--mem=64G', '-t', '08:00:00',
           f'--array=0-{n - 1}%{n}', '-J', JOB + ('l' if light else ''), '-o', str(RUN / f'logs/{kind}_%A_%a.log'),
           '--chdir=' + str(RUN), '--export=ALL', str(RUN / 'code/worker.sbatch'), kind]
    job = subprocess.check_output(cmd, text=True, timeout=60).strip().split(';')[0]
    with (RUN / 'jobs.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind, job=job, workers=n, after=after)) + '\n')
    print('HELDOUT_SUBMITTED', job, n, 'after', after, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'submit':
        opts = sys.argv[3:]
        after = next((o.split('=', 1)[1] for o in opts if o.startswith('--after=')), None)
        submit(sys.argv[2], light='--light' in opts, after=after)
    else:
        raise SystemExit(__doc__)
