"""PR4 preparation: freeze negatives and the frozen TAILOR-F configurations whose false acceptance is measured.

Selection history comes from the frozen TAILOR-F walk: a deployment is first-try when the initial-margin
rank-1 candidate passed, and retry-selected otherwise (rank 2 or 3, the request-local correction, a backoff
margin, or the live-failure backoff). Top1 deploys only first-try configurations; Top3 and Full also deploy
retry-selected ones; a request deployed under several policies gets the same configuration under each. All
retry-selected configurations are measured, together with an equal number of first-try-only configurations
drawn with a fixed seed before any negative is decoded.

Usage:
  python experiments/main_controls/relocate.py RUNF
  python experiments/heldout/freeze_images.py RUNF/inputs/negatives.json --role negatives \\
      --counts A=125,B=125,C=125,D=75,E=50 --seed 20261110 --exclude RUNH/inputs/heldout_images.json \\
      --hash-cache RUNH/inputs/pool_hashes.json
  TAILOR_MAIN_CONTROLS_RUN=RUNF python experiments/fpr/prepare_fpr.py
"""
import os
import collections
import datetime
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'experiments/main_controls'), str(REPO / 'experiments/heldout'), str(REPO / 'experiments')]
from common import FROZEN, RUN, atomic, read, sha  # noqa: E402
import prepare_heldout as PH  # noqa: E402

SEED = 20261110
N_CLEAN, N_WRONG = 500, 100


def selection_kind(rows):
    """spec_id -> 'first_try' or 'retry' from the frozen record chain."""
    kinds = {}
    amended = {r['spec_id']: r for r in read(FROZEN / 'live_failure_fallback_20260921/amended_results.json')['rows']
               if r['cohort'] == 'main' and r['arm'] == '06'}
    fb = {}
    for cls in ('C1', 'C2', 'C3', 'C4', 'C5'):
        for r in read(FROZEN / 'fallback/requests' / f'{cls}_06.json')['rows']:
            fb[(cls, r['i'])] = r
    editing = {r['spec_id']: r for r in read(FROZEN / 'editing_request_extension_20260921/results.json')['rows']
               if (r['variant'], r['arm']) == ('Full', '06')}
    for r in rows:
        sid = r['spec_id']
        if sid in amended:
            a = amended[sid]
            f = fb[(a['cls'], a['i'])]
            first = a['previous_status'] == 'fixed_live_pass' and (f.get('selected') or {}).get('rank') == 1
        else:
            e = editing[sid]
            first = e['stage'] == 'initial' and e['rank'] == 1
        kinds[sid] = 'first_try' if first else 'retry'
    return kinds


def main():
    if (RUN / 'manifest.json').exists():
        raise SystemExit('manifest exists')
    negatives = read(RUN / 'inputs/negatives.json')
    if negatives['n'] != N_CLEAN:
        raise SystemExit('negatives.json must hold 500 images')
    import shutil
    shutil.copy2(Path(os.environ.get('TAILOR_RUNS', 'runs') + '/'
                      'main_controls_20261109/inputs/requests.json'), RUN / 'inputs/requests.json')
    _, rows = PH.accepted_assignments()
    kinds = selection_kind(rows)
    by_cfg = collections.defaultdict(lambda: dict(kinds=collections.Counter(), alphas=collections.Counter(), specs=[]))
    cand = {}
    for r in rows:
        g = by_cfg[r['cfg_id']]
        g['kinds'][kinds[r['spec_id']]] += 1
        g['alphas'][repr(r['request']['fpr'])] += 1
        g['specs'].append(r['spec_id'])
        cand[r['cfg_id']] = r['candidate']
    retry = sorted(c for c, g in by_cfg.items() if g['kinds']['retry'])
    first_only = sorted(c for c, g in by_cfg.items() if not g['kinds']['retry'])
    rng = random.Random(SEED)
    first_sample = sorted(rng.sample(first_only, min(len(retry), len(first_only))))
    tasks = []
    for group, ids in (('retry', retry), ('first_try', first_sample)):
        for cfg_id in ids:
            for kind, n in (('clean', N_CLEAN), ('wrong', N_WRONG)):
                tasks.append(dict(cfg_id=cfg_id, group=group, kind=kind, n=n,
                                  candidate={k: cand[cfg_id][k] for k in ('order', 'fe_on', 's')}))
    git = lambda *a: subprocess.run(['git', '-C', str(REPO), *a], check=True, capture_output=True, text=True).stdout.strip()
    atomic(RUN / 'manifest.json', dict(
        policy='exp_fpr_retries_v1', created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        repository=dict(commit=git('rev-parse', 'HEAD')), negatives_sha256=negatives['images_sha256'],
        negatives_file_sha256=sha(RUN / 'inputs/negatives.json'), seed=SEED,
        deployments=dict(collections.Counter(kinds.values())), configurations=dict(
            retry=len(retry), first_try_only=len(first_only), first_try_sampled=len(first_sample)),
        config_groups={c: dict(kinds=dict(g['kinds']), alphas=dict(g['alphas']), n_specs=len(g['specs']))
                       for c, g in by_cfg.items() if c in set(retry) | set(first_sample)},
        tasks=tasks, n_clean=N_CLEAN, n_wrong=N_WRONG,
        note='negatives are never read by any selection; one decode per (configuration, image) gives every alpha'))
    print('PREPARED', dict(retry=len(retry), first_try_sampled=len(first_sample), tasks=len(tasks)), flush=True)


if __name__ == '__main__':
    main()
