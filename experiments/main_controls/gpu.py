"""Native unified live measurement in the relocated root, including the cross-environment attacks.

A task names one configuration and the attacks whose cells are missing. Cells measured here are written
by the relocated live_topk_full.py under RUN/root/live_topk/cells; cross-environment attacks are decoded
into an exported group under RUN/root and then published as cells. Frozen cells are never copied into
the run's cache.
"""
import fcntl
import os
import runpy
import subprocess
import sys
from pathlib import Path
from common import (ARM, C, CELLS, CP, PY, ROOT, RUN, XENV, atomic, cell_name, cfgkey, group_path, light_eligible,
                    load_cell, read, tag, task_id, verify_sources)


IMAGE_SET = RUN / 'inputs/image_set.json'
IDENTITY_HOOK = RUN / 'identity_hook'           # fresh-identity runs only (experiments/heldout/identity_hook)


def child_env():
    """Environment of the native child: with RUN/identity_hook/identity.json the hook directory leads
    PYTHONPATH, so its sitecustomize shifts the composite's identity index; otherwise unchanged."""
    env = dict(os.environ)
    if (IDENTITY_HOOK / 'identity.json').exists():
        env['PYTHONPATH'] = str(IDENTITY_HOOK) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def install_image_set(path=IMAGE_SET):
    """Point the frozen driver's image sampler at a frozen image set (held-out runs only).

    The frozen driver draws its covers with src.image_pool.sample(N, offset=300). When the run has an
    inputs/image_set.json, that call returns the set's images in their frozen order instead; the frozen
    code is not changed. Without the file nothing is patched (the main controls)."""
    if not Path(path).exists():
        return None
    doc = read(path)
    files = [r['path'] for r in doc['images']]
    import src.image_pool as pool
    if not pool.__file__.startswith(str(C.CF)):
        raise ImportError(f'unexpected image pool module {pool.__file__}')

    def frozen_sample(n, offset=0, composition=None):
        if n != len(files) or offset != 300 or composition is not None:
            raise ValueError(f'frozen image set has {len(files)} images; driver asked n={n} offset={offset}')
        return list(files)
    pool.sample = frozen_sample
    return doc['images_sha256']


def plan_row(c, attacks):
    """The single request row handed to the frozen driver; it is judged before anything is measured."""
    return dict(i=0, attacks=list(attacks), fpr=.01, min_psnr=0., max_ms=1e20, candidates=[dict(c, rank=1)])


def preflight(task):
    """Run the frozen driver's pre-measurement walk on CPU with no cells; a missing candidate field fails
    here instead of on a GPU node."""
    row = plan_row(task['candidate'], task['attacks'])
    CP.walk_rows([row], lambda cand, attacks: {a: None for a in attacks}, .9,
                 lambda cand: CP.exact_cfg_id(cfgkey(cand)))
    assert row['open'] and sorted(row['walk'][0]['pending']) == sorted(task['attacks']), row['walk']
    return row


def device():
    """(GPU name, host) of this allocation; recorded with every task so cells can be traced to a GPU type."""
    import socket
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True, timeout=15)
    names = [r.strip() for r in raw.splitlines() if r.strip()]
    assert len(names) == 1, ('ambiguous GPU allocation', names)
    return names[0], socket.gethostname()


def run(task):
    c, attacks, cls = task['candidate'], task['attacks'], task['cls']
    gpu_name, host = device()
    key = cfgkey(c)
    digest = CP.exact_cfg_id(key)
    folder = RUN / 'gpu_tasks' / task_id(task)
    folder.mkdir(parents=True, exist_ok=True)
    force_run = bool(task.get('run_cells_only'))     # bridging tasks measure even what the frozen cache has
    with (ROOT / 'fixed/checkpoints' / ('cfg_' + digest + '.lock')).open('a') as h:
        try:
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('GPU_CONFIGURATION_BUSY', digest, flush=True)
            raise SystemExit(75)
        missing = [a for a in attacks if load_cell(c, a, frozen=not force_run) is None]
        # A non-A100 GPU measures only attacks a cross-GPU check approved (or, in a check run, has under test).
        if 'A100' not in gpu_name and not light_eligible(missing):
            print('NEEDS_A100', gpu_name, sorted(missing), flush=True)
            raise SystemExit(75)
        if missing:
            atomic(folder / 'rows.json', [plan_row(c, missing)])
            atomic(folder / 'plan.json', dict(cls=cls, arm=ARM, rank=1, configs=[dict(
                key=list(map(list, key)), cand=dict(c, rank=1), attacks=missing)]))
            # End the encoder and in-process attack process before loading the cross-environment models.
            subprocess.run([PY, '-B', '-u', __file__, 'native', str(folder), cls], check=True, env=child_env())
            xm = [a for a in missing if a in XENV and load_cell(c, a, frozen=False) is None]
            if xm:
                p = group_path(cls, key)
                g = read(p)
                item = dict(cls=cls, arm=ARM, gidx=g['gidx'], domain=f'topk_{cls}_{tag()}', required=xm, path=str(p))
                atomic(folder / 'xenv.json', [item])
                C.xenv_task(str(folder / 'xenv.json'), 0)
        for a in attacks:
            rec = load_cell(c, a, frozen=not force_run)
            assert rec is not None, ('missing measurement', digest, a)
            if rec['origin'] == 'run':
                # Publish the run's own cross-environment records for exact reuse; frozen cells stay frozen.
                p = CELLS / cell_name(key, a)
                if not p.exists():
                    atomic(p, dict({k: v for k, v in rec.items() if k != 'origin'}, key=list(map(list, key))))
    atomic(folder / 'complete.json', dict(task=task, complete=True, protocol=CP.PROTOCOL, device=gpu_name, host=host,
                                         measured=missing))
    print('GPU_TASK_VERIFIED', digest, attacks, flush=True)


if __name__ == '__main__':
    verify_sources()
    if sys.argv[1] == 'native':
        folder, cls = Path(sys.argv[2]), sys.argv[3]
        C.boot(ARM)
        image_set = install_image_set()
        if image_set:
            print('FROZEN_IMAGE_SET', image_set, flush=True)
        if (IDENTITY_HOOK / 'identity.json').exists():
            import eval_matrix
            off = getattr(eval_matrix.OursComposite, '_tailor_identity_offset', None)
            if off != read(IDENTITY_HOOK / 'identity.json')['offset']:
                raise RuntimeError(f'identity hook not active in the native child (offset {off})')
            print('FRESH_IDENTITY_OFFSET', off, flush=True)
        os.environ.update(TAG=tag(), FULL_PREFETCH_ROWS=str(folder / 'rows.json'),
                          FULL_RANK_PLAN=str(folder / 'plan.json'))
        source = ROOT / 'fixed/code/live_topk_full.py'
        sys.argv = [str(source), 'measure_rank', str(C.CLASSES.index(cls)), '1', '0', '1', '100']
        runpy.run_path(str(source), run_name='__main__')
    else:
        run(read(sys.argv[1]))
