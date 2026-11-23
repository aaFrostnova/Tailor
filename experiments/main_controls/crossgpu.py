"""Cross-GPU check: do L40S measurements agree with the frozen A100 cells?

Before any L40S worker writes into a run's cache, the nine bridging configurations (bridge_v1 sample of
the main-control run) are measured on L40S in a separate run root, on their 16 in-process attacks only,
and compared with the frozen A100 cells by the bridge_v1 rules (deterministic attacks value by value,
attacks with unseeded randomness by distribution). regen and rinse2x also carry the Stable Diffusion
replacement, so their comparison is informative only together with the A100 bridging result.

Usage:
  python relocate.py RUNX
  TAILOR_MAIN_CONTROLS_RUN=RUNX python crossgpu.py prepare SOURCE_RUN
  TAILOR_MAIN_CONTROLS_RUN=RUNX python RUNX/code/bridge.py submit 3 --light
  TAILOR_MAIN_CONTROLS_RUN=RUNX python RUNX/code/bridge.py decide
"""
import datetime
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import relocate as RL  # noqa: E402
from bridge import IN_PROCESS  # noqa: E402
from common import RUN, atomic, read, sha  # noqa: E402
from prepare import CODE_FILES  # noqa: E402


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def prepare(source_run):
    if (RUN / 'manifest.json').exists():
        raise SystemExit('manifest exists')
    if git('status', '--porcelain', '--', 'experiments/main_controls'):
        raise SystemExit('commit experiments/main_controls first')
    relocation = RL.verify(RUN)
    selection = read(Path(source_run) / 'bridging/selection.json')
    for item in selection['items']:
        item['attacks'] = [a for a in item['attacks'] if a in IN_PROCESS]
    selection['purpose'] = 'cross-GPU check: L40S versus the frozen A100 cells, in-process attacks only'
    selection['source_selection'] = str(Path(source_run) / 'bridging/selection.json')
    atomic(RUN / 'bridging/selection.json', selection)
    code = RUN / 'code'
    code.mkdir(parents=True, exist_ok=False)
    for name in CODE_FILES:
        shutil.copy2(HERE / name, code / name)
    atomic(RUN / 'manifest.json', dict(policy='exp_crossgpu_l40s_v1', created_at=datetime.datetime.now(
        datetime.timezone.utc).isoformat(), repository=dict(commit=git('rev-parse', 'HEAD')),
        device='L40S (gpu, --gres=gpu:l40s:1)', reference='frozen A100 cells',
        configurations=len(selection['items'])))
    paths = [RUN / 'manifest.json', RUN / 'relocation.json', *(code / n for n in CODE_FILES),
             *(RUN / 'root' / rec['path'] for rec in relocation['files'])]
    atomic(RUN / 'sources.json', dict(hashes={str(p): sha(p) for p in paths}))
    print('PREPARED', len(selection['items']), 'configurations', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare(sys.argv[2])
    else:
        raise SystemExit(__doc__)
