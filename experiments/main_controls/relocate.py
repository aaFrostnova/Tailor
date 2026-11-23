"""Build RUN/root: byte-identical copies of the frozen runner and its inputs, one patched shell script.

The frozen runner derives every output path from its own file location (RUN_ROOT = parents[2] of the
code file), so a copy placed under RUN/root writes all new cells, groups and checkpoints under RUN/root
and never into the frozen run root. Only fixed/code/xenv_static.sh names the frozen root by absolute
path; its copy has that prefix replaced and the replacement is recorded in RUN/relocation.json.

Fresh-identity runs pass --identity-hook DIR: the chain script's PYTHONPATH line then starts with DIR, whose
sitecustomize.py (experiments/heldout/identity_hook) shifts the composite's identity index in the decode_xenv
process as well; the second replacement is recorded next to the first.

Usage: python relocate.py RUN [--identity-hook DIR]
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

FROZEN = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/'
              'rigor_editing_only_20260911/unified_ba_20260914')
# The frozen root as xenv_static.sh spells it.
OLD_PREFIX = '"${TAILOR_OUTPUTS:?}"/rigor_editing_only_20260911/unified_ba_20260914'
COPY_DIRS = ('fixed/code', 'fixed/inputs', 'code', 'shared_table_20260916/code')
FRESH_DIRS = ('fixed/checkpoints', 'fixed/plans', 'live_topk/cells')
PATCHED = 'fixed/code/xenv_static.sh'
EXPECTED_OCCURRENCES = 3   # two CDIR assignments and the decode_xenv call
SUFFIXES = ('.py', '.sh', '.sbatch', '.json')


class RelocationError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_files(frozen=FROZEN):
    """Regular top-level files of the copied directories (no caches, no subdirectories)."""
    out = []
    for rel in COPY_DIRS:
        folder = frozen / rel
        if not folder.is_dir():
            raise RelocationError(f'missing frozen directory {folder}')
        for p in sorted(folder.iterdir()):
            if p.is_file() and not p.is_symlink() and p.suffix in SUFFIXES:
                out.append(str(p.relative_to(frozen)))
    return out


PYTHONPATH_LINE = 'export PYTHONPATH="$CF:'


def patch_text(text, new_root):
    count = text.count(OLD_PREFIX)
    if count != EXPECTED_OCCURRENCES:
        raise RelocationError(f'{PATCHED}: expected {EXPECTED_OCCURRENCES} occurrences of the frozen root, '
                              f'found {count}')
    if str(FROZEN) in text:
        raise RelocationError(f'{PATCHED}: also names the resolved frozen root')
    return text.replace(OLD_PREFIX, str(new_root)), count


def hook_text(text, hook):
    if text.count(PYTHONPATH_LINE) != 1:
        raise RelocationError(f'{PATCHED}: expected one PYTHONPATH export')
    return text.replace(PYTHONPATH_LINE, f'export PYTHONPATH="{hook}:$CF:')


def relocate(run, frozen=FROZEN, identity_hook=None):
    run = Path(run).resolve()
    root = run / 'root'
    manifest_path = run / 'relocation.json'
    if manifest_path.exists() or root.exists():
        raise RelocationError(f'{root} already exists; a relocation is never redone in place')
    files = source_files(frozen)
    if PATCHED not in files:
        raise RelocationError(f'{PATCHED} is not among the copied files')
    records = []
    for rel in files:
        src, dst = frozen / rel, root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(src)
        if rel == PATCHED:
            text, count = patch_text(src.read_text(), root)
            extra = {}
            if identity_hook:
                text = hook_text(text, Path(identity_hook).resolve())
                extra = dict(identity_hook=dict(old=PYTHONPATH_LINE, new=f'export PYTHONPATH="'
                                                f'{Path(identity_hook).resolve()}:$CF:', count=1))
            dst.write_text(text)
            os.chmod(dst, src.stat().st_mode & 0o777)
            records.append(dict(path=rel, source_sha256=before, sha256=sha256(dst), identical=False,
                                replaced=dict(old=OLD_PREFIX, new=str(root), count=count), **extra))
        else:
            shutil.copy2(src, dst)
            after = sha256(dst)
            if after != before:
                raise RelocationError(f'copy of {rel} differs from its source')
            records.append(dict(path=rel, source_sha256=before, sha256=after, identical=True))
    for rel in FRESH_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=False)
    manifest = dict(frozen_root=str(frozen), root=str(root), created_at=datetime.datetime.now(
        datetime.timezone.utc).isoformat(), copied_dirs=list(COPY_DIRS), fresh_dirs=list(FRESH_DIRS),
        patched=PATCHED, files=records)
    manifest_path.write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def verify(run, frozen=FROZEN):
    """Every relocated file still matches the relocation record and its frozen source."""
    run = Path(run).resolve()
    manifest = json.loads((run / 'relocation.json').read_text())
    if manifest['frozen_root'] != str(frozen) or manifest['root'] != str(run / 'root'):
        raise RelocationError('relocation record names another root')
    problems = []
    for rec in manifest['files']:
        dst = run / 'root' / rec['path']
        if sha256(dst) != rec['sha256']:
            problems.append(f"{rec['path']}: relocated copy changed")
        if sha256(frozen / rec['path']) != rec['source_sha256']:
            problems.append(f"{rec['path']}: frozen source changed")
    if sorted(r['path'] for r in manifest['files']) != sorted(source_files(frozen)):
        problems.append('the frozen directories gained or lost files since the relocation')
    if problems:
        raise RelocationError('; '.join(problems))
    return manifest


if __name__ == '__main__':
    hook = sys.argv[sys.argv.index('--identity-hook') + 1] if '--identity-hook' in sys.argv else None
    m = relocate(sys.argv[1], identity_hook=hook)
    print(json.dumps(dict(root=m['root'], files=len(m['files']),
                          patched=[r for r in m['files'] if not r['identical']]), indent=1))
