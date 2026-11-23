"""Operational readiness checks; an unfinished dependency is not a failed run."""
import hashlib
import json
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pending_dependencies(root, lane):
    root = Path(root)
    names = ('baselines', 'single_baselines') if lane == 'single_baselines' else ('baselines',)
    pending = []
    for name in names:
        base = root / name
        path = base / 'smoke/audit.json'
        if not path.exists():
            pending.append(str(path))
            continue
        audit = _read(path)
        if audit.get('complete') is not True:
            pending.append(str(path))
            continue
        expected = 'single_fragment_gpu_preflight_v1' if name == 'single_baselines' else 'baseline_gpu_preflight_v1'
        if audit.get('preflight_passed') is not True or audit.get('schema') != expected or audit.get('formal_data') is not False:
            raise ValueError('completed GPU smoke audit is invalid: ' + str(path))
        sources = base / 'protocol_sources.json'
        if audit.get('source_manifest_sha256') != _digest(sources):
            raise ValueError('GPU smoke sources changed: ' + str(path))
        plan = _read(base / 'smoke/plans/smoke_tasks.json')
        if audit.get('plan_sha256') != plan['plan_sha256']:
            raise ValueError('GPU smoke plan changed: ' + str(path))
        for source, expected_hash in _read(sources)['hashes'].items():
            if _digest(source) != expected_hash:
                raise ValueError('GPU smoke source changed: ' + source)
        for evidence in audit['evidence'].values():
            if _digest(evidence['path']) != evidence['sha256']:
                raise ValueError('GPU smoke evidence changed: ' + evidence['path'])
    return pending
