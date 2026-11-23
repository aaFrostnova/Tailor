"""Immutable image roles and acceptance semantics for the corrected experiment."""
import datetime
import hashlib
import json
from pathlib import Path

import os
B = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get('BASELINE_RUN_ROOT', str(B / 'H1_rule'))).resolve()
if ROOT.parent != B or ROOT.name not in {'H1_rule', 'H2_rule', 'smoke'}:
    raise ValueError('BASELINE_RUN_ROOT must name an isolated baseline method root')
SC = Path(os.environ.get('TAILOR_WORKSPACE', 'workspace'))
MAIN = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/rigor_editing_only_20260911/unified_ba_20260914')
SOURCE = SC / 'attack_extension_nfpa_edit_20260911'
PY = os.environ.get('TAILOR_PYTHON', 'python')
COMPOSITION = {'A': 25, 'B': 25, 'C': 25, 'D': 15, 'E': 10}
OFFSETS = {'calibration': 0, 'selection': 300, 'combination': 800, 'smoke': 850, 'final': 950}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


def build_splits(path=None):
    # All methods consume the exact newly frozen main manifest, including its
    # held-out final role; never silently rebuild or substitute image roles.
    path = Path(path or ROOT / 'plans/image_splits.json')
    source = MAIN / 'plans/image_splits.json'
    value = read(source)
    validate_splits(value, verify_files=True)
    if path.exists():
        if read(path) != value:
            raise ValueError('baseline and unified main image manifests differ')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(source.read_bytes())
    return value


def validate_splits(value, verify_files=False):
    assert value['schema'] == 'rigor_image_splits_v1'
    assert set(value['splits']) == set(OFFSETS)
    unsigned = {k: v for k, v in value.items() if k != 'manifest_sha256'}
    assert value['manifest_sha256'] == canonical_hash(unsigned)
    seen = {key: set() for key in ('path', 'sha256', 'pixel_sha256')}
    for role in OFFSETS:
        split = value['splits'][role]
        rows = split['images']
        assert split['n'] == len(rows) == 100
        assert [r['index'] for r in rows] == list(range(100))
        assert split['images_sha256'] == canonical_hash(rows)
        assert {s: sum(r['source'] == s for r in rows) for s in COMPOSITION} == COMPOSITION
        for row in rows:
            for key in seen:
                if role not in ('calibration', 'selection'):
                    assert row[key] not in seen[key], ('image overlap', role, key, row[key])
                seen[key].add(row[key])
            if verify_files:
                assert digest(row['path']) == row['sha256'], row['path']
                from PIL import Image
                with Image.open(row['path']) as im:
                    resized = im.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
                    assert hashlib.sha256(resized.tobytes()).hexdigest() == row['pixel_sha256'], row['path']
    return True


def preregister():
    splits = build_splits()
    value = dict(schema='rigor_protocol_v1', created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 source_campaign=str(SOURCE), active_extension_attacks=['editing_ip2p_s20_v1'], excluded_attacks=['nfpa_sd21_xy40_s10_v1'], restored_C4_requests=100, requests=10000, arm_request_evaluations=20000,
                 image_splits_sha256=digest(ROOT / 'plans/image_splits.json'),
                 formula='composite residual; mean and per-image-rate corrections kept separate',
                 model_optimality_scope='surrogate model only; no real-space global optimum or impossibility claim',
                 combination_audit=dict(tasks=126, images_per_task=100, used_for_model_fitting=False,
                                        final_selection_feedback=False),
                 selection=dict(image_role='selection', final_test_feedback=False),
                 final_test=dict(image_role='final', config_frozen_before_measurement=True,
                                 images_per_attack=100, unmarker_images=30,
                                 empirical_detection_floor=.9, confidence_level=.95,
                                 population_lower_bound_reported_separately=True,
                                 capacity_label='BSC estimate, not demonstrated coded payload'),
                 latency=dict(metric='mean per-image embedding plus complete unified strict-BA decode',
                              required_gpu_model='NVIDIA A100-SXM4-80GB',
                              aggregation='maximum mean over requested attacks',
                              models='warm and resident', batch_size=1,
                              include=['embedding', 'frontend', 'strict-BA decode', 'permitted geometry until first acceptance'],
                              exclude=['model initialization', 'file I/O', 'attack execution'],
                              gpu_synchronization=True, report=['mean', 'median', 'p95', 'maximum'],
                              tail_latency_guarantee=False))
    p = ROOT / 'plans/preregistration.json'
    if not p.exists():
        with p.open('x') as handle:
            json.dump(value, handle, indent=1)
            handle.write('\n')
    return read(p)


if __name__ == '__main__':
    result = preregister()
    print(json.dumps(result, indent=1))
