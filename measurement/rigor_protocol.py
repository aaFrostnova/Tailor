"""Immutable image roles and acceptance semantics for the corrected experiment."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SC = Path('/data/tailor/workspace/wm_dataset10k')
SOURCE = SC / 'attack_extension_nfpa_edit_20260911'
PY = '/data/tailor/home/.conda/envs/fingerprint/bin/python'
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
    from PIL import Image
    path = Path(path or ROOT / 'plans/image_splits.json')
    if path.exists():
        result = read(path)
        validate_splits(result, verify_files=True)
        return result
    result = {'schema': 'rigor_image_splits_v1', 'resolution': 512, 'splits': {},
              'independence_scope': 'disjoint from this campaign calibration, selection, combination and smoke images',
              'final_results_must_not_update_selection': True,
              'development_content_overlaps': [],
              'development_roles': ['calibration', 'selection']}
    paths_seen, bytes_seen, pixels_seen = set(), set(), set()
    for role, offset in OFFSETS.items():
        records = []
        for source, count in COMPOSITION.items():
            files = sorted((SC / 'pool' / source / 'img').glob('*.png'))
            accepted = 0
            for position in range(offset, len(files)):
                p = files[position].resolve()
                raw_hash = digest(p)
                with Image.open(p) as im:
                    image = im.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
                    pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                duplicate = str(p) in paths_seen or raw_hash in bytes_seen or pixel_hash in pixels_seen
                if duplicate:
                    if role in ('calibration', 'selection'):
                        result['development_content_overlaps'].append(dict(role=role, path=str(p),
                            sha256=raw_hash, pixel_sha256=pixel_hash,
                            interpretation='legacy development image; not an independent test sample'))
                    else:
                        continue
                records.append(dict(index=len(records), path=str(p), source=source,
                                    source_position=position, sha256=raw_hash, pixel_sha256=pixel_hash))
                paths_seen.add(str(p)); bytes_seen.add(raw_hash); pixels_seen.add(pixel_hash)
                accepted += 1
                if accepted == count:
                    break
            if accepted != count:
                raise ValueError(f'insufficient unused images: {role}/{source}')
        result['splits'][role] = dict(offset=offset, n=len(records), images=records,
                                     composition=COMPOSITION, images_sha256=canonical_hash(records))
    result['manifest_sha256'] = canonical_hash(result)
    validate_splits(result, verify_files=True)
    with path.open('x') as handle:
        json.dump(result, handle, indent=1, allow_nan=False)
        handle.write('\n')
    return result


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
    value = dict(schema='rigor_protocol_unified_ba_v1', created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 source_campaign=str(SOURCE), active_extension_attacks=['editing_ip2p_s20_v1'], excluded_attacks=['nfpa_sd21_xy40_s10_v1'], restored_C4_requests=100, requests=10000, arm_request_evaluations=20000,
                 image_splits_sha256=digest(ROOT / 'plans/image_splits.json'),
                 formula='composite residual; selected pair overlay replaces only its own solo delta; mean and rate corrections separate',
                 acceptance_protocol='unified_raw_ba_v1',
                 acceptance=dict(n_bits=100, raw_before_ECC=True, zero_LLR_counts_as_match=False,
                     identity_verification_can_accept=False,
                     threshold='smallest integer h with K_max * Binomial(100,0.5) upper tail at h <= input alpha',
                     arithmetic='exact integer tail sum against the exact rational represented by the input float',
                     max_tests='K0(m)+5*m*resync+133*scale+81*tile+65*m*angle; frontend-host compatibility required',
                     early_stop_changes_budget=False, geometry_fusion_acceptance=False,
                     per_query_guarantee='conditional on a fresh query-specific ideal-PRF mask independent of the negative image/view proposals',
                     real_PRF_security_error_additive=True, arbitrary_fixed_key_population_guarantee=False),
                 offline_prior=dict(protocol='legacy measured BA prior', refitted_to_new_acceptance=False,
                     old_identity_flags_used_as_rate_success=False, mandatory_new_live=True),
                 model_optimality_scope='surrogate model only; no real-space global optimum or impossibility claim',
                 combination_audit=dict(tasks=126, images_per_task=100, used_for_model_fitting=False,
                                        final_selection_feedback=False),
                 selection=dict(image_role='selection', final_test_feedback=False),
                 final_test=dict(image_role='final', config_frozen_before_measurement=True,
                                 heldout_reused_across_protocol_revisions=True,
                                 newly_unseen_holdout_claim=False, old_final_outcomes_used_in_selection=False,
                                 images_per_attack=100, unmarker_images=30,
                                 empirical_detection_floor=.9, confidence_level=.95,
                                 population_lower_bound_reported_separately=True,
                                 capacity_label='BSC estimate, not demonstrated coded payload'),
                 latency=dict(metric='mean per-image embedding plus complete raw-BA decoder',
                              required_gpu_model='NVIDIA A100-SXM4-80GB',
                              aggregation='maximum mean over requested attacks',
                              models='warm and resident', batch_size=1,
                              include=['embedding', 'frontend', 'decode', 'cascade if reader executes'],
                              unchanged_embedding_component_reuse='permitted only with byte/factory provenance and same-GPU warm timing evidence; cross-session sum labelled',
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
