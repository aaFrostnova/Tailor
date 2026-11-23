"""Hardware preflight for one immutable candidate decoder revision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', type=Path, required=True)
args = parser.parse_args()
snapshot = args.snapshot.resolve()
sys.path.insert(0, str(snapshot / 'code'))
import unified_detector as UD
sys.path.insert(1, str(ROOT.parent / 'fpr_calibration_20260914/code'))
from measurement_worker import load_composite, configure, read_source_image, atomic_json, utc, sha256

os.environ.update(HF_HOME=os.environ.get('HF_HOME', '~/.cache/huggingface'),
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
source_sha = sha256(snapshot / 'code/unified_detector.py')
plan = json.loads((ROOT.parent / 'fpr_calibration_20260914/plan.json').read_text())
families = plan['families']
selected = []
for stages, count in [([], 1), ([], 2), ([], 3), (['resync'], 3),
                      (['scale'], 3), (['tile'], 3), (['angle'], 3), (['resync', 'scale'], 2)]:
    candidates = [f for f in families if set(f['cfg']['fe_on']) == set(stages) and len(f['cfg']['order']) == count]
    if candidates:
        selected.append(candidates[0])
    elif stages == ['tile']:
        cfg = dict(order=['VINE','TrustMark','VideoSeal'], S=['VINE','TrustMark','VideoSeal'],
                   s={'VINE':0.7,'TrustMark':1.0,'VideoSeal':1.0}, fe_on=['tile'])
        selected.append(dict(family_id='explicit_tile_preflight', cfg=cfg,
                             signature='VINE+TrustMark+VideoSeal/tile'))
    else:
        raise ValueError(('missing required preflight family', stages, count))
print(json.dumps(dict(phase='loading_models', families=len(selected), source_sha256=source_sha)), flush=True)
comp, torch = load_composite('cuda')
records = []
for index, family in enumerate(selected):
    cfg = family['cfg']
    original_record = plan['images']['smoke'][index % len(plan['images']['smoke'])]
    cover = read_source_image(original_record)
    configure(comp, cfg)
    watermarked, secret = comp.embed(cover, 700000 + index)
    for label, img, query in [('positive', watermarked, secret),
                              ('clean_negative', cover, comp.secret_for(800000 + index))]:
        begin = time.perf_counter()
        batch = UD.decode_thresholds_detailed(comp, img, query, plan['request_alphas'])
        torch.cuda.synchronize()
        batch_seconds = time.perf_counter() - begin
        comparisons = []
        for alpha in plan['request_alphas']:
            detail = UD.decode_detailed(comp, img, query, alpha)
            theta = UD.threshold_count(cfg, alpha)
            shared = batch['by_threshold'][str(theta)]
            fields = ['detected', 'threshold_count', 'max_tests', 'accepted_score', 'fused_ba',
                      'per_fragment_ba', 'accepted_path', 'attempted_tests']
            assert all(detail[k] == shared[k] for k in fields), (family['family_id'], label, alpha)
            assert detail['attempted_tests'] <= detail['max_tests']
            assert not detail['detected'] or detail['accepted_score'] >= theta / 100
            assert detail['identity_verification_is_acceptance_route'] is False
            comparisons.append(dict(alpha=alpha, threshold=theta, equal=True))
        row = dict(family_id=family['family_id'], cfg=cfg, image=original_record,
                   label=label, comparisons=comparisons, batch=batch, batch_seconds=batch_seconds)
        records.append(row)
        atomic_json(snapshot / 'progress.json', dict(updated_at=utc(), completed=len(records),
            total=2 * len(selected), family_id=family['family_id'], label=label, batch_seconds=batch_seconds))
        print(json.dumps(dict(phase='measured', completed=len(records), total=2*len(selected),
            family=family['signature'], label=label, batch_seconds=batch_seconds)), flush=True)
assert sha256(snapshot / 'code/unified_detector.py') == source_sha
atomic_json(snapshot / 'report.json', dict(passed=True, acceptance_protocol=UD.PROTOCOL,
    source_sha256=source_sha, completed_at=utc(), families=len(selected), records=records,
    GPU=torch.cuda.get_device_name(), tests='shared-threshold decoder versus independently executed actual early-stop decoder',
    scope='hardware implementation preflight; not positive population success or empirical FPR certification'))
print('UNIFIED_GPU_SMOKE_PASSED', flush=True)
