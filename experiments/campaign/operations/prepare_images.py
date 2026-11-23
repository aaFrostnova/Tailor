"""Preserve development roles and reserve a genuinely unused final image set."""
import os
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from rigor_protocol import canonical_hash, digest, validate_splits

SC = Path(os.environ.get('TAILOR_WORKSPACE', 'workspace'))
OLD = SC / 'rigor_editing_only_20260911/plans/image_splits.json'
NEGATIVE = ROOT.parent / 'fpr_calibration_20260914/plan.json'


def main():
    from PIL import Image
    target = ROOT / 'plans/image_splits.json'
    if target.exists():
        validate_splits(json.loads(target.read_text()), verify_files=True)
        print('Existing immutable image split verified')
        return
    original = json.loads(OLD.read_text())
    negative = json.loads(NEGATIVE.read_text())
    past = [r for s in original['splits'].values() for r in s['images']]
    past += [r for rows in negative['images'].values() for r in rows]
    seen = {key: {r[key] for r in past} for key in ('path', 'sha256', 'pixel_sha256')}
    randomizer = random.Random(2026091402)
    records = []
    composition = {'A': 25, 'B': 25, 'C': 25, 'D': 15, 'E': 10}
    for source, quota in composition.items():
        files = list(enumerate(sorted((SC / 'pool' / source / 'img').glob('*.png'))))
        randomizer.shuffle(files)
        accepted = 0
        for position, filename in files:
            path = filename.resolve()
            if str(path) in seen['path']:
                continue
            sha = digest(path)
            if sha in seen['sha256']:
                continue
            with Image.open(path) as opened:
                image = opened.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
                pixel = hashlib.sha256(image.tobytes()).hexdigest()
            if pixel in seen['pixel_sha256']:
                continue
            row = dict(index=len(records), path=str(path), source=source, source_position=position,
                       sha256=sha, pixel_sha256=pixel)
            for key in seen:
                seen[key].add(row[key])
            records.append(row)
            accepted += 1
            if accepted == quota:
                break
        if accepted != quota:
            raise RuntimeError('Insufficient unused final images for ' + source)
    value = copy.deepcopy(original)
    value.pop('manifest_sha256')
    value['independence_scope'] = ('Final images are disjoint in path, bytes and resized pixels from all original '
        'experiment roles, all superseded FPR campaign roles, and the current development roles.')
    value['historical_exclusion_manifests'] = {str(OLD): digest(OLD), str(NEGATIVE): digest(NEGATIVE)}
    value['splits']['final'] = dict(offset=None, sampling_seed=2026091402, n=100, images=records,
        composition=composition, images_sha256=canonical_hash(records))
    value['manifest_sha256'] = canonical_hash(value)
    validate_splits(value, verify_files=True)
    with target.open('x') as stream:
        json.dump(value, stream, indent=1, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(final_images=len(records), images_sha256=canonical_hash(records),
                         historical_records_excluded=len(past), manifest_sha256=digest(target))))


if __name__ == '__main__':
    main()
