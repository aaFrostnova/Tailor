"""New threshold-indexed BA cells; legacy keyed-verification cells are rejected."""
import math

PROTOCOL = 'unified_raw_ba_v1'
FRAGMENT_KEYS = {'VINE': 'vine', 'TrustMark': 'trustmark', 'VideoSeal': 'videoseal'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cell_of_detailed(order, records):
    require(bool(records), 'no unified decoder observations')
    order = list(order)
    require(order and len(set(order)) == len(order) and set(order) <= set(FRAGMENT_KEYS), 'invalid cell fragment order')
    keys = set(records[0]['by_threshold'])
    require(bool(keys) and all(set(record['by_threshold']) == keys for record in records),
            'threshold coverage differs between images')
    cell = dict(protocol=PROTOCOL, n=len(records), order=order, by_threshold={})
    for key in sorted(keys, key=int):
        theta = int(key)
        require(str(theta) == key and 0 <= theta <= 101, 'invalid integer threshold key')
        values = [record['by_threshold'][key] for record in records]
        row = dict(threshold_count=theta, detected=[], accepted_score=[], best_fragment_ba=[],
                   per_fragment_ba={f: [] for f in order}, used_geometry=[], fused_ba=[])
        for item in values:
            require(item['threshold_count'] == theta, 'threshold-specific observation mismatch')
            require(type(item['detected']) is bool and type(item['used_geometry']) is bool,
                    'nonboolean acceptance or geometry flag')
            require(item['detected'] == (item['accepted_score'] >= theta / 100),
                    'acceptance does not use the declared strict BA threshold')
            for field in ('accepted_score', 'best_fragment_ba', 'fused_ba'):
                value = item[field]
                require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                        'invalid measured BA: ' + field)
                row[field].append(float(value))
            for field in ('detected', 'used_geometry'):
                row[field].append(item[field])
            for fragment in order:
                value = item['per_fragment_ba'].get(fragment, item['per_fragment_ba'].get(FRAGMENT_KEYS[fragment]))
                require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                        'missing or invalid fragment accuracy')
                row['per_fragment_ba'][fragment].append(float(value))
        cell['by_threshold'][key] = row
    return cell


def measure_cell(comp, cfg, images, secrets, alphas):
    from unified_detector import decode_thresholds_detailed
    require(len(images) == len(secrets) and bool(images), 'image/identity counts differ')
    return cell_of_detailed(cfg['order'], [decode_thresholds_detailed(comp, image, secret, alphas)
                                         for image, secret in zip(images, secrets)])
