"""Frozen request cohorts.

The main cohort is the 7,321 request IDs listed in data/main_cohort.json. The ablation
cohort is the 1,111-request set listed under ablation_spec_ids in
data/ablation_evaluation.json. Request inputs come from
data/request_specifications.json and data/editing_requests.json.

Two spec_id schemes are frozen in the data and both are kept (IDs are never re-minted):
  list4        sha256 of the compact JSON list [sorted attacks, fpr, min_psnr, max_ms]
               (request_specifications.json)
  inputs_dict  sha256 of the compact, key-sorted JSON dict of the four inputs with sorted
               attacks (editing_requests.json)
Every row is recomputed under its file's scheme, so a manifest row cannot drift from its inputs.
"""
import hashlib
import json
from pathlib import Path

SPEC_FILES = ('data/request_specifications.json', 'data/editing_requests.json')
MAIN_FILE = 'data/main_cohort.json'
ABLATION_FILE = 'data/ablation_evaluation.json'
SCENARIOS = ('S1', 'S2', 'S3', 'S4', 'S5')
EXPECTED_MAIN = dict(
    n=7321,
    scenarios={'S1': 1551, 'S2': 1532, 'S3': 1373, 'S4': 822, 'S5': 2043},
    attacks=20, attack_sets=240, ctrlregen_s03=548, ctrlregen_s05=274,
)
EXPECTED_ABLATION = dict(n=1111, scenarios={'S1': 195, 'S2': 191, 'S3': 279, 'S4': 150, 'S5': 296})


class CohortError(ValueError):
    """A manifest does not match the frozen cohort definition."""


SPEC_SCHEMES = {'request_specifications': 'list4', 'editing_requests': 'inputs_dict'}


def spec_id(attacks, fpr, min_psnr, max_ms, scheme='list4'):
    if scheme == 'list4':
        key = [sorted(attacks), fpr, min_psnr, max_ms]
        text = json.dumps(key, separators=(',', ':'))
    elif scheme == 'inputs_dict':
        key = dict(attacks=sorted(attacks), fpr=fpr, min_psnr=min_psnr, max_ms=max_ms)
        text = json.dumps(key, sort_keys=True, separators=(',', ':'))
    else:
        raise CohortError(f'unknown spec_id scheme {scheme!r}')
    return hashlib.sha256(text.encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def load_specs(paper):
    """spec_id -> (source file stem, row) over both request-specification files."""
    paper = Path(paper)
    rows = {}
    base = _read(paper / SPEC_FILES[0])['rows']
    editing = _read(paper / SPEC_FILES[1])
    for name, part in ((SPEC_FILES[0], base), (SPEC_FILES[1], editing)):
        stem = Path(name).stem
        for r in part:
            if r['spec_id'] in rows:
                raise CohortError(f"spec_id {r['spec_id']} appears in two specification files")
            rows[r['spec_id']] = (stem, r)
    return rows


def manifest_row(source, r):
    inputs = r['inputs']
    attacks = sorted(inputs['attacks'])
    scheme = SPEC_SCHEMES[source]
    row = dict(spec_id=r['spec_id'], scenario='S' + r['cls'][1:], cls=r['cls'], attacks=attacks,
               alpha=inputs['fpr'], psnr_floor_db=inputs['min_psnr'],
               latency_ceiling_ms=inputs['max_ms'], attack_count=len(attacks), source=source,
               spec_id_scheme=scheme)
    if spec_id(attacks, row['alpha'], row['psnr_floor_db'], row['latency_ceiling_ms'], scheme) != row['spec_id']:
        raise CohortError(f"spec_id {row['spec_id']} does not match its inputs")
    return row


def _rows_for(paper, ids):
    specs = load_specs(paper)
    missing = sorted(set(ids) - set(specs))
    if missing:
        raise CohortError(f'{len(missing)} cohort IDs have no specification row, e.g. {missing[0]}')
    return sorted((manifest_row(*specs[i]) for i in set(ids)), key=lambda r: r['spec_id'])


def _check(cond, message):
    if not cond:
        raise CohortError(message)


def validate_main(rows):
    ids = [r['spec_id'] for r in rows]
    _check(len(ids) == len(set(ids)), 'duplicate spec_id in main manifest')
    _check(len(rows) == EXPECTED_MAIN['n'], f"main manifest has {len(rows)} rows, expected {EXPECTED_MAIN['n']}")
    counts = {s: sum(r['scenario'] == s for r in rows) for s in SCENARIOS}
    _check(counts == EXPECTED_MAIN['scenarios'], f'scenario counts {counts}')
    attacks = {a for r in rows for a in r['attacks']}
    _check(len(attacks) == EXPECTED_MAIN['attacks'], f'{len(attacks)} distinct attacks')
    _check('ctrlregen_s07' not in attacks, 'ctrlregen_s07 is excluded from the main cohort')
    for name in ('ctrlregen_s03', 'ctrlregen_s05'):
        n = sum(name in r['attacks'] for r in rows)
        _check(n == EXPECTED_MAIN[name], f'{name} appears in {n} requests')
    sets = len({tuple(r['attacks']) for r in rows})
    _check(sets == EXPECTED_MAIN['attack_sets'], f'{sets} distinct attack sets')
    four = {(tuple(r['attacks']), r['alpha'], r['psnr_floor_db'], r['latency_ceiling_ms']) for r in rows}
    _check(len(four) == len(rows), 'two requests share the same four inputs')
    return rows


def validate_ablation(rows):
    ids = [r['spec_id'] for r in rows]
    _check(len(ids) == len(set(ids)), 'duplicate spec_id in ablation manifest')
    _check(len(rows) == EXPECTED_ABLATION['n'], f'ablation manifest has {len(rows)} rows')
    counts = {s: sum(r['scenario'] == s for r in rows) for s in SCENARIOS}
    _check(counts == EXPECTED_ABLATION['scenarios'], f'ablation scenario counts {counts}')
    return rows


def main_cohort_rows(paper):
    """The main cohort is defined by the IDs of MAIN_FILE alone."""
    paper = Path(paper)
    retained = _read(paper / MAIN_FILE)['retained_spec_ids']
    return validate_main(_rows_for(paper, retained))


def ablation_cohort_rows(paper):
    ids = _read(Path(paper) / ABLATION_FILE)['ablation_spec_ids']
    return validate_ablation(_rows_for(paper, ids))


def _jsonl_bytes(rows):
    ordered = sorted(rows, key=lambda r: r['spec_id'])
    return ''.join(json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n' for r in ordered).encode()


def cohort_hash(rows):
    return hashlib.sha256(_jsonl_bytes(rows)).hexdigest()


def write_jsonl(rows, path):
    data = _jsonl_bytes(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()
