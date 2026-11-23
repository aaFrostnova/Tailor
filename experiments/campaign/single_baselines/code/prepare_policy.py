"""Freeze full fixed-single populations without consulting any final result.

Input JSON and image-split manifests are copied byte-for-byte. The historical
single_asshipped function alone is extracted by AST; no module is imported.
Only this policy's immutable artifacts are written. Measurement source freezing
and GPU submissions belong to the independent campaign controller.
"""
import argparse
import ast
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from single_policy import METHODS, METHOD_FRAGMENTS, POLICY_VERSION, canonical_hash, method_specs, propose

SB = Path(__file__).resolve().parents[1]
SC = Path(os.environ.get('TAILOR_WORKSPACE', 'workspace'))
N = Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/rigor_editing_only_20260911/unified_ba_20260914')
B = SB.parent / 'baselines'
OLD_POLICY = SC / 'rigorous_baselines.py'
OLD_W = Path(os.environ.get('TAILOR_PROJECT', '.') + '/solver/watermark_smt_v2.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_bytes_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'immutable artifact differs: {path}')
        return
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f'immutable artifact differs: {path}')
    finally:
        os.unlink(temporary)


def write_once(path, value):
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    write_bytes_once(path, payload)


def copy_once(source, target):
    write_bytes_once(target, Path(source).read_bytes())


def historical_definition():
    policy_tree = ast.parse(OLD_POLICY.read_text())
    functions = [n for n in policy_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'single_asshipped']
    if len(functions) != 1:
        raise ValueError('single_asshipped definition missing or ambiguous')
    environment = {'fixed_config': lambda scenario, *configuration: configuration}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(OLD_POLICY), 'exec'), environment)
    alpha_levels = None
    for node in ast.parse(OLD_W.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ALV' for t in node.targets):
            alpha_levels = ast.literal_eval(node.value)
            break
    if alpha_levels is None or alpha_levels[3] != 1.0:
        raise ValueError('historical native alpha mapping changed')
    configurations = {}
    for method, fragment in METHOD_FRAGMENTS.items():
        subset, resync, nested, alpha_index = environment['single_asshipped']({}, fragment)
        if subset != (fragment,) or resync or nested or alpha_index != 3:
            raise ValueError('historical fixed single definition changed')
        configurations[method] = dict(S=list(subset), order=list(subset),
                                      s={fragment: float(alpha_levels[alpha_index])}, fe_on=[])
    if configurations != method_specs()['methods']:
        raise ValueError('proposal definition differs from original fixed single')
    return dict(historical_function='single_asshipped', definition_line=functions[0].lineno,
                alpha_levels=alpha_levels, configurations=configurations,
                matches_three_fixed_methods=True, reproduces_single_ship_aggregate=False)


def original_sources():
    paths = [OLD_POLICY, OLD_W, Path(__file__).resolve(),
             Path(__file__).with_name('single_policy.py').resolve(),
             N / 'plans/image_splits.json', B / 'H1_rule/plans/image_splits.json']
    paths += [N / f'fixed/inputs/C{k}.json' for k in range(1, 6)]
    return {str(path): digest(path) for path in paths}


def prepare(root=SB):
    root = Path(root).resolve()
    if any((root / method / 'final/measurement_started.json').exists() for method in METHODS):
        raise ValueError('single-baseline final already started; cannot prepare new policy')
    before = original_sources()
    history = historical_definition()
    if read(N / 'plans/image_splits.json') != read(B / 'H1_rule/plans/image_splits.json'):
        raise ValueError('main and prior baseline image splits differ')
    copied = {}
    for k in range(1, 6):
        original = N / f'fixed/inputs/C{k}.json'
        target = root / f'inputs/C{k}.json'
        copy_once(original, target)
        copied[str(target)] = digest(target)
    for method in METHODS:
        target = root / method / 'plans/image_splits.json'
        copy_once(N / 'plans/image_splits.json', target)
        copied[str(target)] = digest(target)
    spec_path = root / 'inputs/method_specs.json'
    write_once(spec_path, method_specs())
    copied[str(spec_path)] = digest(spec_path)
    source_hashes = {**before, **copied}
    rows = {method: [] for method in METHODS}
    attack_keys = {method: set() for method in METHODS}
    config_keys = {method: set() for method in METHODS}
    per_class = collections.Counter()
    for k in range(1, 6):
        group = f'C{k}'
        value = read(root / f'inputs/{group}.json')
        records = value['records']
        if value['protocol'] != 'four_inputs_capacity_output_v1':
            raise ValueError('wrong four-input protocol')
        if len(records) != 2000 or {r['i'] for r in records} != set(range(2000)):
            raise ValueError(f'incomplete request population: {group}')
        for record in sorted(records, key=lambda r: r['i']):
            if record['cls'] != group or 'min_bits' in record:
                raise ValueError('wrong request class or input payload')
            request = {key: copy.deepcopy(record[key]) for key in ('attacks', 'fpr', 'min_psnr', 'max_ms')}
            per_class[group] += 1
            for method in METHODS:
                proposal = propose(request, method)
                config_id = canonical_hash(proposal['config'])
                config_keys[method].add(config_id)
                attack_keys[method].update((config_id, a) for a in request['attacks'])
                for arm in ('04', '06'):
                    rows[method].append(dict(row_id=f'{group}/{arm}/{record["i"]}', group=group,
                        arm=arm, request_index=record['i'], request=request,
                        config_id=config_id, **proposal))
    if original_sources() != before or any(digest(p) != sha for p, sha in copied.items()):
        raise ValueError('policy source/input changed while preparing proposals')
    outputs = {}
    for method in METHODS:
        value = dict(schema='baseline_policy_proposals_v1', method=method, protocol=POLICY_VERSION,
            source_hashes=source_hashes,
            policy=dict(path=str(spec_path), sha256=digest(spec_path), **method_specs()),
            rows=rows[method], audit=dict(requests=10000, comparison_rows=20000,
                per_class=dict(per_class), unique_configs=len(config_keys[method]),
                unique_config_attack_keys=len(attack_keys[method]),
                arms_are_paired_references=True, policy_uses_final_observations=False,
                request_dependent_fragment_choice=False, historical_single_asshipped_matches=True,
                historical_single_ship_aggregate_reproduced=False))
        path = root / method / 'plans/proposals.json'
        write_once(path, value)
        outputs[method] = dict(path=str(path), sha256=digest(path), rows=len(rows[method]),
                               unique_configs=len(config_keys[method]),
                               unique_config_attack_keys=len(attack_keys[method]))
    manifest = dict(schema='baseline_policy_freeze_v1', source_hashes=source_hashes,
        policy=dict(path=str(spec_path), sha256=digest(spec_path), **method_specs()),
        historical_definition=history, methods=outputs,
        assertions=dict(all_proposals_frozen_before_baseline_final=True,
            no_main_or_baseline_final_observations_read=True, capacity_is_output=True,
            original_10000_requests_reused=True, image_splits_reused=True,
            separate_fixed_single_methods=True, no_best_single_aggregate=True))
    write_once(root / 'policy_freeze.json', manifest)
    print(json.dumps(manifest, sort_keys=True))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=SB)
    args = parser.parse_args()
    prepare(args.root)
