"""Preregister and evaluate an independent combination-model error audit.

This audit uses no calibration or final-holdout images and never feeds its
outcomes back into the current model. Results describe this fixed design; they
do not establish a global approximation bound or full solver admissibility.
"""
import argparse
import collections
import copy
import hashlib
import functools
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = Path('/data/tailor/workspace/wm_dataset10k/attack_extension_nfpa_edit_20260911')
PLAN_SCHEMA = 'unified_ba_combination_plan_v1'
OBS_SCHEMA = 'unified_ba_combination_observation_v1'
FRAGMENTS = ('VINE', 'TrustMark', 'VideoSeal')
STAGES = {'resync': FRAGMENTS, 'scale': ('VINE',), 'angle': FRAGMENTS, 'tile': ('TrustMark',)}
ATTACKS = ('editing_ip2p_s20_v1',)
FPRS = (.1, .01, 1e-4, 1e-6, 1e-9, 2**-37)
GRID = (.19, .53, .83)
SPOTS = ((.19, .83), (.53, .19), (.83, .53))
SEED = 20260911


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write('\n')


def _strength(table, fragment, fraction):
    lo, hi = table['ranges'][fragment]
    return round(lo + (hi-lo)*fraction, 10)


def _config(order, strengths, stage):
    return dict(order=list(order), s=dict(strengths), fe_on=[] if stage is None else [stage],
                fe={name: name == stage for name in STAGES})


def make_plan(table, attack_specs, image_splits, *, table_source=None, attack_specs_source=None,
              image_splits_source=None):
    if set(ATTACKS) - set(table['attacks']) or set(ATTACKS) - set(attack_specs):
        raise ValueError('the editing attack column and pinned attack specs are required')
    splits = image_splits['splits']
    chosen = splits['combination']
    images = chosen['images']
    if chosen['n'] != 100 or len(images) != 100:
        raise ValueError('combination audit requires exactly 100 preregistered images')
    ids = [item['index'] for item in images]
    if len(set(ids)) != 100 or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('combination image IDs must be unique nonnegative integers')
    for name in ('path', 'sha256', 'pixel_sha256'):
        if len({item[name] for item in images}) != 100:
            raise ValueError(f'duplicate combination images: {name}')
        own = {item[name] for item in images}
        for split_name, split in splits.items():
            if split_name != 'combination' and own & {item[name] for item in split['images']}:
                raise ValueError(f'combination overlaps {split_name} by {name}')
    tasks = []
    def append(order, strength_map, stage, kind, fractions):
        for attack in ATTACKS:
            for fragment in order:
                label = f'{fragment}|{attack}'
                if stage and fragment in STAGES[stage]:
                    knots = table['frontend'][f'base_fe_{stage}_{label}']['xs']
                else:
                    knots = table['base'][label]['xs']
                if any(abs(strength_map[fragment]-x) < 1e-8 for x in knots):
                    raise ValueError('audit strength coincides with a calibration knot')
            task = dict(kind=kind, attack=attack, attack_spec_sha256=digest(attack_specs[attack]),
                        config=_config(order, strength_map, stage), stage=stage,
                        normalized_strengths=fractions)
            task['id'] = digest(task)[:24]
            tasks.append(task)
    for pair in itertools.combinations(FRAGMENTS, 2):
        for order in (pair, tuple(reversed(pair))):
            for x, y in itertools.product(GRID, repeat=2):
                fractions = dict(zip(pair, (x, y)))
                strengths = {f: _strength(table, f, fractions[f]) for f in pair}
                append(order, strengths, None, 'pair_plain_2d_grid', fractions)
            for stage, affected in STAGES.items():
                if not set(pair) & set(affected):
                    continue
                for x, y in SPOTS:
                    fractions = dict(zip(pair, (x, y)))
                    strengths = {f: _strength(table, f, fractions[f]) for f in pair}
                    append(order, strengths, stage, 'pair_frontend_spot', fractions)
    triplet_fractions = dict(zip(FRAGMENTS, (.23, .57, .81)))
    strengths = {f: _strength(table, f, triplet_fractions[f]) for f in FRAGMENTS}
    for index, order in enumerate(itertools.permutations(FRAGMENTS)):
        append(order, strengths, None, 'triple_spot', triplet_fractions)
        append(order, strengths, ('scale', 'tile', 'resync', 'angle', 'scale', 'tile')[index],
               'triple_spot', triplet_fractions)
    assert len(tasks) == len({t['id'] for t in tasks}) == 126
    # Put one pair/plain example of each attack first for an inexpensive smoke
    # subset. The complete plan remains fixed and contains all 126 tasks.
    tasks.sort(key=lambda t: (t['kind'] != 'pair_plain_2d_grid', t['stage'] is not None,
                             tuple(t['config']['order']), tuple(t['config']['s'].values()), t['attack']))
    plan = dict(schema=PLAN_SCHEMA, seed=SEED, n=100, images=copy.deepcopy(images), image_ids=ids,
                image_split='combination', split_offset=chosen.get('offset'), image_splits_sha256=digest(image_splits),
                image_splits_source=image_splits_source, table_sha256=digest(table), table_source=table_source,
                attack_specs_sha256=digest(attack_specs), attack_specs=copy.deepcopy(attack_specs),
                attack_specs_source=attack_specs_source, tasks=tasks,
                fprs=list(FPRS), margins=[dict(name='04', mean=.02, rate=.04), dict(name='06', mean=.02, rate=.06)],
                live_det_min=.9, normalized_plain_grid=list(GRID), normalized_frontend_spots=list(SPOTS),
                measured_host_reference={'VINE': .6, 'TrustMark': 1., 'VideoSeal': 1.},
                role='independent error assessment; never used to fit this run',
                result_policy='report all preregistered tasks, including failed predictions',
                full_solver_admissibility_checked=False, global_error_bound_claimed=False)
    plan['acceptance_protocol'] = 'unified_raw_ba_v1'
    plan['legacy_offline_is_prior'] = True
    plan['execution_source_hashes'] = {str(p):file_digest(p) for p in (
        ROOT/'code/combination_measurement.py',ROOT/'code/combination_evaluation.py',
        ROOT/'code/unified_detector.py',ROOT/'code/unified_measurement.py',
        ROOT/'fixed/code/capacity_protocol.py',EXT/'code/frozen_measurement.py')}
    plan['plan_id'] = digest(plan)
    return plan


def check_plan(plan):
    if plan.get('schema') != PLAN_SCHEMA:
        raise ValueError('wrong combination plan schema')
    payload = {k: v for k, v in plan.items() if k != 'plan_id'}
    if digest(payload) != plan.get('plan_id'):
        raise ValueError('combination plan hash changed')
    if len(plan['tasks']) != 126 or len(set(t['id'] for t in plan['tasks'])) != 126:
        raise ValueError('incomplete combination design')
    return plan


@functools.lru_cache(maxsize=1)
def protocol_module():
    path = ROOT/'fixed/code/capacity_protocol.py'
    spec = importlib.util.spec_from_file_location('_combination_capacity_protocol', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_observation(plan, task, observation):
    if observation.get('schema') != OBS_SCHEMA or not observation.get('complete'):
        raise ValueError('incomplete combination observation')
    if observation.get('plan_id') != plan['plan_id'] or observation.get('task') != task:
        raise ValueError('observation belongs to another plan/configuration')
    if observation.get('image_ids') != plan['image_ids']:
        raise ValueError('observation image identities differ')
    if observation.get('acceptance_protocol') != 'unified_raw_ba_v1':
        raise ValueError('old acceptance protocol is not reusable')
    if not protocol_module().valid_cell(observation.get('cell'), task['config']['order'], 100):
        raise ValueError('observation needs complete finite 100-image vectors')
    from unified_detector import threshold_count
    if {str(threshold_count(task['config'],a)) for a in plan['fprs']} - set(observation['cell']['by_threshold']):
        raise ValueError('combination observation lacks a requested threshold')
    timing = observation.get('decode_ms')
    if not isinstance(timing, list) or len(timing) != 100 or any(not math.isfinite(x) or x < 0 for x in timing):
        raise ValueError('missing measured decode timing')
    return observation


def check_evidence(plan, task, observation):
    """Bind reported vectors to the on-disk decode and embedding evidence."""
    metadata = observation.get('measurement', {})
    if (metadata.get('image_splits_sha256') != plan['image_splits_sha256']
            or metadata.get('attack_spec_sha256') != task['attack_spec_sha256']):
        raise ValueError('observation provenance differs from preregistration')
    evidence = metadata.get('evidence', [])
    by_name = {Path(e['path']).name: e for e in evidence}
    required = {'phase.json', 'inputs.json', 'embed_complete.json', 'summary.json', 'decoded.json'}
    if len(evidence) != 5 or set(by_name) != required:
        raise ValueError('measurement evidence is incomplete')
    for item in evidence:
        if file_digest(item['path']) != item['sha256']:
            raise ValueError('measurement evidence file changed')
    phase = read(by_name['phase.json']['path'])
    decoded = read(by_name['decoded.json']['path'])
    embedded = read(by_name['embed_complete.json']['path'])
    summary = read(by_name['summary.json']['path'])
    if (phase['plan_id'] != plan['plan_id'] or phase['task'] != task
            or phase['images'] != plan['images'] or phase['image_ids'] != plan['image_ids']
            or decoded['phase'] != phase or embedded['phase'] != phase
            or not decoded['complete'] or not embedded['complete']):
        raise ValueError('measurement phase does not match audit configuration/images')
    if (decoded['cell'] != observation['cell'] or decoded['decode_ms'] != observation['decode_ms']
            or embedded['psnr_db'] != observation['psnr_db']
            or embedded['per_image_psnr_db'] != observation['per_image_psnr_db']):
        raise ValueError('reported vectors differ from decode/embedding evidence')
    if (decoded['attack_summary_sha256'] != by_name['summary.json']['sha256']
            or embedded['manifest_sha256'] != by_name['inputs.json']['sha256']
            or summary['manifest_sha256'] != by_name['inputs.json']['sha256']):
        raise ValueError('measurement artifact hash chain is broken')
    # Reuse the worker's CPU-only identity checks, including fixed attack seeds.
    from combination_measurement import validate_inputs, validate_attack_records, validate_reuse
    validate_reuse(embedded)
    manifest = validate_inputs(phase, read(by_name['inputs.json']['path']))
    validate_attack_records(phase, summary, manifest, folder=Path(by_name['summary.json']['path']).parent)
    return observation


def residual_metrics(residuals):
    values = np.asarray(residuals, float)
    if not values.size:
        return dict(n=0)
    absolute = np.abs(values)
    return dict(n=len(values), bias=float(values.mean()), optimistic_bias=float(np.maximum(values, 0).mean()),
                optimistic_fraction=float((values > 0).mean()), mae=float(absolute.mean()),
                rmse=float(np.sqrt(np.mean(values**2))), max_positive=float(max(0., values.max())),
                max_abs=float(absolute.max()),
                absolute_quantiles={str(q): float(np.quantile(absolute, q)) for q in (.5, .9, .95, .99)},
                signed_quantiles={str(q): float(np.quantile(values, q)) for q in (.05, .5, .95)})


def paired_bootstrap(predicted, image_values, *, reps=1000, seed=SEED):
    """Pair image indices across all configs; conditional on this fixed table/design."""
    predicted, observed = np.asarray(predicted, float), np.asarray(image_values, float)
    if observed.ndim != 2 or observed.shape != (len(predicted), 100) or reps < 1:
        raise ValueError('bootstrap requires task-by-100 paired observations and positive reps')
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps):
        ids = rng.integers(0, 100, 100)
        residual = predicted - observed[:, ids].mean(axis=1)
        samples.append([residual.mean(), np.abs(residual).mean(), np.sqrt(np.mean(residual**2))])
    bounds = np.quantile(np.asarray(samples), [.025, .975], axis=0)
    return dict(replicates=reps, seed=seed, resampling_unit='paired image index across fixed configurations',
                conditional_on_fixed_model=True,
                ci95={name: bounds[:, i].tolist() for i, name in enumerate(('bias', 'mae', 'rmse'))})


def evaluate(plan, table, observations, *, predictor=None, tau_from_fpr=None, bootstrap_reps=1000,
             verify_evidence=True):
    """Compare exact mean/rate coverage forecasts with complete live observations."""
    check_plan(plan)
    if digest(table) != plan['table_sha256']:
        raise ValueError('prediction table differs from preregistered table')
    if predictor is None or tau_from_fpr is None:
        sys.path.insert(1, str(ROOT/'fixed/code'))
        from surrogate_model import Surrogate
        import watermark_smt_v2 as W
        from composite_prediction import predict_composite
        sg = Surrogate.from_dict(table)
        predictor = predictor or (lambda cfg, attack, tau, margin: predict_composite(
            sg, cfg, attack, tau=tau, mean_margin=margin['mean'], rate_margin=margin['rate'], det_min=.9))
        from unified_detector import threshold_for_config
        tau_from_fpr = tau_from_fpr or (lambda fpr, cfg: threshold_for_config(cfg, fpr))
    found = {}
    for observation in observations:
        task_id = observation['task']['id']
        if task_id in found:
            raise ValueError('duplicate combination observation')
        found[task_id] = observation
    expected = {task['id'] for task in plan['tasks']}
    if set(found) != expected:
        raise ValueError(f'combination coverage incomplete: missing={len(expected-set(found))}, extra={len(set(found)-expected)}')
    protocol, rows = protocol_module(), []
    for task in plan['tasks']:
        obs = check_observation(plan, task, found[task['id']])
        if verify_evidence:
            check_evidence(plan, task, obs)
        cfg = task['config']
        for fpr in plan['fprs']:
            tau = tau_from_fpr(fpr, cfg)
            values, verified, accepted, _ = protocol.selected_views(obs['cell'], cfg['order'], tau)
            combined = protocol.combine(obs['cell'], cfg['order'], tau, .9)
            combined_values = np.asarray(combined['image_bit_accuracy'], float)
            actual_mean, actual_rate = combined['mean'], combined['rate']
            actual_pass = combined['ok']
            for margin in plan['margins']:
                pred = predictor(cfg, task['attack'], tau, margin)
                fragments = pred['fragments']
                mean_prediction = max(fragments[f]['mean'] for f in cfg['order'])
                rates = [fragments[f]['rate_isotonic'] for f in cfg['order']]
                if any(r is None for r in rates):
                    raise ValueError('missing calibrated rate curve on a preregistered new attack')
                rate_prediction = max(rates)
                predicted_pass = pred['attack_pass']
                if predicted_pass is None:
                    raise ValueError('missing offline rate evidence on a preregistered new attack')
                fragment_errors = {}
                for fi, fragment in enumerate(cfg['order']):
                    actual_fragment_mean = float(values[fi].mean())
                    actual_fragment_rate = float(np.mean(values[fi] >= tau-1e-12))
                    prediction = fragments[fragment]
                    fragment_errors[fragment] = dict(predicted_mean=prediction['mean'], observed_mean=actual_fragment_mean,
                        mean_residual=prediction['mean']-actual_fragment_mean, predicted_rate=prediction['rate_isotonic'],
                        observed_rate=actual_fragment_rate,
                        rate_residual=None if prediction['rate_isotonic'] is None else prediction['rate_isotonic']-actual_fragment_rate)
                rows.append(dict(task_id=task['id'], kind=task['kind'], attack=task['attack'], stage=task['stage'],
                    config=cfg, fpr=fpr, tau=tau, margin=margin, predicted_mean=mean_prediction, observed_mean=actual_mean,
                    predicted_rate_proxy=rate_prediction, observed_rate=actual_rate,
                    mean_residual=mean_prediction-actual_mean,
                    rate_residual=None if rate_prediction is None else rate_prediction-actual_rate,
                    predicted_coverage_pass=bool(predicted_pass), live_pass=bool(actual_pass),
                    false_predicted_pass=bool(predicted_pass and not actual_pass), fragments=fragment_errors,
                    image_bit_accuracy=combined_values.tolist(), image_pass=accepted.tolist()))
    summaries = []
    for attack in ATTACKS:
        for fpr in plan['fprs']:
            for margin in plan['margins']:
                selected = [r for r in rows if r['attack'] == attack and r['fpr'] == fpr and r['margin'] == margin]
                pred_pass = sum(r['predicted_coverage_pass'] for r in selected)
                summary = dict(attack=attack, fpr=fpr, margin=margin, tasks=len(selected), predicted_pass=pred_pass,
                    false_predicted_pass=sum(r['false_predicted_pass'] for r in selected),
                    false_predicted_pass_denominator='all predicted coverage passes',
                    mean_error=residual_metrics([r['mean_residual'] for r in selected]),
                    rate_proxy_error=residual_metrics([r['rate_residual'] for r in selected if r['rate_residual'] is not None]),
                    fragment_mean_error=residual_metrics([x['mean_residual'] for r in selected for x in r['fragments'].values()]),
                    fragment_rate_error=residual_metrics([x['rate_residual'] for r in selected for x in r['fragments'].values() if x['rate_residual'] is not None]))
                summary['false_predicted_pass_fraction'] = summary['false_predicted_pass']/pred_pass if pred_pass else None
                if bootstrap_reps:
                    summary['mean_bootstrap'] = paired_bootstrap([r['predicted_mean'] for r in selected],
                        [r['image_bit_accuracy'] for r in selected], reps=bootstrap_reps)
                    summary['rate_bootstrap'] = paired_bootstrap([r['predicted_rate_proxy'] for r in selected],
                        [r['image_pass'] for r in selected], reps=bootstrap_reps)
                summaries.append(summary)
    design_summaries = []
    for kind in sorted({r['kind'] for r in rows}):
        for stage in [None, *STAGES]:
            selected = [r for r in rows if r['kind'] == kind and r['stage'] == stage]
            if selected:
                design_summaries.append(dict(kind=kind, stage=stage, evaluated_forecasts=len(selected),
                    mean_error=residual_metrics([r['mean_residual'] for r in selected]),
                    rate_proxy_error=residual_metrics([r['rate_residual'] for r in selected]),
                    false_predicted_pass=sum(r['false_predicted_pass'] for r in selected),
                    predicted_pass=sum(r['predicted_coverage_pass'] for r in selected)))
    return dict(schema='independent_combination_error_report_v1', plan_id=plan['plan_id'],
                complete=True, tasks=len(plan['tasks']), rows=rows, summaries=summaries, design_summaries=design_summaries,
                measurement_evidence_verified=verify_evidence,
                residual_sign='prediction minus measurement; positive is optimistic',
                rate_proxy='maximum single-fragment isotonic rate; not a predicted joint-image acceptance probability',
                pass_definition='legacy-prior same-fragment mean/rate coverage versus unified-BA accepted-score mean and >=90% images',
                model_updated_from_audit=False, full_solver_admissibility_checked=False, global_error_bound_claimed=False,
                bootstrap_scope='image-sampling uncertainty conditional on fixed configurations/model; no calibration-fit uncertainty')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    pp = sub.add_parser('plan')
    pp.add_argument('--table', type=Path, required=True)
    pp.add_argument('--attack-specs', type=Path, required=True)
    pp.add_argument('--image-splits', type=Path, required=True)
    pp.add_argument('--out', type=Path, required=True)
    ep = sub.add_parser('evaluate')
    ep.add_argument('--plan', type=Path, required=True)
    ep.add_argument('--table', type=Path, required=True)
    ep.add_argument('--observations', type=Path, required=True)
    ep.add_argument('--out', type=Path, required=True)
    ep.add_argument('--bootstrap-reps', type=int, default=1000)
    args = parser.parse_args()
    if args.mode == 'plan':
        result = make_plan(read(args.table), read(args.attack_specs), read(args.image_splits),
            table_source=str(args.table.resolve()), attack_specs_source=str(args.attack_specs.resolve()),
            image_splits_source=str(args.image_splits.resolve()))
    else:
        result = evaluate(read(args.plan), read(args.table), [read(p) for p in sorted(args.observations.glob('*.json'))],
                          bootstrap_reps=args.bootstrap_reps)
    write_new(args.out, result)
    print(json.dumps(dict(path=str(args.out), plan_id=result['plan_id'], tasks=len(result.get('tasks', []))
                          if isinstance(result.get('tasks'), list) else result.get('tasks'))))
