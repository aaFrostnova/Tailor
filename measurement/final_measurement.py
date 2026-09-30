"""Isolated final-image measurement, with actual request-aware deployment timing.

CLI: python -B final_measurement.py task PLAN INDEX --cache-root DIR
Each task writes DIR/<task_id>/result.json. No selection code runs here.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

from rigor_protocol import ROOT, SOURCE, PY, canonical_hash, digest, read
from final_holdout import (check_frozen, check_plan, check_signed, check_splits, describe,
                           require, write_once)

CF = Path('/data/tailor/project')
SC = Path('/data/tailor/workspace/wm_dataset10k')
BASE = SC / 'topk_capacity_output_20260910'
UNIFIED_ROOT = Path('/data/tailor/home/outputs/rigor_editing_only_20260911/unified_ba_20260914')
sys.path.insert(0, str(UNIFIED_ROOT / 'code'))
import unified_detector as UD
from unified_measurement import cell_of_detailed
ACCEPTANCE_PROTOCOL = 'unified_raw_ba_v1'
NORMAL = {'jpeg25', 'blur', 'noise', 'bright', 'contrast', 'crop75', 'crop50', 'rot9',
          'rs256', 'hflip', 'crop_jpeg', 'border20', 'vaeB', 'vaeC', 'regen', 'rinse2x'}
CRPY = '/data/tailor/assets/.conda/envs/ctrlregen/bin/python'
UMPY = '/data/tailor/assets/.conda/envs/unmarker/bin/python'


def source_fingerprints():
    paths = [Path(__file__), Path(__file__).with_name('final_holdout.py'),
             Path(__file__).with_name('rigor_protocol.py'),
             UNIFIED_ROOT / 'code/unified_detector.py', UNIFIED_ROOT / 'code/unified_measurement.py',
             UNIFIED_ROOT / 'fixed/code/capacity_protocol.py',
             CF / 'scripts/defense/eval_matrix.py', CF / 'scripts/defense/composite_external_eval.py']
    paths += sorted((CF / 'src').glob('*.py'))
    if Path(__file__).resolve().parent != UNIFIED_ROOT / 'code':
        paths += sorted(Path(__file__).parent.glob('*.py'))
        manifest = Path(__file__).resolve().parents[1] / 'protocol_sources.json'
        if manifest.exists():
            paths.append(manifest)
    return {str(p.resolve()): digest(p) for p in paths}


def check_sources(hashes):
    require(hashes, 'missing source provenance')
    for path, sha in hashes.items():
        require(digest(path) == sha, f'measurement/attack source changed: {path}')


def attack_specification(attack):
    require(attack != 'nfpa_sd21_xy40_s10_v1', 'NFPA is excluded from the editing-only revision')
    specs = read(SOURCE / 'attack_specs.json')
    if attack in specs:
        spec = specs[attack]
        check_sources(spec['source_hashes'])
        return dict(kind='extension', attack=attack, spec=spec, extension_spec_sha256=canonical_hash(spec))
    if attack in NORMAL:
        return dict(kind='normal', attack=attack, seed_policy='42 + final image index',
                    source_hashes={str(CF / 'src/attacks.py'): digest(CF / 'src/attacks.py')})
    if attack in {'ctrlregen_s03', 'ctrlregen_s05', 'ctrlregen_s07'}:
        path = CF / 'scripts/attack/ctrlregen_batch.py'
        return dict(kind='ctrlregen', attack=attack, python=CRPY, steps=50,
                    strength=int(attack[-2:]) / 10, seed=1, source_hashes={str(path): digest(path)})
    if attack == 'unmarker':
        paths = [CF / 'scripts/attack/unmarker_batch.py', CF / 'external/ai-watermark/attack_configs/Vine.yaml']
        return dict(kind='unmarker', attack=attack, python=UMPY, seed=1234,
                    n=30, batch=1, config='attack_configs/Vine.yaml',
                    source_hashes={str(p): digest(p) for p in paths})
    raise ValueError(f'unsupported attack, no default pass: {attack}')


def timed_call(function, *args, synchronize=None, clock=time.perf_counter, **kwargs):
    """Only the function itself is charged; callers preload images/models."""
    if synchronize is None:
        import torch
        synchronize = torch.cuda.synchronize
    synchronize()
    start = clock()
    value = function(*args, **kwargs)
    synchronize()
    return value, (clock() - start) * 1000


def time_decode(comp, image, secret, fpr, **timing):
    return timed_call(UD.decode_detailed, comp, image, secret, alpha=fpr, **timing)


def device_metadata(expected=None):
    import torch
    require(torch.cuda.is_available(), 'GPU measurement requires CUDA')
    name = torch.cuda.get_device_name()
    if expected:
        require(name == expected, f'wrong timing hardware: {name}, expected {expected}')
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(gpu_model=name, gpu_memory_bytes=properties.total_memory,
                gpu_uuid=str(getattr(properties, 'uuid', 'unavailable')), host=platform.node(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), torch_version=torch.__version__,
                cuda_version=torch.version.cuda, cpu_threads=torch.get_num_threads(),
                batch_size=1, synchronized=True, clock='time.perf_counter',
                timing_path='unified strict-BA decoder; all enabled views charged to the same FPR budget',
                excluded=['PNG I/O', 'model initialization', 'attack execution', 'separate diagnostic reads (none in unified protocol)'])


def measurement_namespace():
    # Native image codecs only; no historical decoder or diagnostic cache loader.
    for path in (CF, CF / 'scripts', CF / 'scripts/defense', CF / 'external/WatermarkAttacker'):
        sys.path.insert(0, str(path))
    from eval_matrix import OursComposite
    UD.install(OursComposite)
    return {'OursComposite': OursComposite}


def context(plan_path, index, cache_root):
    plan = read(plan_path)
    check_signed(plan, 'plan_sha256')
    require(digest(plan['frozen_path']) == plan['frozen_file_sha256'], 'frozen file changed')
    frozen = read(plan['frozen_path'])
    splits = check_frozen(frozen)
    check_plan(plan, frozen)
    task = plan['tasks'][index]
    check_signed(task, 'task_id')
    require(task['selection_sha256'] == frozen['selection_sha256'] == plan['selection_sha256'], 'selection/task mismatch')
    require(task['images_sha256'] == splits['splits'][task['mode']]['images_sha256'], 'wrong final image role')
    require(task['n'] == (min(30, frozen['n']) if task['attack'] == 'unmarker' else frozen['n']), 'wrong attack sample size')
    check_sources(task['measurement_source_hashes'])
    require(task['attack_spec'] == attack_specification(task['attack']), 'attack specification changed')
    cache_root = Path(cache_root).resolve()
    require(cache_root.is_relative_to(ROOT.resolve()), 'final cache must be inside rigor experiment root')
    require('pool300' not in str(cache_root) and not cache_root.is_relative_to(ROOT / 'fixed'), 'selection cache forbidden')
    embed_identity = dict(selection_sha256=frozen['selection_sha256'], cfg_sha256=task['cfg_sha256'],
                          images_sha256=task['images_sha256'], n=frozen['n'],
                          measurement_source_hashes=task['measurement_source_hashes'],
                          required_gpu_model=task.get('required_gpu_model'))
    return dict(plan_path=str(Path(plan_path).resolve()), plan_sha256=plan['plan_sha256'], index=index,
        cache_root=str(cache_root), task=task, frozen=frozen,
        images=splits['splits'][task['mode']]['images'][:frozen['n']],
        folder=cache_root / task['task_id'], embed_folder=cache_root / 'embeddings' / canonical_hash(embed_identity),
        embed_identity=embed_identity)


def load_context(path):
    saved = read(path)
    value = context(saved['plan_path'], saved['index'], saved['cache_root'])
    require(value['plan_sha256'] == saved['plan_sha256'], 'plan changed during measurement')
    return value


def mark_measurement_started(ctx):
    if ctx['task']['mode'] == 'final':
        guard = Path(ctx['frozen']['split_manifest']).resolve().parent.parent / 'final/measurement_started.json'
        write_once(guard, dict(selection_sha256=ctx['task']['selection_sha256'], no_final_feedback=True))


def load_image(row):
    from PIL import Image
    import hashlib
    require(digest(row['path']) == row['sha256'], 'final input PNG changed')
    with Image.open(row['path']) as opened:
        image = opened.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
    require(hashlib.sha256(image.tobytes()).hexdigest() == row['pixel_sha256'], 'final image pixels changed')
    return image


def composite(ns, cfg):
    keys = {'VINE': 'vine', 'TrustMark': 'trustmark', 'VideoSeal': 'videoseal'}
    fe = set(cfg['fe_on'])
    order = [keys[f] for f in cfg['order']]
    comp = ns['OursComposite']('cuda', tm_variant='B', vine_variant='R', config={
        'order': order, 'frags': order, 'strengths': {},
        'resync': 'resync' in fe, 'nested': 'scale' in fe, 'scale_search': 'scale' in fe,
        'angle_sweep': 'angle' in fe, 'tile': 'tile' in fe})
    comp.strength = dict(comp.DEFAULT_STRENGTH)
    comp.strength.update({keys[f]: float(cfg['s'][f]) for f in cfg['order']})
    return comp


def verify_embedding(ctx):
    folder = ctx['embed_folder']
    result = read(folder / 'complete.json')
    require(result['complete'] is True and result['identity'] == ctx['embed_identity'], 'wrong embedding cache identity')
    if result.get('reused_embedding_pixels'):
        require(result.get('old_detection_results_reused') is False and result.get('embedding_latency_reused') is True,
                'invalid embedding migration provenance')
        check_sources(result['pixel_reuse_source_hashes'])
    require(result['manifest_sha256'] == digest(folder / 'inputs.json'), 'embedding manifest changed')
    items = read(folder / 'inputs.json')
    require(len(items) == len(ctx['images']), 'incomplete final embeddings')
    for row, item in zip(ctx['images'], items):
        require(item['index'] == row['index'] and item['cover_sha256'] == row['sha256'], 'embedding image mismatch')
        require(item['path'] == str(folder / 'images' / f"i{row['index']:05d}.png"), 'foreign embedding image')
        require(digest(item['path']) == item['image_sha256'], 'embedding PNG changed')
    return result, items


def embed(context_path):
    import numpy as np
    ctx = load_context(context_path)
    mark_measurement_started(ctx)
    folder, task = ctx['embed_folder'], ctx['task']
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / 'complete.json').exists():
            verify_embedding(ctx)
            return
        if try_reuse_embedding(ctx):
            verify_embedding(ctx)
            return
        # Frozen selection has already been hash-checked before this first image read.
        covers = [load_image(row) for row in ctx['images']]
        ns = measurement_namespace()
        started = time.perf_counter()
        comp = composite(ns, task['cfg'])
        metadata = device_metadata(task.get('required_gpu_model'))
        import torch
        torch.cuda.synchronize()
        init_seconds = time.perf_counter() - started
        for i in range(task['warmup']):
            comp.embed(covers[i % len(covers)], ctx['images'][i % len(covers)]['index'])
        torch.cuda.synchronize()
        times, psnrs, items = [], [], []
        (folder / 'images').mkdir(exist_ok=True)
        for row, cover in zip(ctx['images'], covers):
            (watermarked, _), elapsed = timed_call(comp.embed, cover, row['index'])
            require(watermarked.size == (512, 512), 'deployment embed changed image resolution')
            times.append(elapsed)
            mse = float(np.mean((np.asarray(cover, dtype=float) - np.asarray(watermarked, dtype=float)) ** 2))
            psnrs.append(10 * math.log10(255**2 / max(mse, 1e-9)))
            path = folder / 'images' / f"i{row['index']:05d}.png"
            watermarked.save(path)
            items.append(dict(index=row['index'], path=str(path), cover_path=row['path'],
                              cover_sha256=row['sha256'], image_sha256=digest(path)))
        write_once(folder / 'inputs.json', items)
        write_once(folder / 'complete.json', dict(complete=True, identity=ctx['embed_identity'],
            manifest_sha256=digest(folder / 'inputs.json'), embed_ms=times, psnr_db=psnrs,
            embedding_quality=describe(psnrs), metadata=metadata, warmup=task['warmup'],
            model_initialization_seconds=init_seconds))


def attack_normal(context_path):
    """An attack-only subprocess keeps diffusion models out of deployment timing."""
    ctx = load_context(context_path)
    task, folder = ctx['task'], ctx['folder']
    require(task['attack'] in NORMAL, 'wrong in-process attack')
    sys.path[:0] = [str(CF), str(CF / 'scripts/defense'), str(CF / 'external/WatermarkAttacker'), str(SC)]
    from src.attacks import attack_pil_any
    from PIL import Image
    import numpy as np
    import torch
    items = read(folder / 'attack_inputs.json')
    images = [Image.open(x['path']).convert('RGB') for x in items]
    attacked = folder / 'attacked'
    attacked.mkdir(exist_ok=True)
    # Lazy model loading is charged to an explicit warmup, never deployment latency.
    started = time.perf_counter()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    attack_pil_any(task['attack'], images[0], dev='cuda')
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started
    seconds = []
    for item, image in zip(items, images):
        seed = 42 + item['index']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        result, elapsed = timed_call(attack_pil_any, task['attack'], image, dev='cuda')
        seconds.append(elapsed / 1000)
        result.convert('RGB').resize((512, 512)).save(attacked / f"i{item['index']:05d}.png")
    write_once(folder / 'attack_runtime.json', dict(attack_seconds=seconds,
        initialization_and_warmup_seconds=warmup_seconds, timing_excluded_from_deployment=True))


def external_command(task, input_dir, output_dir):
    spec = task['attack_spec']
    if spec['kind'] == 'ctrlregen':
        return [CRPY, '-B', str(CF / 'scripts/attack/ctrlregen_batch.py'), '--in_dir', str(input_dir),
                '--out_dir', str(output_dir), '--step', str(spec['strength']), '--steps', '50', '--seed', '1']
    if spec['kind'] == 'unmarker':
        script = str(CF / 'scripts/attack/unmarker_batch.py')
        # Preserve the wrapper exactly, with explicit initial RNG seeds.
        bootstrap = "import random,numpy,torch,runpy;random.seed(1234);numpy.random.seed(1234);torch.manual_seed(1234);runpy.run_path(" + repr(script) + ",run_name='__main__')"
        return [UMPY, '-B', '-c', bootstrap, '--in_dir', str(input_dir), '--out_dir', str(output_dir),
                '--config', spec['config'], '--n', str(task['n']), '--start_idx', '0', '--batch', '1']
    raise ValueError('not a legacy external attack')


def pixel_reuse_entry(context_path, plan_cache=None):
    """Index exact image artifacts without loading any old detection result."""
    context_path = Path(context_path).resolve()
    old_context = read(context_path)
    plan_path = old_context['plan_path']
    cached = None if plan_cache is None else plan_cache.get(plan_path)
    if cached is None:
        old_plan = read(plan_path)
        check_signed(old_plan, 'plan_sha256')
        cached = (old_plan, digest(plan_path))
        if plan_cache is not None:
            plan_cache[plan_path] = cached
    old_plan, plan_file_sha256 = cached
    require(old_context['plan_sha256'] == old_plan['plan_sha256'], 'old pixel context plan mismatch')
    task = old_plan['tasks'][old_context['index']]
    check_signed(task, 'task_id')
    return dict(context_path=str(context_path), context_sha256=digest(context_path),
                plan_path=plan_path, plan_file_sha256=plan_file_sha256,
                cfg_sha256=task['cfg_sha256'], images_sha256=task['images_sha256'],
                attack=task['attack'], n=task['n'], mode=task['mode'])


def build_pixel_reuse_index(context_paths, output_path):
    # Thousands of tasks share a large plan: parse/hash each plan once, then
    # check it again before publication. Task-time import revalidates bytes.
    plans = {}
    entries = [pixel_reuse_entry(path, plans) for path in dict.fromkeys(map(str, context_paths))]
    check_sources({path: value[1] for path, value in plans.items()})
    return write_once(output_path, dict(schema='unified_attack_pixel_index_v1', entries=entries,
        detection_results_reused=False, embedding_latency_reused=False))


def old_pixel_context(ctx, entry, match_attack=True):
    """Validate old image provenance, without inspecting old decode results."""
    task = ctx['task']
    require(entry == pixel_reuse_entry(entry['context_path']), 'old pixel index provenance changed')
    old_context = read(entry['context_path'])
    old_plan = read(entry['plan_path'])
    old_task = old_plan['tasks'][old_context['index']]
    require(old_task['cfg'] == task['cfg'] and old_task['cfg_sha256'] == task['cfg_sha256'],
            'pixel cache exact configuration differs')
    if match_attack:
        require(old_task['attack_spec'] == task['attack_spec'], 'pixel cache attack/source/seed differs')
        require(old_task['n'] == task['n'], 'pixel cache attack image count differs')
    require(old_task['mode'] == task['mode'] and
            old_task['images_sha256'] == task['images_sha256'], 'pixel cache image role differs')
    check_sources(old_task['measurement_source_hashes'])
    old_frozen = read(old_plan['frozen_path'])
    check_signed(old_frozen, 'selection_sha256')
    require(digest(old_plan['frozen_path']) == old_plan['frozen_file_sha256'], 'old pixel selection changed')
    require(old_frozen['selection_sha256'] == old_task['selection_sha256'], 'old pixel selection identity differs')
    require(digest(old_frozen['split_manifest']) == old_frozen['split_manifest_sha256'], 'old image manifest changed')
    old_images = check_splits(old_frozen['split_manifest'])['splits'][task['mode']]['images'][:old_frozen['n']]
    require(old_images == ctx['images'], 'pixel cache source images or identities differ')
    folder = Path(old_context['cache_root']) / old_task['task_id']
    require(Path(entry['context_path']).parent == folder.resolve(), 'foreign old pixel context')
    sources = {str(path): digest(path) for path in (Path(entry['context_path']), Path(entry['plan_path']),
        Path(old_plan['frozen_path']), Path(old_frozen['split_manifest']))}
    sources.update(old_task['measurement_source_hashes'])
    return old_context, old_task, old_frozen, folder, sources


def embedding_migration_certificate(ctx, old_task):
    path = os.environ.get('UNIFIED_EMBED_REUSE_CERTIFICATE')
    require(bool(path), 'embedding migration certificate unavailable')
    cert = read(path)
    check_signed(cert, 'certificate_sha256')
    require(cert['schema'] == 'unified_embedding_migration_v1' and cert['complete'] is True and
            cert['factory_equivalence_reviewed'] is True, 'embedding factory equivalence not reviewed')
    for key in ('source_hashes', 'measurement_files', 'evidence'):
        check_sources(cert[key])
    measurement = str(Path(__file__).resolve())
    require(cert['measurement_files'].get(measurement) == digest(measurement), 'new embedding factory not certified')
    old_files = [p for p in old_task['measurement_source_hashes'] if Path(p).name == 'final_measurement.py']
    require(old_files and all(cert['measurement_files'].get(p) == old_task['measurement_source_hashes'][p]
                             for p in old_files), 'old embedding factory not certified')
    require(bool(cert['representatives']), 'embedding equivalence representatives absent')
    expected_gpu = ctx['task']['required_gpu_model']
    for row in cert['representatives']:
        require(isinstance(row['cfg_sha256'], str) and len(row['cfg_sha256']) == 64 and
                row['old_png_sha256'] == row['new_png_sha256'] and len(row['new_png_sha256']) == 64,
                'embedding representative bytes differ')
        require(row['old_gpu_model'] == row['new_gpu_model'] == expected_gpu and row['latency_check_pass'] is True,
                'embedding representative timing hardware/check differs')
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] > 0
                    for k in ('old_embed_ms', 'new_embed_ms')), 'invalid representative embedding timing')
    sources = {str(Path(path).resolve()): digest(path)}
    for key in ('source_hashes', 'measurement_files', 'evidence'):
        sources.update(cert[key])
    return sources


def validate_embedding_reuse(ctx, entry):
    old_context, task, frozen, _, sources = old_pixel_context(ctx, entry, match_attack=False)
    sources.update(embedding_migration_certificate(ctx, task))
    identity = dict(selection_sha256=frozen['selection_sha256'], cfg_sha256=task['cfg_sha256'],
        images_sha256=task['images_sha256'], n=frozen['n'], measurement_source_hashes=task['measurement_source_hashes'],
        required_gpu_model=task.get('required_gpu_model'))
    folder = Path(old_context['cache_root']) / 'embeddings' / canonical_hash(identity)
    summary = read(folder / 'complete.json')
    require(summary['complete'] is True and summary['identity'] == identity and
            summary['manifest_sha256'] == digest(folder / 'inputs.json'), 'old embedding manifest differs')
    require(summary['warmup'] == ctx['task']['warmup'] >= 1, 'embedding warmup differs')
    meta = summary['metadata']
    require(meta['gpu_model'] == ctx['task']['required_gpu_model'] and meta['batch_size'] == 1 and
            meta['synchronized'] is True, 'old embedding timing hardware or synchronization differs')
    items = read(folder / 'inputs.json')
    require(len(items) == len(ctx['images']), 'old embedding sample differs')
    for key in ('embed_ms', 'psnr_db'):
        require(len(summary[key]) == len(items) and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                                       for v in summary[key]), 'invalid old embedding measurements')
    require(summary['embedding_quality'] == describe(summary['psnr_db']), 'old embedding quality differs')
    for row, item in zip(ctx['images'], items):
        require(item['index'] == row['index'] and item['cover_path'] == row['path'] and
                item['cover_sha256'] == row['sha256'] and digest(row['path']) == row['sha256'],
                'old embedding source identity differs')
        require(item['path'] == str(folder / 'images' / f"i{row['index']:05d}.png") and
                digest(item['path']) == item['image_sha256'], 'old embedded PNG differs')
        sources[item['path']] = item['image_sha256']
        sources[row['path']] = row['sha256']
    for path in (folder / 'complete.json', folder / 'inputs.json'):
        sources[str(path)] = digest(path)
    return summary, items, sources


def try_reuse_embedding(ctx):
    index_path = os.environ.get('UNIFIED_ATTACK_REUSE_INDEX')
    if not index_path or not os.environ.get('UNIFIED_EMBED_REUSE_CERTIFICATE'):
        return False
    index = read(index_path)
    require(index.get('schema') == 'unified_attack_pixel_index_v1', 'invalid pixel reuse index')
    for entry in index['entries']:
        if any(entry.get(key) != ctx['task'][key] for key in ('cfg_sha256', 'images_sha256', 'mode')):
            continue
        try:
            old, items, sources = validate_embedding_reuse(ctx, entry)
        except (OSError, ValueError, KeyError, AssertionError):
            continue
        folder = ctx['embed_folder']
        (folder / 'images').mkdir(parents=True, exist_ok=True)
        new_items = []
        for prior in items:
            path = folder / 'images' / Path(prior['path']).name
            shutil.copyfile(prior['path'], path)
            require(digest(path) == prior['image_sha256'], 'copied embedding PNG differs')
            new_items.append(dict(prior, path=str(path)))
        check_sources(sources)
        write_once(folder / 'inputs.json', new_items)
        write_once(folder / 'complete.json', dict(old, identity=ctx['embed_identity'],
            manifest_sha256=digest(folder / 'inputs.json'), reused_embedding_pixels=True,
            embedding_latency_reused=True, cross_session_components=True,
            pixel_reuse_source_hashes=sources, old_detection_results_reused=False))
        return True
    return False


def validate_pixel_reuse(ctx, entry, items):
    """Return pinned attack artifacts only if all codec/image/seed facts agree."""
    task = ctx['task']
    _, old_task, _, folder, sources = old_pixel_context(ctx, entry)
    attack_path = folder / 'attack_complete.json'
    summary = read(attack_path)
    require(summary['complete'] is True and summary['task_id'] == old_task['task_id'] and
            summary['n'] == len(summary['records']) == task['n'], 'incomplete old attack pixels')
    require(summary['manifest_sha256'] == digest(folder / 'attack_inputs.json') and
            summary['attack_spec_sha256'] == canonical_hash(task['attack_spec']), 'old attack manifest or seed changed')
    for item, old in zip(items, summary['records']):
        require(item['index'] == old['index'] and digest(item['path']) == old['input_sha256'] and
                digest(old['input_path']) == old['input_sha256'], 'new and old watermarked PNG bytes differ')
        require(digest(old['output_path']) == old['output_sha256'], 'old attacked PNG bytes changed')
    for path in (attack_path, folder / 'attack_inputs.json'):
        sources[str(path)] = digest(path)
    for old in summary['records']:
        sources[old['input_path']] = old['input_sha256']
        sources[old['output_path']] = old['output_sha256']
    if task['attack_spec']['kind'] == 'extension':
        path = folder / 'attacked/summary.json'
        require(digest(path) == summary['extension_summary_sha256'], 'old extension summary changed')
        ext = read(path)
        require(ext['complete'] is True and ext['n'] == task['n'] and
                ext['attack_spec_sha256'] == task['attack_spec']['extension_spec_sha256'] and
                ext['manifest_sha256'] == summary['manifest_sha256'] and len(ext['records']) == task['n'],
                'old extension provenance differs')
        sources[str(path)] = digest(path)
        for old, record in zip(summary['records'], ext['records']):
            require(all(old[key] == record[key] for key in ('index', 'input_path', 'input_sha256', 'output_sha256')),
                    'old extension image record differs')
            record_path = folder / 'attacked' / f"i{record['index']:05d}.json"
            require(record == read(record_path), 'old extension sidecar differs')
            seed = 42 if task['attack_spec']['spec']['kind'] == 'editing' else 1234 + record['index']
            require(record['seed'] == seed and record['attack_spec_sha256'] == ext['attack_spec_sha256'],
                    'old extension seed or attack differs')
            sources[str(record_path)] = digest(record_path)
        summary = dict(summary, extension=ext)
    return summary, sources


def try_reuse_attack_pixels(ctx, items):
    index_path = os.environ.get('UNIFIED_ATTACK_REUSE_INDEX')
    if not index_path:
        return False
    index = read(index_path)
    require(index.get('schema') == 'unified_attack_pixel_index_v1', 'invalid pixel reuse index')
    task, folder = ctx['task'], ctx['folder']
    for entry in index['entries']:
        if any(entry.get(key) != task[key] for key in ('cfg_sha256', 'images_sha256', 'attack', 'n', 'mode')):
            continue
        try:
            old, sources = validate_pixel_reuse(ctx, entry, items)
        except (OSError, ValueError, KeyError, AssertionError):
            # An ineligible historical artifact is a cache miss, never a live failure.
            continue
        output = folder / 'attacked'
        output.mkdir(parents=True, exist_ok=True)
        records = []
        for item, prior in zip(items, old['records']):
            path = output / f"i{item['index']:05d}.png"
            shutil.copyfile(prior['output_path'], path)
            require(digest(path) == prior['output_sha256'], 'copied attack pixels differ')
            records.append(dict(index=item['index'], input_path=item['path'], input_sha256=digest(item['path']),
                                output_path=str(path), output_sha256=prior['output_sha256']))
        check_sources(sources)
        value = dict(complete=True, task_id=task['task_id'], n=task['n'],
            records=records, manifest_sha256=digest(folder / 'attack_inputs.json'),
            attack_spec_sha256=canonical_hash(task['attack_spec']), total_attack_subprocess_seconds=0,
            attack_model_init_and_io_excluded_from_deployment=True, reused_attack_pixels=True,
            pixel_reuse_source_hashes=sources, old_detection_results_reused=False,
            old_embedding_latency_reused=False)
        if task['attack_spec']['kind'] == 'extension':
            ext_records = []
            for item, prior in zip(items, old['extension']['records']):
                record = dict(prior, input_path=item['path'])
                write_once(output / f"i{item['index']:05d}.json", record)
                ext_records.append(record)
            extension = dict(old['extension'], records=ext_records, manifest_sha256=value['manifest_sha256'],
                             reused_attack_pixels=True, historical_attack_runtime=True)
            write_once(output / 'summary.json', extension)
            value['extension_summary_sha256'] = digest(output / 'summary.json')
        write_once(folder / 'attack_complete.json', value)
        return True
    return False


def attack(context_path):
    ctx = load_context(context_path)
    task, folder = ctx['task'], ctx['folder']
    _, embedding_items = verify_embedding(ctx)
    items = [dict(index=x['index'], path=x['path']) for x in embedding_items[:task['n']]]
    write_once(folder / 'attack_inputs.json', items)
    summary_path = folder / 'attack_complete.json'
    if summary_path.exists():
        verify_attack(ctx)
        return
    if try_reuse_attack_pixels(ctx, items):
        verify_attack(ctx)
        return
    output = folder / 'attacked'
    output.mkdir(exist_ok=True)
    spec = task['attack_spec']
    started = time.perf_counter()
    if spec['kind'] == 'extension':
        subprocess.run([spec['spec']['python'], '-B', str(SOURCE / 'code/attack_batch.py'),
            '--attack', task['attack'], '--manifest', str(folder / 'attack_inputs.json'), '--output', str(output)], check=True)
    elif spec['kind'] == 'normal':
        subprocess.run([PY, '-B', __file__, '_normal', str(context_path)], check=True)
    else:
        input_dir = folder / 'attack_input_pngs'
        input_dir.mkdir(exist_ok=True)
        for item in items:
            path = input_dir / Path(item['path']).name
            if not path.exists():
                path.symlink_to(item['path'])
        subprocess.run(external_command(task, input_dir, output), cwd=CF, check=True)
    elapsed = time.perf_counter() - started
    records = []
    for item in items:
        path = output / f"i{item['index']:05d}.png"
        require(path.exists(), 'attack output missing')
        records.append(dict(index=item['index'], input_path=item['path'], input_sha256=digest(item['path']),
                            output_path=str(path), output_sha256=digest(path)))
    value = dict(complete=True, task_id=task['task_id'], n=task['n'], records=records,
                 manifest_sha256=digest(folder / 'attack_inputs.json'), attack_spec_sha256=canonical_hash(spec),
                 total_attack_subprocess_seconds=elapsed, attack_model_init_and_io_excluded_from_deployment=True,
                 legacy_attack_runtime_includes_initialization=spec['kind'] in ('ctrlregen', 'unmarker'))
    if spec['kind'] == 'extension':
        value['extension_summary_sha256'] = digest(output / 'summary.json')
    write_once(summary_path, value)
    verify_attack(ctx)


def verify_attack(ctx):
    folder, task = ctx['folder'], ctx['task']
    summary = read(folder / 'attack_complete.json')
    require(summary['complete'] is True and summary['task_id'] == task['task_id'], 'foreign attack cache')
    require(summary['n'] == len(summary['records']) == task['n'], 'wrong attack image count')
    require(summary['manifest_sha256'] == digest(folder / 'attack_inputs.json'), 'attack manifest changed')
    require(summary['attack_spec_sha256'] == canonical_hash(task['attack_spec']), 'attack specification mismatch')
    if summary.get('reused_attack_pixels'):
        require(summary.get('old_detection_results_reused') is False and summary.get('old_embedding_latency_reused') is False,
                'legacy measurements cannot be imported with pixels')
        check_sources(summary['pixel_reuse_source_hashes'])
    for expected, item in enumerate(summary['records']):
        require(item['index'] == expected, 'attack index mismatch')
        require(item['input_path'] == str(ctx['embed_folder'] / 'images' / f'i{expected:05d}.png'), 'foreign attack input')
        require(item['output_path'] == str(folder / 'attacked' / f'i{expected:05d}.png'), 'foreign attack output')
        require(digest(item['input_path']) == item['input_sha256'], 'attacked source PNG changed')
        require(digest(item['output_path']) == item['output_sha256'], 'attacked PNG changed')
    if task['attack_spec']['kind'] == 'extension':
        path = folder / 'attacked/summary.json'
        require(digest(path) == summary['extension_summary_sha256'], 'extension attack summary changed')
        ext = read(path)
        spec_hash = task['attack_spec']['extension_spec_sha256']
        require(ext['complete'] and ext['n'] == task['n'] and ext['attack_spec_sha256'] == spec_hash, 'invalid extension provenance')
        require(ext['manifest_sha256'] == summary['manifest_sha256'], 'extension manifest mismatch')
        require(len(ext['records']) == task['n'], 'extension records missing')
        for item, record in zip(summary['records'], ext['records']):
            require(all(item[key] == record[key] for key in ('index', 'input_path', 'input_sha256', 'output_sha256')), 'extension record mismatch')
            require(record == read(folder / 'attacked' / f"i{item['index']:05d}.json"), 'extension per-image JSON mismatch')
            seed = 42 if task['attack_spec']['spec']['kind'] == 'editing' else 1234 + item['index']
            require(record['seed'] == seed and record['attack_spec_sha256'] == spec_hash, 'extension attack seed/spec mismatch')
    return summary


def decode(context_path):
    from PIL import Image
    ctx = load_context(context_path)
    mark_measurement_started(ctx)
    folder, task = ctx['folder'], ctx['task']
    embedded, _ = verify_embedding(ctx)
    attack_summary = verify_attack(ctx)
    images = [Image.open(item['output_path']).convert('RGB') for item in attack_summary['records']]
    require(all(im.size == (512, 512) for im in images), 'attack output resolution changed')
    ns = measurement_namespace()
    started = time.perf_counter()
    comp = composite(ns, task['cfg'])
    metadata = device_metadata(task.get('required_gpu_model'))
    require(metadata['gpu_model'] == embedded['metadata']['gpu_model'], 'embedding and decoder GPU models differ')
    import torch
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - started
    secrets = [comp.secret_for(item['index']) for item in attack_summary['records']]
    deployment = {}
    detailed_images = [{'by_threshold': {}} for _ in images]
    for fpr in task['fprs']:
        theta = UD.threshold_count(task['cfg'], fpr)
        for i in range(task['warmup']):
            UD.decode_detailed(comp, images[i % len(images)], secrets[i % len(images)], alpha=fpr)
        torch.cuda.synchronize()
        flags, bas, scores, bests, times = [], [], [], [], []
        for index, (image, secret) in enumerate(zip(images, secrets)):
            detail, elapsed = time_decode(comp, image, secret, fpr)
            require(detail['threshold_count'] == theta, 'actual and planned unified thresholds differ')
            flags.append(detail['detected']); bas.append(detail['fused_ba'])
            scores.append(detail['accepted_score']); bests.append(detail['best_fragment_ba']); times.append(elapsed)
            detailed_images[index]['by_threshold'][str(theta)] = detail
        deployment[repr(fpr)] = dict(fpr=fpr, threshold_count=theta, detected=flags, fused_ba=bas,
            accepted_score=scores, best_fragment_ba=bests, decode_ms=times)
    cell = cell_of_detailed(task['cfg']['order'], detailed_images)
    result = dict(schema='rigor_final_measurement_v1', complete=True, task_id=task['task_id'],
        selection_sha256=task['selection_sha256'], images_sha256=task['images_sha256'], mode=task['mode'],
        cfg_sha256=task['cfg_sha256'], attack=task['attack'], attack_spec_sha256=canonical_hash(task['attack_spec']),
        n=task['n'], image_indices=list(range(task['n'])), cell=cell, acceptance_protocol=ACCEPTANCE_PROTOCOL,
        context_path=str(Path(context_path).resolve()),
        deployment=deployment, embed_ms=embedded['embed_ms'][:task['n']],
        embedding_quality=embedded['embedding_quality'], metadata=metadata, embedding_metadata=embedded['metadata'],
        latency_composition=dict(embedding_reused=embedded.get('embedding_latency_reused', False),
            cross_session_components=embedded.get('cross_session_components', False),
            decode_remeasured=True, basis='sum of synchronized warm embed and warm unified decode on the same GPU model'),
        model_initialization_seconds=init_seconds, warmup=task['warmup'], no_final_feedback=True,
        measurement_source_hashes=task['measurement_source_hashes'],
        evidence={str(folder / 'attack_complete.json'): digest(folder / 'attack_complete.json'),
                  str(ctx['embed_folder'] / 'complete.json'): digest(ctx['embed_folder'] / 'complete.json'),
                  str(Path(context_path).resolve()): digest(context_path),
                  str(ctx['plan_path']): digest(ctx['plan_path'])})
    validate_result(task, result)
    write_once(folder / 'result.json', result)


def validate_result(task, result, verify_evidence=True):
    check_signed(task, 'task_id')
    require(result.get('schema') == 'rigor_final_measurement_v1' and result.get('complete') is True, 'incomplete final result')
    require(result.get('acceptance_protocol') == ACCEPTANCE_PROTOCOL, 'legacy detection scores cannot be reused')
    for key in ('task_id', 'selection_sha256', 'images_sha256', 'mode', 'cfg_sha256', 'attack', 'n'):
        require(result.get(key) == task[key], f'wrong cached result: {key}')
    require(result['no_final_feedback'] is True and result['warmup'] == task['warmup'] >= 1, 'leakage/warmup protocol mismatch')
    require(result['image_indices'] == list(range(task['n'])), 'wrong final image indices')
    require(result['attack_spec_sha256'] == canonical_hash(task['attack_spec']), 'result attack spec mismatch')
    require(result['measurement_source_hashes'] == task['measurement_source_hashes'], 'result source mismatch')
    n = task['n']
    def vector(values, low=0, high=None):
        require(isinstance(values, list) and len(values) == n, 'wrong measurement vector size')
        require(all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
                    and v >= low and (high is None or v <= high) for v in values), 'invalid measured values')
    vector(result['embed_ms'])
    require(set(result['deployment']) == {repr(x) for x in task['fprs']}, 'missing threshold-specific actual decode')
    for fpr in task['fprs']:
        row = result['deployment'][repr(fpr)]
        require(row['fpr'] == fpr and len(row['detected']) == n and all(isinstance(v, bool) for v in row['detected']), 'invalid deployment verdict')
        vector(row['decode_ms']); vector(row['fused_ba'], high=1)
    sys.path.insert(0, str(UNIFIED_ROOT / 'fixed/code'))
    import capacity_protocol as CP
    require(CP.valid_cell(result['cell'], task['cfg']['order'], n, cfg=task['cfg'], alphas=task['fprs']),
            'invalid unified raw-BA measurement cell')
    require(result['cell']['n'] == n, 'unified cell sample size mismatch')
    for fpr in task['fprs']:
        theta = UD.threshold_count(task['cfg'], fpr)
        deployment = result['deployment'][repr(fpr)]
        require(deployment.get('threshold_count') == theta, 'wrong complete-detector threshold')
        cellrow = result['cell']['by_threshold'][str(theta)]
        for field in ('detected', 'accepted_score', 'best_fragment_ba', 'fused_ba'):
            require(deployment.get(field) == cellrow[field], 'live/final unified readout mismatch: ' + field)
    for metadata in (result['metadata'], result['embedding_metadata']):
        require(metadata['synchronized'] is True and metadata['batch_size'] == 1, 'invalid timing method')
        require(metadata['gpu_model'] == task.get('required_gpu_model', metadata['gpu_model']), 'wrong measurement GPU')
    require(result['metadata']['gpu_model'] == result['embedding_metadata']['gpu_model'], 'mixed timing GPUs')
    require(result['embedding_quality']['n'] == (100 if task['mode'] == 'final' else result['embedding_quality']['n']), 'wrong embedding quality sample')
    if verify_evidence:
        check_sources(result['measurement_source_hashes'])
        require(result['evidence'], 'missing measurement evidence')
        for path, sha in result['evidence'].items():
            require(digest(path) == sha, f'final measurement evidence changed: {path}')
        ctx = load_context(result['context_path'])
        require(ctx['task'] == task, 'measurement context differs from task')
        embedding, _ = verify_embedding(ctx)
        verify_attack(ctx)
        require(result['embed_ms'] == embedding['embed_ms'][:n], 'result embedding timings differ from evidence')
        require(result['embedding_quality'] == describe(embedding['psnr_db']), 'embedding PSNR summary differs from evidence')
    return True


def run_task(plan_path, index, cache_root):
    ctx = context(plan_path, index, cache_root)
    mark_measurement_started(ctx)
    folder = ctx['folder']
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        context_path = folder / 'context.json'
        write_once(context_path, {k: ctx[k] for k in ('plan_path', 'plan_sha256', 'index', 'cache_root')})
        if (folder / 'result.json').exists():
            validate_result(ctx['task'], read(folder / 'result.json'))
            verify_embedding(ctx); verify_attack(ctx)
            return read(folder / 'result.json')
        for phase in ('_embed', '_attack', '_decode'):
            subprocess.run([PY, '-B', '-u', __file__, phase, str(context_path)], check=True)
        return read(folder / 'result.json')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'pixel-index':
        parser = argparse.ArgumentParser(description='Index old PNG contexts without reading detection results')
        parser.add_argument('mode')
        parser.add_argument('--contexts-list', required=True, help='JSON list of existing context.json paths')
        parser.add_argument('--output', required=True)
        args = parser.parse_args()
        value = build_pixel_reuse_index(read(args.contexts_list), args.output)
        print(json.dumps(dict(index=str(args.output), entries=len(value['entries']))))
        return
    if len(sys.argv) > 1 and sys.argv[1].startswith('_'):
        {'_embed': embed, '_attack': attack, '_normal': attack_normal, '_decode': decode}[sys.argv[1]](sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['task'])
    parser.add_argument('plan')
    parser.add_argument('index', type=int)
    parser.add_argument('--cache-root', default=str(ROOT / 'final/cache'))
    args = parser.parse_args()
    result = run_task(args.plan, args.index, args.cache_root)
    print(json.dumps(dict(complete=True, task_id=result['task_id'], n=result['n'])))


if __name__ == '__main__':
    main()
