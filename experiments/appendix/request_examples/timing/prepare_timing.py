"""CPU preparation for fresh encoder+decoder timing of the three request->solution examples.

Adapted from qualitative_timing_20260925/prepare.py; worker.py is an unchanged copy.
Only the selected TAILOR-F (m0 = 0.06) configurations are timed.  Attacks are
realized on the frozen native encodings: editing reuses the cached live images of
the exact configuration; VAE and regeneration attacks are prepared on the GPU by
worker.py; all other attacks are realized here on the CPU (seed 42 + image index).
"""
from pathlib import Path
import hashlib, json, os, sys, random
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
EX = OUT.parent
B = EX.parent
U = B / 'results/rigor_editing_only_20260911/unified_ba_20260914'
CF = Path(os.environ.get('TAILOR_PROJECT', '.'))
sys.path[:0] = [str(B / 'qualitative_edge_20260925'), str(CF), str(CF / 'external/WatermarkAttacker'),
                str(U / 'shared_table_20260916/priority_20260917/code')]
from image_sources import load_watermarked, configuration_id

CACHED = {'editing_ip2p_s20_v1', 'ctrlregen_s03', 'ctrlregen_s05', 'unmarker'}
GPU_PREP = {'vaeB', 'vaeC', 'regen', 'rinse2x'}


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p, x):
    t = p.with_suffix(p.suffix + '.tmp'); t.write_text(json.dumps(x, indent=2) + '\n'); t.replace(p)


def prepare_task(payload):
    ti, t, total = payload
    from src.attacks import attack_pil_any
    folder = OUT / 'tasks' / t['cfg_id']; folder.mkdir(parents=True, exist_ok=True)
    wp = folder / 'watermarked.npy'
    if not wp.exists():
        np.save(wp, np.stack([load_watermarked(t['cfg_id'], i) for i in range(100)]), allow_pickle=False)
    t['watermarked_npy'] = str(wp); t['watermarked_sha256'] = sha(wp)
    watermarked = np.load(wp, mmap_mode='r')
    for a, ar in t['attacks'].items():
        ap = folder / f'attack_{a}.npy'; mp = folder / f'attack_{a}.json'
        if a in CACHED:
            gp = Path(ar['cell_source']).resolve(); g = read(gp)
            assert configuration_id(dict(order=g['order'], fe_on=g['fe'], s=g['s'])) == t['cfg_id']
            fp = gp.with_suffix('') / f'att_{a}'
            files = [fp / f'i{i:05d}.png' for i in range(ar['n'])]
            assert all(p.is_file() for p in files), fp
            ar['image_paths'] = [str(p) for p in files]
            ar['image_sha256'] = [sha(p) for p in files]
            ar['source_group'] = str(gp); ar['source_group_sha256'] = sha(gp)
            ar['pixel_source'] = 'frozen exact-configuration live attack images'
        elif a in GPU_PREP:
            ar['npy'] = str(ap); ar['metadata'] = str(mp); ar['needs_gpu_preparation'] = True
            ar['pixel_source'] = 'fresh attack realization on frozen native encodings, seed 42+image index'
        else:
            if not mp.exists():
                arr = np.empty((ar['n'], 512, 512, 3), dtype=np.uint8)
                for i in range(ar['n']):
                    random.seed(42 + i); np.random.seed(42 + i)
                    im = attack_pil_any(a, Image.fromarray(watermarked[i]), dev='cpu')
                    if im.size != (512, 512): im = im.resize((512, 512))
                    arr[i] = np.asarray(im.convert('RGB'))
                np.save(ap, arr, allow_pickle=False)
                atomic(mp, dict(complete=True, n=ar['n'], cfg_id=t['cfg_id'], attack=a, sha256=sha(ap),
                                seed_policy='42+image index', prepared_on='CPU'))
            assert read(mp)['sha256'] == sha(ap)
            ar['npy'] = str(ap); ar['metadata'] = str(mp); ar['needs_gpu_preparation'] = False
            ar['pixel_source'] = 'fresh deterministic attack realization on frozen native encodings'
    atomic(folder / 'task.json', t)
    print(json.dumps(dict(task=ti, total=total, cfg=t['cfg_id'][:12], attacks=len(t['attacks']),
                          phase='cpu_prepared')), flush=True)
    return t


def main():
    import common as C
    os.environ['TQDM_DISABLE'] = '1'
    sel = read(EX / 'selected_examples.json')
    tasks = []
    for ex in sel['examples']:
        cfg = ex['cfg']
        assert configuration_id(cfg) == ex['cfg_id']
        attacks = {}
        for a in ex['request']['attacks']:
            n = 30 if a == 'unmarker' else 100
            rec = C.load_cell(cfg, a)
            attacks[a] = dict(n=n, alphas=[ex['request']['fpr']], cell_source=str(rec.get('source')))
        tasks.append(dict(cfg_id=ex['cfg_id'], cfg=cfg, attacks=attacks,
                          bindings=[dict(example=ex['example'], method='TAILOR-F', alpha=ex['request']['fpr'])]))
    images = read(B / 'ssim_preview_20260924/index/ssim_main_both_extended_index.json')['images']
    sources = [U / 'code/final_measurement.py', U / 'code/unified_detector.py', CF / 'solver/eval_matrix.py',
               CF / 'solver/composite_external_eval.py', CF / 'src/attacks.py']
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks = list(pool.map(prepare_task, [(i, t, len(tasks)) for i, t in enumerate(tasks)]))
    atomic(OUT / 'plan.json', dict(schema='fresh_qualitative_latency_v1', tasks=tasks,
        cases=[dict(example=e['example'], request=e['request']) for e in sel['examples']], images=images,
        source_sha256={str(p): sha(p) for p in sources}, warmup=3, repeats=3,
        required_gpu='NVIDIA A100-SXM4-80GB', latency_scope='synchronized warm encoder + single-alpha deployed decoder',
        original_requests_and_budgets_unchanged=True, old_latency_values_reused=False,
        attack_generation_excluded_from_latency=True))
    print('PREPARATION_COMPLETE', len(tasks), flush=True)


if __name__ == '__main__':
    main()
