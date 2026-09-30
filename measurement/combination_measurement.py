"""Execute one frozen combination-audit task, with isolated attack environments."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

from combination_evaluation import (ROOT, EXT, OBS_SCHEMA, check_observation, check_plan,
                                    digest, file_digest, read, write_new)
from unified_measurement import cell_of_detailed
import unified_detector as UD

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PYTHON

PY = PYTHON
# Embeddings are reused from an earlier campaign rather than recomputed, so this phase has
# to be told which one: the directory holding its phases, and the reader whose digest the
# reuse is checked against. A first campaign has neither, and recomputes.
OLD_PHASES = Path(os.environ.get('TAILOR_REUSE_PHASES', ''))
OLD_READER = Path(os.environ.get('TAILOR_REUSE_READER', ''))
ALPHAS = (.1, .01, 1e-4, 1e-6, 1e-9, 2**-37)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    os.replace(tmp, path)


def validate_sources(plan):
    check_plan(plan)
    for source, expected in (('table_source', 'table_sha256'), ('attack_specs_source', 'attack_specs_sha256'),
                             ('image_splits_source', 'image_splits_sha256')):
        if not plan.get(source) or digest(read(plan[source])) != plan[expected]:
            raise ValueError(f'preregistered source changed: {source}')
    for spec in plan['attack_specs'].values():
        for path, expected in spec['source_hashes'].items():
            if file_digest(path) != expected:
                raise ValueError(f'pinned attack source changed: {path}')
    for path, expected in plan.get('execution_source_hashes', {}).items():
        if file_digest(path) != expected:
            raise ValueError(f'pinned combination executor changed: {path}')


def validate_inputs(phase, manifest, *, check_files=True):
    if len(manifest) != 100 or [r['index'] for r in manifest] != phase['image_ids']:
        raise ValueError('embedded input identities differ from preregistration')
    for row, source in zip(manifest, phase['images']):
        if row['cover'] != source['path'] or row['cover_sha256'] != source['sha256']:
            raise ValueError('embedded cover identity differs')
        if check_files and (file_digest(row['path']) != row['image_sha256']
                            or file_digest(row['cover']) != source['sha256']):
            raise ValueError('embedded or cover image changed')
    return manifest


def validate_attack_records(phase, summary, manifest, *, check_files=True, folder=None):
    if (not summary.get('complete') or summary.get('n') != 100
            or summary.get('attack') != phase['task']['attack']
            or summary.get('attack_spec_sha256') != phase['task']['attack_spec_sha256']):
        raise ValueError('attack batch does not match the frozen task')
    records = summary['records']
    if len(records) != 100 or sorted(r['index'] for r in records) != sorted(phase['image_ids']):
        raise ValueError('attack image identities incomplete or duplicated')
    by_id = {r['index']: r for r in records}
    ordered = []
    spec = phase['attack_spec']
    for row in manifest:
        record = by_id[row['index']]
        expected_seed = 42 if spec['kind'] == 'editing' else spec['seed_base'] + row['index']
        if (record['input_path'] != row['path'] or record['input_sha256'] != row['image_sha256']
                or record['attack_spec_sha256'] != phase['task']['attack_spec_sha256']
                or record['attack'] != phase['task']['attack'] or record['seed'] != expected_seed):
            raise ValueError('attack record input, seed, or specification mismatch')
        if check_files and record['output_sha256'] != file_digest(Path(folder)/f"i{row['index']:05d}.png"):
            raise ValueError('attacked image changed')
        ordered.append(record)
    return ordered


def _frozen():
    sys.path.insert(0, str(EXT/'code'))
    from frozen_measurement import load
    ns = load()
    ns['embedding_factory_sha256'] = ns['source_sha256']
    ns['read'] = lambda comp, order, img, iid, tx: UD.decode_thresholds_detailed(comp, img, (iid, tx), ALPHAS)
    ns['cell_of'] = cell_of_detailed
    ns['source_sha256'] = digest(dict(embedding_factory=ns['embedding_factory_sha256'],
        decoder=file_digest(UD.__file__), cell=file_digest(ROOT/'code/unified_measurement.py')))
    return ns


def validate_reuse(done):
    reuse = done.get('pure_image_reuse')
    if not reuse:
        return
    if reuse.get('old_decoder_outputs_read') is not False:
        raise ValueError('old decoder evidence cannot be imported')
    for path, expected in reuse['evidence'].items():
        if file_digest(path) != expected:
            raise ValueError('reused embedding/attack provenance changed')
    if reuse['embedding_factory_sha256'] != file_digest(OLD_READER):
        raise ValueError('embedding factory changed')


def reuse_images(phase_path):
    """Import only immutable embedding/attack artifacts, never old BA results."""
    phase_path = Path(phase_path)
    phase = read(phase_path)
    folder = phase_path.parent
    old = OLD_PHASES/phase['task']['id']
    required = [old/'phase.json',old/'inputs.json',old/'embed_complete.json',old/'attacked/summary.json']
    if not all(p.exists() for p in required):
        return False
    previous, manifest, embedded, summary = map(read, required)
    if any(previous[k] != phase[k] for k in ('task','images','image_ids','attack_spec')):
        return False
    if (embedded.get('phase') != previous or embedded.get('complete') is not True
            or embedded.get('reader_sha256') != file_digest(OLD_READER)
            or embedded.get('manifest_sha256') != file_digest(required[1])
            or summary.get('manifest_sha256') != file_digest(required[1])):
        return False
    validate_inputs(previous, manifest)
    validate_attack_records(previous, summary, manifest, folder=old/'attacked')
    (folder/'attacked').mkdir(exist_ok=True)
    for row in manifest:
        source = old/'attacked'/f"i{row['index']:05d}.png"
        target = folder/'attacked'/source.name
        if target.exists():
            if file_digest(target) != file_digest(source):
                raise ValueError('existing imported attack image differs')
        else:
            target.symlink_to(source)
    for source, target in ((required[1],folder/'inputs.json'),(required[3],folder/'attacked/summary.json')):
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('existing imported manifest differs')
        shutil.copyfile(source,target)
    done = dict(embedded, phase=phase, pure_image_reuse=dict(
        old_decoder_outputs_read=False, embedding_latency_reused=False,
        embedding_factory_sha256=file_digest(OLD_READER),
        evidence={str(p):file_digest(p) for p in required}))
    validate_reuse(done)
    atomic(folder/'embed_complete.json',done)
    print('COMBINATION_PIXELS_REUSED', phase['task']['id'], flush=True)
    return True


def _comp(ns, phase):
    cfg = phase['task']['config']
    return ns['composite_for'](cfg['order'], cfg['fe_on'], [cfg['s'][f] for f in cfg['order']])


def embed(phase_path):
    import numpy as np
    from PIL import Image
    phase_path = Path(phase_path)
    phase = read(phase_path)
    ns = _frozen()
    comp = _comp(ns, phase)
    folder = phase_path.parent/'embed'
    folder.mkdir(exist_ok=True)
    manifest, psnrs = [], []
    for source in phase['images']:
        if file_digest(source['path']) != source['sha256']:
            raise ValueError('preregistered cover bytes changed')
        with Image.open(source['path']) as raw:
            cover = raw.convert('RGB').resize((512, 512), Image.BICUBIC)
        if hashlib.sha256(cover.tobytes()).hexdigest() != source['pixel_sha256']:
            raise ValueError('preregistered cover pixels changed')
        watermarked, _ = comp.embed(cover, source['index'])
        if watermarked.size != (512, 512):
            watermarked = watermarked.resize((512, 512))
        path = folder/f"i{source['index']:05d}.png"
        tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
        watermarked.save(tmp, format='PNG')
        os.replace(tmp, path)
        manifest.append(dict(index=source['index'], path=str(path), cover=source['path'],
                             cover_sha256=source['sha256'], image_sha256=file_digest(path)))
        mse = float(np.mean((np.asarray(cover, float)-np.asarray(watermarked, float))**2))
        psnrs.append(float(10*np.log10(255**2/max(mse, 1e-9))))
    validate_inputs(phase, manifest)
    atomic(phase_path.parent/'inputs.json', manifest)
    atomic(phase_path.parent/'embed_complete.json', dict(phase=phase, complete=True, n=100,
        psnr_db=float(np.mean(psnrs)), per_image_psnr_db=psnrs,
        manifest_sha256=file_digest(phase_path.parent/'inputs.json'), reader_sha256=ns['source_sha256']))


def decode(phase_path):
    import torch
    from PIL import Image
    phase_path = Path(phase_path)
    phase = read(phase_path)
    manifest = validate_inputs(phase, read(phase_path.parent/'inputs.json'))
    summary_path = phase_path.parent/'attacked/summary.json'
    summary = read(summary_path)
    if summary['manifest_sha256'] != file_digest(phase_path.parent/'inputs.json'):
        raise ValueError('attack input manifest changed')
    records = validate_attack_records(phase, summary, manifest, folder=summary_path.parent)
    ns = _frozen()
    comp = _comp(ns, phase)
    rows, timings = [], []
    for item in records:
        with Image.open(summary_path.parent/f"i{item['index']:05d}.png") as raw:
            attacked = raw.convert('RGB')
        if attacked.size != (512, 512):
            raise ValueError('attacked dimensions changed')
        iid, tx = comp.secret_for(item['index'])
        torch.cuda.synchronize()
        start = time.perf_counter()
        rows.append(ns['read'](comp, phase['task']['config']['order'], attacked, iid, tx))
        torch.cuda.synchronize()
        timings.append((time.perf_counter()-start)*1000)
    result = dict(phase=phase, complete=True, cell=ns['cell_of'](phase['task']['config']['order'], rows),
                  decode_ms=timings, attack_summary=str(summary_path),
                  acceptance_protocol=UD.PROTOCOL, shared_measurement_is_deployment_latency=False,
                  attack_summary_sha256=file_digest(summary_path), reader_sha256=ns['source_sha256'])
    atomic(phase_path.parent/'decoded.json', result)


def task(plan_path, index):
    plan = read(plan_path)
    validate_sources(plan)
    if type(index) is not int or not 0 <= index < len(plan['tasks']):
        raise ValueError('task index outside preregistered design')
    item = plan['tasks'][index]
    folder = ROOT/'combination'/plan['plan_id']/'phases'/item['id']
    output = folder.parent.parent/'raw'/f"{item['id']}.json"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'task.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            observation = check_observation(plan, item, read(output))
            for evidence in observation['measurement']['evidence']:
                if file_digest(evidence['path']) != evidence['sha256']:
                    raise ValueError('completed measurement evidence changed')
            return observation
        phase = dict(plan_id=plan['plan_id'], task=item, images=plan['images'], image_ids=plan['image_ids'],
                     attack_spec=plan['attack_specs'][item['attack']])
        phase_path = folder/'phase.json'
        if phase_path.exists():
            if read(phase_path) != phase:
                raise ValueError('phase differs from preregistration')
        else:
            write_new(phase_path, phase)
        if not (folder/'embed_complete.json').exists():
            if not reuse_images(phase_path):
                subprocess.run([PY, '-u', __file__, 'embed', str(phase_path)], check=True)
        done = read(folder/'embed_complete.json')
        validate_reuse(done)
        if done['phase'] != phase or not done['complete'] or done['manifest_sha256'] != file_digest(folder/'inputs.json'):
            raise ValueError('embedding evidence mismatch')
        manifest = validate_inputs(phase, read(folder/'inputs.json'))
        result_path = folder/'decoded.json'
        if not result_path.exists():
            spec = phase['attack_spec']
            if not (folder/'attacked/summary.json').exists():
                subprocess.run([spec['python'], '-u', str(EXT/'code/attack_batch.py'), '--attack', item['attack'],
                                '--manifest', str(folder/'inputs.json'), '--output', str(folder/'attacked')], check=True)
            subprocess.run([PY, '-u', __file__, 'decode', str(phase_path)], check=True)
        result = read(result_path)
        if (result['phase'] != phase or not result['complete']
                or result.get('acceptance_protocol') != UD.PROTOCOL):
            raise ValueError('decode phase mismatch')
        summary_path = Path(result['attack_summary'])
        if file_digest(summary_path) != result['attack_summary_sha256']:
            raise ValueError('decoded attack evidence changed')
        validate_attack_records(phase, read(summary_path), manifest, folder=summary_path.parent)
        evidence_files = [phase_path, folder/'inputs.json', folder/'embed_complete.json', summary_path, result_path]
        observation = dict(schema=OBS_SCHEMA, complete=True, plan_id=plan['plan_id'], task=item,
            image_ids=plan['image_ids'], cell=result['cell'], decode_ms=result['decode_ms'],
            psnr_db=done['psnr_db'], per_image_psnr_db=done['per_image_psnr_db'],
            acceptance_protocol=UD.PROTOCOL, shared_measurement_is_deployment_latency=False,
            measurement=dict(executor_sha256=file_digest(__file__), reader_sha256=result['reader_sha256'],
                image_splits_sha256=plan['image_splits_sha256'], attack_spec_sha256=item['attack_spec_sha256'],
                completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                evidence=[dict(path=str(p), sha256=file_digest(p)) for p in evidence_files]))
        check_observation(plan, item, observation)
        write_new(output, observation)
        print('COMBINATION_TASK_COMPLETE', index, item['id'], item['attack'], flush=True)
        return observation


if __name__ == '__main__':
    if sys.argv[1] == 'task':
        task(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == 'embed':
        embed(sys.argv[2])
    elif sys.argv[1] == 'decode':
        decode(sys.argv[2])
    else:
        raise SystemExit('expected task <plan> <index>, embed <phase>, or decode <phase>')
