"""Bind the reviewed unchanged embedding factory to immutable migration evidence."""
import os
from pathlib import Path
import hashlib,json,datetime
U=Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/rigor_editing_only_20260911/unified_ba_20260914')
SC=Path(os.environ.get('TAILOR_WORKSPACE', 'workspace'))
CF=Path(os.environ.get('TAILOR_PROJECT', '.'))

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
s=U/'smoke/embedding_migration';raw=read(s/'rawproof.json');old=read(s/'old.json');new=read(s/'new.json')
assert raw['complete'] and raw['all_png_and_pixels_equal'] and raw['all_secrets_equal'] and raw['all_timing_sanity_checks_pass']
assert old['native_source_hashes']==new['native_source_hashes']
source=dict(old['native_source_hashes'])
for p in [SC/'topk_capacity_output_20260910/code/certify_full_frozen.py']:
 source[str(p)]=digest(p)
for name,sha in source.items():assert digest(name)==sha,name
files=[SC/'rigor_editing_only_20260911/code/final_measurement.py',
 U.parent/'baselines/code/final_measurement.py',U.parent/'single_baselines_20260914/code/final_measurement.py',
 U/'code/final_measurement.py',U/'baselines/code/final_measurement.py',U/'single_baselines/code/final_measurement.py']
measurements={str(p):digest(p) for p in files}
assert measurements[old['measurement_file']]==old['measurement_file_final_sha256']
assert measurements[new['measurement_file']]==new['measurement_file_final_sha256']
weights=set();h=Path(os.environ.get('HF_HOME', '~/.cache/huggingface') + '/hub')
for name in ('models--Shilin-LU--VINE-R-Enc','models--Shilin-LU--VINE-R-Dec','models--stabilityai--sd-turbo'):
 ref=h/name/'refs/main';source[str(ref)]=digest(ref);root=h/name/'snapshots'/ref.read_text().strip()
 for p in root.rglob('*'):
  if p.is_file():
   if p.suffix in ('.safetensors','.bin','.pth','.ckpt'):weights.add(p)
   else:source[str(p)]=digest(p)
for p in (Path(__import__('trustmark').__file__).parent/'models').glob('*_B.ckpt'):weights.add(p)
weights.add(CF/'external/videoseal/ckpts/videoseal_y_256b_img.pth')
for root in (Path(__import__('trustmark').__file__).parent,CF/'external/videoseal/videoseal',Path(os.environ.get('TAILOR_VINE_REPO', 'external/VINE') + '/vine/src')):
 for p in root.rglob('*.py'):source[str(p)]=digest(p)
model_hashes={str(p):digest(p) for p in sorted(weights)}
evidence={str(s/'rawproof.json'):digest(s/'rawproof.json'),**raw['evidence']}
for p,sha in evidence.items():assert digest(p)==sha,p
cert=dict(schema='unified_embedding_migration_v1',complete=True,factory_equivalence_reviewed=True,
 created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=source,
 measurement_files=measurements,evidence=evidence,representatives=raw['pairs'],
 model_weight_hashes=model_hashes,
 model_weight_validation='Pinned by the global campaign source freeze; checked at startup, transitions and final audit. Per-image importer checks the certificate and code/configuration provenance.',
 review='The old extracted factory and the new native factory use the same OursComposite constructor, frontend embedding parameters, exact strengths and secret_for. The unified change affects acceptance only. Seven representative configurations (35 paired images) are byte-identical.',
 timing_scope='Same A100, warm models, synchronized batch-one embedding component. Migration median tolerance is a sanity check, not a statistical equivalence guarantee. Reused embedding plus newly measured decoder is labelled a cross-session sum.',
 historical_weight_hashes_available=False,
 historical_cache_validation='Historical images are checked against their recorded SHA256; model weights are bound at migration, and old/new factory equivalence is measured using these weights.')
cert['certificate_sha256']=canonical(cert)
p=U/'plans/embedding_migration.json'
assert not p.exists()
p.write_text(json.dumps(cert,sort_keys=True,indent=1)+'\n')
print(json.dumps(dict(path=str(p),representatives=len(cert['representatives']),source_files=len(source),weights=len(weights),weight_bytes=sum(p.stat().st_size for p in weights),sha256=digest(p))))
