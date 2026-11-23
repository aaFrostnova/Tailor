from pathlib import Path
import json,subprocess,sys,time,os,datetime
U=Path(os.environ.get('TAILOR_OUTPUTS', 'outputs') + '/rigor_editing_only_20260911/unified_ba_20260914')
PY=os.environ.get('TAILOR_PYTHON', 'python')
sys.path.insert(0,str(U/'code'))
from rigor_protocol import atomic,digest,read
from final_holdout import summarize
from combination_evaluation import check_observation,check_evidence
started=time.monotonic()
while not (U/'smoke/embedding_migration/rawproof.json').exists():
 if time.monotonic()-started>900: raise RuntimeError('embedding probe did not finish')
 time.sleep(5)
assert read(U/'smoke/embedding_migration/rawproof.json')['complete']
plan=U/'combination_scope/plan.json'
subprocess.run([PY,'-B','-u',str(U/'code/combination_measurement.py'),'task',str(plan),'0'],check=True)
p=read(plan);task=p['tasks'][0];raw=U/'combination'/p['plan_id']/'raw'/f"{task['id']}.json"
obs=check_observation(p,task,read(raw));check_evidence(p,task,obs)
smoke=U/'plans/smoke_editing_revision_tasks.json';cache=U/'smoke/pipeline/cache'
for i in range(len(read(smoke)['tasks'])):
 subprocess.run([PY,'-B','-u',str(U/'code/final_measurement.py'),'task',str(smoke),str(i),'--cache-root',str(cache)],check=True)
result=U/'smoke/pipeline/result.json'
r=summarize(U/'plans/smoke_editing_revision_selection.json',smoke,cache,result)
assert r['passed'] is True
atomic(U/'smoke/pipeline/report.json',dict(passed=True,acceptance_protocol='unified_raw_ba_v1',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),combination_images=100,final_smoke_images=2,final_smoke_tasks=len(read(smoke)['tasks']),evidence={str(q):digest(q) for q in [raw,result,smoke,plan]},main_final_outcomes_used=False))
print('UNIFIED_PIPELINE_SMOKE_PASSED',flush=True)
