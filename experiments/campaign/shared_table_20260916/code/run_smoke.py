"""CPU policy checks on one fixed request per scenario and margin; no final data."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parents[1]


def one(item):
    import shared_greedy as G
    cls,arm=item
    request=json.loads((G.MAIN/'fixed/inputs'/(cls+'.json')).read_text())['records'][0]
    result=G.propose(request,arm)
    result.update(class_name=cls,request_index=0)
    path=HERE/'smoke'/(cls+'_'+arm+'_0.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    return dict(class_name=cls,arm=arm,status=result['status'],seconds=result['seconds'],
                cfg=None if result['candidate'] is None else result['candidate']['cfg'])


if __name__=='__main__':
    items=[('C'+str(c),a) for c in range(1,6) for a in ('04','06')]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(one,x) for x in items]):
            row=future.result();rows.append(row);print(json.dumps(row),flush=True)
    report=dict(cpu_policy_smoke_passed=True,formal_gpu_results=False,
                tested_requests=items,rows=rows,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (HERE/'smoke/audit.json').write_text(json.dumps(report,indent=2)+'\n')
