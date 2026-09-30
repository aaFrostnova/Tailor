"""CPU integration checks for exact-cache reuse and the preregistered launch gate."""
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import RUN_ROOT
U = Path(RUN_ROOT)
sys.path[:0] = [str(U/'code'), str(U/'fixed/code')]
import capacity_protocol as CP


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = load('_unified_fixed_startup_campaign', U/'fixed/code/campaign.py')
saved = sys.modules.get('campaign')
sys.modules['campaign'] = campaign
controller = load('_unified_fixed_startup_controller', U/'fixed/code/controller.py')
prefetch = load('_unified_startup_prefetch', U/'fixed/code/prefetch.py')
if saved is None:
    sys.modules.pop('campaign', None)
else:
    sys.modules['campaign'] = saved
audit = load('_unified_startup_audit', U/'fallback/code/pipeline_audit.py')


def batches():
    rows = [[k,a,i] for k in campaign.CLASSES for a in campaign.ARMS for i in range(2000)]
    return [rows[i:i+10] for i in range(0, len(rows), 10)]


def measured_cell(cfg, n=100):
    from unified_detector import threshold_count
    data = dict(protocol=CP.CELL_PROTOCOL, n=n, order=cfg['order'], by_threshold={})
    for theta in {threshold_count(cfg,alpha) for alpha in CP.REQUEST_ALPHAS}:
        data['by_threshold'][str(theta)] = dict(threshold_count=theta, detected=[True]*n,
            accepted_score=[1.0]*n, best_fragment_ba=[1.0]*n, fused_ba=[1.0]*n,
            used_geometry=[False]*n, per_fragment_ba={f:[1.0]*n for f in cfg['order']})
    return data


class StartupTests(unittest.TestCase):
    def test_all_twenty_thousand_and_no_duplicate(self):
        rows=batches()
        controller.validate_enumeration_manifest(rows)
        invalid=copy.deepcopy(rows); invalid[-1][-1]=invalid[0][0]
        with self.assertRaises(ValueError): controller.validate_enumeration_manifest(invalid)
        with self.assertRaises(ValueError): controller.validate_enumeration_manifest(rows[:-1])

    def test_freeze_gate_rejects_changed_worker(self):
        old_root,old_code=controller.ROOT,controller.CODE
        try:
            with tempfile.TemporaryDirectory() as td:
                root=Path(td); controller.ROOT=root/'fixed';controller.CODE=root/'fixed/code'
                payload={
                    controller.ROOT/'prepared.json':dict(protocol=CP.PROTOCOL,acceptance_protocol=CP.CELL_PROTOCOL,arm_request_evaluations=20000),
                    controller.ROOT/'plans/enumeration.json':batches(),
                    controller.CODE/'controller.py':{},controller.CODE/'campaign.py':{},controller.CODE/'capacity_protocol.py':{},
                    root/'preflight_audit.json':{},root/'combination_scope/plan.json':{}}
                for p,v in payload.items(): campaign.atomic(p,v)
                hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in payload}
                campaign.atomic(root/'source_hashes.json',hashes)
                self.assertEqual(len(controller.verify_startup()),2000)
                (controller.CODE/'campaign.py').write_text('changed')
                with self.assertRaisesRegex(ValueError,'frozen source changed'): controller.verify_startup()
        finally:
            controller.ROOT,controller.CODE=old_root,old_code

    def test_prefetch_does_not_merge_nearby_strengths(self):
        old={k:getattr(prefetch,k) for k in ('ROOT','RUN_ROOT','CLASSES','ARMS','task_status')}
        try:
            with tempfile.TemporaryDirectory() as td:
                root=Path(td); prefetch.RUN_ROOT=root;prefetch.ROOT=root/'fixed';prefetch.CLASSES=['C1'];prefetch.ARMS=['04']
                prefetch.task_status=lambda manifest,index:Path(manifest).with_suffix('.status.json')
                for i,x in enumerate((.5000000000000001,.5000000000000002)):
                    row=dict(i=i,protocol=CP.PROTOCOL,attacks=['jpeg25'],candidates=[dict(order=['VINE'],fe_on=[],s={'VINE':x})])
                    campaign.atomic(prefetch.ROOT/'checkpoints/C1_04'/f'q{i:04d}.json',row)
                first=prefetch.next_task();campaign.atomic(prefetch.task_status(first,0),dict(complete=True))
                second=prefetch.next_task()
                self.assertIsNotNone(second);self.assertNotEqual(first,second)
                a=campaign.read(campaign.read(first)[0]['plan'])['configs'][0]['key']
                b=campaign.read(campaign.read(second)[0]['plan'])['configs'][0]['key']
                self.assertNotEqual(CP.exact_cfg_id(a),CP.exact_cfg_id(b))
                c=dict(order=['VINE'],fe_on=[],s={'VINE':a[2][0]})
                self.assertEqual(audit.cfg_id(c),CP.exact_cfg_id(a))
        finally:
            for key,value in old.items(): setattr(prefetch,key,value)

    def test_fallback_xenv_reads_new_exact_cache(self):
        source=(U/'fallback/code/campaign.py').read_text()
        function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='xenv_task')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); froot=root/'fallback'; stage=froot/'stages/02'; (stage/'checkpoints').mkdir(parents=True)
            cfg=dict(order=['VINE'],fe_on=[]); key=CP.exact_cfg_key(cfg['order'],[],[.5000000000000001]); cid=CP.exact_cfg_id(key)
            group=root/'certify/g1.json'; manifest=stage/'plans/xenv.json'
            campaign.atomic(group,dict(order=cfg['order'],fe=[],s={'VINE':key[2][0]},cfg_id=cid,attacks={},psnr_db=40))
            campaign.atomic(manifest,[dict(path=str(group),required=['editing_ip2p_s20_v1'])])
            cache=root/'live_topk/cells'/f'{cid}_editing_ip2p_s20_v1_pool300_n100.json'
            campaign.atomic(cache,dict(cell=measured_cell(cfg),psnr_db=40,source='new_exact_cache'))
            def fail(*args): raise AssertionError('complete new cache should avoid another GPU task')
            import fcntl
            namespace=dict(read=campaign.read,atomic=campaign.atomic,exact_cfg_key=CP.exact_cfg_key,exact_cfg_id=CP.exact_cfg_id,
                           ROOT=stage,FROOT=froot,valid_cell=CP.valid_cell,fcntl=fcntl,base=types.SimpleNamespace(xenv_task=fail),PROTOCOL=CP.PROTOCOL)
            exec(compile(ast.Module(body=[function],type_ignores=[]),'<fallback-xenv-test>','exec'),namespace)
            namespace['xenv_task'](manifest,0)
            self.assertTrue(CP.valid_cell(campaign.read(group)['attacks']['editing_ip2p_s20_v1'],cfg['order'],100,cfg=cfg))
            self.assertEqual(campaign.read(cache)['source'],'new_exact_cache')


if __name__=='__main__': unittest.main()
