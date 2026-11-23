"""Behavioral checks: missing data, refinement, and fallback remain distinct."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import controller as D
from common import atomic

class Flow(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.p=patch.object(D,'output',lambda t:self.root/(t['stage']+'.json'));self.p.start()
        self.t=dict(variant='SingleOptimized',arm='04',cls='C4',i=1,spec_id='x',request={})
        self.c=dict(rank=1,order=['VINE'],fe_on=[],s={'VINE':.5})
    def tearDown(self): self.p.stop();self.temp.cleanup()
    def put(self,stage,cs): atomic(self.root/(stage+'.json'),dict(candidates=cs))
    def inspect(self,passed=False,pending=()):
        cpu,gpu=[],{}
        with patch.object(D,'judge',return_value={'pass':passed,'pending':list(pending)}):
            result=D.inspect_target(self.t,cpu,gpu)
        return result,cpu,gpu
    def test_missing_never_refines(self):
        self.put('initial',[self.c]);r,c,g=self.inspect(pending=['unmarker'])
        self.assertEqual(r['status'],'pending_live');self.assertFalse(c);self.assertEqual(len(g),1)
    def test_failed_complete_refines_once(self):
        self.put('initial',[self.c]);r,c,g=self.inspect()
        self.assertEqual(c[0]['stage'],'initial_patch')
        self.put('initial_patch',[]);r,c,g=self.inspect()
        self.assertEqual(r['status'],'pending_fallback');self.assertEqual(c[0]['stage'],'02')
    def test_initial_unsat_backoff(self):
        self.put('initial',[]);r,c,g=self.inspect();self.assertEqual(c[0]['stage'],'02')
        self.put('02',[self.c]);self.put('02_patch',[])
        r,c,g=self.inspect();self.assertEqual(c[0]['stage'],'00')
    def test_pass_stops(self):
        self.put('initial',[self.c]);r,c,g=self.inspect(passed=True)
        self.assertEqual(r['status'],'accepted');self.assertFalse(c);self.assertFalse(g)
    def test_failed_patch_backs_off(self):
        self.put('initial',[self.c]);self.put('initial_patch',[dict(self.c,rank=4)])
        r,c,g=self.inspect();self.assertEqual(c[0]['stage'],'02')
        self.assertEqual([x['rank'] for x in r['walk']],[1,4])
    def test_pending_patch_blocks_backoff(self):
        self.put('initial',[self.c]);self.put('initial_patch',[dict(self.c,rank=4)])
        cpu,gpu=[],{}
        def judge(req,c):return {'pass':False,'pending':['noise'] if c['rank']==4 else []}
        with patch.object(D,'judge',side_effect=judge):r=D.inspect_target(self.t,cpu,gpu)
        self.assertEqual(r['status'],'pending_live');self.assertFalse(cpu)
    def test_06_retries_04_first_after_live_failure(self):
        self.t['arm']='06';self.put('initial',[self.c]);self.put('initial_patch',[])
        r,c,g=self.inspect();self.assertEqual(c[0]['stage'],'04')
    def test_zero_margin_exhaustion_terminates(self):
        for stage in ['initial','02','00']:
            self.put(stage,[self.c]);self.put(stage+'_patch',[])
        r,c,g=self.inspect();self.assertEqual(r['status'],'live_exhausted');self.assertFalse(c);self.assertFalse(g)
    def test_backoff_does_not_change_user_request(self):
        self.t['request']={'attacks':['noise'],'fpr':1e-9,'min_psnr':35.,'max_ms':2000.}
        self.put('initial',[self.c]);self.put('initial_patch',[])
        r,c,g=self.inspect();self.assertEqual(c[0]['request'],self.t['request'])

if __name__=='__main__': unittest.main()
