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
        self.t=dict(variant='Full',arm='04',cls='C4',i=1,spec_id='x',request={})
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
        self.assertEqual(r['status'],'live_exhausted');self.assertFalse(c)
    def test_initial_unsat_backoff_only(self):
        self.put('initial',[]);r,c,g=self.inspect();self.assertEqual(c[0]['stage'],'02')
        self.put('02',[self.c]);self.put('02_patch',[])
        r,c,g=self.inspect();self.assertEqual(c[0]['stage'],'00')
    def test_greedy_stall_is_not_unsat(self):
        self.t['variant']='G_shared';self.put('initial',[])
        r,c,g=self.inspect();self.assertEqual(r['status'],'greedy_stalled');self.assertFalse(c)
    def test_s1_has_no_patch(self):
        self.t['variant']='S1';self.put('initial',[self.c]);r,c,g=self.inspect()
        self.assertEqual(r['status'],'live_exhausted');self.assertFalse(c)
    def test_pass_stops(self):
        self.put('initial',[self.c]);r,c,g=self.inspect(passed=True)
        self.assertEqual(r['status'],'accepted');self.assertFalse(c);self.assertFalse(g)

if __name__=='__main__': unittest.main()
