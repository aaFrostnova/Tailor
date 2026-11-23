import unittest
from types import SimpleNamespace
import z3
from common import add_restriction

class Restrictions(unittest.TestCase):
    def setup_model(self,variant):
        o=z3.Solver();u={f:z3.Bool(f) for f in ['A','B','C']}
        o._svars={f:z3.Real('s'+f) for f in u}
        o.add(z3.Or(*u.values()))
        add_restriction(variant,(o,u))
        return o,u
    def test_single_excludes_pairs_but_retains_strength(self):
        o,u=self.setup_model('SingleOptimized')
        o.push();o.add(u['A'],u['B']);self.assertEqual(o.check(),z3.unsat);o.pop()
        o.add(u['A'],o._svars['A']==z3.RealVal('.73'))
        self.assertEqual(o.check(),z3.sat)
    def test_unit_constrains_every_selected_member(self):
        o,u=self.setup_model('UnitStrength')
        o.add(u['A'],u['B'],z3.Not(u['C']),o._svars['C']==0)
        self.assertEqual(o.check(),z3.sat)
        self.assertEqual(str(o.model().eval(o._svars['A'])),'1')
        self.assertEqual(str(o.model().eval(o._svars['B'])),'1')
        o.add(o._svars['A']==z3.RealVal('.73'))
        self.assertEqual(o.check(),z3.unsat)

if __name__=='__main__':unittest.main()
