"""Every C2 rule against the frozen solver itself.

The solver's model for a request (watermark_smt_v2.build in surrogate mode, as campaign.enumerate_one calls it,
plus watermark_smt_topk.no_passenger_clause) is pinned to one on-knot configuration at a time: selection,
strengths, embed order (precedence literals) and stages. The pinned model is satisfiable exactly when C2 in FITTED
mode finds the configuration feasible (ranges and clean floors, stage responsibility and admissibility, latency,
fidelity floor, coverage with mean and per-image rate, no passenger), and then the model's distortion and latency
equal C2's. Stages with no curve on any requested column are not compared: the solver leaves them free but they
are not part of a skeleton and C2 does not enumerate them.
"""
from pathlib import Path
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grid import frozen  # noqa: E402

AVAILABLE = frozen.available()
if AVAILABLE:
    from grid.grid_enumerator import FRAGMENTS, GridSearch, margins, structures  # noqa: E402
    from grid.tables import GridTables  # noqa: E402

REQUESTS = (
    (0, 'initial'),          # signal + VAE; scale and tile responsible
    (4000, '02'),            # geometry; scale optimum
    (6300, 'initial'),       # geometry with angle/resync
)


@unittest.skipUnless(AVAILABLE, 'relocated run root not present')
class SolverEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.ns = frozen.boot()
        cls.TF = GridTables(mode='fitted', ns=cls.ns)
        cls.requests = {r['i']: r for r in json.loads(frozen.requests_path().read_text())}

    def pinned(self, built, solver, order, stage, strengths):
        import z3
        o, u = built[0], built[1]
        solver.push()
        try:
            for f in FRAGMENTS:
                solver.add(u[f] == z3.BoolVal(f in order))
            for f in order:
                solver.add(o._svars[f] == z3.RealVal(strengths[f]))
            pos = {f: i for i, f in enumerate(order)}
            for (f, g), v in o._pvars.items():
                if f in pos and g in pos:
                    solver.add(v == z3.BoolVal(pos[f] < pos[g]))
            for name, v in o._fevars.items():
                solver.add(v == z3.BoolVal(name == stage))
            result = solver.check()
            self.assertIn(result, (z3.sat, z3.unsat))
            if result != z3.sat:
                return False, None, None
            m = solver.model()
            return True, -float(m.eval(built[6]).as_fraction()), float(m.eval(built[7]).as_fraction())
        finally:
            solver.pop()

    def test_pinned_solver_agrees_with_fitted_c2(self):
        import z3
        W, T, CP = self.ns.W, self.ns.T, self.ns.CP
        rng = random.Random(11)
        sat_seen = unsat_seen = 0
        for i, stage in REQUESTS:
            request = self.requests[i]['request']
            rate_margin, mean_margin = margins(self.ns, stage)
            scen = CP.solver_scenario(request, margin=mean_margin)
            built = W.build(**scen, enable_order=True, continuous_strength=True, surrogate=self.TF.sg,
                            rate_margin=rate_margin)
            solver = z3.Solver()
            solver.add(built[0].assertions())
            solver.add(T.no_passenger_clause(built, scen['attacks']))
            gs = GridSearch(self.TF, request, rate_margin=rate_margin, mean_margin=mean_margin)
            self.assertEqual(gs.fe_resp, built[0]._fe_responsible)
            self.assertEqual({f: gs.lo[f] for f in FRAGMENTS}, built[0]._clean_floors)
            comparable = [s for s in structures() if s.stage is None or s.stage in gs.fe_gain]
            configs = []
            feasible = np.flatnonzero(gs.evaluate_positions(0, gs.n_ranked))
            for pos in rng.sample(list(feasible), min(8, len(feasible))):
                s = gs.structures[int(gs.c_sid[pos])]
                kn = gs.c_kn[pos]
                configs.append((s, {f: self.TF.knots[f][int(kn[FRAGMENTS.index(f)])] for f in s.order}))
            for _ in range(14):
                s = rng.choice(comparable)
                configs.append((s, {f: rng.choice(self.TF.knots[f]) for f in s.order}))
            for s, strengths in configs:
                ev = gs.evaluate(s.order, s.stage, strengths, details=False)
                sat, D, ms = self.pinned(built, solver, s.order, s.stage, strengths)
                self.assertEqual(sat, ev['feasible'], (i, s, strengths, ev['violations']))
                if sat:
                    sat_seen += 1
                    self.assertAlmostEqual(D, ev['D_pred'], delta=1e-9 * max(1.0, D))
                    self.assertAlmostEqual(ms, ev['ms'], delta=1e-9)
                else:
                    unsat_seen += 1
        self.assertGreater(sat_seen, 10)
        self.assertGreater(unsat_seen, 10)


if __name__ == '__main__':
    unittest.main()
