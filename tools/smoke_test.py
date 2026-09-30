"""Check that a fresh checkout can answer requests, with nothing installed but the solver.

    python tools/smoke_test.py

Needs z3-solver, numpy and scipy, no models and no GPU. It reads the bundled database, the
bundled requests and the solver, and checks the answers the README quotes. If this passes,
the half of the repository that needs no download works; if it fails, the failure says
which part.
"""
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "solver")]


class Database(unittest.TestCase):
    """The measured database is an input, so its shape is part of the contract."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "inputs", "surrogate_canonical.json")) as f:
            cls.db = json.load(f)

    def test_three_fragments_and_twenty_attacks(self):
        self.assertEqual(self.db["fragments"], ["VINE", "TrustMark", "VideoSeal"])
        self.assertEqual(len(self.db["attacks"]), 20)

    def test_curve_counts(self):
        self.assertEqual(len(self.db["base"]), 60)          # recovery, per fragment and attack
        self.assertEqual(len(self.db["delta"]), 120)        # interference, per ordered pair
        self.assertEqual(len(self.db["frontend"]), 827)     # the geometric stages
        self.assertEqual(len(self.db["perimage"]), 181)     # the scores behind the means

    def test_every_attack_has_a_curve_for_every_fragment(self):
        for fragment in self.db["fragments"]:
            for attack in self.db["attacks"]:
                self.assertIn(f"{fragment}|{attack}", self.db["base"])

    def test_requests_are_the_four_inputs(self):
        with open(os.path.join(ROOT, "inputs", "requests.json")) as f:
            bundle = json.load(f)
        self.assertEqual(bundle["n"], 7321)
        self.assertEqual(len(bundle["requests"]), bundle["n"])
        self.assertEqual(sum(bundle["counts"].values()), bundle["n"])
        self.assertEqual(sorted(bundle["requests"][0]),
                         ["attacks", "fpr", "id", "max_ms", "min_psnr", "scenario"])
        for request in bundle["requests"]:
            self.assertTrue(set(request["attacks"]) <= set(bundle["attacks"]),
                            request["attacks"])
            self.assertGreater(request["fpr"], 0.0)
            self.assertIn(request["scenario"], bundle["scenarios"])

    def test_the_editing_attack_is_the_one_column_with_no_curve(self):
        """The evaluated requests name one attack the bundled database has no curve for, the
        W-Bench editing operator that is not part of this release. Nothing else is missing,
        so the gap is exactly that one column and the release says so."""
        with open(os.path.join(ROOT, "inputs", "requests.json")) as f:
            bundle = json.load(f)
        requested = set()
        for request in bundle["requests"]:
            requested |= set(request["attacks"])
        self.assertEqual(sorted(requested - set(self.db["attacks"])),
                         ["editing_ip2p_s20_v1"])


class Protocol(unittest.TestCase):
    """The budget the request states, and the bit accuracy the solver derives from it."""

    def test_budget_sets_the_accuracy(self):
        import watermark_smt_v2 as W
        for fpr, ba in ((1e-1, 0.57), (1e-2, 0.63), (1e-4, 0.69),
                        (1e-6, 0.74), (1e-9, 0.80), (2 ** -37, 0.90)):
            self.assertAlmostEqual(W.beta_from_fpr(fpr), ba, places=2, msg=f"FPR {fpr:g}")

    def test_a_tighter_budget_never_asks_for_less(self):
        import watermark_smt_v2 as W
        budgets = [1e-1, 1e-2, 1e-4, 1e-6, 1e-9]
        accuracies = [W.beta_from_fpr(b) for b in budgets]
        self.assertEqual(accuracies, sorted(accuracies))

    def test_the_deployment_derives_it_the_same_way(self):
        import capacity_protocol as CP
        import watermark_smt_v2 as W
        request = dict(attacks=["jpeg25"], fpr=1e-6, min_psnr=38.0, max_ms=500.0)
        self.assertAlmostEqual(CP.solver_scenario(request)["min_ba"],
                               W.beta_from_fpr(1e-6), places=9)


class Solver(unittest.TestCase):
    """The answers the README quotes, through the command line it quotes them for."""

    def solve(self, attacks, fpr, min_psnr, max_ms):
        out = subprocess.run(
            [sys.executable, os.path.join("solver", "watermark_smt_v2.py"),
             "--attacks", *attacks, "--fpr", str(fpr),
             "--min_psnr", str(min_psnr), "--max_ms", str(max_ms)],
            cwd=ROOT, capture_output=True, text=True, timeout=600)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        return out.stdout

    def test_compression_only_is_one_fragment(self):
        answer = self.solve(["jpeg25"], 1e-2, 42, 100)
        self.assertIn("VINE", answer)
        self.assertNotIn("UNSAT", answer)

    def test_a_harder_attack_set_grows_the_composition(self):
        answer = self.solve(["jpeg25", "crop75", "regen"], 1e-6, 38, 500)
        self.assertIn("VINE", answer)
        self.assertIn("TrustMark", answer)

    def test_asking_for_too_much_is_unsat(self):
        answer = self.solve(["jpeg25", "crop75", "crop50", "rot9", "regen", "rinse"],
                            1e-6, 46, 200)
        self.assertIn("UNSAT", answer)

    def test_the_budget_alone_can_decide_a_request(self):
        loose = self.solve(["jpeg25", "regen"], 1e-1, 40, 300)
        tight = self.solve(["jpeg25", "regen"], 1e-2, 40, 300)
        self.assertNotIn("UNSAT", loose)
        self.assertIn("UNSAT", tight)

    def test_the_query_line_shows_both(self):
        answer = self.solve(["jpeg25"], 1e-2, 42, 100)
        self.assertIn("FPR<=0.01", answer)
        self.assertIn("ba>=0.63", answer)


class UnmeasuredAttacks(unittest.TestCase):
    """An attack the tables do not cover is refused, not reported impossible."""

    def test_refused_with_the_measured_set(self):
        import watermark_smt_v2 as W
        with self.assertRaises(ValueError) as caught:
            W.build(38.0, 500.0, ["jpeg25", "border20"], 0.63, True, True)
        message = str(caught.exception)
        # the unmeasured clause names only the unmeasured attack, and the measured set is
        # listed after it, so jpeg25 belongs in the message but not in that clause
        self.assertIn("no measured bit accuracy for ['border20']", message)
        self.assertIn("'jpeg25'", message.split("Measured:")[1])

    def test_the_command_line_says_it_without_a_traceback(self):
        out = subprocess.run(
            [sys.executable, os.path.join("solver", "watermark_smt_v2.py"),
             "--attacks", "border20", "--fpr", "1e-2", "--min_psnr", "38", "--max_ms", "500"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertNotIn("Traceback", out.stderr)
        self.assertIn("no measured bit accuracy", out.stderr + out.stdout)

    def test_every_documented_attack_is_actually_accepted(self):
        """The README lists what this command takes; each name has to reach a verdict."""
        import watermark_smt_v2 as W
        documented = """jpeg25 jpeg50 blur noise bright contrast crop90 crop75 crop50 rot9
            rot30 vaeB vaeC regen rinse ctrlregen ctrlregen_s03 ctrlregen_s05 ctrlregen_s07
            unmarker""".split()
        self.assertEqual(sorted(documented), sorted(W.DISCRETE_ATTACKS))
        for attack in documented:
            W.build(34.0, 2000.0, [attack], 0.57, True, True)     # must not raise


class Paths(unittest.TestCase):
    """Nothing outside inputs/ is a fixed path, and the fetcher reports without fetching."""

    def test_every_location_is_overridable(self):
        from src import paths
        with open(os.path.join(ROOT, "src", "paths.py")) as f:
            source = f.read()
        for var, attribute in (("TAILOR_WORKSPACE", "WORKSPACE"),
                               ("TAILOR_MODELS", "MODELS"),
                               ("TAILOR_POOL", "POOL"),
                               ("TAILOR_VINE_REPO", "VINE_REPO"),
                               ("TAILOR_SYNCSEAL_JIT", "SYNCSEAL_JIT")):
            self.assertTrue(hasattr(paths, attribute), attribute)
            self.assertIn(var, source, f"{attribute} is not settable through {var}")

    def test_no_location_points_outside_the_checkout(self):
        from src import paths
        for attribute in ("WORKSPACE", "MODELS", "POOL", "RUN_ROOT"):
            value = getattr(paths, attribute)
            self.assertTrue(os.path.abspath(value).startswith(os.path.abspath(ROOT)),
                            f"{attribute} defaults outside the checkout: {value}")

    def test_fetcher_reports_without_fetching(self):
        out = subprocess.run(
            [sys.executable, os.path.join("tools", "fetch_models.py"), "--check", "--all"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        for component in ("trustmark", "vine", "videoseal", "regen", "ctrlregen",
                          "unmarker", "syncseal", "maskwm", "sd"):
            self.assertIn(component, out.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
