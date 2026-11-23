import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import local_worker as LW


class LocalWorkerTests(unittest.TestCase):
    def test_owned_cache_checks_stats_and_rehashes_changes_and_force_full(self):
        with tempfile.TemporaryDirectory(dir=LW.OPS) as folder, patch.dict(os.environ):
            root = Path(folder); ops = root / 'operations'; ops.mkdir()
            relative = ['code/unified_detector.py', 'code/rigor_controller.py',
                        'code/combination_measurement.py', 'fixed/code/campaign.py', 'weight.bin']
            for name in relative:
                p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'frozen')
            hashes = {str(root / name): LW.sha(root / name) for name in relative}
            (root / 'source_hashes.json').write_text(json.dumps(hashes))
            original_sha = LW.sha
            calls = []
            def counted(path):
                calls.append(str(path)); return original_sha(path)
            with patch.object(LW, 'U', root), patch.object(LW, 'OPS', ops), patch.object(LW, '_CACHE_OWNER', False):
                LW.initialize_source_cache()
                with patch.object(LW, 'sha', side_effect=counted):
                    LW.verify_freeze()
                    self.assertEqual(sum(p in hashes for p in calls), len(hashes))
                    calls.clear(); LW.verify_freeze()
                    self.assertEqual(sum(p in hashes for p in calls), 0)
                    # Inherited child credentials can read, not rewrite, the certificate.
                    with patch.object(LW, '_CACHE_OWNER', False):
                        calls.clear(); LW.verify_freeze()
                        self.assertEqual(sum(p in hashes for p in calls), 0)
                    weight = root / 'weight.bin'
                    weight.touch(); calls.clear(); LW.verify_freeze()
                    self.assertEqual(sum(p in hashes for p in calls), 1)
                    calls.clear(); LW.verify_freeze(force_full=True)
                    self.assertEqual(sum(p in hashes for p in calls), len(hashes))
                    os.environ[LW._CACHE_TOKEN_ENV] = 'different-process-token'
                    calls.clear(); LW.verify_freeze()
                    self.assertEqual(sum(p in hashes for p in calls), len(hashes))
                    weight.write_bytes(b'broken')
                    with self.assertRaisesRegex(ValueError, 'frozen source changed'):
                        LW.verify_freeze()

    def test_original_native_lock_competition_and_release_across_processes(self):
        with tempfile.TemporaryDirectory(dir=LW.OPS) as folder:
            lockpath = Path(folder) / 'native.lock'
            script = """import sys,fcntl
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import local_worker as W
try:
 with W.NonblockingNativeLocks():
  with Path(sys.argv[2]).open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   Path(sys.argv[3]).write_text('executed')
except BlockingIOError:
 sys.exit(75)
"""
            output = Path(folder) / 'result'
            argv = [sys.executable, '-B', '-c', script, str(LW.OPS), str(lockpath), str(output)]
            with lockpath.open('a') as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(argv, timeout=5, capture_output=True, text=True)
                self.assertEqual(result.returncode, 75, result.stderr)
                self.assertFalse(output.exists())
            result = subprocess.run(argv, timeout=5, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text(), 'executed')

    def test_wrapper_only_changes_u_exclusive_locks_and_restores_function(self):
        original = fcntl.flock
        with tempfile.TemporaryFile() as external:
            called = []
            with patch.object(fcntl, 'flock', side_effect=lambda fd, operation: called.append(operation)):
                before = fcntl.flock
                with LW.NonblockingNativeLocks():
                    fcntl.flock(external, fcntl.LOCK_EX)
                    fcntl.flock(external, fcntl.LOCK_UN)
                self.assertIs(fcntl.flock, before)
            self.assertEqual(called, [fcntl.LOCK_EX, fcntl.LOCK_UN])
        self.assertIs(fcntl.flock, original)

    def test_nonbusy_oserror_is_not_swallowed(self):
        with tempfile.NamedTemporaryFile(dir=LW.OPS) as handle:
            with patch.object(fcntl, 'flock', side_effect=OSError(5, 'I/O error')):
                with LW.NonblockingNativeLocks():
                    with self.assertRaises(OSError) as found:
                        fcntl.flock(handle, fcntl.LOCK_EX)
                    self.assertEqual(found.exception.errno, 5)

    def test_manifest_and_linked_plan_are_part_of_task_identity(self):
        with tempfile.TemporaryDirectory(dir=LW.OPS) as folder:
            folder = Path(folder)
            plan, manifest = folder / 'plan.json', folder / 'manifest.json'
            plan.write_text('{"cfg":1}')
            manifest.write_text(json.dumps([{'plan': str(plan), 'index': 0}]))
            first = LW.spec_for('measure_task', manifest, 0)
            plan.write_text('{"cfg":2}')
            second = LW.spec_for('measure_task', manifest, 0)
            self.assertNotEqual(LW.task_identity(first), LW.task_identity(second))
            with patch.object(LW, 'native_setup', return_value=None):
                with self.assertRaisesRegex(ValueError, 'linked configuration plan'):
                    LW.validate(first)

    def test_foreign_manifest_rejected_and_wait_pid_not_reused(self):
        with tempfile.NamedTemporaryFile() as handle:
            with self.assertRaises(ValueError):
                LW.spec_for('final', handle.name, 0)
        self.assertFalse(LW.pid_matches(os.getpid(), 'not_the_current_command_a9123'))


if __name__ == '__main__':
    unittest.main()
