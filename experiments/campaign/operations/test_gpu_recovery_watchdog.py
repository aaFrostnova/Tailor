"""Temporary mock Slurm responses only; never submits real jobs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('watchdog', Path(__file__).with_name('gpu_recovery_watchdog.py'))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ops = Path(self.temp.name) / 'operations'
        self.ops.mkdir()
        (self.ops / 'gpu_worker.sbatch').write_text('#!/bin/bash\n')
        self.job = dict(job_id='123', submitted_at='2026-09-15T00:00:00+00:00',
                        script=str(self.ops / 'gpu_worker.sbatch'), role='independent_A100_participant',
                        sbatch_sha256=M.digest(self.ops / 'gpu_worker.sbatch'))
        M.atomic(self.ops / 'gpu_worker_job.json', self.job)
        self.calls, self.answers = [], []
        self.w = M.Watchdog(self.ops, self.runner)

    def runner(self, argv, **kwargs):
        self.calls.append(argv)
        expected, answer = self.answers.pop(0)
        self.assertEqual(argv[0], expected)
        if isinstance(answer, Exception):
            raise answer
        return subprocess.CompletedProcess(argv, 0, stdout=answer, stderr='')

    def tick(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.w.tick()

    def test_running_does_not_submit(self):
        self.answers = [('squeue', '123|RUNNING|unified_gpu_participant|\n')]
        self.assertFalse(self.tick())
        self.assertEqual(len(self.calls), 1)

    def test_queue_communication_failure_does_not_submit(self):
        self.answers = [('squeue', subprocess.TimeoutExpired('squeue', 45))]
        with self.assertRaises(subprocess.TimeoutExpired):
            self.tick()
        self.assertEqual(len(self.calls), 1)

    def test_queue_absence_needs_accounting(self):
        self.answers = [('squeue', ''), ('sacct', '')]
        with self.assertRaisesRegex(RuntimeError, 'without unique accounting'):
            self.tick()
        self.assertEqual(len(self.calls), 2)

    def test_normal_timeout_rotates_and_records_job(self):
        self.w.state['consecutive_worker_failures'] = 2
        self.answers = [('squeue', ''), ('sacct', '123|TIMEOUT|unified_gpu_participant|86400||'),
                        ('sbatch', '124')]
        self.assertFalse(self.tick())
        self.assertEqual(M.read(self.ops / 'gpu_worker_job.json')['job_id'], '124')
        self.assertEqual(self.w.state['consecutive_worker_failures'], 0)
        self.assertEqual(self.calls[-1][-1], str(self.ops / 'gpu_worker.sbatch'))

    def test_third_worker_failure_stops_before_submit(self):
        self.w.state['consecutive_worker_failures'] = 2
        self.answers = [('squeue', ''), ('sacct', '123|FAILED|unified_gpu_participant|12||')]
        with self.assertRaisesRegex(RuntimeError, 'three consecutive GPU'):
            self.tick()
        self.assertEqual(len(self.calls), 2)

    def test_ambiguous_submission_cannot_be_repeated(self):
        self.answers = [('squeue', ''), ('sacct', '123|TIMEOUT|unified_gpu_participant|86400||'),
                        ('sbatch', subprocess.TimeoutExpired('sbatch', 45))]
        with self.assertRaises(subprocess.TimeoutExpired):
            self.tick()
        # Simulate a process restart: the durable intent still blocks resubmission.
        self.w = M.Watchdog(self.ops, self.runner)
        self.answers = [('squeue', ''), ('sacct', '')]
        with self.assertRaisesRegex(RuntimeError, 'no new submission'):
            self.tick()
        self.assertEqual(sum(x[0] == 'sbatch' for x in self.calls), 1)

    def test_ambiguous_submission_is_recovered_by_identity(self):
        self.answers = [('squeue', ''), ('sacct', '123|TIMEOUT|unified_gpu_participant|86400||'),
                        ('sbatch', subprocess.TimeoutExpired('sbatch', 45))]
        with self.assertRaises(subprocess.TimeoutExpired):
            self.tick()
        marker = self.w.state['submissions'][0]['marker']
        self.answers = [('squeue', '124|PENDING|unified_gpu_participant|' + marker), ('sacct', '')]
        self.assertFalse(self.tick())
        self.assertEqual(M.read(self.ops / 'gpu_worker_job.json')['job_id'], '124')
        self.assertEqual(sum(x[0] == 'sbatch' for x in self.calls), 1)

    def test_three_query_failures_leave_error_alert(self):
        self.answers = [('squeue', subprocess.TimeoutExpired('squeue', 45)) for _ in range(3)]
        with patch.object(M.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.w.run(1), 2)
        self.assertEqual(M.read(self.ops / 'gpu_recovery_alert.json')['status'], 'error')
        self.assertEqual(len(self.calls), 3)

    def test_completed_main_cancels_only_owned_named_job(self):
        self.w.main_complete = lambda: True
        self.answers = [('squeue', '123|RUNNING|unified_gpu_participant|'), ('scancel', '')]
        self.assertFalse(self.tick())
        self.assertEqual(self.calls[-1], ['scancel', '123'])
        self.answers = [('squeue', ''), ('sacct', '123|CANCELLED|unified_gpu_participant|24||')]
        self.assertTrue(self.tick())

    def test_wrong_job_name_is_never_cancelled(self):
        self.w.main_complete = lambda: True
        self.answers = [('squeue', '123|RUNNING|another_users_work|')]
        with self.assertRaisesRegex(RuntimeError, 'name differs'):
            self.tick()
        self.assertEqual(len(self.calls), 1)

    def test_changed_gpu_script_is_rejected(self):
        (self.ops / 'gpu_worker.sbatch').write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'script hash differs'):
            self.tick()
        self.assertFalse(self.calls)

    def test_stalled_current_owned_job_cancelled_once(self):
        stamp = (M.datetime.datetime.now(M.datetime.timezone.utc) - M.datetime.timedelta(seconds=1000)).isoformat()
        M.atomic(self.ops / 'local_worker_state.json', dict(started_at=self.job['submitted_at'],
                 heartbeat_at=stamp, phase='running', validated_task_keys=[]))
        self.answers = [('squeue', '123|RUNNING|unified_gpu_participant|'), ('scancel', '')]
        self.assertFalse(self.tick())
        self.assertEqual(M.read(self.ops / 'gpu_recovery_stall_alert.json')['job_id'], '123')
        self.answers = [('squeue', '123|RUNNING|unified_gpu_participant|')]
        self.assertFalse(self.tick())
        self.assertEqual(sum(x[0] == 'scancel' for x in self.calls), 1)

    def test_old_allocation_heartbeat_cannot_cancel_new_gpu(self):
        M.atomic(self.ops / 'local_worker_state.json', dict(started_at='2026-09-14T00:00:00+00:00',
                 heartbeat_at='2026-09-14T00:00:00+00:00', phase='running'))
        self.answers = [('squeue', '123|RUNNING|unified_gpu_participant|')]
        self.assertFalse(self.tick())
        self.assertEqual(len(self.calls), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
