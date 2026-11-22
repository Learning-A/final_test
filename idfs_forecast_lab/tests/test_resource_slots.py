"""Real Linux flock compatibility; no models, GPUs or training resources used."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from idfs_lab.resource_slots import slot_locks, run_leased, OWNERS


@unittest.skipUnless(sys.platform == 'linux', 'real flock tests require Linux')
class ResourceSlots(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.coord = Path(self.tmp.name)
        (self.coord / 'gpu_training.lock').touch()
        self.a, self.b = list(OWNERS)
        self.ga, self.gb = OWNERS[self.a][1], OWNERS[self.b][1]

    def test_distinct_slots_parallel_same_slot_and_owner_rejected(self):
        with slot_locks(self.coord, self.ga, self.a):
            with slot_locks(self.coord, self.gb, self.b):
                pass
            with self.assertRaises(BlockingIOError):
                with slot_locks(self.coord, self.ga, self.b):
                    pass
            with self.assertRaises(BlockingIOError):
                with slot_locks(self.coord, self.gb, self.a):
                    pass

    def test_legacy_exclusive_blocks_new_and_new_blocks_legacy(self):
        import fcntl
        with (self.coord / 'gpu_training.lock').open('r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                with slot_locks(self.coord, self.gb, self.b):
                    pass
            fcntl.flock(f, fcntl.LOCK_UN)
            with slot_locks(self.coord, self.gb, self.b):
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_child_keeps_all_locks_after_supervisor_closes(self):
        import fcntl
        with slot_locks(self.coord, self.gb, self.b) as fds:
            child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                     pass_fds=fds, stdin=subprocess.PIPE)
        try:
            with self.assertRaises(BlockingIOError):
                with slot_locks(self.coord, self.gb, self.b):
                    pass
            with (self.coord / 'gpu_training.lock').open('r+') as f:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            child.communicate(timeout=5)
        with slot_locks(self.coord, self.gb, self.b):
            pass

    def test_lock_symlink_rejected_and_gate_contents_preserved(self):
        gate = self.coord / 'gpu_training.lock'
        gate.write_text('legacy record')
        with slot_locks(self.coord, self.gb, self.b):
            self.assertEqual(gate.read_text(), 'legacy record')
        gate.unlink()
        gate.symlink_to(self.coord / 'missing')
        with self.assertRaises(OSError):
            with slot_locks(self.coord, self.gb, self.b):
                pass

    def test_supervisor_records_failure_timeout_and_releases_after_exit(self):
        import json
        env = {k: '4' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}
        env['CUDA_VISIBLE_DEVICES'] = ''
        for name, script, error in (
                ('fail', 'raise SystemExit(3)', RuntimeError),
                ('timeout', 'import time; time.sleep(20)', subprocess.TimeoutExpired)):
            job = self.coord / name
            job.mkdir()
            with patch('idfs_lab.resource_slots.require_training_host', return_value=job), \
                    patch('idfs_lab.resource_slots.COORD', self.coord), patch.dict(os.environ, env):
                with self.assertRaises(error):
                    run_leased([sys.executable, '-c', script], job_id=name, owner_thread=self.b,
                               code_sha='a'*40, data_hashes={'test':'b'*64}, budget_seconds=1)
            state = json.loads((self.coord / (name+'.json')).read_text())
            self.assertEqual(state['state'], 'timed_out' if name == 'timeout' else 'failed_or_interrupted')
            with self.assertRaises(ProcessLookupError):
                os.kill(state['pid'], 0)
            with slot_locks(self.coord, self.gb, self.b):
                pass
