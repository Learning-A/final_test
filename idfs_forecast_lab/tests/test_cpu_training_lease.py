import json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from contextlib import ExitStack,contextmanager
from idfs_lab.cpu_training_lease import cpu_lease


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
        root=Path(tmp);job=root/'cpu_job';job.mkdir();coord=root/'coordination';coord.mkdir()
        lock=coord/'gpu_training.lock';lock.write_text('previous published record')
        fcntl=types.SimpleNamespace(flock=Mock(),LOCK_EX=2,LOCK_NB=4)
        stack.enter_context(patch('idfs_lab.cpu_training_lease.require_training_host',return_value=job))
        stack.enter_context(patch.dict(sys.modules,{'fcntl':fcntl}))
        stack.enter_context(patch.object(os,'getuid',return_value=coord.stat().st_uid,create=True))
        stack.enter_context(patch.object(os,'O_NOFOLLOW',getattr(os,'O_NOFOLLOW',0),create=True))
        stack.enter_context(patch.dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=''))
        yield coord,lock,fcntl


def lease(**patches):
    args=dict(job_id='cpu_job',code_sha='a'*40,data_hashes={'train':'b'*64})
    args.update(patches)
    return cpu_lease(**args)


class CPUTrainingLeaseTests(unittest.TestCase):
    def test_cpu_metadata_uses_same_lock_and_records_full_lifetime(self):
        with fixture() as (coord,lock,flock):
            with lease() as record:
                self.assertEqual(record['state'],'reserved')
                self.assertIsNone(record['gpu_uuid'])
                self.assertEqual(json.loads((coord/'cpu_job.json').read_text())['data_sha256'],{'train':'b'*64})
                self.assertEqual(json.loads(lock.read_text())['code_sha'],'a'*40)
            self.assertEqual(json.loads((coord/'cpu_job.json').read_text())['state'],'released')
            self.assertEqual(flock.flock.call_count,1)

    def test_busy_lock_does_not_change_record_or_shared_file(self):
        with fixture() as (coord,lock,flock):
            flock.flock.side_effect=BlockingIOError('busy')
            with self.assertRaises(BlockingIOError):
                with lease():self.fail('entered busy lease')
            self.assertFalse((coord/'cpu_job.json').exists())
            self.assertEqual(lock.read_text(),'previous published record')

    def test_child_failure_preserved_and_existing_job_never_overwritten(self):
        with fixture() as (coord,lock,flock):
            with self.assertRaises(RuntimeError):
                with lease():raise RuntimeError('child failed')
            saved=(coord/'cpu_job.json').read_text()
            self.assertEqual(json.loads(saved)['state'],'failed_or_interrupted')
            with self.assertRaises(FileExistsError):
                with lease():self.fail('reused old job')
            self.assertEqual((coord/'cpu_job.json').read_text(),saved)

    def test_budget_hashes_environment_and_host_are_required(self):
        with fixture() as (coord,lock,flock):
            for args in ({'budget_seconds':1801},{'budget_seconds':True},{'code_sha':'pending'},{'data_hashes':{}}):
                with self.assertRaises(ValueError):
                    with lease(**args):self.fail('invalid resource request')
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'):
                with self.assertRaises(ValueError):
                    with lease():self.fail('GPU visible CPU worker')
            self.assertEqual(flock.flock.call_count,0)
        with patch('idfs_lab.cpu_training_lease.require_training_host',side_effect=RuntimeError('Windows forbidden')):
            with self.assertRaises(RuntimeError):
                with lease():self.fail('wrong host')


if __name__=='__main__':unittest.main()
