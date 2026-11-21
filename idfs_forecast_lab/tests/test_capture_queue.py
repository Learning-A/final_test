from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from idfs_lab.capture_queue import exclusive_lock, tick
from idfs_lab.io import load, save


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.paths=[]
        for name in ('site','protocol','workloads'):
            path=self.base/(name+'.json');save(path,{'fixture':name});self.paths.append(path)
        self.root=self.base/'batch'
        self.lock=self.base/'controller.lock'
        self.healthy=Mock(return_value={'healthy':True,'idle':True})
        self.capture=Mock(return_value=[{'state':'captured_processed'}])

    def tick(self, **kw):
        options=dict(execute=True, check=self.healthy, capture=self.capture,
                     sleep=lambda _:None, clock=lambda:1000)
        options.update(kw)
        return tick(self.root,*self.paths,self.lock,**options)

    def test_plan_does_not_connect_or_create_queue(self):
        self.assertEqual(self.tick(execute=False)['state'],'plan_only')
        self.healthy.assert_not_called();self.capture.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_busy_environment_does_not_consume_attempts(self):
        self.healthy.return_value={'healthy':True,'idle':False}
        self.assertEqual(self.tick()['state'],'waiting_environment')
        self.capture.assert_not_called()
        self.assertEqual(load(self.root/'queue.json')['rounds'][0]['attempts'],[])
        self.assertEqual(self.tick()['state'],'cooldown')

    def test_failed_attempt_retried_separately_then_completed_round_skipped(self):
        self.capture.side_effect=[RuntimeError('fault'),[{'state':'captured_processed'}],[]]
        first=self.tick(cooldown_s=0)
        self.assertEqual(first['state'],'cooldown')
        second=self.tick(cooldown_s=0)
        self.assertEqual(second['attempt'],2)
        roots=[c.args[0] for c in self.capture.call_args_list]
        self.assertNotEqual(roots[0],roots[1])
        third=self.tick(cooldown_s=0)
        self.assertNotEqual(second['run_id'],third['run_id'])
        state=load(self.root/'queue.json')
        self.assertEqual(state['rounds'][0]['attempts'][0]['state'],'failed')
        self.assertEqual(state['rounds'][0]['state'],'captured_pending_qa')

    def test_restart_preserves_interrupted_attempt_and_blocks_while_remote_busy(self):
        self.tick()
        path=self.root/'queue.json';state=load(path)
        state['rounds'][0]['state']='running'
        state['rounds'][0]['attempts'][0]['state']='running'
        save(path,state)
        self.healthy.return_value={'healthy':True,'idle':False}
        self.tick()
        item=load(path)['rounds'][0]
        self.assertEqual(item['attempts'][0]['state'],'interrupted')
        self.assertEqual(len(item['attempts']),1)
        self.assertEqual(self.capture.call_count,1)

    def test_retry_limit_does_not_execute_fourth_attempt(self):
        self.capture.side_effect=RuntimeError('fault')
        for _ in range(3):self.tick(cooldown_s=0)
        self.assertEqual(self.tick(cooldown_s=0)['state'],'retry_limit')
        self.assertEqual(self.capture.call_count,3)

    def test_changed_configuration_refuses_resume(self):
        self.tick()
        save(self.paths[1],{'protocol':'changed'})
        with self.assertRaisesRegex(ValueError,'configuration changed'):
            self.tick()
        self.assertEqual(self.capture.call_count,1)

    def test_lock_prevents_second_controller_and_releases(self):
        with exclusive_lock(self.lock):
            with self.assertRaisesRegex(RuntimeError,'owns the lock'):
                with exclusive_lock(self.lock):pass
        with exclusive_lock(self.lock):pass

    def test_exception_during_preflight_never_starts_capture(self):
        self.healthy.side_effect=TimeoutError('connection')
        self.assertEqual(self.tick()['state'],'waiting_environment')
        self.capture.assert_not_called()
