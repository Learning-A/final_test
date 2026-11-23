import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[2]/'scripts/run_v2_i1_study.py'
spec=importlib.util.spec_from_file_location('i1_runner',path)
i1=importlib.util.module_from_spec(spec)
spec.loader.exec_module(i1)


class I1Metrics(unittest.TestCase):
    def test_formal_mape_excludes_only_zero(self):
        y=np.tile(np.array([[0]*8,[1]*8]),(5,1)).astype(float)
        p=y+1
        workloads=sum(([f'w{k}']*2 for k in range(5)),[])
        got=i1.summary(np,p,y,workloads)
        self.assertEqual(got['macro_mape_percent'],100)
        self.assertEqual(got['groups_mape_lt10'],0)
        self.assertEqual(got['groups']['w0']['idfs44']['mae'],1)
        self.assertEqual(got['summed_node_cluster']['w0']['mae'],8)

    def test_screen_rejects_lower_mape_with_workload_mae_regression(self):
        baseline=dict(macro_mape_percent=100,worst_group_mape_percent=200,workload_mean_node_mae={'w':10})
        candidate=dict(macro_mape_percent=80,worst_group_mape_percent=190,workload_mean_node_mae={'w':11})
        self.assertFalse(i1.screen(candidate,baseline)['passes'])
        candidate['workload_mean_node_mae']['w']=10.1
        self.assertTrue(i1.screen(candidate,baseline)['passes'])


if __name__=='__main__':
    unittest.main()
