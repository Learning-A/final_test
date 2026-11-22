"""Exercise real NumPy scalar reductions that previously broke result persistence."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest


class R1Scalars(unittest.TestCase):
    def test_numpy_metrics_and_screen_counts_are_json_serializable(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy unavailable; no model fitting is involved')
        scripts=Path(__file__).resolve().parents[2]/'scripts'
        sys.path.insert(0,str(scripts))
        try:
            spec=importlib.util.spec_from_file_location('r1_report_scalar_test',scripts/'run_r1_input_study.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            result=module.group_metrics(np.array([0.,4.,90.,1000.]),np.array([0.,5.,100.,2000.]))
            count=sum(x<10 for x in [result['mape_percent']])
            encoded=json.loads(json.dumps({'metrics':result,'passing_groups':count,'screen':result['mae_iops']<10}))
            self.assertEqual(encoded['metrics']['n'],4)
            self.assertEqual(encoded['metrics']['zero_actual'],1)
            self.assertEqual(encoded['metrics']['positive_bands']['positive_le_10']['n'],1)
        finally:
            sys.path.remove(str(scripts))
