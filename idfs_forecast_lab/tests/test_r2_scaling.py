"""Hand-supplied state and synthetic schemas only; never fit statistics here."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None
else:
    from idfs_lab.r2_scaling import (feature_contract,contract_sha,transform,
        fit_observed_scaling,fit_target_scales,run_equal_node_weights,HOSTS)


@unittest.skipIf(np is None, 'NumPy not available')
class R2ScalingTests(unittest.TestCase):
    def setUp(self):
        self.c=feature_contract();self.names=self.c['names']
        self.x=np.zeros((2,258));self.x[:,110]=1;self.x[:,111]=16
        self.state=dict(schema=self.c['schema'],contract_sha256=contract_sha(self.c),hosts=list(HOSTS),
            source_role='train',partition='fit_run01_02',center=[0.]*258,
            scale=[1.]*258,observed_count=[0]*258)

    def test_schema_dimensions_and_names_match_pure_input_builders(self):
        from idfs_lab.masked_inputs import build_masked_input
        from idfs_lab.fine_inputs import build_fine_input,SECOND
        t=100*SECOND;ix={}
        for h in HOSTS:
            for second in range(80,100):
                at=second*SECOND
                ix[('host_oss',h,at)]=dict(window_start_ns=at,window_end_ns=at+SECOND,n_ticks=10,
                    available_epoch_ns=at+SECOND,complete=True,read_iops=7,write_iops=3,total_iops=10)
        base=build_masked_input(ix,t,HOSTS[0],HOSTS,'synthetic-contract',80*SECOND)
        fine=build_fine_input({},t,HOSTS[0],'synthetic-contract',80*SECOND,processing_budget_ns=50_000_000)
        self.assertEqual(base['feature_names']+fine['feature_names'],self.names)
        self.assertEqual(len(self.c['continuous']),181)
        self.assertEqual(len(self.c['preserved']),77)
        actual=transform(np.asarray([base['features']+fine['features']]),self.names,self.state)
        self.assertEqual(actual.shape,(1,258))

    def test_observed_zero_differs_from_missing_and_signed_values_preserved(self):
        self.x[0,38]=1 # observed true zero for optional first lag
        self.x[:,30]=[-10,10]
        for col in (35,30):
            self.state['center'][col]=5;self.state['scale'][col]=2;self.state['observed_count'][col]=20
        self.x[1,35]=0 # missing same numeric raw placeholder
        original=self.x.copy();got=transform(self.x,self.names,self.state)
        self.assertEqual(got[0,35],-2.5);self.assertEqual(got[1,35],0)
        np.testing.assert_array_equal(got[:,30],[-7.5,2.5])
        np.testing.assert_array_equal(got[:,self.c['preserved']],self.x[:,self.c['preserved']])
        np.testing.assert_array_equal(self.x,original)

    def test_never_observed_training_column_keeps_frozen_defaults(self):
        self.x[1,147]=1;self.x[1,144]=9;self.x[1,145]=1;self.x[1,146]=10
        got=transform(self.x,self.names,self.state)
        self.assertEqual(got[1,144],9)
        self.assertEqual(self.state['observed_count'][144],0)
        self.assertEqual(got[0,144],0)

    def test_rejects_feature_order_bad_state_and_missing_placeholders(self):
        names=list(self.names);names[0],names[1]=names[1],names[0]
        with self.assertRaises(ValueError):transform(self.x,names,self.state)
        for key,col,value in [('center',38,2),('scale',1,0),('observed_count',1,-1),('center',35,4)]:
            bad=copy.deepcopy(self.state);bad[key][col]=value
            with self.assertRaises(ValueError):transform(self.x,self.names,bad)
        for col,val in [(35,5),(38,.5),(244,1.1),(148,1),(248,.5),(110,0),(0,float('nan'))]:
            bad=self.x.copy();bad[0,col]=val
            with self.assertRaises(ValueError):transform(bad,self.names,self.state)

    def test_fit_entry_points_fail_closed_on_nontraining_host_before_statistics(self):
        with patch('idfs_lab.placement.socket.gethostname',return_value='windows-test'):
            with self.assertRaisesRegex(RuntimeError,'restricted'):
                fit_observed_scaling(self.x,self.names,source_role='train',partition='fit_run01_02')
            with self.assertRaisesRegex(RuntimeError,'restricted'):
                fit_target_scales(np.ones((2,8)),source_role='train',partition='fit_run01_02')

    def test_role_and_partition_refused_before_statistics_even_on_allowed_placement(self):
        # Mock placement only to exercise rejection; no fit function is reached.
        with patch('idfs_lab.r2_scaling.require_training_host',return_value=Path('/not-used')):
            for role,part in [('validation','fit_run01_02'),('train','internal_run03')]:
                with self.assertRaises(ValueError):fit_observed_scaling(self.x,self.names,source_role=role,partition=part)
                with self.assertRaises(ValueError):fit_target_scales(np.ones((2,8)),source_role=role,partition=part)

    def test_fixed_run_weights_balance_runs_nodes_and_whole_dataset(self):
        ids=[str(i) for i in range(10) for _ in range(i+1)]
        weights=run_equal_node_weights(iter(ids))
        self.assertAlmostEqual(float(weights.sum()),1)
        for run in set(ids):self.assertAlmostEqual(float(weights[np.asarray(ids)==run].sum()),.1)
        np.testing.assert_allclose(weights.sum(axis=0),np.ones(8)/8)
        with self.assertRaises(ValueError):run_equal_node_weights(['only_one_run'])
