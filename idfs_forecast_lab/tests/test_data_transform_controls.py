"""Forward-only synthetic checks; no local fitting of transform statistics."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from idfs_lab.data_transform_controls import (LogTargetMLP, apply_input_transform,
    fit_input_transform, transform_contract)
from idfs_lab.r2_scaling import feature_contract


class DataTransformControlsTests(unittest.TestCase):
    def test_partition_signed_missing_and_preserved(self):
        c = transform_contract()
        self.assertEqual([len(c[k]) for k in ('nonnegative', 'signed', 'preserved')], [178, 3, 77])
        x = np.zeros((2, 258)); x[:,110] = 1.; x[:,111] = 16.
        x[:,30] = [-5.,5.]; x[0,38] = 1.; x[0,35] = 9.
        state = dict(contract=c, method='log1p', source_role='train', partition='fit_run01_02',
                     center=[0.]*258, scale=[1.]*258, observed_count=[0]*258, lambdas=[1.]*258)
        state['center'][35] = 1.; state['observed_count'][35] = 1
        result = apply_input_transform(x,c['names'],state)
        np.testing.assert_array_equal(result[:,30],x[:,30])
        np.testing.assert_array_equal(result[:,c['preserved']],x[:,c['preserved']])
        self.assertAlmostEqual(result[0,35],np.log(10)-1.)
        self.assertEqual(result[1,35],0.)

    def test_fit_refuses_local_host_before_statistics(self):
        with patch('idfs_lab.placement.socket.gethostname',return_value='local-test'):
            with self.assertRaisesRegex(RuntimeError,'restricted'):
                fit_input_transform(None,None,method='log1p',source_role='train',partition='fit_run01_02')

    def test_log_output_initializes_to_original_training_scales(self):
        scales=[100.,200.,300.,400.,500.,600.,700.,800.]
        model=LogTargetMLP(scales)
        with torch.inference_mode():
            pred=model(torch.zeros((8,258)),torch.arange(8))
        np.testing.assert_allclose(pred.numpy(),scales,rtol=1e-6)
        self.assertEqual(sum(p.numel() for p in model.parameters()),21089)
