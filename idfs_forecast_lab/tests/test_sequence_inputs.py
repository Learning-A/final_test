import unittest
from unittest.mock import patch
from idfs_lab.models_torch import sequence_inputs, fit_sequence


class TrainingGuardTests(unittest.TestCase):
    def test_opt_in_cannot_bypass_training_host_guard(self):
        with patch('idfs_lab.placement.require_training_host',side_effect=RuntimeError('wrong host')):
            with self.assertRaisesRegex(RuntimeError,'wrong host'):
                fit_sequence([],[],[],[],'gru','unused.ts',normalize_inputs=True)


class SequenceInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('optional torch absent; transform inference not tested')
        cls.torch=torch

    def test_fixed_scaler_temporal_order_and_ignored_mean_feature(self):
        torch=self.torch
        # Predetermined inference constants, no data-derived parameter fitting.
        features=torch.cat((torch.expm1(torch.tensor([[1.,2.,3.,4.,5.]])),torch.tensor([[99999.]])),dim=1)
        actual=sequence_inputs(features,3.,2.).squeeze(-1)
        self.assertTrue(torch.allclose(actual,torch.tensor([[1.,.5,0.,-.5,-1.]])))
        self.assertTrue(torch.allclose(sequence_inputs(features),torch.tensor([[[5.],[4.],[3.],[2.],[1.]]])))

    def test_normalization_is_embedded_in_cpu_trace(self):
        torch=self.torch
        class Transform(torch.nn.Module):
            def forward(self, features):
                return sequence_inputs(features,3.,2.)
        model=Transform()
        features=torch.tensor([[0.,1.,3.,7.,15.,999.]])
        traced=torch.jit.trace(model,features)
        other=features*2
        self.assertTrue(torch.allclose(traced(other),model(other)))


if __name__=='__main__': unittest.main()
