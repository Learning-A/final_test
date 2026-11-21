import unittest
from unittest.mock import patch
from idfs_lab.models_torch import normalized_sequence_output, fit_sequence


class ResidualGuardTests(unittest.TestCase):
    def test_residual_does_not_bypass_training_host(self):
        with patch('idfs_lab.placement.require_training_host',side_effect=RuntimeError('wrong host')):
            with self.assertRaisesRegex(RuntimeError,'wrong host'):
                fit_sequence([],[],[],[],'gru','unused.ts',residual_output=True)


class ResidualInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('optional torch absent; residual inference not checked')
        cls.torch=torch

    def test_zero_correction_is_persistence_including_zero(self):
        torch=self.torch
        x=torch.tensor([[0.,1.,2.,3.,4.,99.],[3000.,1.,2.,3.,4.,99.]],dtype=torch.float64)
        head=torch.zeros(2,dtype=torch.float64)
        normalized=normalized_sequence_output(head,x,5.,2.,True)
        decoded=torch.expm1(normalized*2+5).clamp_min(0)
        self.assertTrue(torch.allclose(decoded,x[:,0]))
        self.assertTrue(torch.equal(normalized_sequence_output(head,x,5.,2.),head))

    def test_trace_embeds_skip_and_correction_has_expected_log_effect(self):
        torch=self.torch
        class FixedCorrection(torch.nn.Module):
            def forward(self, x):
                h=torch.ones(x.shape[0],dtype=x.dtype)*.25
                z=normalized_sequence_output(h,x,5.,2.,True)
                return torch.expm1(z*2+5).clamp_min(0)
        x=torch.tensor([[100.,1.,2.,3.,4.,20.]])
        model=FixedCorrection()
        trace=torch.jit.trace(model,x)
        other=x*3
        self.assertTrue(torch.allclose(trace(other),model(other)))
        self.assertTrue(torch.allclose(model(x)+1,(x[:,0]+1)*torch.exp(torch.tensor(.5))))


if __name__=='__main__': unittest.main()
