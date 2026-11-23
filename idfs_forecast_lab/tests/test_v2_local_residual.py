import unittest
import numpy as np
import torch
from idfs_lab.v2_local_residual import LocalResidualMLP,local_context


class LocalResidualTests(unittest.TestCase):
    def raw(self,values):
        x=np.zeros((1,8,258))
        x[:,:,[2,5,8,11,14]]=np.asarray(values)
        return x

    def test_local_statistics_use_only_five_observed_history_totals(self):
        x=self.raw([0,10,20,30,40])
        b,s,ctx=local_context(x,[100]*8)
        np.testing.assert_allclose(b,20)
        np.testing.assert_allclose(s,np.std([0,10,20,30,40]))
        x[:,:,32]=99999
        b2,s2,c2=local_context(x,[100]*8)
        np.testing.assert_array_equal(ctx,c2)

    def test_zero_history_has_fixed_baseline_and_training_scale_floor(self):
        b,s,c=local_context(self.raw([0]*5),[100]*8)
        np.testing.assert_allclose(b,1)
        np.testing.assert_allclose(s,5)
        np.testing.assert_array_equal(c[:,:,3],1)

    def test_initial_prediction_is_history_base_and_output_is_nonnegative(self):
        model=LocalResidualMLP([100]*8)
        x=torch.zeros((8,262));x[:,-4]=.2;x[:,-3]=.05
        nodes=torch.arange(8)
        torch.testing.assert_close(model(x,nodes),torch.full((8,),20.))
        with torch.no_grad():model.backbone[-1].bias.fill_(-10)
        torch.testing.assert_close(model(x,nodes),torch.zeros(8))
        self.assertEqual(sum(p.numel() for p in model.parameters()),21345)

    def test_constructor_preserves_rng_and_residual_has_gradient(self):
        before=torch.random.get_rng_state().clone()
        model=LocalResidualMLP([10]*8)
        torch.testing.assert_close(torch.random.get_rng_state(),before)
        x=torch.zeros((8,262));x[:,-4]=1;x[:,-3]=.1
        model(x,torch.arange(8)).sum().backward()
        self.assertGreater(float(model.backbone[-1].bias.grad),0)


if __name__=='__main__':
    unittest.main()
