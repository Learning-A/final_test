import unittest
import torch
from idfs_lab.masked_moe import MaskedForecast,training_scaler,objective


class MaskedMoETests(unittest.TestCase):
    def test_masked_placeholders_do_not_change_scaler(self):
        x=torch.zeros(2,8,144);y=x.clone();y[:,:,35:38]=100000
        a,b=training_scaler(x);c,d=training_scaler(y)
        self.assertTrue(torch.equal(a,c));self.assertTrue(torch.equal(b,d))

    def test_scripted_residual_anchor_and_gates(self):
        x=torch.zeros(2,8,144);x[:,:,2]=3
        expected=torch.tensor([[3.]*8+[24.]]*2)
        for mode in ('single','uniform','moe'):
            model=MaskedForecast(torch.zeros(144),torch.ones(144),torch.ones(9),mode)
            exported=torch.jit.script(model)
            pred,lower,upper=exported.details(x)
            self.assertTrue(torch.allclose(pred,expected,atol=1e-5))
            self.assertEqual(tuple(lower.shape),(2,8,3,2))
            self.assertTrue(torch.allclose(lower.sum(-1),torch.ones(2,8,3)))
            self.assertTrue(torch.allclose(upper.sum(-1),torch.ones(2,8)))
            if mode=='uniform':self.assertTrue(torch.allclose(upper,torch.ones_like(upper)/3))

    def test_composite_loss_keeps_zero_and_low_positive_finite(self):
        pred=torch.ones(2,9);truth=torch.tensor([[0.,.1,1,2,3,4,5,6,21.1]]*2)
        value=objective(pred,truth,torch.ones(2,8,3,2)/2,torch.ones(2,8,3)/3,torch.ones(9),torch.ones(9),True,True)
        self.assertTrue(torch.isfinite(value));self.assertGreater(value.item(),0)


if __name__=='__main__':unittest.main()
