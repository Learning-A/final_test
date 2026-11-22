import unittest
import torch
from torch import nn
from idfs_lab.legacy_adapter import LegacyAdapter


class ZeroCore(nn.Module):
    def forward(self,x):return x[:,-1]*0


class LegacyAdapterTests(unittest.TestCase):
    def test_window_and_global_inverse_and_script(self):
        x=torch.arange(30,dtype=torch.float32).reshape(2,15,1)
        a=torch.jit.script(LegacyAdapter(ZeroCore(),True))
        self.assertTrue(torch.allclose(a(x),x.mean(1).squeeze(-1)))
        b=torch.jit.script(LegacyAdapter(ZeroCore(),False,3,2))
        self.assertTrue(torch.equal(b(x),torch.tensor([3.,3.])))
        self.assertTrue(torch.isfinite(a(torch.zeros_like(x))).all())


if __name__=='__main__':unittest.main()
