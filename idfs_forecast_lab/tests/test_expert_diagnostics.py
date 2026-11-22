import unittest
import torch
from idfs_lab.masked_moe import MaskedForecast
from idfs_lab.expert_diagnostics import routed_prediction


class ExpertDiagnosticsTests(unittest.TestCase):
    def test_no_removal_matches_nonzero_model_and_removal_is_normalized(self):
        m=MaskedForecast(torch.zeros(144),torch.ones(144),torch.ones(9),'moe').double().eval()
        x=torch.ones(3,8,144,dtype=torch.double)
        with torch.no_grad():
            m.node_head.weight.fill_(.02)
            m.cluster_head[-1].weight.fill_(.01)
            base=m(x);direct,_=routed_prediction(m,x)
            self.assertTrue(torch.allclose(base,direct,rtol=1e-12,atol=1e-12))
            for g in range(3):
                pred,weights=routed_prediction(m,x,g)
                self.assertTrue(torch.isfinite(pred).all())
                self.assertEqual(float(weights[:,:,g].sum()),0)
                self.assertTrue(torch.allclose(weights.sum(-1),torch.ones(3,8,dtype=torch.double)))


if __name__=='__main__':unittest.main()
