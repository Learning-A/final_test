import unittest

import torch

from idfs_lab.r2_loss_controls import R2SharedMLP
from idfs_lab.v2_workload_adapter import WorkloadAdapterMLP


class WorkloadAdapterTests(unittest.TestCase):
    def test_zero_adapters_match_frozen_backbone_and_unknown_uses_shared_output(self):
        core = torch.zeros(6, 258)
        nodes = torch.arange(6)
        features = torch.column_stack((core, torch.arange(6)))
        original = R2SharedMLP([10] * 8)
        adapted = WorkloadAdapterMLP([10] * 8)
        expected = original(core, nodes)
        torch.testing.assert_close(adapted(features, nodes), expected, rtol=0, atol=0)
        with torch.no_grad():
            adapted.adapter_bias.fill_(1)
        predictions = adapted(features, nodes)
        self.assertTrue((predictions[:5] > expected[:5]).all())
        torch.testing.assert_close(predictions[5], expected[5], rtol=0, atol=0)
