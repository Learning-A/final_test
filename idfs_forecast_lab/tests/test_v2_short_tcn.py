import unittest

import torch

from idfs_lab.r2_loss_controls import R2SharedMLP
from idfs_lab.v2_short_tcn import ShortCausalTCN


class ShortCausalTests(unittest.TestCase):
    def test_later_slot_cannot_change_earlier_states(self):
        model = ShortCausalTCN([10] * 8)
        x = torch.zeros(1, 258)
        x[:, 147:244:5] = 1
        before, _ = model.temporal_states(x)
        x[:, 144:147] = 100
        after, _ = model.temporal_states(x)
        torch.testing.assert_close(before[:, :, :-1], after[:, :, :-1], rtol=0, atol=0)

    def test_initial_outputs_match_r2_even_with_all_slots_missing(self):
        model = ShortCausalTCN([10] * 8)
        x = torch.zeros(8, 258)
        nodes = torch.arange(8)
        expected = R2SharedMLP([10] * 8)(x, nodes)
        torch.testing.assert_close(model(x, nodes), expected, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 23457)
