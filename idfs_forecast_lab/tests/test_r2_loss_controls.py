"""Pure forward checks against the R2 card, using hand-supplied fixtures."""
import math
import unittest

try:
    import torch
except ImportError:
    torch = None
else:
    from idfs_lab.r2_loss_controls import ARMS, R2SharedMLP, weighted_loss


@unittest.skipIf(torch is None, 'PyTorch not available')
class R2LossTests(unittest.TestCase):
    def test_shared_initialization_and_scaled_epoch_zero(self):
        scales = torch.arange(1, 9, dtype=torch.float32)
        indices = torch.arange(8)
        features = torch.arange(8 * 258, dtype=torch.float32).reshape(8, 258) / 100
        before = torch.get_rng_state()
        models = [R2SharedMLP(scales) for _ in ARMS]
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for model in models:
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 21089)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, models[0].state_dict()[name]), name)
            with torch.inference_mode():
                torch.testing.assert_close(model(features, indices), scales)

    def test_fixed_losses_include_real_zero_and_use_huber_not_smooth_l1(self):
        prediction = torch.tensor([2.0, 6.0], dtype=torch.float64)
        actual = torch.tensor([0.0, 2.0], dtype=torch.float64)
        scale = torch.tensor([2.0, 2.0], dtype=torch.float64)
        floor = torch.tensor([1.0, 2.0], dtype=torch.float64)
        weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
        scaled_huber = 0.25 * 0.5 + 0.75 * 1.5
        log_huber = (0.25 * 0.1 * (math.log(3) - 0.05)
                     + 0.75 * 0.1 * (math.log(7 / 3) - 0.05))
        expected = {'L0': 1.75, 'L1': scaled_huber, 'L2': log_huber,
                    'L3': 0.5 * log_huber + 0.25 * scaled_huber + 0.25 * 1.5}
        for arm, expected_value in expected.items():
            with self.subTest(arm=arm):
                loss, parts = weighted_loss(arm, prediction, actual, scale, floor,
                                            weights, population_size=2)
                self.assertAlmostEqual(loss.item(), expected_value)
                self.assertAlmostEqual(sum(value.item() for value in parts.values()), expected_value)

    def test_short_batch_uses_total_population_and_rejects_broadcasting(self):
        zero = torch.tensor([0.0])
        one = torch.tensor([1.0])
        loss, _ = weighted_loss('L0', one, zero, one, one, one / 10, population_size=10)
        self.assertAlmostEqual(loss.item(), 1.0)
        with self.assertRaises(ValueError):
            weighted_loss('L0', one, zero[:, None], one, one, one, population_size=1)
