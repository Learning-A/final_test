"""Shared R2 backbone and four fixed pointwise loss controls."""
import math

import torch
from torch import nn
from torch.nn import functional as F


ARMS = ('L0', 'L1', 'L2', 'L3')
SEED = 2026


class R2SharedMLP(nn.Module):
    """One 258-feature model across eight nodes, with fixed output scales."""

    def __init__(self, output_scales):
        super().__init__()
        scales = torch.as_tensor(output_scales, dtype=torch.float32).detach().cpu().clone()
        if scales.shape != (8,) or not torch.isfinite(scales).all() or (scales < 1).any():
            raise ValueError('eight finite training scales of at least one are required')
        self.register_buffer('output_scales', scales)
        # Every arm starts from the same parameters without advancing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SEED)
            self.node_embedding = nn.Embedding(8, 4, device='cpu')
            self.backbone = nn.Sequential(
                nn.Linear(262, 64, device='cpu'),
                nn.ReLU(),
                nn.Linear(64, 64, device='cpu'),
                nn.ReLU(),
                nn.Linear(64, 1, device='cpu'),
            )
        nn.init.zeros_(self.backbone[-1].weight)
        nn.init.constant_(self.backbone[-1].bias, math.log(math.expm1(1)))

    def forward(self, features, node_indices):
        if (features.ndim != 2 or features.shape[1] != 258
                or node_indices.shape != (len(features),) or node_indices.dtype != torch.long):
            raise ValueError('expected 258 input features and one integer node ID per row')
        combined = torch.cat((features, self.node_embedding(node_indices)), dim=1)
        normalized = F.softplus(self.backbone(combined).squeeze(-1))
        return normalized * self.output_scales.index_select(0, node_indices)


def weighted_loss(arm, prediction, actual, scale, floor, weights, *, population_size):
    """Return total loss and coefficient-weighted components for the batch.

    The runner supplies finite node-level vectors and weights whose sum over the
    fitting population is one. The same reduction applies to full and short batches.
    """
    if arm not in ARMS or type(population_size) is not int or population_size < 1:
        raise ValueError('fixed loss arm and positive fitting population size required')
    if (prediction.ndim != 1 or not 0 < len(prediction) <= population_size
            or any(x.shape != prediction.shape for x in (actual, scale, floor, weights))):
        raise ValueError('loss inputs must be matching nonempty node-level vectors')

    def reduce(values):
        return population_size * torch.mean(weights * values)

    error = prediction - actual
    if arm == 'L0':
        components = {'scaled_mae': reduce(torch.abs(error / scale))}
    elif arm == 'L1':
        residual = error / scale
        components = {'scaled_huber': reduce(F.huber_loss(
            residual, torch.zeros_like(residual), reduction='none', delta=1.0))}
    else:
        log_residual = torch.log1p(prediction) - torch.log1p(actual)
        log_huber = reduce(F.huber_loss(log_residual, torch.zeros_like(log_residual),
                                        reduction='none', delta=0.1))
        components = {'log_huber': log_huber}
        if arm == 'L3':
            scaled = error / scale
            relative = error / torch.maximum(actual, floor)
            components = {
                'log_huber': 0.5 * log_huber,
                'scaled_huber': 0.25 * reduce(F.huber_loss(
                    scaled, torch.zeros_like(scaled), reduction='none', delta=1.0)),
                'relative_huber': 0.25 * reduce(F.huber_loss(
                    relative, torch.zeros_like(relative), reduction='none', delta=1.0)),
            }
    return sum(components.values()), components
