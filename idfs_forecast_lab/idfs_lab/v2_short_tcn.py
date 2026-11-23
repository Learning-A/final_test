"""Short causal convolution over the original masked 100 ms input slots."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .r2_loss_controls import SEED


class ShortCausalTCN(nn.Module):
    def __init__(self, output_scales):
        super().__init__()
        scales = torch.as_tensor(output_scales, dtype=torch.float32).detach().cpu().clone()
        if scales.shape != (8,) or not torch.isfinite(scales).all() or (scales < 1).any():
            raise ValueError('Eight finite training target scales required')
        self.register_buffer('output_scales', scales)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SEED)
            self.node_embedding = nn.Embedding(8, 4, device='cpu')
            self.convolutions = nn.ModuleList([
                nn.Conv1d(5, 32, 3, dilation=1, device='cpu'),
                nn.Conv1d(32, 32, 3, dilation=2, device='cpu'),
                nn.Conv1d(32, 32, 3, dilation=4, device='cpu'),
            ])
            self.backbone = nn.Sequential(
                nn.Linear(194, 64, device='cpu'), nn.ReLU(),
                nn.Linear(64, 64, device='cpu'), nn.ReLU(),
                nn.Linear(64, 1, device='cpu'),
            )
        nn.init.zeros_(self.backbone[-1].weight)
        nn.init.constant_(self.backbone[-1].bias, math.log(math.expm1(1)))

    def temporal_states(self, features):
        # C stores lag1 first. Convolutions process oldest to newest.
        sequence = features[:, 144:244].reshape(-1, 20, 5).flip(1).transpose(1, 2)
        observed = sequence[:, 3:4]
        for layer in self.convolutions:
            sequence = F.relu(layer(F.pad(sequence, (2 * layer.dilation[0], 0))))
        return sequence, observed

    def forward(self, features, node_indices):
        if features.ndim != 2 or features.shape[1] != 258:
            raise ValueError('The original C258 features are required')
        if node_indices.shape != (len(features),) or node_indices.dtype != torch.long:
            raise ValueError('One integer node index per feature row is required')
        sequence, mask = self.temporal_states(features)
        pooled = (sequence * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        combined = torch.cat((
            features[:, :144], features[:, 244:], pooled, self.node_embedding(node_indices),
        ), dim=1)
        return F.softplus(self.backbone(combined).squeeze(-1)) * self.output_scales.index_select(0, node_indices)
