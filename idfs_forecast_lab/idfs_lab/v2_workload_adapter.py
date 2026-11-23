"""Small additive workload adapters on the original shared R2 backbone."""

import torch
from torch import nn
from torch.nn import functional as F

from .r2_loss_controls import R2SharedMLP


class WorkloadAdapterMLP(R2SharedMLP):
    """A sixth, fixed-zero adapter gives unknown workloads the shared output."""

    def __init__(self, output_scales):
        super().__init__(output_scales)
        self.adapter_weight = nn.Parameter(torch.zeros(5, 64))
        self.adapter_bias = nn.Parameter(torch.zeros(5))

    def forward(self, features, node_indices):
        if features.ndim != 2 or features.shape[1] != 259:
            raise ValueError('C258 plus one workload index required')
        if node_indices.shape != (len(features),) or node_indices.dtype != torch.long:
            raise ValueError('One integer node index required per feature row')
        workload = features[:, 258].to(torch.long)
        hidden = torch.cat((features[:, :258], self.node_embedding(node_indices)), dim=1)
        for layer in list(self.backbone)[:-1]:
            hidden = layer(hidden)
        shared = self.backbone[-1](hidden).squeeze(-1)
        weights = F.pad(self.adapter_weight, (0, 0, 0, 1))
        biases = F.pad(self.adapter_bias, (0, 1))
        correction = (hidden * weights.index_select(0, workload)).sum(dim=-1)
        correction = correction + biases.index_select(0, workload)
        return F.softplus(shared + correction) * self.output_scales.index_select(0, node_indices)
