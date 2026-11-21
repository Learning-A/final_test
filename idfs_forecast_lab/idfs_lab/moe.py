"""Real temporal/graph/cross-layer experts with learned hierarchical gates.

Torch is imported only by explicit neural jobs or CPU architecture tests.
Current task is client-layer, one-second forecasting; no multi-horizon claim.
"""
import torch
from torch import nn
from torch.nn import functional as F


class JointMoE(nn.Module):
    def __init__(self, adjacency, mean=None, std=None):
        super().__init__()
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.register_buffer("mean", torch.zeros(415) if mean is None else mean)
        self.register_buffer("std", torch.ones(415) if std is None else std)
        self.temporal = nn.GRU(3, 32, batch_first=True)
        self.node_embedding = nn.Embedding(96, 8)
        self.graph1 = nn.Linear(11, 32)
        self.graph2 = nn.Linear(32, 32)
        self.attention = nn.Linear(32, 1)
        self.cross = nn.Sequential(nn.Linear(80, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head_temporal = nn.Linear(32, 2)
        self.head_spatial = nn.Linear(32, 2)
        self.head_cross = nn.Linear(32, 2)
        self.gate_layer = nn.Linear(96, 1)
        self.gate_storage = nn.Linear(96, 1)

    @torch.jit.export
    def details(self, x):
        x=x.to(dtype=self.mean.dtype)
        z = (torch.log1p(x.clamp_min(0))-self.mean)/self.std
        sequence = z[:, :15].reshape(-1, 5, 3)
        temporal_sequence, _ = self.temporal(sequence)
        temporal = temporal_sequence[:, -1, :]
        raw_nodes = x[:, 15:399].reshape(-1, 96, 4)
        nodes = z[:, 15:399].reshape(-1, 96, 4)
        mask = raw_nodes[:, :, 3:4]
        ids = torch.arange(96, device=x.device)
        embedding = self.node_embedding(ids).unsqueeze(0).expand(x.size(0), -1, -1)
        node_input = torch.cat((nodes[:, :, :3], embedding), dim=2)*mask
        # Re-normalize messages over observed neighbours; masked nodes send no data.
        denominator = torch.matmul(self.adjacency, mask).clamp_min(1e-6)
        messages = torch.matmul(self.adjacency, node_input)/denominator
        hidden = F.relu(self.graph1(messages))*mask
        hidden = F.relu(self.graph2(torch.matmul(self.adjacency, hidden)/denominator))*mask
        weights = torch.softmax(self.attention(hidden).squeeze(-1).masked_fill(mask.squeeze(-1)<0.5, -1e9), dim=1)
        spatial = (hidden*weights.unsqueeze(-1)).sum(dim=1)
        cross_mask = x[:, 407:415]
        cross_values = z[:, 399:407]*cross_mask
        cross = self.cross(torch.cat((temporal, spatial, cross_values, cross_mask), dim=1))
        context = torch.cat((temporal, spatial, cross), dim=1)
        client_gate = torch.sigmoid(self.gate_layer(context))
        spatial_gate = torch.sigmoid(self.gate_storage(context))
        gates = torch.cat((client_gate, (1-client_gate)*spatial_gate,
                           (1-client_gate)*(1-spatial_gate)), dim=1)
        experts = torch.stack((F.softplus(self.head_temporal(temporal)),
                               F.softplus(self.head_spatial(spatial)),
                               F.softplus(self.head_cross(cross))), dim=1)
        log_rw = (experts*gates.unsqueeze(-1)).sum(dim=1)
        read_write = torch.expm1(log_rw.clamp(max=30)).clamp_min(0)
        # Columns: total, read, write, temporal/spatial/cross gate weights.
        return torch.cat((read_write.sum(dim=1, keepdim=True), read_write, gates), dim=1)

    def forward(self, x):
        return self.details(x)[:, 0]


def training_scaler(x):
    """Call on training rows only. Masks are consumed raw, never as observations."""
    values = torch.log1p(x.clamp_min(0))
    observed = torch.ones_like(x)
    nodes = x[:, 15:399].reshape(-1, 96, 4)
    observed[:, 15:399] = nodes[:, :, 3:4].expand(-1, -1, 4).reshape(-1, 384)
    observed[:, 399:407] = x[:, 407:415]
    count = observed.sum(dim=0).clamp_min(1)
    mean = (values*observed).sum(dim=0)/count
    # Small smoke captures contain constant dimensions. A tiny denominator
    # magnifies an ordinary later change by 10000x; retain raw observations and
    # use an explicit scale floor in log space, without clipping source spikes.
    std = (((values-mean).square()*observed).sum(dim=0)/count).sqrt().clamp_min(0.25)
    return mean, std
