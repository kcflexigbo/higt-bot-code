"""Phase 4 GNN baselines: GAT and GIN.

Both expect PyG `Data` with `x` (node features), `edge_index`, optional
`edge_attr`, and `y` (node labels). Output: per-node logits, shape [N, 2].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv


class GINBaseline(nn.Module):
    """3-layer GIN with epsilon learned, sum aggregation.

    The 2024 GraphSAINT+GIN paper on CTU-13 is the number to match per the
    plan's Phase 4 gate.
    """

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 2,
                 num_layers: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **_) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class GATBaseline(nn.Module):
    """2-layer GAT: concat heads in layer 1, average in layer 2.

    The "attentional but flat" reference. Per the plan, tune heads/dropout
    rather than chasing more depth.
    """

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 2,
                 heads: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        # Layer 1: heads concatenated → hidden*heads
        self.conv1 = GATConv(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
        # Layer 2: average heads → out_dim or hidden
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, concat=False, dropout=dropout)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **_) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.head(x)


def build_model(name: str, in_dim: int, **kw) -> nn.Module:
    name = name.lower()
    if name == "gin":
        return GINBaseline(in_dim, **kw)
    if name == "gat":
        return GATBaseline(in_dim, **kw)
    raise ValueError(f"unknown model {name!r}")
