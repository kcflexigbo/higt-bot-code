"""Phase 4 GNN baselines: GAT and GIN/GINE.

Both expect PyG `Data` with `x` (node features), `edge_index`, optional
`edge_attr`, and `y` (node labels). Output: per-node logits, shape [N, 2].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, GINEConv, JumpingKnowledge


class GINBaseline(nn.Module):
    """3-layer GIN with epsilon learned, sum aggregation, residual + JK.

    Improvements over plain `GINConv` per the recent literature on
    botnet-detection GNNs:
    - **GINEConv** (Hu et al.) uses edge features — our 10-dim edge attrs
      (protocol one-hot, byte/packet counts, duration) were being ignored by
      plain GIN, which is the main reason RF/XGB at 0.96 beat plain GIN at 0.94.
    - **Residual connections** wrapping each block (XG-BoT, IoT Elsevier 2023).
      Lets gradients flow even when GINE blocks underperform individual identity
      mappings.
    - **JumpingKnowledge** across all layer outputs (Xu et al.). Bot hosts have
      variable receptive fields — 1-hop for C2 victims, 3-hop+ for P2P peers —
      so concatenating layer outputs lets the classifier pick the right depth.

    Set `use_edge_features=False` to fall back to plain GIN (kept for an
    ablation row in Phase 8).
    """

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 2,
                 num_layers: int = 3, dropout: float = 0.3,
                 edge_dim: int | None = None,
                 use_edge_features: bool = True,
                 use_residual: bool = True,
                 jk_mode: str | None = "cat") -> None:
        super().__init__()
        self.use_edge_features = use_edge_features and edge_dim is not None
        self.use_residual = use_residual
        self.jk_mode = jk_mode

        # Project raw node features to hidden, so residual works from layer 1.
        self.input_proj = nn.Linear(in_dim, hidden)

        # Project edges to hidden_dim for GINEConv (it requires edge_dim == hidden).
        if self.use_edge_features:
            self.edge_proj = nn.Linear(edge_dim, hidden)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            if self.use_edge_features:
                self.convs.append(GINEConv(mlp, train_eps=True))
            else:
                self.convs.append(GINConv(mlp, train_eps=True))
        self.dropout = dropout

        # JumpingKnowledge over per-layer outputs.
        if jk_mode is not None:
            self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden, num_layers=num_layers)
            head_in = hidden * num_layers if jk_mode == "cat" else hidden
        else:
            self.jk = None
            head_in = hidden

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None, **_) -> torch.Tensor:
        h = self.input_proj(x)
        edge_emb = self.edge_proj(edge_attr) if (self.use_edge_features and edge_attr is not None) else None

        layer_outputs: list[torch.Tensor] = []
        for conv in self.convs:
            if self.use_edge_features and edge_emb is not None:
                out = conv(h, edge_index, edge_emb)
            else:
                out = conv(h, edge_index)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            if self.use_residual:
                out = out + h
            layer_outputs.append(out)
            h = out

        h = self.jk(layer_outputs) if self.jk is not None else h
        return self.head(h)


class GATBaseline(nn.Module):
    """2-layer GAT: concat heads in layer 1, average in layer 2.

    The "attentional but flat" reference. Per the plan, tune heads/dropout
    rather than chasing more depth. When `edge_dim` is given, edge features
    are fed into the attention computation (PyG's GATConv handles the
    projection internally), giving GAT a fair shot at the edge signal that
    pushed GINE past the tree baselines.
    """

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 2,
                 heads: int = 4, dropout: float = 0.3,
                 edge_dim: int | None = None) -> None:
        super().__init__()
        self.edge_dim = edge_dim
        # Layer 1: heads concatenated → hidden*heads
        self.conv1 = GATConv(in_dim, hidden, heads=heads, concat=True,
                              dropout=dropout, edge_dim=edge_dim)
        # Layer 2: average heads → hidden
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, concat=False,
                              dropout=dropout, edge_dim=edge_dim)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None, **_) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr if self.edge_dim else None))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr if self.edge_dim else None))
        return self.head(x)


def build_model(name: str, in_dim: int, edge_dim: int | None = None, **kw) -> nn.Module:
    name = name.lower()
    if name == "gin":
        # Plain GIN ablation — node features only, no residual, no JK.
        return GINBaseline(in_dim, edge_dim=None, use_edge_features=False,
                           use_residual=False, jk_mode=None, **kw)
    if name == "gine":
        # The improved version: edge-aware GIN + residual + JK over layers.
        return GINBaseline(in_dim, edge_dim=edge_dim, use_edge_features=True,
                           use_residual=True, jk_mode="cat", **kw)
    if name == "gat":
        # Edge-aware GAT — same fairness as GINE
        return GATBaseline(in_dim, edge_dim=edge_dim, **kw)
    raise ValueError(f"unknown model {name!r}")
