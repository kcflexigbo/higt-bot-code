"""TemporalGINE — Phase 5 hybrid model.

Pipeline:
    (per-node flow sequence) ─► TemporalFlowEncoder ─► [N, d_model]
                                                       │
                                                       ▼
                                       GINBaseline (edge-aware, residual, JK)
                                                       │
                                                       ▼
                                                [N, num_classes]

Inputs are a PyG `Batch` carrying:
  - flows:     [sum_N, max_flows, flow_feat_dim]
  - flow_mask: [sum_N, max_flows]
  - edge_index, edge_attr, y, batch (set by Batch.from_data_list)

The encoder runs on the flat [sum_N, ...] stack — no graph-level batching
is needed because each row is one node.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.gnn_baselines import GINBaseline
from src.models.temporal import TemporalFlowEncoder


class TemporalGINE(nn.Module):
    def __init__(
        self,
        *,
        flow_feat_dim: int = 13,
        edge_dim: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_flows: int = 256,
        encoder_dropout: float = 0.1,
        gin_hidden: int = 64,
        gin_layers: int = 3,
        out_dim: int = 2,
        dropout: float = 0.3,
        encoder_chunk_size: int = 64,
        raw_feat_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder_chunk_size = encoder_chunk_size
        self.raw_feat_dim = raw_feat_dim
        self.encoder = TemporalFlowEncoder(
            flow_feat_dim=flow_feat_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, max_flows=max_flows, dropout=encoder_dropout,
        )
        gin_in = d_model + (raw_feat_dim or 0)
        self.gnn = GINBaseline(
            in_dim=gin_in, hidden=gin_hidden, out_dim=out_dim,
            num_layers=gin_layers, dropout=dropout, edge_dim=edge_dim,
            use_edge_features=True, use_residual=True, jk_mode="cat",
        )

    def _encode_flows(self, flows: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Run the temporal encoder in node chunks to cap VRAM."""
        n = flows.size(0)
        chunk = self.encoder_chunk_size
        if n <= chunk:
            return self.encoder(flows, pad_mask)
        parts = [
            self.encoder(flows[i : i + chunk], pad_mask[i : i + chunk])
            for i in range(0, n, chunk)
        ]
        return torch.cat(parts, dim=0)

    def forward(self, batch) -> torch.Tensor:
        node_emb = self._encode_flows(batch.flows, batch.flow_mask)
        if self.raw_feat_dim is not None:
            node_emb = torch.cat([node_emb, batch.x], dim=-1)
        edge_attr = getattr(batch, "edge_attr", None)
        return self.gnn(node_emb, batch.edge_index, edge_attr=edge_attr)

    @torch.no_grad()
    def encode_nodes(self, batch) -> torch.Tensor:
        """Inference-only helper used by scripts/cache_embeddings.py."""
        self.eval()
        return self._encode_flows(batch.flows, batch.flow_mask)
