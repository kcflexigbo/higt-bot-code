"""Phase 6: DiffPool hierarchical pooling on top of cached encoder embeddings.

Architecture (initial single-level):

    [node_emb (64-d) || raw_x (9-d)]
        │
        ├─ GIN block (2 layers, hidden=128)            ──► Z, [N, 128]
        ├─ GIN block (2 layers, hidden=K)              ──► S = softmax(.), [N, K]
        ▼
        DiffPool: X' = S^T Z, A' = S^T A S
        │
        ├─ GIN block (1 layer, hidden=128) on coarsened ──► Z2
        ▼
        Readout: graph emb = sum(Z2)
        │
        ├─ MLP head (graph-level) ──► graph_logits
        ▼
        Project back to nodes:    node_logits = S @ graph_logits   (broadcast)
                                  (or use last-layer Z and S for per-node head)

For node classification (the task), we run a node-level head on the
pre-pool Z (skip), and a graph-level auxiliary head on the pooled graph.
Final node logits combine both via a learned gate. Aux losses (link, entropy)
are returned and added by the training loop.

PyG's `dense_diff_pool` handles the math:
    x_new, adj_new, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, JumpingKnowledge, dense_diff_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


class _GINEBlock(nn.Module):
    """A small stack of GINEConv layers with residual + JK; outputs [N, hidden]."""
    def __init__(self, in_dim: int, hidden: int, edge_dim: int, num_layers: int = 2,
                 dropout: float = 0.3, jk: str | None = "cat") -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
        self.dropout = dropout
        self.jk = JumpingKnowledge(jk, channels=hidden, num_layers=num_layers) if jk else None
        self.out_dim = hidden * num_layers if jk == "cat" else hidden

    def forward(self, x, edge_index, edge_attr):
        h = self.input_proj(x)
        e = self.edge_proj(edge_attr)
        outs = []
        for conv in self.convs:
            h2 = F.relu(conv(h, edge_index, e))
            h2 = F.dropout(h2, p=self.dropout, training=self.training)
            h = h2 + h
            outs.append(h)
        return self.jk(outs) if self.jk is not None else h


class HiGTBotDiffPool(nn.Module):
    """Single-level DiffPool model for Phase 6.

    Consumes cached encoder embeddings (`batch.node_emb`, [N, d_model]) +
    raw scaled features (`batch.x`, [N, raw_dim]). One DiffPool layer
    coarsens N → K = ceil(max_nodes * ratio) super-nodes.

    Output: per-node logits [N, 2]. Aux losses (link, entropy) are returned
    in a dict for the training loop to add to the main classification loss.
    """

    def __init__(
        self,
        *,
        d_model: int = 64,
        raw_feat_dim: int = 9,
        edge_dim: int = 10,
        hidden: int = 128,
        max_nodes: int = 400,
        pool_ratio: float = 0.25,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        K1 = max(int(max_nodes * pool_ratio), 8)
        self.K1 = K1
        in_dim = d_model + raw_feat_dim

        # Pre-pool embedder + assignment GNN
        self.embed_gnn = _GINEBlock(in_dim, hidden, edge_dim, num_layers=2,
                                     dropout=dropout, jk="cat")
        # Project embed output back to `hidden` for the dense pool layer
        self.embed_proj = nn.Linear(self.embed_gnn.out_dim, hidden)
        self.pool_gnn = _GINEBlock(in_dim, K1, edge_dim, num_layers=2,
                                    dropout=dropout, jk=None)
        # post-pool GNN operating on coarsened DENSE graph (uses MLP since
        # the coarsened adj is dense [K, K] and PyG GINEConv expects sparse).
        self.post_pool = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

        # Node-level head: combines node embedding (pre-pool) + projected
        # graph-level info from the pooled representation.
        self.node_head = nn.Sequential(
            nn.Linear(hidden + hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch) -> dict:
        # Build node input: [encoder_emb || raw_x]
        x = torch.cat([batch.node_emb, batch.x], dim=-1)
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        # Pre-pool node embeddings (sparse PyG conv)
        z = self.embed_gnn(x, edge_index, edge_attr)
        z = self.embed_proj(z)                               # [sumN, hidden]
        s_logits = self.pool_gnn(x, edge_index, edge_attr)   # [sumN, K1]

        # Convert to dense for DiffPool
        z_dense, mask = to_dense_batch(z, batch.batch, max_num_nodes=self.max_nodes)
        s_dense, _ = to_dense_batch(s_logits, batch.batch, max_num_nodes=self.max_nodes)
        adj = to_dense_adj(edge_index, batch.batch, max_num_nodes=self.max_nodes)  # [B, N, N]

        # DiffPool: returns coarsened (x', adj', aux losses)
        x_coarse, adj_coarse, link_loss, ent_loss = dense_diff_pool(
            z_dense, adj, s_dense, mask
        )  # x_coarse: [B, K, hidden], adj_coarse: [B, K, K]

        # Post-pool processing on the coarsened graph (per-cluster MLP)
        x_coarse = F.relu(self.post_pool(x_coarse))           # [B, K, hidden]

        # Graph-level readout: mean over clusters (could use attention later)
        graph_repr = x_coarse.mean(dim=1)                     # [B, hidden]

        # Project graph repr back to each node via S (sparse view)
        # s_dense is [B, N, K]; argmax for hard cluster, or use softmax(s) @ x_coarse
        s_soft = F.softmax(s_dense, dim=-1)                   # [B, N, K]
        node_from_cluster = s_soft @ x_coarse                 # [B, N, hidden]

        # Flatten with mask to recover the original sparse node count
        node_from_cluster_flat = node_from_cluster[mask]       # [sumN, hidden]

        # Concat per-node embedding (z) with cluster-aware representation
        node_in = torch.cat([z, node_from_cluster_flat], dim=-1)
        node_logits = self.node_head(node_in)                  # [sumN, 2]

        return {
            "logits": node_logits,
            "link_loss": link_loss,
            "ent_loss": ent_loss,
            "graph_repr": graph_repr,
            "assignments": s_soft,
        }
