"""Phase 6.4 — Sparse hierarchical pooling (SAGPool variant).

Motivation: Phase 6.3 (SSL pretrain) showed that SSL helps iot23-35-1 *at the
flat-node-classification level* (Phase 5 score 0.62), but DiffPool's dense
soft-assignment then averages the rare bot signal into majority-benign super-
nodes — Phase 6 SSL-FT scores iot23-35-1 only 0.27.

SAGPool / TopKPool are *sparse*: each node is scored individually and the
graph is reduced by keeping the top-k scoring nodes intact (no averaging
into clusters). Rare-but-discriminative bot nodes can survive the coarsening
if their score is high enough, instead of being folded into a soft cluster.

Architecture (single sparse pooling level):
    x = [node_emb || raw_x]            -> [N, in_dim]
    z = GINE-stack(x, edge_index)      -> [N, hidden]   (pre-pool repr)
    keep, edge_index', edge_attr',
      batch', perm, score = SAGPool(z) -> [K, hidden]
    z2 = GINE-stack(keep, ...)         -> [K, hidden]   (post-pool repr)
    g_repr = mean+max readout(z2)      -> [B, 2*hidden] (graph summary)
    node_logits = head([z || g_repr@v])-> [N, num_classes]

For each node, the head sees the node's own pre-pool embedding plus the
graph-level summary of the coarsened representation. Surviving nodes also
contributed directly to the summary; dropped nodes still get a fair logit
because z[v] still carries the per-node features.

No aux losses (link/entropy) — SAGPool's scoring layer is trained end-to-end
through the classification loss only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, JumpingKnowledge, SAGPooling, global_max_pool, global_mean_pool


class _GINEBlock(nn.Module):
    """GINEConv stack with residual + JK; outputs [N, hidden] (jk=cat doubles)."""

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


class HiGTBotSparsePool(nn.Module):
    """Single-level sparse hierarchical pooling for node-level bot detection.

    `pool_ratio` controls coarsening (fraction of nodes kept). Lower = more
    aggressive coarsening. With node-level prediction, the rare bot node's
    *own* embedding always reaches the head — coarsening only changes the
    graph summary, not the per-node identity feature.
    """

    def __init__(
        self,
        *,
        d_model: int = 64,
        raw_feat_dim: int = 9,
        edge_dim: int = 10,
        hidden: int = 128,
        pool_ratio: float = 0.5,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        in_dim = d_model + raw_feat_dim

        self.embed_gnn = _GINEBlock(in_dim, hidden, edge_dim, num_layers=2,
                                     dropout=dropout, jk="cat")
        self.embed_proj = nn.Linear(self.embed_gnn.out_dim, hidden)

        # SAGPool scoring uses a GraphConv by default (depends only on
        # edge_index, not edge_attr). Sparse, per-node, learnable.
        self.pool = SAGPooling(hidden, ratio=pool_ratio)

        # Post-pool GINE on the coarsened (still sparse) graph. After pooling,
        # the surviving edges keep their edge_attr, so GINE-with-edge-features
        # still applies.
        self.post_pool_gnn = _GINEBlock(hidden, hidden, edge_dim, num_layers=1,
                                         dropout=dropout, jk=None)

        # Graph readout: mean + max over surviving nodes → [B, 2*hidden]
        # Node head sees [z_pre_pool_v || graph_repr_for_v's_graph]
        self.node_head = nn.Sequential(
            nn.Linear(hidden + 2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch) -> dict:
        x = torch.cat([batch.node_emb, batch.x], dim=-1)
        ei, ea, b = batch.edge_index, batch.edge_attr, batch.batch

        # Pre-pool node embeddings (sparse PyG conv) — these go to the head.
        z = self.embed_gnn(x, ei, ea)
        z = self.embed_proj(z)                                  # [N, hidden]

        # Sparse pool: drop low-scoring nodes, keep edges between survivors.
        # Returns: kept x, new edge_index, new edge_attr (if input had it),
        #          new batch, perm (kept node indices), score.
        z_pool, ei_pool, ea_pool, b_pool, perm, score = self.pool(
            z, ei, edge_attr=ea, batch=b,
        )

        # Post-pool GINE on coarsened sparse graph.
        z_pool2 = self.post_pool_gnn(z_pool, ei_pool, ea_pool)  # [K, hidden]

        # Graph readout (mean + max over surviving nodes per graph).
        gm = global_mean_pool(z_pool2, b_pool)                  # [B, hidden]
        gx = global_max_pool(z_pool2, b_pool)                   # [B, hidden]
        g_repr = torch.cat([gm, gx], dim=-1)                    # [B, 2*hidden]

        # Broadcast graph summary back to each original node via its batch idx.
        g_per_node = g_repr[b]                                   # [N, 2*hidden]

        # Per-node logits: [own z, graph summary] -> 2-class.
        node_in = torch.cat([z, g_per_node], dim=-1)
        node_logits = self.node_head(node_in)

        return {
            "logits": node_logits,
            "pool_score": score,
            "kept_perm": perm,
            "graph_repr": g_repr,
            # No aux losses for sparse pooling; training loop should set
            # link/ent weights to 0 (or skip them entirely).
            "link_loss": torch.tensor(0.0, device=node_logits.device),
            "ent_loss": torch.tensor(0.0, device=node_logits.device),
        }
