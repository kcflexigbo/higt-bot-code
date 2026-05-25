"""Phase 7 — Full HiGT-Bot model.

Builds on the Phase 6.4 SAGPool architecture by inserting a Graph Transformer
block between the post-pool GINE and the readout. The plan calls for two
variants implemented as an ablation:

  - variant="edge"   → PyG TransformerConv (edge-aware attention over the
                       sparse coarsened graph; keeps edge_attr signal).
  - variant="global" → nn.MultiheadAttention; treat surviving super-nodes
                       as a sequence and let every super-node attend to
                       every other (no edge bias). Padded to dense via
                       `to_dense_batch`.
  - variant="hybrid" → alternating edge/global blocks: layer 0 is edge-aware,
                       layer 1 is global, repeat. Idea: edge layer sharpens
                       per-cluster representation along real connectivity;
                       global layer then mixes long-range across all clusters.
  - variant="none"   → bypass the GT block (= Phase 6.4 SAGPool, kept here
                       so train_phase7.py can run the no-GT ablation through
                       the same script).

Per-node prediction is preserved: the head still receives the *pre-pool*
embedding `z[v]` concatenated with a graph-level summary, so dropped nodes
get a fair logit. The GT block changes only what `g_repr` looks like — it
gives the summary access to long-range super-community structure.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    JumpingKnowledge,
    SAGPooling,
    TransformerConv,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch


class _GINEBlock(nn.Module):
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


class _EdgeGTBlock(nn.Module):
    """Variant A: edge-aware Graph Transformer via PyG TransformerConv."""

    def __init__(self, d: int, edge_dim: int, *, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        # concat=False averages heads so output stays d-dim → cheap stacking.
        self.conv1 = TransformerConv(d, d, heads=nhead, concat=False,
                                       dropout=dropout, edge_dim=edge_dim)
        self.conv2 = TransformerConv(d, d, heads=nhead, concat=False,
                                       dropout=dropout, edge_dim=edge_dim)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        self.norm3 = nn.LayerNorm(d)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(self.norm1(x), edge_index, edge_attr)
        x = x + F.dropout(h, p=0.1, training=self.training)
        h = self.conv2(self.norm2(x), edge_index, edge_attr)
        x = x + F.dropout(h, p=0.1, training=self.training)
        x = x + self.ff(self.norm3(x))
        return x


class _GlobalAttnBlock(nn.Module):
    """Variant B: edge-free global self-attention over padded super-node seqs."""

    def __init__(self, d: int, *, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, K, d]  key_padding_mask: [B, K]  True at padded positions
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + F.dropout(h, p=0.1, training=self.training)
        x = x + self.ff(self.norm2(x))
        return x


class HiGTBot(nn.Module):
    """Full HiGT-Bot — Phase 5 encoder → SAGPool → GT → per-node head.

    The encoder output is consumed via `batch.node_emb` (cached parquet) so
    this class doesn't re-run the temporal Transformer.

    `gt_variant`:
        "edge"   — TransformerConv (edge-aware, sparse on coarsened graph).
        "global" — MultiheadAttention (dense, padded over super-nodes).
        "hybrid" — alternating edge/global layers (edge first, then global,
                   repeating). Combines structural sharpening with long-range
                   mixing.
        "none"   — skip GT (= Phase 6.4 SAGPool baseline).
    """

    def __init__(
        self,
        *,
        d_model: int = 64,
        raw_feat_dim: int = 9,
        edge_dim: int = 10,
        hidden: int = 128,
        pool_ratio: float = 0.5,
        gt_variant: str = "edge",
        gt_layers: int = 2,
        gt_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if gt_variant not in {"edge", "global", "hybrid", "none"}:
            raise ValueError(
                f"gt_variant must be edge/global/hybrid/none, got {gt_variant!r}"
            )
        self.gt_variant = gt_variant
        in_dim = d_model + raw_feat_dim

        self.embed_gnn = _GINEBlock(in_dim, hidden, edge_dim, num_layers=2,
                                     dropout=dropout, jk="cat")
        self.embed_proj = nn.Linear(self.embed_gnn.out_dim, hidden)

        self.pool = SAGPooling(hidden, ratio=pool_ratio)

        self.post_pool_gnn = _GINEBlock(hidden, hidden, edge_dim, num_layers=1,
                                         dropout=dropout, jk=None)

        if gt_variant == "edge":
            self.gt_blocks = nn.ModuleList([
                _EdgeGTBlock(hidden, edge_dim, nhead=gt_heads, dropout=0.1)
                for _ in range(gt_layers)
            ])
        elif gt_variant == "global":
            self.gt_blocks = nn.ModuleList([
                _GlobalAttnBlock(hidden, nhead=gt_heads, dropout=0.1)
                for _ in range(gt_layers)
            ])
        elif gt_variant == "hybrid":
            blocks = []
            for i in range(gt_layers):
                if i % 2 == 0:
                    blocks.append(_EdgeGTBlock(hidden, edge_dim, nhead=gt_heads, dropout=0.1))
                else:
                    blocks.append(_GlobalAttnBlock(hidden, nhead=gt_heads, dropout=0.1))
            self.gt_blocks = nn.ModuleList(blocks)
        else:
            self.gt_blocks = nn.ModuleList()

        self.node_head = nn.Sequential(
            nn.Linear(hidden + 2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, batch) -> dict:
        x = torch.cat([batch.node_emb, batch.x], dim=-1)
        ei, ea, b = batch.edge_index, batch.edge_attr, batch.batch

        z = self.embed_gnn(x, ei, ea)
        z = self.embed_proj(z)                                      # [N, h]

        z_pool, ei_pool, ea_pool, b_pool, perm, score = self.pool(
            z, ei, edge_attr=ea, batch=b,
        )

        z_pool2 = self.post_pool_gnn(z_pool, ei_pool, ea_pool)      # [K_total, h]

        # Apply the Graph Transformer block(s).
        if self.gt_variant == "edge":
            for blk in self.gt_blocks:
                z_pool2 = blk(z_pool2, ei_pool, ea_pool)
            gt_out_sparse = z_pool2
        elif self.gt_variant == "global":
            # Pad to [B, K_max, h] so MultiheadAttention can run over a fixed
            # super-node sequence per graph.
            dense, mask = to_dense_batch(z_pool2, b_pool)           # [B, Kmax, h], [B, Kmax]
            key_padding = ~mask                                      # True where padded
            for blk in self.gt_blocks:
                dense = blk(dense, key_padding)
            gt_out_sparse = dense[mask]                              # back to [K_total, h]
        elif self.gt_variant == "hybrid":
            # Alternate edge (sparse) and global (dense) blocks. The dense ↔
            # sparse conversion is cheap on the coarsened graph (K ≪ N).
            for blk in self.gt_blocks:
                if isinstance(blk, _EdgeGTBlock):
                    z_pool2 = blk(z_pool2, ei_pool, ea_pool)
                else:                                              # _GlobalAttnBlock
                    dense, mask = to_dense_batch(z_pool2, b_pool)
                    dense = blk(dense, ~mask)
                    z_pool2 = dense[mask]
            gt_out_sparse = z_pool2
        else:
            gt_out_sparse = z_pool2

        # Readout per graph (mean + max over surviving super-nodes).
        gm = global_mean_pool(gt_out_sparse, b_pool)                 # [B, h]
        gx = global_max_pool(gt_out_sparse, b_pool)                  # [B, h]
        g_repr = torch.cat([gm, gx], dim=-1)                         # [B, 2h]

        g_per_node = g_repr[b]                                       # [N, 2h]
        node_in = torch.cat([z, g_per_node], dim=-1)                 # [N, 3h]
        node_logits = self.node_head(node_in)

        return {
            "logits": node_logits,
            "pool_score": score,
            "kept_perm": perm,
            "graph_repr": g_repr,
            "link_loss": torch.tensor(0.0, device=node_logits.device),
            "ent_loss": torch.tensor(0.0, device=node_logits.device),
        }
