"""E-GraphSAGE baseline (Lo et al., NOMS 2022, arXiv:2103.16329).

Faithful re-implementation of the E-GraphSAGE encoder, adapted to our
host-level (node) classification task in two ways.

Encoder (as in the paper / reference DGL code, github.com/waimorris/E-GraphSAGE):
- Node input is an all-ones vector (dimension = edge feature dim, as in the
  reference code). Our 9 engineered node features are deliberately NOT used.
- K = 2 layers, hidden = 128, ReLU, dropout 0.2 between layers.
- Layer k:   m_uv      = W_msg^k [ h_u^{k-1} || e_uv ]
             h_N(v)^k  = MEAN_{u in N(v)} m_uv
             h_v^k     = ReLU( W_apply^k [ h_v^{k-1} || h_N(v)^k ] )
  This is the reference implementation's message function. (Algorithm 1 in
  the paper writes the aggregate over edge features only; the released code
  also concatenates the source embedding into the message, which is what
  makes K = 2 reach 2 hops. We follow the code.)
- The paper builds its graph as an undirected multigraph converted to a
  directed one, so every flow passes messages both ways. We do the same:
  each directed host-pair edge u->v is mirrored v->u with the same edge
  features before aggregation, i.e. the mean runs over all incident edges.
- Edge embedding z_uv = [ z_u || z_v ] (u = originator, v = responder),
  classified by a single linear layer (the paper's MLPPredictor).

Node-classification heads:
- variant "node":      logits_v = Linear(z_v)                    -> [N, 2]
- variant "edge2node": logits_uv = Linear([z_u || z_v])           -> [E, 2]
  trained with edge labels y_uv = y[u] (edge is malicious iff its SOURCE
  host is a bot under the active label version); a node is then scored by
  the max P(malicious) over its outgoing edges (0 if it has none).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class ESAGELayer(nn.Module):
    """One E-GraphSAGE layer (mean aggregator over incident edges)."""

    def __init__(self, node_in: int, edge_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W_msg = nn.Linear(node_in + edge_dim, out_dim)
        self.W_apply = nn.Linear(node_in + out_dim, out_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        msg = self.W_msg(torch.cat([h[src], edge_attr], dim=-1))
        h_neigh = scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="mean")
        return F.relu(self.W_apply(torch.cat([h, h_neigh], dim=-1)))


class EGraphSAGE(nn.Module):
    """E-GraphSAGE encoder + node or edge->node head.

    forward(x, edge_index, edge_attr) returns:
      - variant "node":      node logits [N, 2]
      - variant "edge2node": edge logits [E, 2] for the given (directed) edges
    `x` is only used for its row count / device: the input is all-ones.
    """

    def __init__(self, edge_dim: int, hidden: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, variant: str = "node",
                 out_dim: int = 2) -> None:
        super().__init__()
        if variant not in ("node", "edge2node"):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.edge_dim = edge_dim
        self.node_in = edge_dim  # reference code: ones of size = #edge features
        self.dropout = dropout
        dims = [self.node_in] + [hidden] * num_layers
        self.layers = nn.ModuleList(
            ESAGELayer(dims[i], edge_dim, dims[i + 1]) for i in range(num_layers)
        )
        head_in = hidden if variant == "node" else 2 * hidden
        self.head = nn.Linear(head_in, out_dim)

    def encode(self, num_nodes: int, edge_index: torch.Tensor,
               edge_attr: torch.Tensor) -> torch.Tensor:
        h = torch.ones(num_nodes, self.node_in, device=edge_attr.device,
                       dtype=edge_attr.dtype)
        # Mirror edges so the mean runs over all incident edges (paper's
        # undirected-to-directed construction).
        ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        ea = torch.cat([edge_attr, edge_attr], dim=0)
        for i, layer in enumerate(self.layers):
            h = layer(h, ei, ea)
            if i < len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None, **_) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("E-GraphSAGE requires edge_attr")
        z = self.encode(x.size(0), edge_index, edge_attr)
        if self.variant == "node":
            return self.head(z)
        z_uv = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        return self.head(z_uv)


def edge_labels_from_nodes(y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Edge label = label of the edge's SOURCE (originator) host."""
    return y[edge_index[0]]


def edge_to_node_scores(edge_proba: torch.Tensor, edge_index: torch.Tensor,
                        num_nodes: int) -> torch.Tensor:
    """Node score = max P(malicious) over its outgoing edges; 0 if none."""
    out = torch.zeros(num_nodes, device=edge_proba.device, dtype=edge_proba.dtype)
    if edge_proba.numel() == 0:
        return out
    return out.scatter_reduce(0, edge_index[0], edge_proba, reduce="amax",
                              include_self=True)
