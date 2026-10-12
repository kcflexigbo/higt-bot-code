"""Direction-aware, ego-separated GINE layer (opt-in via HIGT_DIRECTED=1).

PyG message passing runs source -> target, and our edges point from flow
originator to responder, so a plain GINEConv delivers the evidence on an
infected host's *outgoing* edges to its victims and never to the host itself
(see manuscript/review/fable_audit_and_decision_2026-10-11.md). This layer
aggregates in-neighbours and out-neighbours separately and keeps the node's own
representation as a third, unmixed input (Dir-GNN, Rossi et al. 2023; ego /
neighbour separation, H2GCN, Zhu et al. 2020):

    m_in(v)  = sum_{u -> v} ReLU(h_u + e_uv)
    m_out(v) = sum_{v -> w} ReLU(h_w + e_vw)
    h'(v)    = MLP( W [ (1 + eps) h_v || m_in(v) || m_out(v) ] )

With HIGT_DIRECTED unset, `make_gine` returns the original GINEConv, so every
existing script and checkpoint behaves exactly as before.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

DIRECTED = os.environ.get("HIGT_DIRECTED", "0") == "1"


class DirGINEConv(nn.Module):
    def __init__(self, mlp: nn.Module, hidden: int) -> None:
        super().__init__()
        self.mlp = mlp
        self.eps = nn.Parameter(torch.zeros(1))
        self.combine = nn.Linear(3 * hidden, hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        m_in = torch.zeros_like(x).index_add_(0, dst, F.relu(x[src] + edge_attr))
        m_out = torch.zeros_like(x).index_add_(0, src, F.relu(x[dst] + edge_attr))
        return self.mlp(self.combine(torch.cat([(1 + self.eps) * x, m_in, m_out], dim=-1)))


def make_gine(mlp: nn.Module, hidden: int) -> nn.Module:
    return DirGINEConv(mlp, hidden) if DIRECTED else GINEConv(mlp, train_eps=True)
