"""End-to-end shape and gradient test for the hybrid temporal-GINE model."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from src.models.hybrid import TemporalGINE


def _fake_graph(num_nodes: int, num_edges: int, max_flows: int) -> Data:
    g = Data(
        x=torch.zeros(num_nodes, 1),   # ignored — encoder produces real x
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 10),
        y=torch.randint(0, 2, (num_nodes,)),
        graph_y=torch.tensor([1]),
    )
    g.flows = torch.randn(num_nodes, max_flows, 13)
    g.flow_mask = torch.zeros(num_nodes, max_flows, dtype=torch.bool)
    return g


def test_hybrid_forward_shapes() -> None:
    torch.manual_seed(0)
    model = TemporalGINE(flow_feat_dim=13, edge_dim=10, d_model=32, nhead=4,
                          num_layers=2, max_flows=16, gin_hidden=32,
                          gin_layers=2, out_dim=2, dropout=0.1)
    batch = Batch.from_data_list([_fake_graph(5, 8, 16), _fake_graph(7, 10, 16)])
    logits = model(batch)
    assert logits.shape == (12, 2)
    assert torch.isfinite(logits).all()


def test_hybrid_backward_grads_flow_through_encoder() -> None:
    """Loss against per-node logits must produce non-zero gradient on the
    temporal encoder's CLS token — otherwise the hybrid is degenerate."""
    torch.manual_seed(0)
    model = TemporalGINE(flow_feat_dim=13, edge_dim=10, d_model=32, nhead=4,
                          num_layers=2, max_flows=16, gin_hidden=32,
                          gin_layers=2, out_dim=2, dropout=0.0)
    batch = Batch.from_data_list([_fake_graph(5, 8, 16)])
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, batch.y)
    loss.backward()
    grad = model.encoder.cls_token.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0
