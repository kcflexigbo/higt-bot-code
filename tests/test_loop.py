"""Unit tests for src/training/loop.py — only the parts that don't require
spinning up a full training run. The training loop itself is exercised end-to-end
by the GNN baselines scripts."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.training.loop import TrainConfig, _class_weights_from_train


def _g(n_nodes: int, y_vals: list[int]) -> Data:
    return Data(
        x=torch.randn(n_nodes, 4),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, 2),
        y=torch.tensor(y_vals, dtype=torch.long),
        graph_y=torch.tensor([int(any(y_vals))], dtype=torch.long),
    )


def test_class_weights_balanced() -> None:
    """Equal class frequencies → equal weights (both 1.0)."""
    g = _g(4, [0, 0, 1, 1])
    w_neg, w_pos = _class_weights_from_train([g])
    assert abs(w_neg - 1.0) < 1e-6
    assert abs(w_pos - 1.0) < 1e-6


def test_class_weights_imbalanced() -> None:
    """Rare positive class gets a higher weight."""
    g = _g(10, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    w_neg, w_pos = _class_weights_from_train([g])
    # inverse-frequency: w_neg = 10/(2*9), w_pos = 10/(2*1)
    assert abs(w_neg - 10 / 18) < 1e-6
    assert abs(w_pos - 10 / 2) < 1e-6
    # Positive (rare) class must be weighted more heavily.
    assert w_pos > w_neg


def test_class_weights_no_positives() -> None:
    """All-negative input must not divide by zero."""
    g = _g(5, [0, 0, 0, 0, 0])
    w_neg, w_pos = _class_weights_from_train([g])
    # n_pos floored at 1 → weights are finite.
    assert w_neg > 0 and w_pos > 0
    assert w_pos > w_neg  # synthetic single-positive penalty


def test_train_config_defaults() -> None:
    cfg = TrainConfig()
    assert cfg.lr == 1e-3
    assert cfg.weight_decay == 1e-5
    assert cfg.patience == 20
    assert cfg.grad_clip == 1.0
    assert cfg.batch_size == 32
