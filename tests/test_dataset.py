"""Unit tests for src/data/dataset.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.data.dataset import (
    SplitSpec,
    _chronological_split,
    apply_feature_scaler,
    fit_feature_scaler,
    load_split,
)


def _make_graph(n_nodes: int, n_feat: int, scenario: str, idx: int) -> Data:
    g = Data(
        x=torch.randn(n_nodes, n_feat),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, 4),
        y=torch.zeros(n_nodes, dtype=torch.long),
        graph_y=torch.tensor([0], dtype=torch.long),
    )
    g.scenario = scenario
    g.window_idx = idx
    return g


def test_chronological_split_proportions() -> None:
    files = [Path(f"window_{i:05d}.pt") for i in range(100)]
    out = _chronological_split(files, train_frac=0.6, val_frac=0.2)
    assert len(out["train"]) == 60
    assert len(out["val"]) == 20
    assert len(out["test"]) == 20
    # Chronological order is preserved.
    assert out["train"] == files[:60]
    assert out["test"] == files[80:]


def test_chronological_split_handles_remainder() -> None:
    """7 windows with 0.6/0.2: 4 train, 1 val, 2 test."""
    files = [Path(f"w_{i}.pt") for i in range(7)]
    out = _chronological_split(files, train_frac=0.6, val_frac=0.2)
    assert len(out["train"]) == 4
    assert len(out["val"]) == 1
    assert len(out["test"]) == 2


def test_chronological_split_no_overlap() -> None:
    """No file appears in more than one split."""
    files = [Path(f"w_{i}.pt") for i in range(50)]
    out = _chronological_split(files, train_frac=0.6, val_frac=0.2)
    tr = set(out["train"]); va = set(out["val"]); te = set(out["test"])
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert tr | va | te == set(files)


def test_split_spec_load() -> None:
    spec = SplitSpec.load()
    assert spec.version >= 1
    assert 0.0 < spec.train_frac < 1.0
    assert 0.0 < spec.val_frac < 1.0
    assert spec.train_frac + spec.val_frac <= 1.0
    assert spec.test_frac > 0.0
    assert len(spec.train_scenarios) > 0
    # Train scenarios and holdouts are disjoint.
    assert set(spec.train_scenarios).isdisjoint(spec.holdout_test_scenarios)


def test_fit_apply_feature_scaler() -> None:
    graphs = [_make_graph(50, 6, "synthetic", i) for i in range(4)]
    # Set known means so we can verify scaler.
    for g in graphs:
        g.x = g.x * 100.0 + 50.0
    mean, std = fit_feature_scaler(graphs)
    assert mean.shape == (6,)
    assert std.shape == (6,)
    assert (std > 0).all()

    apply_feature_scaler(graphs, mean, std)
    pooled = torch.cat([g.x for g in graphs], dim=0)
    # After standardization, mean ≈ 0 and std ≈ 1 per feature.
    assert torch.allclose(pooled.mean(dim=0), torch.zeros(6), atol=1e-5)
    assert torch.allclose(pooled.std(dim=0), torch.ones(6), atol=1e-2)


def test_feature_scaler_floors_zero_std() -> None:
    """Constant features should not divide-by-zero."""
    g = _make_graph(10, 3, "synthetic", 0)
    g.x = torch.ones(10, 3) * 5.0  # constant
    mean, std = fit_feature_scaler([g])
    # All means = 5, all stds floored at 1e-6 (not 0).
    assert torch.allclose(mean, torch.full((3,), 5.0))
    assert (std >= 1e-6).all()
    apply_feature_scaler([g], mean, std)
    # Result is (5-5)/floor = 0; no NaN.
    assert torch.isfinite(g.x).all()


def test_load_split_returns_data_objects() -> None:
    """Smoke test against the live graphs directory.

    Skips gracefully if data/graphs/ is empty (CI / fresh checkout).
    """
    spec = SplitSpec.load()
    train_dir = Path("data/graphs") / spec.train_scenarios[0]
    if not train_dir.exists() or not any(train_dir.glob("window_*.pt")):
        pytest.skip("no graphs on disk — skipping integration test")
    tr = load_split("train", spec)
    assert len(tr) > 0
    g = tr[0]
    assert hasattr(g, "x") and hasattr(g, "y") and hasattr(g, "edge_index")
    assert g.x.dim() == 2
