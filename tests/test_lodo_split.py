"""LODO (leave-one-dataset-out) unit tests for src/data/dataset.py.

These tests exercise the optional LODO branch added in the LODO experiment.
They never touch the canonical configs/split.yaml — instead they construct
SplitSpec objects directly so the tests are independent of disk state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.dataset import SplitSpec


def test_split_spec_lodo_fields_default_to_none_and_empty() -> None:
    """SplitSpec should remain backward compatible: when LODO fields are not
    set, holdout_dataset is None and lodo_test_scenarios is empty."""
    spec = SplitSpec(
        version=1,
        seed=42,
        train_frac=0.6,
        val_frac=0.2,
        train_scenarios=["scenario-a", "scenario-b"],
        holdout_test_scenarios=["scenario-c"],
    )
    assert spec.holdout_dataset is None
    assert spec.lodo_test_scenarios == []


def test_split_spec_lodo_fields_settable() -> None:
    """LODO fields accept explicit values."""
    spec = SplitSpec(
        version=2,
        seed=42,
        train_frac=0.8,
        val_frac=0.2,
        train_scenarios=["a", "b"],
        holdout_test_scenarios=[],
        holdout_dataset="iot23",
        lodo_test_scenarios=["iot23-1-1", "iot23-3-1"],
    )
    assert spec.holdout_dataset == "iot23"
    assert spec.lodo_test_scenarios == ["iot23-1-1", "iot23-3-1"]
