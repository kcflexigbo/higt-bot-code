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


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "split.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_split_spec_load_without_lodo_fields(tmp_path: Path) -> None:
    """Loading a YAML without LODO keys yields holdout_dataset=None."""
    cfg = _write_yaml(tmp_path, """
split_version: 7
seed: 42
train_frac: 0.6
val_frac: 0.2
train: [a, b]
holdout_test: [c]
""")
    spec = SplitSpec.load(cfg)
    assert spec.version == 7
    assert spec.holdout_dataset is None
    assert spec.lodo_test_scenarios == []
    assert spec.train_scenarios == ["a", "b"]


def test_split_spec_load_with_lodo_fields(tmp_path: Path) -> None:
    """Loading a YAML with LODO keys populates the new fields."""
    cfg = _write_yaml(tmp_path, """
split_version: lodo-iot23-v1
seed: 42
train_frac: 0.8
val_frac: 0.2
holdout_dataset: iot23
train: [ctu13-1, ctu13-2, medbiot-bashlite_mal_CC_all]
lodo_test_scenarios: [iot23-1-1, iot23-3-1, iot23-8-1]
excluded: [ctu13-5]
""")
    spec = SplitSpec.load(cfg)
    assert spec.holdout_dataset == "iot23"
    assert spec.lodo_test_scenarios == ["iot23-1-1", "iot23-3-1", "iot23-8-1"]
    assert spec.train_scenarios == ["ctu13-1", "ctu13-2", "medbiot-bashlite_mal_CC_all"]
    assert spec.excluded == ["ctu13-5"]
