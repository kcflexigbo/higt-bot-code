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


def _populate_graphs(tmp_graphs: Path, scenarios: dict[str, int]) -> None:
    """Create empty window_*.pt files for each scenario name."""
    for scenario, n in scenarios.items():
        d = tmp_graphs / scenario
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"window_{i:05d}.pt").write_bytes(b"")


def test_load_split_files_lodo_mode_assigns_test_correctly(tmp_path: Path) -> None:
    """In LODO mode, every window of lodo_test_scenarios goes to test."""
    from src.data.dataset import load_split_files

    graphs_dir = tmp_path / "graphs"
    _populate_graphs(graphs_dir, {
        "ctu13-1": 10,
        "ctu13-2": 10,
        "iot23-1-1": 5,
        "iot23-3-1": 7,
    })
    spec = SplitSpec(
        version="lodo-iot23-v1",
        seed=42,
        train_frac=0.8,
        val_frac=0.2,
        train_scenarios=["ctu13-1", "ctu13-2"],
        holdout_test_scenarios=[],
        holdout_dataset="iot23",
        lodo_test_scenarios=["iot23-1-1", "iot23-3-1"],
    )

    tr = load_split_files("train", spec, graphs_dir)
    va = load_split_files("val", spec, graphs_dir)
    te = load_split_files("test", spec, graphs_dir)

    # Train scenarios chronologically split 0.8/0.2 each: ctu13-1 -> 8/2, ctu13-2 -> 8/2.
    assert len(tr) == 16
    assert len(va) == 4
    # Test = full window sequences of every LODO test scenario: 5 + 7 = 12.
    assert len(te) == 12
    # No test scenario file comes from the train_scenarios pool.
    assert all("ctu13" not in str(p) for p in te)
    # No train/val file comes from the LODO test pool.
    assert all("iot23" not in str(p) for p in tr + va)


def test_load_split_files_lodo_mode_ignores_holdout_test_scenarios(tmp_path: Path) -> None:
    """The existing holdout_test_scenarios field is ignored in LODO mode."""
    from src.data.dataset import load_split_files

    graphs_dir = tmp_path / "graphs"
    _populate_graphs(graphs_dir, {
        "ctu13-1": 10,
        "medbiot-torii_mal_all": 4,
        "iot23-1-1": 5,
    })
    spec = SplitSpec(
        version="lodo-iot23-v1",
        seed=42,
        train_frac=0.8,
        val_frac=0.2,
        train_scenarios=["ctu13-1", "medbiot-torii_mal_all"],
        holdout_test_scenarios=["should-not-appear-anywhere"],  # ignored in LODO mode
        holdout_dataset="iot23",
        lodo_test_scenarios=["iot23-1-1"],
    )

    te = load_split_files("test", spec, graphs_dir)
    # Only iot23-1-1 windows: 5 of them. holdout_test_scenarios is ignored.
    assert len(te) == 5
    assert all("iot23-1-1" in str(p) for p in te)


def test_load_split_files_non_lodo_mode_unchanged(tmp_path: Path) -> None:
    """Regression: when holdout_dataset is None, behavior matches the old code path
    (chronological 60/20/20 + holdout_test_scenarios appended to test)."""
    from src.data.dataset import load_split_files

    graphs_dir = tmp_path / "graphs"
    _populate_graphs(graphs_dir, {
        "a": 10,
        "b": 10,
        "hot": 5,
    })
    spec = SplitSpec(
        version=1,
        seed=42,
        train_frac=0.6,
        val_frac=0.2,
        train_scenarios=["a", "b"],
        holdout_test_scenarios=["hot"],
    )

    tr = load_split_files("train", spec, graphs_dir)
    va = load_split_files("val", spec, graphs_dir)
    te = load_split_files("test", spec, graphs_dir)

    # 60/20/20 chronological: 6/2/2 per scenario, x2 = 12/4/4. Plus 5 holdout in test.
    assert len(tr) == 12
    assert len(va) == 4
    assert len(te) == 4 + 5
