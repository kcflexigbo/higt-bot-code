"""Load PyG graphs from `data/graphs/<scenario>/window_*.pt` and apply the
shared train/val/test split described in `configs/split.yaml`.

This is the *one* place that owns splitting logic. Every Phase 4+ baseline
loads its data through here so results are directly comparable.

Split rule (per-scenario chronological):
- Sort `window_*.pt` files by window index (already chronological).
- Take first `train_frac` to train, next `val_frac` to val, rest to test.
- Holdout-test scenarios contribute their *entire* window sequence to test
  (used to measure cross-scenario generalization).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch_geometric.data import Data

GRAPHS_DIR = Path("data/graphs")
DEFAULT_SPLIT = Path("configs/split.yaml")

SplitName = Literal["train", "val", "test"]


@dataclass
class SplitSpec:
    version: int
    seed: int
    train_frac: float
    val_frac: float
    train_scenarios: list[str]
    holdout_test_scenarios: list[str]
    excluded: list[str] = field(default_factory=list)

    @property
    def test_frac(self) -> float:
        return max(0.0, 1.0 - self.train_frac - self.val_frac)

    @classmethod
    def load(cls, path: Path = DEFAULT_SPLIT) -> "SplitSpec":
        with path.open() as f:
            cfg = yaml.safe_load(f)
        return cls(
            version=int(cfg["split_version"]),
            seed=int(cfg["seed"]),
            train_frac=float(cfg["train_frac"]),
            val_frac=float(cfg["val_frac"]),
            train_scenarios=list(cfg["train"]),
            holdout_test_scenarios=list(cfg.get("holdout_test", [])),
            excluded=list(cfg.get("excluded", [])),
        )


def _list_window_files(scenario: str, graphs_dir: Path = GRAPHS_DIR) -> list[Path]:
    """Sorted window_*.pt files for a scenario (chronological by index)."""
    d = graphs_dir / scenario
    return sorted(d.glob("window_*.pt"))


def _chronological_split(
    files: list[Path], train_frac: float, val_frac: float
) -> dict[SplitName, list[Path]]:
    """First train_frac → train, next val_frac → val, rest → test."""
    n = len(files)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    return {
        "train": files[:n_tr],
        "val": files[n_tr : n_tr + n_va],
        "test": files[n_tr + n_va :],
    }


def load_graphs(files: list[Path]) -> list[Data]:
    """Load PyG Data objects from a list of .pt files."""
    return [torch.load(fp, weights_only=False) for fp in files]


def fit_feature_scaler(graphs: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std on the pooled node features of `graphs`
    (typically the training split). Use these to standardize every split so
    GNNs see comparable magnitudes across the byte_rate (millions) vs
    port_entropy (0–10) range. Floors std at 1e-6 to avoid division blowup.
    """
    X = torch.cat([g.x for g in graphs], dim=0)
    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp(min=1e-6)
    return mean, std


def apply_feature_scaler(graphs: list[Data], mean: torch.Tensor, std: torch.Tensor) -> None:
    """Standardize in-place. Mutates each graph's `x`."""
    for g in graphs:
        g.x = ((g.x - mean) / std).float()


def load_split(
    name: SplitName,
    spec: SplitSpec | None = None,
    graphs_dir: Path = GRAPHS_DIR,
) -> list[Data]:
    """Load all graphs belonging to one split of the canonical config.

    `test` includes both the chronological tail of train scenarios AND every
    graph from holdout_test scenarios. train and val draw only from
    train_scenarios.
    """
    if spec is None:
        spec = SplitSpec.load()
    files: list[Path] = []
    for sc in spec.train_scenarios:
        per_scenario = _chronological_split(
            _list_window_files(sc, graphs_dir), spec.train_frac, spec.val_frac
        )
        files.extend(per_scenario[name])
    if name == "test":
        for sc in spec.holdout_test_scenarios:
            files.extend(_list_window_files(sc, graphs_dir))
    return load_graphs(files)


def split_summary(spec: SplitSpec, graphs_dir: Path = GRAPHS_DIR) -> dict:
    """Return per-split counts (graphs, nodes, bot-nodes, bot-graphs)."""
    out: dict = {}
    for name in ("train", "val", "test"):
        graphs = load_split(name, spec, graphs_dir)  # type: ignore[arg-type]
        n_nodes = sum(int(g.num_nodes) for g in graphs)
        n_bot_nodes = sum(int(g.y.sum()) for g in graphs)
        n_bot_graphs = sum(int(g.graph_y.item()) for g in graphs)
        out[name] = {
            "graphs": len(graphs),
            "nodes": n_nodes,
            "bot_nodes": n_bot_nodes,
            "bot_graphs": n_bot_graphs,
            "bot_node_frac": (n_bot_nodes / n_nodes) if n_nodes else 0.0,
        }
    return out
