"""Glue between Phase 3 graph files and Phase 5 flow sequences.

Two responsibilities:
  1. `attach_flow_sequences(graphs, window_flows, ...)` — for an in-memory
     list of PyG `Data` objects, attach `.flows` and `.flow_mask` tensors
     aligned to each graph's existing `node_ips` ordering.
  2. `load_flow_sequences_into(graphs, root)` — read precomputed
     `data/flow_seqs/<scenario>/window_<NNNN>.pt` files (written by
     scripts/build_flow_sequences.py) and attach them to the graphs.

The cache path is the fast path used during training; `attach_flow_sequences`
is used by the precompute script and by tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.flow_seq import FLOW_FEATURE_DIM, build_node_sequences

FLOW_SEQS_DIR = Path("data/flow_seqs")


def attach_flow_sequences(
    graphs: Iterable[Data],
    *,
    window_flows: dict[tuple[str, int], pd.DataFrame],
    window_seconds: float,
    max_flows: int = 256,
    seed: int = 42,
    eval_mode: bool = False,
) -> None:
    """In-place: for each graph, look up the matching flow DataFrame by
    (scenario, window_idx), build sequences aligned to graph.node_ips, attach
    `.flows: [N, max_flows, F]` and `.flow_mask: [N, max_flows]`.

    If a graph has no matching DataFrame, raise — silent zero-fill would be a
    nasty bug.
    """
    for g in graphs:
        key = (g.scenario, int(g.window_idx))
        if key not in window_flows:
            raise KeyError(f"no flows for {key}")
        df = window_flows[key]
        ws = pd.Timestamp(g.window_start)
        flows_arr, mask_arr = build_node_sequences(
            df,
            node_ips=list(g.node_ips),
            window_start=ws,
            window_seconds=window_seconds,
            max_flows=max_flows,
            seed=seed,
            eval_mode=eval_mode,
        )
        g.flows = torch.from_numpy(flows_arr)
        g.flow_mask = torch.from_numpy(mask_arr)


def cache_path(scenario: str, window_idx: int, root: Path = FLOW_SEQS_DIR) -> Path:
    return root / scenario / f"window_{window_idx:05d}.pt"


def save_flow_sequences(graph: Data, root: Path = FLOW_SEQS_DIR) -> Path:
    """Persist {flows, flow_mask, node_ips, scenario, window_idx} to disk so the
    training loop loads sequences without re-walking parquets."""
    out = cache_path(graph.scenario, int(graph.window_idx), root)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "flows": graph.flows,
        "flow_mask": graph.flow_mask,
        "node_ips": list(graph.node_ips),
        "scenario": graph.scenario,
        "window_idx": int(graph.window_idx),
    }, out)
    return out


def load_flow_sequences_into(
    graphs: Iterable[Data], root: Path = FLOW_SEQS_DIR
) -> None:
    """Read cached `flows`/`flow_mask` from disk and attach to graphs.

    Asserts that cached node_ips match graph.node_ips — order is the contract.
    """
    for g in graphs:
        p = cache_path(g.scenario, int(g.window_idx), root)
        blob = torch.load(p, weights_only=False)
        assert list(blob["node_ips"]) == list(g.node_ips), (
            f"flow_seq/graph node_ips mismatch for {g.scenario} w{g.window_idx}"
        )
        assert blob["flows"].shape[2] == FLOW_FEATURE_DIM
        g.flows = blob["flows"]
        g.flow_mask = blob["flow_mask"]


class FlowSeqGraphDataset(Dataset):
    """Lazy loader: graph structure + flow sequences read per batch from disk.

    Avoids pinning ~4k windows x [N,256,13] float tensors in RAM at startup
    (which can exceed 10 GB and freeze the machine before epoch 1).
    """

    def __init__(
        self,
        graph_files: list[Path],
        *,
        flow_root: Path = FLOW_SEQS_DIR,
        edge_mean: torch.Tensor | None = None,
        edge_std: torch.Tensor | None = None,
        node_mean: torch.Tensor | None = None,
        node_std: torch.Tensor | None = None,
    ) -> None:
        self.graph_files = list(graph_files)
        self.flow_root = flow_root
        self.edge_mean = edge_mean
        self.edge_std = edge_std
        self.node_mean = node_mean
        self.node_std = node_std

    def __len__(self) -> int:
        return len(self.graph_files)

    def __getitem__(self, idx: int) -> Data:
        g = torch.load(self.graph_files[idx], weights_only=False)
        p = cache_path(g.scenario, int(g.window_idx), self.flow_root)
        blob = torch.load(p, weights_only=False)
        if list(blob["node_ips"]) != list(g.node_ips):
            raise ValueError(
                f"flow_seq/graph node_ips mismatch for {g.scenario} w{g.window_idx}"
            )
        g.flows = blob["flows"]
        g.flow_mask = blob["flow_mask"]
        if self.edge_mean is not None and self.edge_std is not None and g.edge_attr is not None:
            g.edge_attr = ((g.edge_attr - self.edge_mean) / self.edge_std).float()
        if self.node_mean is not None and self.node_std is not None and g.x is not None:
            g.x = ((g.x - self.node_mean) / self.node_std).float()
        return g
