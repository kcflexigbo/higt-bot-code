"""Window-of-flows → PyTorch Geometric Data graph.

This is the bridge between Phase 2 (parsing) and the rest of the pipeline.
It reads only the canonical flow DataFrame schema (src/data/schema.py).

Design decisions baked in here, with citations to the plan:
- Node identity = (IP). Drop IPs with fewer than `min_flows_per_node` flows
  in the window (default 3) — Phase 3 §"Design decisions".
- Edge = one *directed* edge per (src_ip, dst_ip), aggregating all flows
  between them in the window.
- Node label: "any bot flow involving the node → bot". This is the primary
  rule per the revised plan; the strict >50% rule is an ablation row.
- Graph label: bot iff any node is bot.
- Max nodes per graph: 400. Required for DiffPool dense adjacency on 16 GB
  VRAM. Windows exceeding the cap are filtered down by keeping the
  highest-activity nodes (flow count).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.features import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    compute_edges,
    compute_node_features,
)


@dataclass
class GraphConfig:
    """Knobs for graph construction."""

    min_flows_per_node: int = 3
    max_nodes: int = 400


def _select_active_nodes(flows: pd.DataFrame, cfg: GraphConfig) -> list[str]:
    """Pick which IPs become graph nodes.

    Rule: include any IP that appears (as src or dst) in at least
    `min_flows_per_node` flows. Then, if we exceed `max_nodes`, keep the
    top-N most active by total flow count.
    """
    src_counts = flows["src_ip"].value_counts()
    dst_counts = flows["dst_ip"].value_counts()
    total = src_counts.add(dst_counts, fill_value=0)
    eligible = total[total >= cfg.min_flows_per_node]
    if eligible.empty:
        return []
    sorted_ips = eligible.sort_values(ascending=False)
    keep = sorted_ips.iloc[: cfg.max_nodes].index.tolist()
    keep.sort()                              # deterministic node ordering
    return keep


def _node_labels(flows: pd.DataFrame, nodes: list[str]) -> np.ndarray:
    """1 if the node participated (as src or dst) in any bot flow."""
    bot_ips = pd.concat([
        flows.loc[flows["label"] == "bot", "src_ip"],
        flows.loc[flows["label"] == "bot", "dst_ip"],
    ]).unique()
    bot_set = set(bot_ips)
    return np.array([1 if ip in bot_set else 0 for ip in nodes], dtype=np.int64)


def build_graph(
    flows: pd.DataFrame,
    *,
    scenario: str,
    window_idx: int,
    window_start: pd.Timestamp,
    window_seconds: float,
    cfg: GraphConfig | None = None,
) -> Data | None:
    """Build a single PyG Data object from one window of flows.

    Returns None if the window has no eligible nodes.
    """
    if cfg is None:
        cfg = GraphConfig()

    nodes = _select_active_nodes(flows, cfg)
    if not nodes:
        return None

    # Restrict flows to those between selected nodes (so node features and
    # edges agree on the node universe).
    keep_mask = flows["src_ip"].isin(nodes) & flows["dst_ip"].isin(nodes)
    win_flows = flows.loc[keep_mask].reset_index(drop=True)

    x = compute_node_features(win_flows, nodes, window_seconds)
    edge_index, edge_attr = compute_edges(win_flows, nodes)
    y_node = _node_labels(win_flows, nodes)
    y_graph = np.int64(1) if int(y_node.sum()) > 0 else np.int64(0)

    data = Data(
        x=torch.from_numpy(x).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr).float(),
        y=torch.from_numpy(y_node).long(),
        graph_y=torch.tensor([y_graph], dtype=torch.long),
    )
    data.scenario = scenario
    data.window_idx = int(window_idx)
    data.window_start = pd.Timestamp(window_start).isoformat()
    data.num_node_features_named = len(NODE_FEATURE_NAMES)
    data.num_edge_features_named = len(EDGE_FEATURE_NAMES)
    data.node_ips = nodes  # stash for visualization / debugging
    return data


def graph_summary(g: Data) -> dict[str, Any]:
    """One-line stats dict for a Data object — useful for logging."""
    n_nodes = int(g.num_nodes)
    n_edges = int(g.edge_index.size(1)) if g.edge_index is not None else 0
    return {
        "scenario": getattr(g, "scenario", "?"),
        "window_idx": getattr(g, "window_idx", -1),
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "n_bot_nodes": int(g.y.sum()),
        "graph_label": int(g.graph_y.item()) if hasattr(g, "graph_y") else -1,
    }
