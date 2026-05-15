"""Unit tests for Phase 3: window iteration, feature extraction, graph build."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    compute_edges,
    compute_node_features,
)
from src.data.graph import GraphConfig, build_graph
from src.data.schema import FLOW_COLUMNS, coerce_to_schema
from src.data.window import WindowConfig, iter_windows


# --------------------------------------------------------------------------- #
# Synthetic flow fixture                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_flows() -> pd.DataFrame:
    """30 flows over 600 s, 6 distinct hosts, two bots (A, B) talking to each
    other plus three benign hosts and one server."""
    rng = np.random.default_rng(42)
    rows = []
    base = pd.Timestamp("2020-01-01", tz="UTC")
    flow_id = 0

    def emit(t_off: float, src: str, dst: str, lbl: str, port: int = 80, proto: str = "tcp"):
        nonlocal flow_id
        rows.append({
            "flow_id": flow_id,
            "scenario": "synthetic",
            "src_ip": src, "dst_ip": dst,
            "src_port": np.int32(rng.integers(1024, 65535)),
            "dst_port": np.int32(port),
            "protocol": proto,
            "start_time": base + pd.Timedelta(seconds=t_off),
            "end_time": base + pd.Timedelta(seconds=t_off + 0.5),
            "duration_s": 0.5,
            "bytes_fwd": 500, "bytes_bwd": 1000,
            "pkts_fwd": 5, "pkts_bwd": 10,
            "mean_iat_ms": float("nan"),
            "std_iat_ms": float("nan"),
            "min_pkt_size": float("nan"),
            "max_pkt_size": float("nan"),
            "label": lbl,
            "detailed_label": lbl,
        })
        flow_id += 1

    # Window 1: 0..299 s — busy bot mesh + benign traffic
    for t in range(0, 100, 10):
        emit(float(t), "10.0.0.1", "10.0.0.2", "bot", port=6667)   # A→B
        emit(float(t) + 1, "10.0.0.2", "10.0.0.1", "bot", port=6667)  # B→A
        emit(float(t) + 2, "10.0.0.3", "8.8.8.8", "benign", port=53, proto="udp")  # benign
    # Window 2: 300..599 s — only benign
    for t in range(300, 400, 10):
        emit(float(t), "10.0.0.4", "8.8.8.8", "benign", port=53, proto="udp")
        emit(float(t) + 5, "10.0.0.5", "1.1.1.1", "benign", port=443)
        emit(float(t) + 8, "10.0.0.5", "1.1.1.1", "benign", port=443)

    df = pd.DataFrame(rows)
    df = coerce_to_schema(df[FLOW_COLUMNS])
    return df


# --------------------------------------------------------------------------- #
# Window iteration                                                            #
# --------------------------------------------------------------------------- #


def test_iter_windows_count(tiny_flows: pd.DataFrame) -> None:
    cfg = WindowConfig(window_s=300, min_flows=5, min_nodes=2)
    wins = list(iter_windows(tiny_flows, cfg))
    # 30 flows, 30/30 = ~30 (some in window 1, some in window 2).
    assert len(wins) == 2
    # Window indices are dense and start at 0.
    assert [w[0] for w in wins] == [0, 1]


def test_iter_windows_skips_tiny(tiny_flows: pd.DataFrame) -> None:
    # Each window has 30 flows. Raising min_flows above that should skip both.
    cfg = WindowConfig(window_s=300, min_flows=31, min_nodes=2)
    wins = list(iter_windows(tiny_flows, cfg))
    assert len(wins) == 0


# --------------------------------------------------------------------------- #
# Feature extraction                                                          #
# --------------------------------------------------------------------------- #


def test_compute_node_features_shape(tiny_flows: pd.DataFrame) -> None:
    win = tiny_flows.iloc[:30]
    nodes = sorted(pd.concat([win["src_ip"], win["dst_ip"]]).unique())
    feats = compute_node_features(win, nodes, window_seconds=300.0)
    assert feats.shape == (len(nodes), len(NODE_FEATURE_NAMES))
    assert feats.dtype == np.float32
    # Bot A and bot B should have non-zero fan_out/fan_in to each other.
    a = nodes.index("10.0.0.1")
    assert feats[a, 0] > 0  # fan_out
    assert feats[a, 1] > 0  # fan_in


def test_compute_edges_shape(tiny_flows: pd.DataFrame) -> None:
    win = tiny_flows.iloc[:30]
    nodes = sorted(pd.concat([win["src_ip"], win["dst_ip"]]).unique())
    ei, ea = compute_edges(win, nodes)
    assert ei.dtype == np.int64
    assert ea.dtype == np.float32
    assert ei.shape[0] == 2
    assert ei.shape[1] == ea.shape[0]
    assert ea.shape[1] == len(EDGE_FEATURE_NAMES)
    # All edge endpoints are valid node indices.
    assert ei.min() >= 0 and ei.max() < len(nodes)


# --------------------------------------------------------------------------- #
# Graph build                                                                 #
# --------------------------------------------------------------------------- #


def test_build_graph_basic(tiny_flows: pd.DataFrame) -> None:
    win = tiny_flows.iloc[:30]
    g = build_graph(
        win, scenario="synthetic", window_idx=0,
        window_start=win["start_time"].min(), window_seconds=300.0,
        cfg=GraphConfig(min_flows_per_node=1, max_nodes=400),
    )
    assert g is not None
    assert g.num_nodes >= 4
    assert g.edge_index.size(1) > 0
    assert g.x.size(1) == len(NODE_FEATURE_NAMES)
    assert g.edge_attr.size(1) == len(EDGE_FEATURE_NAMES)
    # Bot A and bot B are present and labeled bot.
    ip_to_idx = {ip: i for i, ip in enumerate(g.node_ips)}
    assert int(g.y[ip_to_idx["10.0.0.1"]]) == 1
    assert int(g.y[ip_to_idx["10.0.0.2"]]) == 1
    assert int(g.y[ip_to_idx["10.0.0.3"]]) == 0  # benign
    assert int(g.graph_y.item()) == 1


def test_build_graph_max_nodes_cap(tiny_flows: pd.DataFrame) -> None:
    g = build_graph(
        tiny_flows.iloc[:30], scenario="synthetic", window_idx=0,
        window_start=tiny_flows["start_time"].min(), window_seconds=300.0,
        cfg=GraphConfig(min_flows_per_node=1, max_nodes=3),
    )
    assert g is not None
    assert g.num_nodes <= 3
