"""Unit tests for src/data/flow_seq.py — per-flow feature extraction and
per-(node, window) sequence builder. Pure numpy / pandas — no torch
required for these tests (torch tensors are produced upstream)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.flow_seq import (
    FLOW_FEATURE_DIM,
    FLOW_FEATURE_NAMES,
    build_node_sequences,
    flow_features_for_node,
)


def _sample_flows(window_start: pd.Timestamp) -> pd.DataFrame:
    """Three flows in a 300s window: A→B (tcp), B→A (tcp), A→C (udp)."""
    rows = [
        # flow 0: A → B  tcp, t=0s
        {"flow_id": 0, "scenario": "test", "src_ip": "A", "dst_ip": "B",
         "src_port": 1000, "dst_port": 80, "protocol": "tcp",
         "start_time": window_start + pd.Timedelta(seconds=0),
         "end_time":   window_start + pd.Timedelta(seconds=1),
         "duration_s": 1.0, "bytes_fwd": 100, "bytes_bwd": 50,
         "pkts_fwd": 2, "pkts_bwd": 1,
         "mean_iat_ms": 10.0, "std_iat_ms": 1.0,
         "min_pkt_size": 60.0, "max_pkt_size": 1500.0,
         "label": "bot", "detailed_label": ""},
        # flow 1: B → A  tcp, t=30s
        {"flow_id": 1, "scenario": "test", "src_ip": "B", "dst_ip": "A",
         "src_port": 80, "dst_port": 1000, "protocol": "tcp",
         "start_time": window_start + pd.Timedelta(seconds=30),
         "end_time":   window_start + pd.Timedelta(seconds=31),
         "duration_s": 1.0, "bytes_fwd": 200, "bytes_bwd": 80,
         "pkts_fwd": 3, "pkts_bwd": 2,
         "mean_iat_ms": float("nan"), "std_iat_ms": float("nan"),
         "min_pkt_size": float("nan"), "max_pkt_size": float("nan"),
         "label": "benign", "detailed_label": ""},
        # flow 2: A → C  udp, t=150s
        {"flow_id": 2, "scenario": "test", "src_ip": "A", "dst_ip": "C",
         "src_port": 5353, "dst_port": 5353, "protocol": "udp",
         "start_time": window_start + pd.Timedelta(seconds=150),
         "end_time":   window_start + pd.Timedelta(seconds=151),
         "duration_s": 1.0, "bytes_fwd": 64, "bytes_bwd": 0,
         "pkts_fwd": 1, "pkts_bwd": 0,
         "mean_iat_ms": 5.0, "std_iat_ms": 0.0,
         "min_pkt_size": 64.0, "max_pkt_size": 64.0,
         "label": "benign", "detailed_label": ""},
    ]
    df = pd.DataFrame(rows)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"]   = pd.to_datetime(df["end_time"],   utc=True)
    return df


def test_feature_names_match_dim() -> None:
    assert len(FLOW_FEATURE_NAMES) == FLOW_FEATURE_DIM == 13


def test_flow_features_for_node_a_outgoing_and_incoming() -> None:
    """Node A sees flow 0 (outgoing tcp), flow 1 (incoming tcp), flow 2 (outgoing udp).
    The sequence is sorted by start_time and direction is relative to A."""
    ws = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    df = _sample_flows(ws)
    seq = flow_features_for_node(df, focal="A", window_start=ws, window_seconds=300.0)
    assert seq.shape == (3, 13)
    # flow 0: tcp + outgoing
    assert seq[0, 0] == 1.0 and seq[0, 4] == 1.0
    # flow 1: tcp + incoming
    assert seq[1, 0] == 1.0 and seq[1, 4] == 0.0
    # flow 2: udp + outgoing
    assert seq[2, 1] == 1.0 and seq[2, 4] == 1.0
    # norm_time strictly increasing
    assert seq[0, 12] < seq[1, 12] < seq[2, 12]
    # has_iat: True on flows 0 and 2, False on 1
    assert seq[0, 9] == 1.0 and seq[1, 9] == 0.0 and seq[2, 9] == 1.0
    # log1p_total_bytes for flow 0 = log1p(150)
    assert seq[0, 6] == pytest.approx(np.log1p(150.0), rel=1e-6)


def test_flow_features_for_node_excludes_unrelated() -> None:
    """Node C only sees flow 2; nothing else."""
    ws = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    df = _sample_flows(ws)
    seq = flow_features_for_node(df, focal="C", window_start=ws, window_seconds=300.0)
    assert seq.shape == (1, 13)
    assert seq[0, 1] == 1.0     # udp
    assert seq[0, 4] == 0.0     # C is dst → incoming


def test_build_node_sequences_padding_and_mask() -> None:
    """Three nodes, max_flows=2 → flow_seqs is [3, 2, 13]; pad mask True on padded rows."""
    ws = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    df = _sample_flows(ws)
    flows, mask = build_node_sequences(
        df, node_ips=["A", "B", "C"], window_start=ws,
        window_seconds=300.0, max_flows=2, seed=42,
    )
    assert flows.shape == (3, 2, 13)
    assert mask.shape == (3, 2)
    # A has 3 flows → 2 kept, both unpadded
    assert not mask[0, 0] and not mask[0, 1]
    # C has 1 flow → 1 unpadded, 1 padded
    assert not mask[2, 0] and mask[2, 1]
    # Padded row is all zeros
    assert np.all(flows[2, 1] == 0.0)


def test_build_node_sequences_evenly_spaced_in_eval() -> None:
    """eval=True selects evenly-spaced sub-samples deterministically."""
    ws = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    df = _sample_flows(ws)
    # Replicate A's outgoing flow 100 times to exceed max_flows.
    rows_a = df[df["src_ip"] == "A"].copy()
    big = pd.concat([df] + [rows_a.assign(flow_id=lambda r, k=k: 100 + k * len(rows_a))
                             for k in range(40)], ignore_index=True)
    big["flow_id"] = np.arange(len(big), dtype=np.int64)
    flows1, mask1 = build_node_sequences(big, node_ips=["A"], window_start=ws,
                                          window_seconds=300.0, max_flows=8,
                                          seed=0, eval_mode=True)
    flows2, mask2 = build_node_sequences(big, node_ips=["A"], window_start=ws,
                                          window_seconds=300.0, max_flows=8,
                                          seed=999, eval_mode=True)
    # Determinism: same input → same output regardless of seed in eval mode.
    np.testing.assert_array_equal(flows1, flows2)
    np.testing.assert_array_equal(mask1, mask2)


def test_build_node_sequences_random_sample_uses_seed() -> None:
    """eval_mode=False: same seed → same sample; different seed → (usually) different."""
    ws = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    df = _sample_flows(ws)
    rows_a = df[df["src_ip"] == "A"].copy()
    big = pd.concat([df] + [rows_a] * 40, ignore_index=True)
    big["flow_id"] = np.arange(len(big), dtype=np.int64)
    flows_a, _ = build_node_sequences(big, node_ips=["A"], window_start=ws,
                                       window_seconds=300.0, max_flows=8,
                                       seed=7, eval_mode=False)
    flows_b, _ = build_node_sequences(big, node_ips=["A"], window_start=ws,
                                       window_seconds=300.0, max_flows=8,
                                       seed=7, eval_mode=False)
    np.testing.assert_array_equal(flows_a, flows_b)


def test_build_node_sequences_random_sample_different_seeds_differ() -> None:
    """eval_mode=False: different seeds should (almost always) yield different samples.
    Use distinct per-flow feature vectors so a permutation actually shows up in the output."""
    ws = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    # 64 distinct outgoing flows from A → various destinations, spaced 1s apart.
    rows = []
    for k in range(64):
        rows.append({
            "flow_id": k, "scenario": "test", "src_ip": "A", "dst_ip": f"D{k}",
            "src_port": 1000 + k, "dst_port": 80, "protocol": "tcp",
            "start_time": ws + pd.Timedelta(seconds=k),
            "end_time":   ws + pd.Timedelta(seconds=k + 1),
            "duration_s": 1.0, "bytes_fwd": 100 + k, "bytes_bwd": 50,
            "pkts_fwd": 2, "pkts_bwd": 1,
            "mean_iat_ms": float(k), "std_iat_ms": 1.0,
            "min_pkt_size": 60.0, "max_pkt_size": 1500.0,
            "label": "benign", "detailed_label": ""})
    big = pd.DataFrame(rows)
    big["start_time"] = pd.to_datetime(big["start_time"], utc=True)
    big["end_time"]   = pd.to_datetime(big["end_time"],   utc=True)
    flows_a, _ = build_node_sequences(big, node_ips=["A"], window_start=ws,
                                       window_seconds=300.0, max_flows=8,
                                       seed=1, eval_mode=False)
    flows_b, _ = build_node_sequences(big, node_ips=["A"], window_start=ws,
                                       window_seconds=300.0, max_flows=8,
                                       seed=2, eval_mode=False)
    # With 64 flows sampled to 8 and per-node seed = seed+0, different seeds
    # should produce different index sets → different feature arrays.
    assert not np.array_equal(flows_a, flows_b)


def test_build_node_sequences_node_with_no_flows() -> None:
    """A node that touches no flows in the window must produce a fully-padded
    row (mask all True, features all zero)."""
    ws = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    df = _sample_flows(ws)
    flows, mask = build_node_sequences(
        df, node_ips=["A", "Z"], window_start=ws,
        window_seconds=300.0, max_flows=4, seed=0,
    )
    # Node Z (index 1) touches no flows → entirely padded.
    assert mask[1].all()
    assert np.all(flows[1] == 0.0)
