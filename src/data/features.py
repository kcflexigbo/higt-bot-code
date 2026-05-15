"""Node and edge feature extractors for one time window of flows.

Inputs are a sub-DataFrame in canonical-schema (one row per flow within
the window). Outputs are plain numpy arrays / pandas Series aligned to
the caller's node ordering.

The features chosen here are the ones the plan calls out for Phase 3 plus
a couple of derived ratios — they are also what the Random Forest sanity
gate trains on, so changes here propagate everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# Stable order — every protocol one-hot uses this column order.
PROTOCOLS = ("tcp", "udp", "icmp", "other")

NODE_FEATURE_NAMES = (
    "fan_out",            # distinct dst IPs contacted
    "fan_in",             # distinct src IPs that contacted this host
    "port_entropy",       # Shannon entropy of dst ports the node contacts
    "byte_rate",          # bytes sent + received / window duration
    "packet_rate",        # packets sent + received / window duration
    "protocol_entropy",   # Shannon entropy of protocols used
    "out_in_ratio",       # bytes_sent / (bytes_sent + bytes_received + eps)
    "avg_flow_duration",  # mean duration_s across involved flows
    "log1p_total_flows",  # log1p(num flows touching this node)
)

EDGE_FEATURE_NAMES = (
    "total_bytes",
    "total_packets",
    "distinct_dst_ports",
    "distinct_src_ports",
    "mean_flow_duration",
    "mean_iat_ms",        # NaN-tolerant: averaged where available, else 0
    *(f"proto_{p}" for p in PROTOCOLS),  # one-hot of dominant protocol
)


def _shannon_entropy(values: pd.Series) -> float:
    """Shannon entropy in bits over the value distribution."""
    counts = values.value_counts(dropna=True)
    if counts.empty:
        return 0.0
    p = counts.values / counts.sum()
    return float(-(p * np.log2(p)).sum())


def compute_node_features(
    flows: pd.DataFrame, nodes: Sequence[str], window_seconds: float
) -> np.ndarray:
    """Return [num_nodes, len(NODE_FEATURE_NAMES)] feature matrix.

    `flows` is the window's flow DataFrame; `nodes` is a stable sorted list
    of IPs in the window. `window_seconds` is used for byte/packet rates.
    """
    eps = 1e-9
    win = max(window_seconds, eps)

    feats = np.zeros((len(nodes), len(NODE_FEATURE_NAMES)), dtype=np.float32)
    index = {ip: i for i, ip in enumerate(nodes)}

    # Group-by-src for outgoing stats, group-by-dst for incoming.
    by_src = flows.groupby("src_ip", sort=False)
    by_dst = flows.groupby("dst_ip", sort=False)

    for src, sub in by_src:
        i = index.get(src)
        if i is None:
            continue
        feats[i, 0] = sub["dst_ip"].nunique()                  # fan_out
        feats[i, 2] = _shannon_entropy(sub["dst_port"])         # port_entropy (on contacted dst ports)
        feats[i, 3] = (sub["bytes_fwd"].sum() + sub["bytes_bwd"].sum()) / win  # provisional byte_rate; refined below
        feats[i, 4] = (sub["pkts_fwd"].sum() + sub["pkts_bwd"].sum()) / win    # provisional packet_rate
        feats[i, 5] = _shannon_entropy(sub["protocol"])         # protocol_entropy
        feats[i, 7] = float(sub["duration_s"].mean())           # avg_flow_duration
        feats[i, 8] = float(np.log1p(len(sub)))                 # log1p_total_flows

    for dst, sub in by_dst:
        i = index.get(dst)
        if i is None:
            continue
        feats[i, 1] = sub["src_ip"].nunique()                  # fan_in
        # Add bytes/packets received to the rates (so far we only counted bytes sent).
        feats[i, 3] += (sub["bytes_fwd"].sum() + sub["bytes_bwd"].sum()) / win
        feats[i, 4] += (sub["pkts_fwd"].sum() + sub["pkts_bwd"].sum()) / win
        # If a node only appears as dst, populate the cells that the src loop skipped.
        if feats[i, 8] == 0:
            feats[i, 7] = float(sub["duration_s"].mean())
            feats[i, 8] = float(np.log1p(len(sub)))

    # out_in_ratio: bytes_sent / (bytes_sent + bytes_received).
    # bytes_sent at node = sum over flows with src=node of (bytes_fwd) + sum over flows with dst=node of (bytes_bwd).
    bytes_sent = np.zeros(len(nodes), dtype=np.float64)
    bytes_recv = np.zeros(len(nodes), dtype=np.float64)
    for src, sub in by_src:
        i = index.get(src)
        if i is None:
            continue
        bytes_sent[i] += float(sub["bytes_fwd"].sum())
        bytes_recv[i] += float(sub["bytes_bwd"].sum())
    for dst, sub in by_dst:
        i = index.get(dst)
        if i is None:
            continue
        bytes_sent[i] += float(sub["bytes_bwd"].sum())
        bytes_recv[i] += float(sub["bytes_fwd"].sum())
    total = bytes_sent + bytes_recv
    feats[:, 6] = np.where(total > 0, bytes_sent / (total + eps), 0.0)

    return feats


def compute_edges(
    flows: pd.DataFrame, nodes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Build the edge_index and edge_attr arrays for one window.

    Each unique (src_ip, dst_ip) pair becomes one directed edge, with all
    flows in that pair aggregated. Returns:
        edge_index: [2, num_edges]   (int64)
        edge_attr:  [num_edges, len(EDGE_FEATURE_NAMES)]   (float32)
    """
    index = {ip: i for i, ip in enumerate(nodes)}
    # Keep only flows where both endpoints are in our node set (defensive).
    mask = flows["src_ip"].isin(index) & flows["dst_ip"].isin(index)
    flows = flows.loc[mask]

    if flows.empty:
        return (np.empty((2, 0), dtype=np.int64),
                np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32))

    grp = flows.groupby(["src_ip", "dst_ip"], sort=False, as_index=False)

    agg = grp.agg(
        total_bytes=("bytes_fwd", "sum"),
        total_bytes_bwd=("bytes_bwd", "sum"),
        total_pkts_fwd=("pkts_fwd", "sum"),
        total_pkts_bwd=("pkts_bwd", "sum"),
        distinct_dst_ports=("dst_port", "nunique"),
        distinct_src_ports=("src_port", "nunique"),
        mean_flow_duration=("duration_s", "mean"),
        mean_iat_ms=("mean_iat_ms", "mean"),
    )
    agg["total_bytes"] = agg["total_bytes"] + agg["total_bytes_bwd"]
    agg["total_packets"] = agg["total_pkts_fwd"] + agg["total_pkts_bwd"]
    agg["mean_iat_ms"] = agg["mean_iat_ms"].fillna(0.0)

    # Dominant protocol per edge → one-hot
    proto = grp["protocol"].agg(lambda s: s.value_counts().idxmax())
    proto_oh = np.zeros((len(agg), len(PROTOCOLS)), dtype=np.float32)
    proto_map = {p: i for i, p in enumerate(PROTOCOLS)}
    for i, p in enumerate(proto["protocol"].values):
        proto_oh[i, proto_map.get(p, proto_map["other"])] = 1.0

    edge_attr = np.column_stack([
        agg["total_bytes"].values.astype(np.float32),
        agg["total_packets"].values.astype(np.float32),
        agg["distinct_dst_ports"].values.astype(np.float32),
        agg["distinct_src_ports"].values.astype(np.float32),
        agg["mean_flow_duration"].values.astype(np.float32),
        agg["mean_iat_ms"].values.astype(np.float32),
        proto_oh,
    ])

    src_idx = agg["src_ip"].map(index).values.astype(np.int64)
    dst_idx = agg["dst_ip"].map(index).values.astype(np.int64)
    edge_index = np.vstack([src_idx, dst_idx])
    return edge_index, edge_attr
