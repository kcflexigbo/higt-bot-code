"""Per-flow feature extraction and per-(node, window) sequence builder for
the Phase 5 temporal Transformer.

Inputs are a canonical-schema flow DataFrame for one window (see
src/data/schema.py). Outputs are numpy arrays aligned to the caller's
`node_ips` ordering — the same ordering used by Phase 3 graph construction,
so the Transformer's `[N, d_model]` output drops in as a replacement for
`data.x`.

Direction is encoded relative to each focal node: the same flow appears in
two nodes' sequences with opposite `is_outgoing` bits.

Sampling rule when a node has more than `max_flows` flows in the window:
- eval_mode=True  → evenly-spaced subsample (deterministic).
- eval_mode=False → random subsample using a numpy RNG seeded with `seed`.

`window_start` must be tz-aware (UTC). A tz-naive timestamp is rejected to
prevent silent timezone-offset bugs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

PROTOCOLS = ("tcp", "udp", "icmp", "other")

FLOW_FEATURE_NAMES = (
    "proto_tcp", "proto_udp", "proto_icmp", "proto_other",
    "is_outgoing",
    "log1p_duration_s",
    "log1p_total_bytes",
    "log1p_total_packets",
    "log1p_mean_iat_ms", "has_iat",
    "log1p_max_pkt_size", "has_max_pkt",
    "norm_time",
)
FLOW_FEATURE_DIM = len(FLOW_FEATURE_NAMES)

_PROTO_IDX = {p: i for i, p in enumerate(PROTOCOLS)}


def _proto_onehot(protocol: pd.Series) -> np.ndarray:
    """[N, 4] one-hot over (tcp, udp, icmp, other). Unknown → other."""
    n = len(protocol)
    out = np.zeros((n, len(PROTOCOLS)), dtype=np.float32)
    if n == 0:
        return out
    codes = protocol.map(_PROTO_IDX).fillna(_PROTO_IDX["other"]).astype(np.int64).values
    out[np.arange(n), codes] = 1.0
    return out


def flow_features_for_node(
    flows: pd.DataFrame,
    *,
    focal: str,
    window_start: pd.Timestamp,
    window_seconds: float,
) -> np.ndarray:
    """Return [num_flows_touching_focal, FLOW_FEATURE_DIM], sorted by start_time."""
    if window_start.tzinfo is None:
        raise ValueError("window_start must be tz-aware (pass tz='UTC')")
    mask = (flows["src_ip"] == focal) | (flows["dst_ip"] == focal)
    sub = flows.loc[mask].sort_values("start_time").reset_index(drop=True)
    n = len(sub)
    out = np.zeros((n, FLOW_FEATURE_DIM), dtype=np.float32)
    if n == 0:
        return out

    out[:, 0:4] = _proto_onehot(sub["protocol"])
    out[:, 4]   = (sub["src_ip"].values == focal).astype(np.float32)        # is_outgoing
    out[:, 5]   = np.log1p(np.clip(sub["duration_s"].values.astype(np.float64), 0, None)).astype(np.float32)
    out[:, 6]   = np.log1p((sub["bytes_fwd"].values + sub["bytes_bwd"].values).astype(np.float64)).astype(np.float32)
    out[:, 7]   = np.log1p((sub["pkts_fwd"].values + sub["pkts_bwd"].values).astype(np.float64)).astype(np.float32)

    iat = sub["mean_iat_ms"].values.astype(np.float64)
    out[:, 8]   = np.log1p(np.where(np.isnan(iat), 0.0, np.clip(iat, 0, None))).astype(np.float32)
    out[:, 9]   = (~np.isnan(iat)).astype(np.float32)

    mxp = sub["max_pkt_size"].values.astype(np.float64)
    out[:, 10]  = np.log1p(np.where(np.isnan(mxp), 0.0, np.clip(mxp, 0, None))).astype(np.float32)
    out[:, 11]  = (~np.isnan(mxp)).astype(np.float32)

    # norm_time ∈ [0, 1]
    win = max(float(window_seconds), 1e-9)
    ws_utc = window_start.tz_convert("UTC")
    ws_ns = np.datetime64(ws_utc.tz_localize(None), "ns")
    dt = (sub["start_time"].dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[ns]") - ws_ns).astype("timedelta64[ns]")
    dt_s = dt.astype(np.float64) / 1e9
    out[:, 12]  = np.clip(dt_s / win, 0.0, 1.0).astype(np.float32)
    return out


def _select_indices(n: int, max_flows: int, *, eval_mode: bool, seed: int) -> np.ndarray:
    if n <= max_flows:
        return np.arange(n)
    if eval_mode:
        # Evenly spaced sub-sample, including endpoints.
        return np.linspace(0, n - 1, num=max_flows).round().astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_flows, replace=False)
    idx.sort()
    return idx


def build_node_sequences(
    flows: pd.DataFrame,
    *,
    node_ips: Sequence[str],
    window_start: pd.Timestamp,
    window_seconds: float,
    max_flows: int = 256,
    seed: int = 0,
    eval_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (flows_arr, pad_mask).

      flows_arr: float32 [N, max_flows, FLOW_FEATURE_DIM]   zero-padded
      pad_mask:  bool    [N, max_flows]   True = padded, False = real flow
    """
    n_nodes = len(node_ips)
    flows_arr = np.zeros((n_nodes, max_flows, FLOW_FEATURE_DIM), dtype=np.float32)
    pad_mask = np.ones((n_nodes, max_flows), dtype=bool)  # default = padded

    for i, ip in enumerate(node_ips):
        per = flow_features_for_node(flows, focal=ip,
                                      window_start=window_start,
                                      window_seconds=window_seconds)
        if per.shape[0] == 0:
            continue
        # Per-node seed so different nodes get distinct samples but the same
        # (scenario, window_idx) is fully reproducible.
        idx = _select_indices(per.shape[0], max_flows,
                               eval_mode=eval_mode, seed=seed + i)
        per = per[idx]
        k = per.shape[0]
        flows_arr[i, :k] = per
        pad_mask[i, :k] = False
    return flows_arr, pad_mask
