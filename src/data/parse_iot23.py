"""Parse an IoT-23 scenario directory into the canonical flow schema.

Input:  data/raw/IoT-23/CTU-IoT-Malware-Capture-<N>-1/bro/conn.log.labeled
Output: pandas DataFrame matching src/data/schema.FLOW_COLUMNS

Zeek `conn.log` is bidirectional. We map orig_* → fwd and resp_* → bwd. The
Stratosphere enrichment appends `label` and `detailed-label` (3-space-separated
from `tunnel_parents`), already normalized by load_iot23_conn.

Per-packet timing isn't exposed by Zeek either — mean_iat / pkt_size remain NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.inspect import load_iot23_conn
from src.data.schema import FLOW_COLUMNS, coerce_to_schema, validate_flow_df

PROTO_MAP = {
    "tcp": "tcp", "udp": "udp", "icmp": "icmp",
    "icmpv6": "icmp", "ipv6-icmp": "icmp",
}


def _normalize_protocol(raw: object) -> str:
    if not isinstance(raw, str):
        return "other"
    return PROTO_MAP.get(raw.strip().lower(), "other")


def parse_iot23_scenario(scenario_dir: Path, scenario_id: str | None = None) -> pd.DataFrame:
    """Parse the conn.log.labeled inside an IoT-23 scenario directory.

    Args:
        scenario_dir: e.g. data/raw/IoT-23/CTU-IoT-Malware-Capture-48-1/
        scenario_id: tag; defaults to "iot23-<short capture id>".
    """
    if scenario_id is None:
        # CTU-IoT-Malware-Capture-48-1 → iot23-48-1
        name = scenario_dir.name
        suffix = name.split("Capture-")[-1] if "Capture-" in name else name
        scenario_id = f"iot23-{suffix}"

    candidates = list(scenario_dir.rglob("conn.log.labeled"))
    if not candidates:
        raise FileNotFoundError(f"No conn.log.labeled under {scenario_dir}")

    frames = [_parse_one_conn(path, scenario_id) for path in candidates]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("start_time").reset_index(drop=True)
    df["flow_id"] = np.arange(len(df), dtype=np.int64)
    df = coerce_to_schema(df)
    validate_flow_df(df)
    return df


def _parse_one_conn(path: Path, scenario_id: str) -> pd.DataFrame:
    raw = load_iot23_conn(path)
    raw = raw.dropna(subset=["ts", "id.orig_h", "id.resp_h"])

    duration = pd.to_numeric(raw["duration"], errors="coerce").fillna(0.0).astype("float64")
    start = raw["ts"]
    end = start + pd.to_timedelta(duration, unit="s")

    orig_bytes = pd.to_numeric(raw["orig_bytes"], errors="coerce").fillna(0).astype("int64")
    resp_bytes = pd.to_numeric(raw["resp_bytes"], errors="coerce").fillna(0).astype("int64")
    orig_pkts = pd.to_numeric(raw["orig_pkts"], errors="coerce").fillna(0).astype("int64")
    resp_pkts = pd.to_numeric(raw["resp_pkts"], errors="coerce").fillna(0).astype("int64")

    # Filter zero-packet flows (no statistics worth keeping).
    keep = (orig_pkts + resp_pkts) >= 1
    if not keep.all():
        raw = raw.loc[keep].reset_index(drop=True)
        start = start.loc[keep].reset_index(drop=True)
        end = end.loc[keep].reset_index(drop=True)
        duration = duration.loc[keep].reset_index(drop=True)
        orig_bytes = orig_bytes.loc[keep].reset_index(drop=True)
        resp_bytes = resp_bytes.loc[keep].reset_index(drop=True)
        orig_pkts = orig_pkts.loc[keep].reset_index(drop=True)
        resp_pkts = resp_pkts.loc[keep].reset_index(drop=True)

    n = len(raw)
    nan = np.full(n, np.nan, dtype="float64")

    out = pd.DataFrame({
        "flow_id": np.arange(n, dtype="int64"),
        "scenario": scenario_id,
        "src_ip": raw["id.orig_h"].values,
        "dst_ip": raw["id.resp_h"].values,
        "src_port": pd.to_numeric(raw["id.orig_p"], errors="coerce").fillna(0).astype("int32").values,
        "dst_port": pd.to_numeric(raw["id.resp_p"], errors="coerce").fillna(0).astype("int32").values,
        "protocol": [_normalize_protocol(p) for p in raw["proto"].values],
        "start_time": start.values,
        "end_time": end.values,
        "duration_s": duration.values,
        "bytes_fwd": orig_bytes.values,
        "bytes_bwd": resp_bytes.values,
        "pkts_fwd": orig_pkts.values,
        "pkts_bwd": resp_pkts.values,
        "mean_iat_ms": nan,
        "std_iat_ms": nan,
        "min_pkt_size": nan,
        "max_pkt_size": nan,
        "label": raw["class"].values,  # already normalized to bot/benign/background
        "detailed_label": raw["detailed-label"].values,
    })
    return out[FLOW_COLUMNS]
