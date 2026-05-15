"""Parse a CTU-13 scenario directory into the canonical flow schema.

Input:  data/raw/CTU-13-Dataset/<N>/*.binetflow
Output: pandas DataFrame matching src/data/schema.FLOW_COLUMNS

CTU-13 binetflow is already bidirectional — each row is one A↔B conversation.
We map columns directly; aggregate per-packet stats (mean_iat / pkt_size) are
NaN because the binetflow doesn't expose packet-level information.

Notes on the source:
- Timestamps are local Czech time (Europe/Prague). We localize and convert to UTC.
- `SrcBytes` is bytes sent by the source; bytes_bwd = TotBytes - SrcBytes.
- TotPkts is bidirectional; CTU-13 does not split per-direction packet counts,
  so we approximate pkts_fwd via the byte ratio. Honest about this in code.
- Filter out flows where TotPkts is missing or zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.inspect import (
    CTU13_BINETFLOW_COLUMNS,
    load_ctu13_binetflow,
    normalize_ctu13_label,
)
from src.data.schema import FLOW_COLUMNS, coerce_to_schema, validate_flow_df

CTU13_TZ = "Europe/Prague"


PROTO_MAP = {
    "tcp": "tcp", "udp": "udp", "icmp": "icmp",
    "icmpv6": "icmp", "ipv6-icmp": "icmp",
}


def _normalize_protocol(raw: str) -> str:
    if not isinstance(raw, str):
        return "other"
    return PROTO_MAP.get(raw.strip().lower(), "other")


def _port_to_int(raw: object) -> int:
    """Robustly coerce CTU-13's port strings (may be '0x1234', '0', '', NaN) to int."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return 0
    s = str(raw).strip()
    if not s:
        return 0
    try:
        return int(s, 0) & 0xFFFF  # base=0 handles 0x hex; mask to uint16 range
    except ValueError:
        return 0


def _ports_vectorized(series: pd.Series) -> np.ndarray:
    """Vectorized port parser. Handles decimal and 0x-hex strings; bad → 0."""
    s = series.fillna("0").astype(str).str.strip()
    # Try plain decimal first via pd.to_numeric.
    dec = pd.to_numeric(s, errors="coerce")
    # Hex fallback for the ones that failed.
    hex_mask = dec.isna() & s.str.startswith(("0x", "0X"), na=False)
    if hex_mask.any():
        hex_vals = s.where(hex_mask).dropna().apply(lambda v: int(v, 16))
        dec.loc[hex_mask] = hex_vals
    return dec.fillna(0).clip(0, 65535).astype("int32").values


def parse_ctu13_scenario(scenario_dir: Path, scenario_id: str | None = None) -> pd.DataFrame:
    """Parse all .binetflow files in a CTU-13 scenario directory.

    Args:
        scenario_dir: e.g. data/raw/CTU-13-Dataset/10/
        scenario_id: tag for the `scenario` column; defaults to "ctu13-<dirname>".

    Returns:
        DataFrame in canonical schema, sorted by start_time.
    """
    if scenario_id is None:
        scenario_id = f"ctu13-{scenario_dir.name}"

    files = sorted(scenario_dir.rglob("*.binetflow"))
    if not files:
        raise FileNotFoundError(f"No .binetflow files in {scenario_dir}")

    frames = [_parse_one_binetflow(f, scenario_id) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("start_time").reset_index(drop=True)
    df["flow_id"] = np.arange(len(df), dtype=np.int64)
    df = coerce_to_schema(df)
    validate_flow_df(df)
    return df


def _parse_one_binetflow(path: Path, scenario_id: str) -> pd.DataFrame:
    raw = load_ctu13_binetflow(path)
    # Drop rows with missing essential fields.
    raw = raw.dropna(subset=["StartTime", "SrcAddr", "DstAddr"])

    # Time: local Prague → UTC. load_ctu13_binetflow returns naive datetimes.
    start = raw["StartTime"].dt.tz_localize(CTU13_TZ, nonexistent="shift_forward",
                                              ambiguous="NaT").dt.tz_convert("UTC")

    duration = pd.to_numeric(raw["Dur"], errors="coerce").fillna(0.0)
    end = start + pd.to_timedelta(duration, unit="s")

    tot_bytes = pd.to_numeric(raw["TotBytes"], errors="coerce").fillna(0).astype("int64")
    src_bytes = pd.to_numeric(raw["SrcBytes"], errors="coerce").fillna(0).astype("int64")
    tot_pkts = pd.to_numeric(raw["TotPkts"], errors="coerce").fillna(0).astype("int64")

    bytes_fwd = src_bytes.clip(lower=0)
    bytes_bwd = (tot_bytes - src_bytes).clip(lower=0)

    # CTU-13 doesn't split packets by direction. Approximate via the byte ratio,
    # falling back to half/half. Honest fallback: pkts_fwd is approximate; the
    # graph constructor mainly uses bytes_fwd/bytes_bwd anyway.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tot_bytes > 0, bytes_fwd / tot_bytes, 0.5)
    pkts_fwd = np.rint(tot_pkts.values * ratio).astype("int64")
    pkts_bwd = (tot_pkts.values - pkts_fwd).astype("int64")
    pkts_fwd = np.maximum(pkts_fwd, 0)
    pkts_bwd = np.maximum(pkts_bwd, 0)

    # Drop zero-packet rows: they have no statistics worth keeping.
    keep = tot_pkts.values >= 1
    if not keep.all():
        raw = raw.loc[keep].reset_index(drop=True)
        start = start.loc[keep].reset_index(drop=True)
        end = end.loc[keep].reset_index(drop=True)
        duration = duration.loc[keep].reset_index(drop=True)
        bytes_fwd = bytes_fwd.loc[keep].reset_index(drop=True)
        bytes_bwd = bytes_bwd.loc[keep].reset_index(drop=True)
        pkts_fwd = pkts_fwd[keep]
        pkts_bwd = pkts_bwd[keep]

    n = len(raw)
    nan = np.full(n, np.nan, dtype="float64")

    out = pd.DataFrame({
        "flow_id": np.arange(n, dtype="int64"),  # rewritten by caller after concat
        "scenario": scenario_id,
        "src_ip": raw["SrcAddr"].values,
        "dst_ip": raw["DstAddr"].values,
        "src_port": _ports_vectorized(raw["Sport"]),
        "dst_port": _ports_vectorized(raw["Dport"]),
        "protocol": raw["Proto"].fillna("other").astype(str).str.lower().map(PROTO_MAP).fillna("other").values,
        "start_time": start.values,
        "end_time": end.values,
        "duration_s": duration.values.astype("float64"),
        "bytes_fwd": bytes_fwd.values,
        "bytes_bwd": bytes_bwd.values,
        "pkts_fwd": pkts_fwd,
        "pkts_bwd": pkts_bwd,
        "mean_iat_ms": nan,
        "std_iat_ms": nan,
        "min_pkt_size": nan,
        "max_pkt_size": nan,
        "label": raw["Label"].map(normalize_ctu13_label).values,
        "detailed_label": raw["Label"].values,
    })
    return out[FLOW_COLUMNS]
