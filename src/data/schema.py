"""Canonical flow schema — the contract between Phase 2 (parsing) and Phase 3
(graph construction).

Every parser (CTU-13 binetflow, IoT-23 Zeek conn.log, MedBIoT pcap) emits
this same DataFrame layout. Phase 3 reads only this schema; it never touches
raw data. That single hop is what lets us swap datasets in/out without
touching graph code.

Some fields are NaN-able because CTU-13 and IoT-23 don't expose per-packet
information. The Phase 5 temporal Transformer will pull `mean_iat_ms` /
`std_iat_ms` / `min_pkt_size` / `max_pkt_size` where available and substitute
zero+mask elsewhere.
"""

from __future__ import annotations

import pandas as pd

# Canonical column order. Parsers MUST return DataFrames with exactly these
# columns, in this order. The Phase 3 graph constructor relies on this.
FLOW_COLUMNS = [
    "flow_id",          # int64, unique within scenario
    "scenario",         # string, e.g. "ctu13-10" / "iot23-48-1" / "medbiot-bashlite-CC"
    "src_ip",           # string
    "dst_ip",           # string
    "src_port",         # int32, 0 for protocols without ports (icmp etc.)
    "dst_port",         # int32
    "protocol",         # string: tcp / udp / icmp / other
    "start_time",       # datetime64[ns, UTC]
    "end_time",         # datetime64[ns, UTC]
    "duration_s",       # float64
    "bytes_fwd",        # int64, src→dst
    "bytes_bwd",        # int64, dst→src
    "pkts_fwd",         # int64
    "pkts_bwd",         # int64
    "mean_iat_ms",      # float64, NaN if unavailable (CTU-13, IoT-23)
    "std_iat_ms",       # float64, NaN if unavailable
    "min_pkt_size",     # float64, NaN if unavailable (kept float for NaN support)
    "max_pkt_size",     # float64, NaN if unavailable
    "label",            # string: bot / benign / background
    "detailed_label",   # string, optional family info (e.g. "Mirai-Okiru")
]


# Expected dtypes after validation/casting.
FLOW_DTYPES = {
    "flow_id": "int64",
    "scenario": "string",
    "src_ip": "string",
    "dst_ip": "string",
    "src_port": "int32",
    "dst_port": "int32",
    "protocol": "string",
    "duration_s": "float64",
    "bytes_fwd": "int64",
    "bytes_bwd": "int64",
    "pkts_fwd": "int64",
    "pkts_bwd": "int64",
    "mean_iat_ms": "float64",
    "std_iat_ms": "float64",
    "min_pkt_size": "float64",
    "max_pkt_size": "float64",
    "label": "string",
    "detailed_label": "string",
}


VALID_LABELS = {"bot", "benign", "background"}
VALID_PROTOCOLS = {"tcp", "udp", "icmp", "other"}


class SchemaError(ValueError):
    """Raised when a DataFrame does not conform to the canonical schema."""


def validate_flow_df(df: pd.DataFrame, *, strict: bool = True) -> None:
    """Raise SchemaError if `df` does not match the canonical schema.

    Checks:
      1. Columns present, in correct order.
      2. Labels are subset of VALID_LABELS.
      3. Protocols are subset of VALID_PROTOCOLS.
      4. flow_id is unique within the frame.
      5. (strict only) dtypes match FLOW_DTYPES.
      6. Time columns are tz-aware UTC.
    """
    if list(df.columns) != FLOW_COLUMNS:
        raise SchemaError(
            "Column mismatch.\n"
            f"  expected: {FLOW_COLUMNS}\n"
            f"  got:      {list(df.columns)}"
        )

    bad_labels = set(df["label"].dropna().unique()) - VALID_LABELS
    if bad_labels:
        raise SchemaError(f"Invalid labels: {sorted(bad_labels)}")

    bad_protos = set(df["protocol"].dropna().unique()) - VALID_PROTOCOLS
    if bad_protos:
        raise SchemaError(f"Invalid protocols: {sorted(bad_protos)}")

    if df["flow_id"].duplicated().any():
        n_dup = int(df["flow_id"].duplicated().sum())
        raise SchemaError(f"flow_id is not unique within frame ({n_dup} duplicates)")

    if not pd.api.types.is_datetime64_any_dtype(df["start_time"]):
        raise SchemaError("start_time must be datetime")
    if not pd.api.types.is_datetime64_any_dtype(df["end_time"]):
        raise SchemaError("end_time must be datetime")
    if df["start_time"].dt.tz is None:
        raise SchemaError("start_time must be tz-aware (UTC)")
    if df["end_time"].dt.tz is None:
        raise SchemaError("end_time must be tz-aware (UTC)")

    if strict:
        for col, want in FLOW_DTYPES.items():
            got = str(df[col].dtype)
            # Allow nullable-equivalent matches: int64 == Int64, etc.
            if got.lower().replace("[pyarrow]", "") not in {want.lower(), want.lower() + "[pyarrow]"}:
                # Permissive: accept obvious aliases
                if want == "string" and got in ("object", "string"):
                    continue
                if want.startswith("int") and got.lower().startswith(want.lower()):
                    continue
                raise SchemaError(f"dtype mismatch for {col!r}: want {want}, got {got}")


def empty_flow_df() -> pd.DataFrame:
    """Build a zero-row DataFrame that already validates."""
    df = pd.DataFrame({col: pd.Series(dtype=FLOW_DTYPES.get(col, "object"))
                       for col in FLOW_COLUMNS})
    df["start_time"] = pd.to_datetime([], utc=True)
    df["end_time"] = pd.to_datetime([], utc=True)
    return df


def coerce_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cast a DataFrame's columns to the canonical dtypes, in canonical order.

    Use this at the END of every parser, right before validate_flow_df.
    """
    df = df.loc[:, FLOW_COLUMNS].copy()
    for col, want in FLOW_DTYPES.items():
        if want == "string":
            df[col] = df[col].astype("string")
        elif want.startswith("int"):
            df[col] = df[col].astype(want)
        elif want.startswith("float"):
            df[col] = df[col].astype("float64")
    if df["start_time"].dt.tz is None:
        df["start_time"] = df["start_time"].dt.tz_localize("UTC")
    if df["end_time"].dt.tz is None:
        df["end_time"] = df["end_time"].dt.tz_localize("UTC")
    return df
