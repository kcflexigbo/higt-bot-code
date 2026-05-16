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

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.inspect import iter_iot23_conn_chunks, load_iot23_conn
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


def _chunk_to_canonical(raw: pd.DataFrame, scenario_id: str) -> pd.DataFrame:
    """Transform one Zeek conn chunk to the canonical FLOW_COLUMNS schema
    (without flow_id — caller assigns it after a global sort)."""
    raw = raw.dropna(subset=["ts", "id.orig_h", "id.resp_h"])

    duration = pd.to_numeric(raw["duration"], errors="coerce").fillna(0.0).astype("float64")
    start = raw["ts"]
    end = start + pd.to_timedelta(duration, unit="s")

    orig_bytes = pd.to_numeric(raw["orig_bytes"], errors="coerce").fillna(0).astype("int64")
    resp_bytes = pd.to_numeric(raw["resp_bytes"], errors="coerce").fillna(0).astype("int64")
    orig_pkts = pd.to_numeric(raw["orig_pkts"], errors="coerce").fillna(0).astype("int64")
    resp_pkts = pd.to_numeric(raw["resp_pkts"], errors="coerce").fillna(0).astype("int64")

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
        "flow_id": np.zeros(n, dtype="int64"),  # placeholder; reassigned post-sort
        "scenario": scenario_id,
        "src_ip": raw["id.orig_h"].values,
        "dst_ip": raw["id.resp_h"].values,
        "src_port": pd.to_numeric(raw["id.orig_p"], errors="coerce").fillna(0).astype("int32").values,
        "dst_port": pd.to_numeric(raw["id.resp_p"], errors="coerce").fillna(0).astype("int32").values,
        "protocol": raw["proto"].fillna("other").astype(str).str.lower().map(PROTO_MAP).fillna("other").values,
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
        "label": raw["class"].values,
        "detailed_label": raw["detailed-label"].values,
    })
    return out[FLOW_COLUMNS]


def parse_iot23_scenario_streaming(
    scenario_dir: Path,
    dest: Path,
    scenario_id: str | None = None,
    chunksize: int = 1_000_000,
) -> int:
    """Streaming parser for huge IoT-23 scenarios (e.g. 7+ GB conn.log).

    Two passes:
      1. Stream Zeek chunks → transform → append to a temp parquet (unsorted).
      2. Read temp parquet via PyArrow, sort by start_time, assign flow_id,
         write final parquet. Free temp.
    """
    if scenario_id is None:
        name = scenario_dir.name
        suffix = name.split("Capture-")[-1] if "Capture-" in name else name
        scenario_id = f"iot23-{suffix}"

    candidates = list(scenario_dir.rglob("conn.log.labeled"))
    if not candidates:
        raise FileNotFoundError(f"No conn.log.labeled under {scenario_dir}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.parquet")
    if tmp.exists():
        tmp.unlink()

    # Pass 1: stream + append to temp parquet.
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    schema: pa.Schema | None = None
    for path in candidates:
        for raw in iter_iot23_conn_chunks(path, chunksize=chunksize):
            canon = _chunk_to_canonical(raw, scenario_id)
            if canon.empty:
                continue
            canon = coerce_to_schema(canon)
            table = pa.Table.from_pandas(canon, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema, compression="snappy")
            else:
                table = table.cast(schema)
            writer.write_table(table)
            total_rows += len(canon)
            print(f"  ...chunk +{len(canon):,} rows  (total {total_rows:,})", flush=True)
    if writer is not None:
        writer.close()

    if total_rows == 0:
        raise RuntimeError(f"No rows produced for {scenario_dir}")

    # Pass 2: sort by start_time and re-assign flow_id, validate, write final.
    print(f"  sorting {total_rows:,} rows by start_time...", flush=True)
    table = pq.read_table(tmp)
    indices = pa.compute.sort_indices(table, sort_keys=[("start_time", "ascending")])
    table = table.take(indices)
    table = table.set_column(
        table.schema.get_field_index("flow_id"),
        "flow_id",
        pa.array(np.arange(total_rows, dtype=np.int64)),
    )
    pq.write_table(table, dest, compression="snappy")
    tmp.unlink()

    # Light validation on the final file.
    df_check = pd.read_parquet(dest)
    validate_flow_df(df_check)
    return total_rows


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
        "protocol": raw["proto"].fillna("other").astype(str).str.lower().map(PROTO_MAP).fillna("other").values,
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
