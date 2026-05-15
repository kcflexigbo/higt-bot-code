"""Parse a MedBIoT pcap into the canonical flow schema.

Input:  data/raw/medbiot/bulk/raw_dataset/{malware,normal}/*.pcap
Output: pandas DataFrame matching src/data/schema.FLOW_COLUMNS

Unlike CTU-13 and IoT-23, pcaps give us per-packet access — so we compute
real values for mean_iat_ms, std_iat_ms, min_pkt_size, max_pkt_size. This is
where the Phase 5 temporal Transformer gets the richest input.

Labels come from the filename (per data/README.md): `*_mal_*.pcap` → bot,
`*_leg_*.pcap` → benign. detailed_label preserves the malware family
(mirai/bashlite/torii) so downstream code can split by family if needed.

Flow grouping rule:
  - Canonical bidirectional 5-tuple key: (min(endpoint), max(endpoint), proto)
    where endpoint = (ip, port).
  - First-seen direction defines src/dst. bytes_fwd / pkts_fwd accumulate
    packets in that direction.
"""

from __future__ import annotations

import socket
from pathlib import Path

import dpkt
import numpy as np
import pandas as pd

from src.data.inspect import (
    medbiot_family_from_filename,
    medbiot_label_from_filename,
)
from src.data.schema import FLOW_COLUMNS, coerce_to_schema, validate_flow_df

# IP proto numbers we care about
PROTO_NUM_NAME = {6: "tcp", 17: "udp", 1: "icmp", 58: "icmp"}  # 58 = ICMPv6


def _proto_name(p: int) -> str:
    return PROTO_NUM_NAME.get(int(p), "other")


def parse_medbiot_pcap(
    path: Path,
    scenario_id: str | None = None,
    max_packets: int | None = None,
) -> pd.DataFrame:
    """Parse a single MedBIoT pcap into canonical-schema flows.

    Args:
        path: e.g. data/raw/medbiot/bulk/raw_dataset/malware/torii_mal_all.pcap
        scenario_id: tag; default derived from filename
            (e.g. "medbiot-torii-mal").
        max_packets: cap on packets read (None = read all). The 6 GB
            bashlite_leg.pcap is the only file where you'd want this.

    Returns:
        DataFrame in canonical schema, one row per bidirectional 5-tuple flow.
    """
    if scenario_id is None:
        scenario_id = _default_scenario_id(path.name)

    label_cls = medbiot_label_from_filename(path.name)
    family = medbiot_family_from_filename(path.name)
    detailed = f"{family}-{Path(path.name).stem}"

    flows: dict[tuple, dict] = {}
    n_pkts = 0

    with open(path, "rb") as f:
        try:
            reader = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)
        linktype = reader.datalink()

        for ts, buf in reader:
            if max_packets is not None and n_pkts >= max_packets:
                break
            n_pkts += 1
            try:
                if linktype == 1:
                    ip = dpkt.ethernet.Ethernet(buf).data
                elif linktype == 113:
                    ip = dpkt.sll.SLL(buf).data
                elif linktype == 276:
                    ip = dpkt.sll2.SLL2(buf).data
                else:
                    continue
                if not isinstance(ip, dpkt.ip.IP):
                    continue
                proto = int(ip.p)
                if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    sport = int(ip.data.sport)
                    dport = int(ip.data.dport)
                else:
                    sport = dport = 0
                src = socket.inet_ntoa(ip.src)
                dst = socket.inet_ntoa(ip.dst)
                size = int(ip.len)

                # Canonical key: sorted endpoints + proto.
                a, b = (src, sport), (dst, dport)
                if a > b:
                    a, b = b, a
                key = (a, b, proto)

                flow = flows.get(key)
                if flow is None:
                    flow = flows[key] = {
                        "src_ip": src, "dst_ip": dst,
                        "src_port": sport, "dst_port": dport,
                        "protocol": proto,
                        "first_ts": ts, "last_ts": ts,
                        "bytes_fwd": 0, "bytes_bwd": 0,
                        "pkts_fwd": 0, "pkts_bwd": 0,
                        "sizes": [],
                        "timestamps": [],
                    }
                # Direction relative to the first-seen src/dst.
                is_fwd = (src == flow["src_ip"]) and (sport == flow["src_port"])
                if is_fwd:
                    flow["bytes_fwd"] += size
                    flow["pkts_fwd"] += 1
                else:
                    flow["bytes_bwd"] += size
                    flow["pkts_bwd"] += 1

                if ts < flow["first_ts"]:
                    flow["first_ts"] = ts
                if ts > flow["last_ts"]:
                    flow["last_ts"] = ts
                flow["sizes"].append(size)
                flow["timestamps"].append(ts)
            except (dpkt.dpkt.UnpackError, AttributeError):
                continue

    if not flows:
        # Return an empty-but-valid frame.
        from src.data.schema import empty_flow_df

        return empty_flow_df()

    rows = []
    for flow in flows.values():
        sizes = np.asarray(flow["sizes"], dtype=np.int64)
        ts_arr = np.asarray(flow["timestamps"], dtype=np.float64)
        ts_arr.sort()
        if ts_arr.size >= 2:
            iat_s = np.diff(ts_arr)
            iat_ms = iat_s * 1000.0
            mean_iat = float(iat_ms.mean())
            std_iat = float(iat_ms.std()) if iat_ms.size >= 2 else 0.0
        else:
            mean_iat = std_iat = float("nan")
        rows.append({
            "src_ip": flow["src_ip"], "dst_ip": flow["dst_ip"],
            "src_port": flow["src_port"], "dst_port": flow["dst_port"],
            "protocol": _proto_name(flow["protocol"]),
            "first_ts": flow["first_ts"], "last_ts": flow["last_ts"],
            "bytes_fwd": flow["bytes_fwd"], "bytes_bwd": flow["bytes_bwd"],
            "pkts_fwd": flow["pkts_fwd"], "pkts_bwd": flow["pkts_bwd"],
            "mean_iat_ms": mean_iat,
            "std_iat_ms": std_iat,
            "min_pkt_size": float(sizes.min()),
            "max_pkt_size": float(sizes.max()),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("first_ts").reset_index(drop=True)

    start = pd.to_datetime(df["first_ts"], unit="s", utc=True)
    end = pd.to_datetime(df["last_ts"], unit="s", utc=True)

    out = pd.DataFrame({
        "flow_id": np.arange(len(df), dtype="int64"),
        "scenario": scenario_id,
        "src_ip": df["src_ip"].values,
        "dst_ip": df["dst_ip"].values,
        "src_port": df["src_port"].astype("int32").values,
        "dst_port": df["dst_port"].astype("int32").values,
        "protocol": df["protocol"].values,
        "start_time": start.values,
        "end_time": end.values,
        "duration_s": (df["last_ts"] - df["first_ts"]).astype("float64").values,
        "bytes_fwd": df["bytes_fwd"].astype("int64").values,
        "bytes_bwd": df["bytes_bwd"].astype("int64").values,
        "pkts_fwd": df["pkts_fwd"].astype("int64").values,
        "pkts_bwd": df["pkts_bwd"].astype("int64").values,
        "mean_iat_ms": df["mean_iat_ms"].values,
        "std_iat_ms": df["std_iat_ms"].values,
        "min_pkt_size": df["min_pkt_size"].values,
        "max_pkt_size": df["max_pkt_size"].values,
        "label": label_cls,
        "detailed_label": detailed,
    })
    out = coerce_to_schema(out[FLOW_COLUMNS])
    validate_flow_df(out)
    return out


def _default_scenario_id(filename: str) -> str:
    stem = Path(filename).stem  # mirai_mal_CC_all
    return f"medbiot-{stem}"
