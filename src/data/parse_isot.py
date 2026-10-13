"""Parse the ISOT Botnet dataset (2010) pcap into the canonical flow schema.

Source: University of Victoria ISOT lab, https://onlineacademiccommunity.uvic.ca/isot/
(ISOT_Botnet_DataSet_2010.pcap, 11.4 GB). Malicious traffic (Storm and Waledac P2P
botnets from the French Honeynet chapter, plus Zeus) was replayed onto hosts of a
background trace and is marked by synthetic MAC addresses (dataset overview, Table 2):

    BB:BB:BB:BB:BB:BB  Storm (UDP, Overnet P2P)     172.16.2.11
    AA:AA:AA:AA:AA:AA  Storm / Waledac SMTP spam   172.16.0.2, .0.11, .0.12
    CC:CC:CC:CC:CC:CC  Zeus                        172.16.2.12
    CC:CC:CC:DD:DD:DD  Zeus C&C                    172.16.2.12

A flow is labelled "bot" if any of its packets carries one of these MACs (source or
destination); everything else is "benign". detailed_label names the family.
Flow grouping and the per-flow statistics follow src/data/parse_medbiot.py exactly
(bidirectional 5-tuple, first-seen direction = src, online IAT/size statistics).
"""
from __future__ import annotations

import socket
from pathlib import Path

import dpkt
import numpy as np
import pandas as pd

from src.data.parse_medbiot import _proto_name
from src.data.schema import FLOW_COLUMNS, coerce_to_schema, validate_flow_df

MAL_MACS = {
    bytes.fromhex("bbbbbbbbbbbb"): "Storm-UDP-P2P",
    bytes.fromhex("aaaaaaaaaaaa"): "Storm/Waledac-SMTP-spam",
    bytes.fromhex("cccccccccccc"): "Zeus",
    bytes.fromhex("cccccc dddddd".replace(" ", "")): "Zeus-CC",
}


def parse_isot_pcap(path: Path, scenario_id: str = "isot-2010", max_packets: int | None = None) -> pd.DataFrame:
    flows: dict[tuple, dict] = {}
    n_pkts = 0
    with open(path, "rb") as f:
        try:
            reader = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)
        if reader.datalink() != 1:
            raise ValueError("ISOT pcap is expected to be Ethernet (linktype 1)")
        for ts, buf in reader:
            if max_packets is not None and n_pkts >= max_packets:
                break
            n_pkts += 1
            if n_pkts % 5_000_000 == 0:
                print(f"  ...{n_pkts:,} packets, {len(flows):,} flows so far", flush=True)
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data
                if not isinstance(ip, dpkt.ip.IP):
                    continue
                fam = MAL_MACS.get(eth.src) or MAL_MACS.get(eth.dst)
                proto = int(ip.p)
                if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    sport, dport = int(ip.data.sport), int(ip.data.dport)
                else:
                    sport = dport = 0
                src, dst = socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst)
                size = int(ip.len)
                a, b = (src, sport), (dst, dport)
                if a > b:
                    a, b = b, a
                key = (a, b, proto)
                flow = flows.get(key)
                if flow is None:
                    flow = flows[key] = {
                        "src_ip": src, "dst_ip": dst, "src_port": sport, "dst_port": dport, "protocol": proto,
                        "first_ts": ts, "last_ts": ts, "bytes_fwd": 0, "bytes_bwd": 0, "pkts_fwd": 0, "pkts_bwd": 0,
                        "min_size": size, "max_size": size, "iat_n": 0, "iat_mean": 0.0, "iat_M2": 0.0,
                        "prev_ts": ts, "fam": None,
                    }
                if fam is not None and flow["fam"] is None:
                    flow["fam"] = fam
                if (src == flow["src_ip"]) and (sport == flow["src_port"]):
                    flow["bytes_fwd"] += size; flow["pkts_fwd"] += 1
                else:
                    flow["bytes_bwd"] += size; flow["pkts_bwd"] += 1
                if ts < flow["first_ts"]:
                    flow["first_ts"] = ts
                if ts > flow["last_ts"]:
                    delta_ms = (ts - flow["prev_ts"]) * 1000.0
                    flow["iat_n"] += 1
                    n = flow["iat_n"]
                    d = delta_ms - flow["iat_mean"]
                    flow["iat_mean"] += d / n
                    flow["iat_M2"] += d * (delta_ms - flow["iat_mean"])
                    flow["prev_ts"] = ts
                    flow["last_ts"] = ts
                if size < flow["min_size"]:
                    flow["min_size"] = size
                if size > flow["max_size"]:
                    flow["max_size"] = size
            except (dpkt.dpkt.UnpackError, AttributeError):
                continue
    rows = []
    for fl in flows.values():
        n_iat = fl["iat_n"]
        mean_iat = fl["iat_mean"] if n_iat >= 1 else float("nan")
        std_iat = ((fl["iat_M2"] / n_iat) ** 0.5 if n_iat >= 2 else 0.0) if n_iat >= 1 else float("nan")
        rows.append({"src_ip": fl["src_ip"], "dst_ip": fl["dst_ip"], "src_port": fl["src_port"],
                     "dst_port": fl["dst_port"], "protocol": _proto_name(fl["protocol"]),
                     "first_ts": fl["first_ts"], "last_ts": fl["last_ts"],
                     "bytes_fwd": fl["bytes_fwd"], "bytes_bwd": fl["bytes_bwd"],
                     "pkts_fwd": fl["pkts_fwd"], "pkts_bwd": fl["pkts_bwd"],
                     "mean_iat_ms": mean_iat, "std_iat_ms": std_iat,
                     "min_pkt_size": float(fl["min_size"]), "max_pkt_size": float(fl["max_size"]),
                     "fam": fl["fam"]})
    df = pd.DataFrame(rows).sort_values("first_ts").reset_index(drop=True)
    start = pd.to_datetime(df["first_ts"], unit="s", utc=True)
    end = pd.to_datetime(df["last_ts"], unit="s", utc=True)
    out = pd.DataFrame({
        "flow_id": np.arange(len(df), dtype="int64"), "scenario": scenario_id,
        "src_ip": df["src_ip"].values, "dst_ip": df["dst_ip"].values,
        "src_port": df["src_port"].astype("int32").values, "dst_port": df["dst_port"].astype("int32").values,
        "protocol": df["protocol"].values, "start_time": start.values, "end_time": end.values,
        "duration_s": (df["last_ts"] - df["first_ts"]).astype("float64").values,
        "bytes_fwd": df["bytes_fwd"].astype("int64").values, "bytes_bwd": df["bytes_bwd"].astype("int64").values,
        "pkts_fwd": df["pkts_fwd"].astype("int64").values, "pkts_bwd": df["pkts_bwd"].astype("int64").values,
        "mean_iat_ms": df["mean_iat_ms"].values, "std_iat_ms": df["std_iat_ms"].values,
        "min_pkt_size": df["min_pkt_size"].values, "max_pkt_size": df["max_pkt_size"].values,
        "label": np.where(df["fam"].notna(), "bot", "benign"),
        "detailed_label": df["fam"].fillna("benign").values,
    })
    out = coerce_to_schema(out[FLOW_COLUMNS])
    validate_flow_df(out)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", type=Path, default=Path("data/raw/p2p_isot/ISOT_Botnet_DataSet_2010/ISOT_Botnet_DataSet_2010.pcap"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/isot-2010.parquet"))
    ap.add_argument("--max-packets", type=int, default=None)
    a = ap.parse_args()
    df = parse_isot_pcap(a.pcap, max_packets=a.max_packets)
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df):,} flows, bot flows {int((df.label == 'bot').sum()):,}")
    print(df[df.label == "bot"].groupby("detailed_label").src_ip.agg(["count", "nunique"]))
