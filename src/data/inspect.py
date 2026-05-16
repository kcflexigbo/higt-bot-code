"""Phase 1 dataset inspector.

Goal: understand the data viscerally before writing parsers. For a given
file or directory, print:
  - row count and time range
  - distinct source / destination IP counts
  - label distribution
  - top-10 most communicative source IPs and their roles

Supports CTU-13 binetflow, IoT-23 Zeek conn.log.labeled, and MedBIoT pcap
(via NFStream — labels inferred from filename).

Usage:
    uv run python -m src.data.inspect --path data/raw/CTU-13-Dataset/10 --format ctu13
    uv run python -m src.data.inspect --path data/raw/IoT-23/CTU-IoT-Malware-Capture-48-1 --format iot23
    uv run python -m src.data.inspect --path data/raw/medbiot/bulk/raw_dataset/malware/torii_mal_all.pcap --format medbiot
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

CTU13_BINETFLOW_COLUMNS = [
    "StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir",
    "DstAddr", "Dport", "State", "sTos", "dTos",
    "TotPkts", "TotBytes", "SrcBytes", "Label",
]

CTU13_LABEL_RE = re.compile(r"flow=(?P<cls>[A-Za-z]+)")


def normalize_ctu13_label(raw: str) -> str:
    """Map a raw CTU-13 label string to one of {bot, benign, background}.

    Examples:
        'flow=Background-Established'                        -> background
        'flow=From-Botnet-V52-1-TCP-CC42-Custom-Encryption'  -> bot
        'flow=To-Botnet-V52-UDP-Attempt'                     -> bot
        'flow=Normal-V52-Stribika-Web'                       -> benign
        'flow=Background-attempt-cmpgw-CVUT'                 -> background
    """
    if not isinstance(raw, str):
        return "background"
    if "Botnet" in raw:
        return "bot"
    if raw.startswith("flow=Normal") or "Normal-" in raw:
        return "benign"
    return "background"


def load_ctu13_binetflow(path: Path) -> pd.DataFrame:
    """Load a single CTU-13 binetflow CSV with proper dtypes.

    The file has a header row, so we let pandas use it but verify column
    names match what we expect.
    """
    df = pd.read_csv(
        path,
        dtype={"SrcAddr": str, "DstAddr": str, "Sport": str, "Dport": str,
               "Proto": str, "State": str, "Dir": str, "Label": str},
        low_memory=False,
    )
    if list(df.columns) != CTU13_BINETFLOW_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {path}:\n  got     {list(df.columns)}\n"
            f"  expect  {CTU13_BINETFLOW_COLUMNS}"
        )
    df["StartTime"] = pd.to_datetime(df["StartTime"], errors="coerce")
    df["class"] = df["Label"].map(normalize_ctu13_label)
    return df


def find_binetflow_files(path: Path) -> list[Path]:
    """If `path` is a file, return [path]. If a directory, find all .binetflow."""
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*.binetflow"))
    if not files:
        raise FileNotFoundError(f"No .binetflow files under {path}")
    return files


def summarize_ctu13(df: pd.DataFrame, source: str) -> None:
    """Print a one-page overview of a binetflow dataframe."""
    n = len(df)
    src_ips = df["SrcAddr"].nunique()
    dst_ips = df["DstAddr"].nunique()
    all_ips = pd.concat([df["SrcAddr"], df["DstAddr"]]).nunique()
    t_min, t_max = df["StartTime"].min(), df["StartTime"].max()
    duration = (t_max - t_min).total_seconds() if pd.notna(t_min) and pd.notna(t_max) else None

    print(f"=== {source} ===")
    print(f"  rows                  {n:,}")
    print(f"  distinct src IPs      {src_ips:,}")
    print(f"  distinct dst IPs      {dst_ips:,}")
    print(f"  distinct IPs (union)  {all_ips:,}")
    print(f"  time range            {t_min}  ->  {t_max}")
    if duration is not None:
        print(f"  duration              {duration:,.0f} s ({duration / 3600:.2f} h)")
    print()

    # Label distribution
    counts = df["class"].value_counts()
    print("  Label distribution (normalized class):")
    for cls in ("bot", "benign", "background"):
        c = int(counts.get(cls, 0))
        pct = (c / n * 100) if n else 0.0
        print(f"    {cls:12s} {c:>10,}  ({pct:5.2f}%)")
    print()

    # Top 10 most communicative source IPs (by flow count) and their dominant class
    top_src = (
        df.groupby("SrcAddr")
          .agg(flows=("SrcAddr", "size"),
               dominant_class=("class", lambda s: s.value_counts().idxmax()),
               bot_flows=("class", lambda s: int((s == "bot").sum())))
          .sort_values("flows", ascending=False)
          .head(10)
    )
    print("  Top 10 source IPs (by flow count):")
    print(f"    {'src_ip':<22s} {'flows':>10s} {'bot_flows':>10s}  dominant")
    for ip, row in top_src.iterrows():
        print(f"    {ip:<22s} {int(row['flows']):>10,} {int(row['bot_flows']):>10,}  {row['dominant_class']}")
    print()

    # Bot source IPs across the whole file (the "who is a bot" answer for Phase 1 gate)
    bot_srcs = df.loc[df["class"] == "bot", "SrcAddr"].value_counts()
    print(f"  Bot source IPs        {len(bot_srcs)}")
    if len(bot_srcs):
        print(f"    {'src_ip':<22s} {'bot_flows':>10s}")
        for ip, c in bot_srcs.head(10).items():
            print(f"    {ip:<22s} {int(c):>10,}")
    print()


def inspect_ctu13(path: Path) -> None:
    files = find_binetflow_files(path)
    print(f"Found {len(files)} binetflow file(s) under {path}\n")

    aggregate = []
    for f in files:
        df = load_ctu13_binetflow(f)
        summarize_ctu13(df, source=str(f.relative_to(path) if path.is_dir() else f.name))
        aggregate.append(df["class"].value_counts())

    if len(files) > 1:
        total = pd.concat(aggregate, axis=1).sum(axis=1)
        print(f"=== AGGREGATE across {len(files)} files ===")
        for cls in ("bot", "benign", "background"):
            print(f"  {cls:12s} {int(total.get(cls, 0)):>12,}")


# --------------------------------------------------------------------------- #
# IoT-23 Zeek conn.log.labeled                                                 #
# --------------------------------------------------------------------------- #

# Standard Zeek conn fields (21) + IoT-23 enrichment (label, detailed-label) = 23.
IOT23_ZEEK_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents", "label", "detailed-label",
]


def normalize_iot23_label(raw: str) -> str:
    """Map IoT-23 `label` field to {bot, benign, background}.

    IoT-23 uses `Malicious` / `Benign` (case-sensitive). Anything else
    (missing, '-', empty) is treated as background.
    """
    if not isinstance(raw, str):
        return "background"
    s = raw.strip()
    if s == "Malicious":
        return "bot"
    if s == "Benign":
        return "benign"
    return "background"


def load_iot23_conn(path: Path) -> pd.DataFrame:
    """Load a single IoT-23 conn.log.labeled file.

    Quirk: the original 21 Zeek columns are tab-separated, but `label` and
    `detailed-label` are appended after `tunnel_parents` with 3-space
    separation. To stay on pandas' C engine (10-30× faster than the python
    engine on these multi-million-row files), we:
      1. Read with sep='\\t' — this collapses `tunnel_parents   label   detailed-label`
         into one trailing string column.
      2. Split that string by whitespace into the three real fields.
    """
    # The line has 20 tabs (21 fields). The first 20 fields are ts..resp_ip_bytes;
    # the 21st field is "<tunnel_parents>   <label>   <detailed-label>".
    raw_cols = IOT23_ZEEK_COLUMNS[:-3] + ["tunnel_label_blob"]  # 21 cols
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=raw_cols,
        engine="c",
        na_values=["-", "(empty)"],
        dtype={
            "id.orig_h": str, "id.resp_h": str,
            "id.orig_p": str, "id.resp_p": str,
            "proto": str, "service": str, "conn_state": str,
            "history": str, "tunnel_label_blob": str,
        },
        low_memory=False,
    )

    # Split "<tunnel_parents>   <label>   <detailed-label>" by whitespace.
    split = df["tunnel_label_blob"].fillna("-   -   -").str.split(r"\s+", n=2, expand=True, regex=True)
    df["tunnel_parents"] = split[0]
    df["label"] = split[1].fillna("-")
    df["detailed-label"] = split[2].fillna("-")
    df = df.drop(columns=["tunnel_label_blob"])
    # Restore canonical column order
    df = df[IOT23_ZEEK_COLUMNS]

    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
    df["class"] = df["label"].map(normalize_iot23_label)
    return df


def iter_iot23_conn_chunks(path: Path, chunksize: int = 1_000_000):
    """Stream a conn.log.labeled in row chunks. Yields the same per-chunk
    DataFrame shape that `load_iot23_conn` returns for the whole file.

    Use this for huge scenarios (e.g. IoT-23 17-1, 33-1 at 7+ GB) where
    a single read would blow up RAM.
    """
    raw_cols = IOT23_ZEEK_COLUMNS[:-3] + ["tunnel_label_blob"]
    reader = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=raw_cols,
        engine="c",
        na_values=["-", "(empty)"],
        dtype={
            "id.orig_h": str, "id.resp_h": str,
            "id.orig_p": str, "id.resp_p": str,
            "proto": str, "service": str, "conn_state": str,
            "history": str, "tunnel_label_blob": str,
        },
        low_memory=False,
        chunksize=chunksize,
        iterator=True,
    )
    for df in reader:
        split = df["tunnel_label_blob"].fillna("-   -   -").str.split(r"\s+", n=2, expand=True, regex=True)
        df["tunnel_parents"] = split[0]
        df["label"] = split[1].fillna("-")
        df["detailed-label"] = split[2].fillna("-")
        df = df.drop(columns=["tunnel_label_blob"])
        df = df[IOT23_ZEEK_COLUMNS]
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
        df["class"] = df["label"].map(normalize_iot23_label)
        yield df


def find_iot23_conn_files(path: Path) -> list[Path]:
    """If file, return [path]. If scenario dir, look in bro/. If parent, recurse."""
    if path.is_file():
        return [path]
    files = sorted(path.rglob("conn.log.labeled"))
    if not files:
        raise FileNotFoundError(f"No conn.log.labeled files under {path}")
    return files


def summarize_iot23(df: pd.DataFrame, source: str) -> None:
    """Print a one-page overview of an IoT-23 conn dataframe."""
    n = len(df)
    src_ips = df["id.orig_h"].nunique()
    dst_ips = df["id.resp_h"].nunique()
    all_ips = pd.concat([df["id.orig_h"], df["id.resp_h"]]).nunique()
    t_min, t_max = df["ts"].min(), df["ts"].max()
    duration = (t_max - t_min).total_seconds() if pd.notna(t_min) and pd.notna(t_max) else None

    print(f"=== {source} ===")
    print(f"  rows                  {n:,}")
    print(f"  distinct src IPs      {src_ips:,}")
    print(f"  distinct dst IPs      {dst_ips:,}")
    print(f"  distinct IPs (union)  {all_ips:,}")
    print(f"  time range            {t_min}  ->  {t_max}  (UTC)")
    if duration is not None:
        print(f"  duration              {duration:,.0f} s ({duration / 3600:.2f} h)")
    print()

    counts = df["class"].value_counts()
    print("  Label distribution (normalized class):")
    for cls in ("bot", "benign", "background"):
        c = int(counts.get(cls, 0))
        pct = (c / n * 100) if n else 0.0
        print(f"    {cls:12s} {c:>10,}  ({pct:5.2f}%)")

    # Detailed-label breakdown — IoT-23's malware-family info
    detailed = df.loc[df["class"] == "bot", "detailed-label"].value_counts()
    if len(detailed):
        print()
        print("  Top detailed-labels (bot flows only):")
        for lbl, c in detailed.head(8).items():
            print(f"    {str(lbl):40s} {int(c):>10,}")
    print()

    top_src = (
        df.groupby("id.orig_h")
          .agg(flows=("id.orig_h", "size"),
               dominant_class=("class", lambda s: s.value_counts().idxmax()),
               bot_flows=("class", lambda s: int((s == "bot").sum())))
          .sort_values("flows", ascending=False)
          .head(10)
    )
    print("  Top 10 source IPs (by flow count):")
    print(f"    {'src_ip':<22s} {'flows':>10s} {'bot_flows':>10s}  dominant")
    for ip, row in top_src.iterrows():
        print(f"    {ip:<22s} {int(row['flows']):>10,} {int(row['bot_flows']):>10,}  {row['dominant_class']}")
    print()

    bot_srcs = df.loc[df["class"] == "bot", "id.orig_h"].value_counts()
    print(f"  Bot source IPs        {len(bot_srcs)}")
    if len(bot_srcs):
        print(f"    {'src_ip':<22s} {'bot_flows':>10s}")
        for ip, c in bot_srcs.head(10).items():
            print(f"    {ip:<22s} {int(c):>10,}")
    print()


def inspect_iot23(path: Path) -> None:
    files = find_iot23_conn_files(path)
    print(f"Found {len(files)} conn.log.labeled file(s) under {path}\n")

    aggregate = []
    for f in files:
        df = load_iot23_conn(f)
        rel = f.relative_to(path) if path.is_dir() else f.name
        summarize_iot23(df, source=str(rel))
        aggregate.append(df["class"].value_counts())

    if len(files) > 1:
        total = pd.concat(aggregate, axis=1).sum(axis=1)
        print(f"=== AGGREGATE across {len(files)} files ===")
        for cls in ("bot", "benign", "background"):
            print(f"  {cls:12s} {int(total.get(cls, 0)):>12,}")


# --------------------------------------------------------------------------- #
# MedBIoT pcap (via NFStream)                                                 #
# --------------------------------------------------------------------------- #

# Label rule from data/README.md:
#   <malware>_mal_*.pcap  → all flows are bot (the malware family)
#   <malware>_leg_*.pcap  → all flows are benign (legitimate traffic captured
#                            during malware deployment, on uninfected hosts)


def medbiot_label_from_filename(filename: str) -> str:
    """Map MedBIoT pcap filename to class label.

    Returns 'bot' for *_mal_*.pcap and 'benign' for *_leg_*.pcap.
    Returns 'background' if the filename matches neither convention.
    """
    name = Path(filename).name.lower()
    if "_mal_" in name or name.endswith("_mal.pcap") or "_mal." in name:
        return "bot"
    if "_leg_" in name or name.endswith("_leg.pcap") or "_leg." in name:
        return "benign"
    return "background"


def medbiot_family_from_filename(filename: str) -> str:
    """Extract malware family (mirai/bashlite/torii) from pcap filename."""
    name = Path(filename).name.lower()
    for family in ("mirai", "bashlite", "torii"):
        if name.startswith(family):
            return family
    return "unknown"


def load_medbiot_pcap(
    path: Path, max_flows: int | None = None, max_packets: int | None = None
) -> pd.DataFrame:
    """Stream packets from a MedBIoT pcap via dpkt, group into 5-tuple
    bidirectional flows, return a DataFrame.

    We use dpkt (not NFStream) because nfstream's bundled nDPI binding is
    broken on Apple Silicon macOS — flat-namespace symbol resolution fails
    even with `brew install ndpi`. dpkt is pure Python where it matters,
    fast in the inner loop, and adequate for Phase 1 inspection (no L7
    classification needed — labels come from the filename).

    Args:
        path: pcap file (classic pcap or pcapng).
        max_flows: cap on emitted flows (None = no cap).
        max_packets: cap on packets read (None = no cap, useful for the
            6 GB bashlite_leg pcap).
    """
    import socket

    import dpkt

    flows: dict[tuple, dict] = {}
    n_pkts = 0
    n_skipped = 0

    with open(path, "rb") as f:
        # Try classic pcap first; fall back to pcapng on magic mismatch.
        try:
            reader = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)

        linktype = reader.datalink()
        # 1 = DLT_EN10MB (Ethernet), 113 = DLT_LINUX_SLL, 276 = DLT_LINUX_SLL2
        for ts, buf in reader:
            if max_packets is not None and n_pkts >= max_packets:
                break
            n_pkts += 1
            try:
                if linktype == 1:
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data
                elif linktype == 113:
                    sll = dpkt.sll.SLL(buf)
                    ip = sll.data
                elif linktype == 276:
                    sll2 = dpkt.sll2.SLL2(buf)
                    ip = sll2.data
                else:
                    n_skipped += 1
                    continue
                if not isinstance(ip, dpkt.ip.IP):
                    n_skipped += 1
                    continue

                src = socket.inet_ntoa(ip.src)
                dst = socket.inet_ntoa(ip.dst)
                proto = ip.p
                if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    sport, dport = int(ip.data.sport), int(ip.data.dport)
                else:
                    sport = dport = 0

                # Canonical bidirectional 5-tuple key (sort endpoints).
                a, b = (src, sport), (dst, dport)
                if a > b:
                    a, b = b, a
                key = (a, b, proto)

                flow = flows.get(key)
                if flow is None:
                    if max_flows is not None and len(flows) >= max_flows:
                        n_skipped += 1
                        continue
                    flow = flows[key] = {
                        "src_ip": src, "dst_ip": dst,
                        "src_port": sport, "dst_port": dport,
                        "protocol": int(proto),
                        "first_seen_s": ts, "last_seen_s": ts,
                        "packets": 0, "bytes": 0,
                    }
                flow["first_seen_s"] = min(flow["first_seen_s"], ts)
                flow["last_seen_s"] = max(flow["last_seen_s"], ts)
                flow["packets"] += 1
                flow["bytes"] += int(ip.len)
            except (dpkt.dpkt.UnpackError, AttributeError):
                n_skipped += 1
                continue

    if n_skipped:
        print(f"  (skipped {n_skipped:,} non-IP / malformed packets)")

    if not flows:
        return pd.DataFrame()

    df = pd.DataFrame(list(flows.values()))
    df["first_seen"] = pd.to_datetime(df["first_seen_s"], unit="s", utc=True)
    df["last_seen"] = pd.to_datetime(df["last_seen_s"], unit="s", utc=True)
    df["duration_s"] = df["last_seen_s"] - df["first_seen_s"]
    df["class"] = medbiot_label_from_filename(path.name)
    df["family"] = medbiot_family_from_filename(path.name)
    return df


def find_medbiot_pcaps(path: Path) -> list[Path]:
    """If file, return [path]. If directory, find all .pcap files recursively."""
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*.pcap"))
    if not files:
        raise FileNotFoundError(f"No .pcap files under {path}")
    return files


def summarize_medbiot(df: pd.DataFrame, source: str) -> None:
    """Print a one-page overview of a MedBIoT pcap flow dataframe."""
    n = len(df)
    if n == 0:
        print(f"=== {source} ===  (empty)")
        return

    src_ips = df["src_ip"].nunique()
    dst_ips = df["dst_ip"].nunique()
    all_ips = pd.concat([df["src_ip"], df["dst_ip"]]).nunique()
    t_min, t_max = df["first_seen"].min(), df["last_seen"].max()
    duration = (t_max - t_min).total_seconds() if pd.notna(t_min) and pd.notna(t_max) else None
    cls = str(df["class"].iloc[0])
    family = str(df["family"].iloc[0])

    print(f"=== {source} ===")
    print(f"  flows                 {n:,}")
    print(f"  inferred class        {cls}  (family: {family})")
    print(f"  distinct src IPs      {src_ips:,}")
    print(f"  distinct dst IPs      {dst_ips:,}")
    print(f"  distinct IPs (union)  {all_ips:,}")
    print(f"  time range            {t_min}  ->  {t_max}  (UTC)")
    if duration is not None:
        print(f"  duration              {duration:,.0f} s ({duration / 3600:.2f} h)")
    print(f"  total bytes           {int(df['bytes'].sum()):,}")
    print(f"  total packets         {int(df['packets'].sum()):,}")
    print()

    top_src = (
        df.groupby("src_ip")
          .agg(flows=("src_ip", "size"),
               pkts=("packets", "sum"),
               bytes_=("bytes", "sum"))
          .sort_values("flows", ascending=False)
          .head(10)
    )
    print("  Top 10 source IPs (by flow count):")
    print(f"    {'src_ip':<22s} {'flows':>10s} {'pkts':>10s} {'bytes':>14s}")
    for ip, row in top_src.iterrows():
        print(f"    {ip:<22s} {int(row['flows']):>10,} {int(row['pkts']):>10,} {int(row['bytes_']):>14,}")
    print()

    top_ports = df["dst_port"].value_counts().head(10)
    print("  Top 10 destination ports:")
    for port, c in top_ports.items():
        print(f"    {int(port):<6d}  {int(c):>10,}")
    print()


def inspect_medbiot(
    path: Path, max_flows: int | None = None, max_packets: int | None = None
) -> None:
    files = find_medbiot_pcaps(path)
    print(f"Found {len(files)} pcap file(s) under {path}")
    if max_flows is not None:
        print(f"  (capping at {max_flows:,} flows per file)")
    if max_packets is not None:
        print(f"  (capping at {max_packets:,} packets per file)")
    print()

    for f in files:
        df = load_medbiot_pcap(f, max_flows=max_flows, max_packets=max_packets)
        rel = f.relative_to(path) if path.is_dir() else f.name
        summarize_medbiot(df, source=str(rel))


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", type=Path, required=True, help="File or directory to inspect")
    p.add_argument("--format", choices=["ctu13", "iot23", "medbiot"], default="ctu13",
                   help="Dataset format")
    p.add_argument("--max-flows", type=int, default=None,
                   help="MedBIoT only: cap flows per pcap")
    p.add_argument("--max-packets", type=int, default=None,
                   help="MedBIoT only: cap packets read per pcap (useful for the 6 GB legitimate captures)")
    args = p.parse_args()

    if not args.path.exists():
        raise SystemExit(f"path does not exist: {args.path}")

    if args.format == "ctu13":
        inspect_ctu13(args.path)
    elif args.format == "iot23":
        inspect_iot23(args.path)
    elif args.format == "medbiot":
        inspect_medbiot(args.path, max_flows=args.max_flows, max_packets=args.max_packets)


if __name__ == "__main__":
    main()
