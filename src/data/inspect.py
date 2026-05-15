"""Phase 1 dataset inspector.

Goal: understand the data viscerally before writing parsers. For a given
file or directory, print:
  - row count and time range
  - distinct source / destination IP counts
  - label distribution
  - top-10 most communicative source IPs and their roles

Supports CTU-13 binetflow CSVs and IoT-23 Zeek conn.log.labeled files.
MedBIoT (pcap) is added when Phase 2 needs it.

Usage:
    uv run python -m src.data.inspect --path data/raw/CTU-13-Dataset/10 --format ctu13
    uv run python -m src.data.inspect --path data/raw/IoT-23/CTU-IoT-Malware-Capture-48-1 --format iot23
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
    print(f"  time range            {t_min}  →  {t_max}")
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

    Quirk: the original 21 Zeek columns are tab-separated, but the appended
    `label` and `detailed-label` are space-separated from the preceding
    `tunnel_parents`. Pandas with sep=r'\\s+' treats both consistently — safe
    here because no field contains internal whitespace.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=IOT23_ZEEK_COLUMNS,
        engine="python",
        na_values=["-", "(empty)"],
        dtype={
            "id.orig_h": str, "id.resp_h": str,
            "id.orig_p": str, "id.resp_p": str,
            "proto": str, "service": str, "conn_state": str,
            "history": str, "tunnel_parents": str,
            "label": str, "detailed-label": str,
        },
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
    df["class"] = df["label"].map(normalize_iot23_label)
    return df


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
    print(f"  time range            {t_min}  →  {t_max}  (UTC)")
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
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", type=Path, required=True, help="File or directory to inspect")
    p.add_argument("--format", choices=["ctu13", "iot23"], default="ctu13",
                   help="Dataset format")
    args = p.parse_args()

    if not args.path.exists():
        raise SystemExit(f"path does not exist: {args.path}")

    if args.format == "ctu13":
        inspect_ctu13(args.path)
    elif args.format == "iot23":
        inspect_iot23(args.path)


if __name__ == "__main__":
    main()
