"""Parse one or more scenarios from data/raw/ into canonical-schema parquet
files under data/processed/.

Each scenario produces a single .parquet file with the schema defined in
src/data/schema.FLOW_COLUMNS.

Examples
--------
# A single CTU-13 scenario
uv run python scripts/parse_all.py --kind ctu13 --scenario 10

# All CTU-13 scenarios
uv run python scripts/parse_all.py --kind ctu13 --all

# One IoT-23 scenario
uv run python scripts/parse_all.py --kind iot23 --scenario 48-1

# One MedBIoT pcap
uv run python scripts/parse_all.py --kind medbiot \\
    --pcap data/raw/medbiot/bulk/raw_dataset/malware/torii_mal_all.pcap

# All MedBIoT malware pcaps
uv run python scripts/parse_all.py --kind medbiot --all-malware
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.data.parse_ctu13 import parse_ctu13_scenario
from src.data.parse_iot23 import parse_iot23_scenario
from src.data.parse_medbiot import parse_medbiot_pcap
from src.data.schema import FLOW_COLUMNS

RAW = Path("data/raw")
OUT = Path("data/processed")


def _write(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    print(f"  → {dest}  ({len(df):,} flows, {dest.stat().st_size / 1e6:.1f} MB)")


def _parse_ctu13_one(scenario: str) -> None:
    src = RAW / "CTU-13-Dataset" / scenario
    out = OUT / f"ctu13-{scenario}.parquet"
    print(f"\n[ctu13] {src}")
    t0 = time.perf_counter()
    df = parse_ctu13_scenario(src, scenario_id=f"ctu13-{scenario}")
    dt = time.perf_counter() - t0
    print(f"  parsed in {dt:.1f}s — bot {(df['label']=='bot').sum():,}  "
          f"benign {(df['label']=='benign').sum():,}  "
          f"background {(df['label']=='background').sum():,}")
    _write(df, out)


def _parse_iot23_one(scenario: str) -> None:
    src = RAW / "IoT-23" / f"CTU-IoT-Malware-Capture-{scenario}"
    out = OUT / f"iot23-{scenario}.parquet"
    print(f"\n[iot23] {src}")
    t0 = time.perf_counter()
    df = parse_iot23_scenario(src, scenario_id=f"iot23-{scenario}")
    dt = time.perf_counter() - t0
    print(f"  parsed in {dt:.1f}s — bot {(df['label']=='bot').sum():,}  "
          f"benign {(df['label']=='benign').sum():,}  "
          f"background {(df['label']=='background').sum():,}")
    _write(df, out)


def _parse_medbiot_pcap(pcap: Path, max_packets: int | None) -> None:
    out = OUT / f"medbiot-{pcap.stem}.parquet"
    print(f"\n[medbiot] {pcap}")
    t0 = time.perf_counter()
    df = parse_medbiot_pcap(pcap, max_packets=max_packets)
    dt = time.perf_counter() - t0
    print(f"  parsed in {dt:.1f}s — {len(df):,} flows  label={df['label'].iloc[0] if len(df) else '—'}")
    _write(df, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["ctu13", "iot23", "medbiot"], required=True)
    ap.add_argument("--scenario", help="CTU-13: '1'..'13'; IoT-23: '48-1' etc.")
    ap.add_argument("--pcap", type=Path, help="MedBIoT: path to a single .pcap")
    ap.add_argument("--all", action="store_true", help="Parse every scenario for the chosen kind")
    ap.add_argument("--all-malware", action="store_true",
                    help="MedBIoT only: parse every malware pcap")
    ap.add_argument("--max-packets", type=int, default=None,
                    help="MedBIoT only: cap packets read (useful for the 6 GB bashlite_leg)")
    args = ap.parse_args()

    if args.kind == "ctu13":
        if args.all:
            for d in sorted((RAW / "CTU-13-Dataset").iterdir()):
                if d.is_dir() and d.name.isdigit():
                    _parse_ctu13_one(d.name)
        elif args.scenario:
            _parse_ctu13_one(args.scenario)
        else:
            ap.error("ctu13 requires --scenario or --all")

    elif args.kind == "iot23":
        if args.all:
            for d in sorted((RAW / "IoT-23").iterdir()):
                if d.is_dir() and "Malware-Capture-" in d.name:
                    short = d.name.split("Capture-")[-1]
                    _parse_iot23_one(short)
        elif args.scenario:
            _parse_iot23_one(args.scenario)
        else:
            ap.error("iot23 requires --scenario or --all")

    elif args.kind == "medbiot":
        if args.all_malware:
            for p in sorted((RAW / "medbiot/bulk/raw_dataset/malware").glob("*.pcap")):
                _parse_medbiot_pcap(p, args.max_packets)
        elif args.pcap:
            _parse_medbiot_pcap(args.pcap, args.max_packets)
        else:
            ap.error("medbiot requires --pcap or --all-malware")

    # Spot-check: read back the last parquet, print sample bot rows.
    parquets = sorted(OUT.glob("*.parquet"))
    if parquets:
        last = parquets[-1]
        df = pd.read_parquet(last)
        print(f"\n--- spot-check {last.name} ---")
        print(f"  columns: {list(df.columns) == FLOW_COLUMNS=}")
        bots = df[df["label"] == "bot"]
        print(f"  bot rows: {len(bots):,}")
        if len(bots):
            print(bots[["src_ip", "dst_ip", "dst_port", "protocol",
                        "bytes_fwd", "bytes_bwd", "detailed_label"]].head(20).to_string())


if __name__ == "__main__":
    main()
