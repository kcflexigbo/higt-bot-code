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

# All MedBIoT pcaps (malware + normal/leg)
uv run python scripts/parse_all.py --kind medbiot --all

# Everything (CTU-13 + IoT-23 + MedBIoT)
uv run python scripts/parse_all.py --everything --streaming
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.data.parse_ctu13 import parse_ctu13_scenario
from src.data.parse_iot23 import parse_iot23_scenario, parse_iot23_scenario_streaming
from src.data.parse_medbiot import parse_medbiot_pcap
from src.data.schema import FLOW_COLUMNS

RAW = Path("data/raw")
OUT = Path("data/processed")


def _write(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    print(f"  -> {dest}  ({len(df):,} flows, {dest.stat().st_size / 1e6:.1f} MB)")


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


def _iot23_scenario_dirs() -> list[Path]:
    """Every IoT-23 capture dir that contains conn.log.labeled."""
    root = RAW / "IoT-23"
    dirs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "Capture-" not in d.name:
            continue
        if list(d.rglob("conn.log.labeled")):
            dirs.append(d)
    return dirs


def _iot23_out_stem(scenario_dir: Path) -> str:
    suffix = scenario_dir.name.split("Capture-")[-1]
    if "Honeypot" in scenario_dir.name:
        return f"iot23-honeypot-{suffix}"
    return f"iot23-{suffix}"


def _parse_iot23_dir(scenario_dir: Path, streaming: bool = False) -> None:
    stem = _iot23_out_stem(scenario_dir)
    out = OUT / f"{stem}.parquet"
    print(f"\n[iot23] {scenario_dir}  (streaming={streaming})")
    t0 = time.perf_counter()
    if streaming:
        n_rows = parse_iot23_scenario_streaming(
            scenario_dir, out, scenario_id=stem.replace(".parquet", "")
        )
        dt = time.perf_counter() - t0
        size_mb = out.stat().st_size / 1e6
        print(f"  streamed in {dt:.1f}s — {n_rows:,} rows  -> {out}  ({size_mb:.1f} MB)")
    else:
        df = parse_iot23_scenario(scenario_dir, scenario_id=stem)
        dt = time.perf_counter() - t0
        print(f"  parsed in {dt:.1f}s — bot {(df['label']=='bot').sum():,}  "
              f"benign {(df['label']=='benign').sum():,}  "
              f"background {(df['label']=='background').sum():,}")
        _write(df, out)


def _parse_iot23_one(scenario: str, streaming: bool = False) -> None:
    """Parse by short id, e.g. 48-1 or honeypot-4-1."""
    if scenario.startswith("honeypot-"):
        src = RAW / "IoT-23" / f"CTU-Honeypot-Capture-{scenario.removeprefix('honeypot-')}"
    else:
        src = RAW / "IoT-23" / f"CTU-IoT-Malware-Capture-{scenario}"
    if not src.is_dir():
        raise FileNotFoundError(src)
    _parse_iot23_dir(src, streaming=streaming)


def _medbiot_pcap_paths(*, malware_only: bool = False) -> list[Path]:
    root = RAW / "medbiot/bulk/raw_dataset"
    paths: list[Path] = []
    if malware_only:
        paths.extend(sorted((root / "malware").glob("*.pcap")))
    else:
        for sub in ("malware", "normal"):
            paths.extend(sorted((root / sub).glob("*.pcap")))
    return paths


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
    ap.add_argument("--kind", choices=["ctu13", "iot23", "medbiot"],
                    help="Required unless --everything is set")
    ap.add_argument("--scenario", action="append",
                    help="CTU-13: '1'..'13'; IoT-23: '48-1' etc. "
                         "Pass multiple times to parse several scenarios.")
    ap.add_argument("--pcap", type=Path, help="MedBIoT: path to a single .pcap")
    ap.add_argument("--all", action="store_true", help="Parse every scenario for the chosen kind")
    ap.add_argument("--all-malware", action="store_true",
                    help="MedBIoT only: parse every malware pcap (subset of --all)")
    ap.add_argument("--everything", action="store_true",
                    help="Parse CTU-13 + IoT-23 + all MedBIoT pcaps")
    ap.add_argument("--max-packets", type=int, default=None,
                    help="MedBIoT only: cap packets read (useful for the 6 GB bashlite_leg)")
    ap.add_argument("--streaming", action="store_true",
                    help="IoT-23 only: stream large conn.log files in chunks "
                         "(use for 7+ GB scenarios like 17-1, 33-1)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip outputs that already exist under data/processed/")
    args = ap.parse_args()

    if not args.everything and not args.kind:
        ap.error("provide --kind or --everything")

    if args.everything:
        for d in sorted((RAW / "CTU-13-Dataset").iterdir()):
            if d.is_dir() and d.name.isdigit():
                out = OUT / f"ctu13-{d.name}.parquet"
                if args.skip_existing and out.exists():
                    print(f"[skip] {out.name}")
                    continue
                _parse_ctu13_one(d.name)
        for scenario_dir in _iot23_scenario_dirs():
            out = OUT / f"{_iot23_out_stem(scenario_dir)}.parquet"
            if args.skip_existing and out.exists():
                print(f"[skip] {out.name}")
                continue
            _parse_iot23_dir(scenario_dir, streaming=args.streaming)
        for pcap in _medbiot_pcap_paths():
            out = OUT / f"medbiot-{pcap.stem}.parquet"
            if args.skip_existing and out.exists():
                print(f"[skip] {out.name}")
                continue
            _parse_medbiot_pcap(pcap, args.max_packets)
    elif args.kind == "ctu13":
        if args.all:
            for d in sorted((RAW / "CTU-13-Dataset").iterdir()):
                if d.is_dir() and d.name.isdigit():
                    _parse_ctu13_one(d.name)
        elif args.scenario:
            for s in args.scenario:
                _parse_ctu13_one(s)
        else:
            ap.error("ctu13 requires --scenario or --all")

    elif args.kind == "iot23":
        if args.all:
            for scenario_dir in _iot23_scenario_dirs():
                out = OUT / f"{_iot23_out_stem(scenario_dir)}.parquet"
                if args.skip_existing and out.exists():
                    print(f"[skip] {out.name}")
                    continue
                _parse_iot23_dir(scenario_dir, streaming=args.streaming)
        elif args.scenario:
            for s in args.scenario:
                _parse_iot23_one(s, streaming=args.streaming)
        else:
            ap.error("iot23 requires --scenario or --all")

    elif args.kind == "medbiot":
        if args.all or args.all_malware:
            for p in _medbiot_pcap_paths(malware_only=args.all_malware and not args.all):
                out = OUT / f"medbiot-{p.stem}.parquet"
                if args.skip_existing and out.exists():
                    print(f"[skip] {out.name}")
                    continue
                _parse_medbiot_pcap(p, args.max_packets)
        elif args.pcap:
            _parse_medbiot_pcap(args.pcap, args.max_packets)
        else:
            ap.error("medbiot requires --pcap, --all-malware, or --all")

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
