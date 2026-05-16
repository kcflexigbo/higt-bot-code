"""Programmatic replacement for the Phase 1 "open in Wireshark" step.

For a given pcap and a target source IP (one infected host), compute the
intuition-shaping statistics the plan asks you to look at by hand:

  - timing: packet inter-arrival distribution (heartbeat? bursts?)
  - sizes:  packet-size distribution and protocol breakdown
  - peers:  destination IP fan-out over time (P2P churn signal)
  - ports:  top destination ports

Outputs both a text summary (stdout + log file) and a 4-panel PNG figure.

The point of this script is to feed Phase 2 feature engineering: every
statistic here corresponds to a candidate feature in src/data/parse.py.

Usage:
    uv run python scripts/bot_traffic_profile.py \\
        --pcap data/raw/CTU-13-Dataset/10/botnet-capture-20110818-bot.pcap \\
        --src 147.32.84.205 --max-packets 2000000
"""

from __future__ import annotations

import argparse
import socket
from collections import Counter, defaultdict
from pathlib import Path

import dpkt
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scan_pcap_for_src(
    pcap_path: Path, target_src: str, max_packets: int | None
) -> tuple[list[dict], dict[str, int]]:
    """Stream a pcap, keep packets whose IPv4 source matches `target_src`.

    Returns:
        rows: list of per-packet dicts (timestamp, dst_ip, dst_port, proto, size)
        global_stats: counters for protocol / linktype / skipped reasons
    """
    rows: list[dict] = []
    stats: dict[str, int] = Counter()
    target_bytes = socket.inet_aton(target_src)

    with open(pcap_path, "rb") as f:
        try:
            reader = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)
        linktype = reader.datalink()

        for ts, buf in reader:
            stats["total_packets"] += 1
            if max_packets is not None and stats["total_packets"] >= max_packets:
                stats["truncated_at"] = max_packets
                break
            try:
                if linktype == 1:
                    ip = dpkt.ethernet.Ethernet(buf).data
                elif linktype == 113:
                    ip = dpkt.sll.SLL(buf).data
                elif linktype == 276:
                    ip = dpkt.sll2.SLL2(buf).data
                else:
                    stats["skipped_linktype"] += 1
                    continue
                if not isinstance(ip, dpkt.ip.IP):
                    stats["skipped_non_ipv4"] += 1
                    continue
                if ip.src != target_bytes:
                    stats["skipped_other_src"] += 1
                    continue
                proto = ip.p
                if isinstance(ip.data, dpkt.tcp.TCP):
                    proto_name = "tcp"
                    dport = int(ip.data.dport)
                elif isinstance(ip.data, dpkt.udp.UDP):
                    proto_name = "udp"
                    dport = int(ip.data.dport)
                elif isinstance(ip.data, dpkt.icmp.ICMP):
                    proto_name = "icmp"
                    dport = 0
                else:
                    proto_name = f"proto-{proto}"
                    dport = 0
                rows.append({
                    "ts": ts,
                    "dst_ip": socket.inet_ntoa(ip.dst),
                    "dst_port": dport,
                    "protocol": proto_name,
                    "size": int(ip.len),
                })
            except (dpkt.dpkt.UnpackError, AttributeError):
                stats["skipped_malformed"] += 1
    return rows, dict(stats)


def print_profile(df: pd.DataFrame, src_ip: str) -> str:
    """Compute and print the text profile. Returns the joined output string."""
    lines: list[str] = []
    p = lines.append

    n = len(df)
    p(f"\n=== Bot traffic profile: {src_ip} ===")
    p(f"  packets matched          {n:,}")
    if n == 0:
        return "\n".join(lines)

    t_min, t_max = df["ts"].min(), df["ts"].max()
    duration = t_max - t_min
    p(f"  time span                {duration:,.1f} s ({duration / 60:,.1f} min)")
    p(f"  mean packet rate         {n / duration:,.2f} pkts/s")
    p("")

    # Protocol breakdown
    p("  Protocol breakdown:")
    for proto, c in df["protocol"].value_counts().items():
        p(f"    {proto:<10s} {int(c):>10,}  ({c/n*100:5.2f}%)")
    p("")

    # Packet size distribution
    s = df["size"]
    p("  Packet size (IP-layer bytes):")
    p(f"    min      {int(s.min()):>6d}")
    p(f"    p25      {int(s.quantile(0.25)):>6d}")
    p(f"    median   {int(s.quantile(0.50)):>6d}")
    p(f"    p75      {int(s.quantile(0.75)):>6d}")
    p(f"    p99      {int(s.quantile(0.99)):>6d}")
    p(f"    max      {int(s.max()):>6d}")
    p(f"    mean     {s.mean():>6.1f}")
    p(f"    std      {s.std():>6.1f}")
    p("")

    # Inter-arrival times
    iat = df["ts"].diff().dropna()
    iat_pos = iat[iat > 0]
    p("  Inter-arrival times (consecutive packets from this bot):")
    p(f"    p1       {iat_pos.quantile(0.01) * 1000:>8.3f} ms")
    p(f"    p25      {iat_pos.quantile(0.25) * 1000:>8.3f} ms")
    p(f"    median   {iat_pos.quantile(0.50) * 1000:>8.3f} ms")
    p(f"    p75      {iat_pos.quantile(0.75) * 1000:>8.3f} ms")
    p(f"    p99      {iat_pos.quantile(0.99) * 1000:>8.3f} ms")
    p(f"    max      {iat_pos.max():>8.3f} s")
    p(f"    mean     {iat_pos.mean() * 1000:>8.3f} ms")

    # Burstiness: coefficient of variation
    if iat_pos.mean() > 0:
        cv = iat_pos.std() / iat_pos.mean()
        p(f"    CV       {cv:>8.3f}  ({'bursty' if cv > 1 else 'regular'})")
    p("")

    # Top destinations (peer fan-out)
    distinct_peers = df["dst_ip"].nunique()
    p(f"  Distinct destinations contacted   {distinct_peers:,}")
    p("  Top 10 destination IPs:")
    for ip, c in df["dst_ip"].value_counts().head(10).items():
        p(f"    {ip:<22s} {int(c):>10,}")
    p("")

    # Top destination ports
    p("  Top 10 destination ports:")
    for port, c in df["dst_port"].value_counts().head(10).items():
        p(f"    {int(port):<6d} {int(c):>10,}")
    p("")

    # Peer churn — how fast do new destinations appear?
    df_sorted = df.sort_values("ts")
    seen: set[str] = set()
    timeline = []
    for ts_v, dst in zip(df_sorted["ts"].values, df_sorted["dst_ip"].values):
        seen.add(dst)
        timeline.append((ts_v - t_min, len(seen)))
    quarter = len(timeline) // 4 or 1
    p("  Peer churn (cumulative distinct peers over time):")
    for i, (t_rel, cum) in enumerate(timeline[quarter - 1::quarter][:4], start=1):
        p(f"    after {i*25:3d}% of packets ({t_rel:>7.1f} s)  ->  {cum:,} distinct peers")
    p("")

    out = "\n".join(lines)
    print(out)
    return out


def plot_profile(df: pd.DataFrame, src_ip: str, out_path: Path) -> None:
    """Render the 4-panel intuition figure: IAT, sizes, ports, peer churn."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Bot traffic profile — {src_ip}", fontsize=13)

    # 1. Inter-arrival time histogram (log scale)
    iat = df["ts"].diff().dropna()
    iat_pos = iat[iat > 0] * 1000  # ms
    if len(iat_pos):
        axes[0, 0].hist(np.log10(iat_pos.values + 1e-6), bins=80, color="steelblue", edgecolor="white")
        axes[0, 0].set_title("Inter-arrival time (log10 ms)")
        axes[0, 0].set_xlabel("log10(IAT in ms)")
        axes[0, 0].set_ylabel("packets")

    # 2. Packet size histogram
    axes[0, 1].hist(df["size"].values, bins=80, color="darkorange", edgecolor="white")
    axes[0, 1].set_title("Packet size (IP-layer bytes)")
    axes[0, 1].set_xlabel("bytes")
    axes[0, 1].set_ylabel("packets")

    # 3. Top destination ports (bar)
    top_ports = df["dst_port"].value_counts().head(15)
    axes[1, 0].bar([str(int(p)) for p in top_ports.index], top_ports.values,
                    color="seagreen", edgecolor="white")
    axes[1, 0].set_title("Top 15 destination ports")
    axes[1, 0].set_xlabel("port")
    axes[1, 0].set_ylabel("packets")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 4. Cumulative unique peers over time
    df_s = df.sort_values("ts")
    t_rel = df_s["ts"].values - df_s["ts"].values[0]
    _, idx = np.unique(df_s["dst_ip"].values, return_index=True)
    first_seen = np.zeros(len(df_s), dtype=bool)
    first_seen[idx] = True
    cum_peers = np.cumsum(first_seen)
    axes[1, 1].plot(t_rel, cum_peers, color="crimson")
    axes[1, 1].set_title("Cumulative distinct peers")
    axes[1, 1].set_xlabel("seconds from first packet")
    axes[1, 1].set_ylabel("distinct destinations")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  figure saved -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap", type=Path, required=True)
    ap.add_argument("--src", required=True, help="Source IP to filter on, e.g. 147.32.84.205")
    ap.add_argument("--max-packets", type=int, default=2_000_000,
                    help="Cap on packets scanned (the bot pcap is 66 GB)")
    ap.add_argument("--log-out", type=Path, default=None,
                    help="Where to save the text profile (default: data/inspection_logs/bot_profile_<src>.txt)")
    ap.add_argument("--fig-out", type=Path, default=None,
                    help="Where to save the figure (default: data/inspection_logs/figures/bot_profile_<src>.png)")
    args = ap.parse_args()

    log_out = args.log_out or Path(f"data/inspection_logs/bot_profile_{args.src}.txt")
    fig_out = args.fig_out or Path(f"data/inspection_logs/figures/bot_profile_{args.src}.png")

    print(f"Scanning {args.pcap}  (cap={args.max_packets:,} packets)")
    rows, stats = scan_pcap_for_src(args.pcap, args.src, args.max_packets)
    print(f"  packets seen       {stats.get('total_packets', 0):,}")
    print(f"  matched (src={args.src})  {len(rows):,}")
    for k in ("skipped_non_ipv4", "skipped_other_src", "skipped_linktype", "skipped_malformed", "truncated_at"):
        if stats.get(k):
            print(f"  {k:<22s} {stats[k]:,}")

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\nNo packets from {args.src} in the first {args.max_packets:,} packets. Increase --max-packets or try another bot IP.")
        return

    text = print_profile(df, args.src)
    log_out.parent.mkdir(parents=True, exist_ok=True)
    log_out.write_text(text)
    print(f"\n  text profile saved -> {log_out}")

    plot_profile(df, args.src, fig_out)


if __name__ == "__main__":
    main()
