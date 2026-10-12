"""Relabel existing window graphs with the infected-source ground truth.

The original rule (`src/data/graph.py::_node_labels`) marks a host as a bot if
it appears as source OR destination of any bot flow in the window, so victims,
scan targets and C&C servers become positives. This script writes a copy of
every graph to `data/graphs_src/` in which a host is a bot only if it
*originates* at least one bot flow in the window (among flows between the
graph's selected nodes, i.e. the same flow universe the original labels used).

Graph structure, node order and features are untouched; only `y` and
`graph_y` change, so cached flow sequences and embeddings stay aligned.

Window membership is reconstructed from `window_start` and the window length;
as a check, the original any-endpoint labels are recomputed from the same
flows and must match the stored `y` exactly.

    uv run python scripts/relabel_infected_src.py
Train on the result with HIGT_GRAPHS_DIR=data/graphs_src.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
import torch  # noqa: E402  (import before pandas on Windows)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = Path("data/graphs")
DST = Path("data/graphs_src")
PROCESSED = Path("data/processed")
# MedBIoT labels are per capture file, so every originator in a *_mal_* capture is
# "bot": external scan responders and the C&C servers included. With
# --medbiot-devices-only, positives are restricted to the testbed devices
# (192.168.10.x) minus the C&C servers: .61 bashlite and .60 mirai (each originates
# flows to 76-84 devices); .109 torii is inferred the same way (14 devices).
MEDBIOT_CC = {"192.168.10.61", "192.168.10.60", "192.168.10.109", "192.168.10.1"}


def bot_flows(scenario: str) -> pd.DataFrame:
    t = pq.read_table(
        PROCESSED / f"{scenario}.parquet",
        columns=["start_time", "src_ip", "dst_ip", "label"],
        filters=[("label", "==", "bot")],
    )
    df = t.to_pandas()
    return df.sort_values("start_time").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-s", type=int, default=300)
    ap.add_argument("--out-json", type=Path,
                    default=Path("data/inspection_logs/relabel_infected_src.json"))
    ap.add_argument("--out-dir", type=Path, default=DST)
    ap.add_argument("--any-dst", action="store_true",
                    help="count bot flows to ANY destination (v2); default keeps v1 behaviour "
                         "(only flows between selected nodes)")
    ap.add_argument("--medbiot-devices-only", action="store_true",
                    help="MedBIoT positives = 192.168.10.x devices minus C&C servers")
    args = ap.parse_args()

    summary: dict[str, dict] = {}
    for sc_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        sc = sc_dir.name
        files = sorted(sc_dir.glob("window_*.pt"))
        if not files:
            continue
        flows = bot_flows(sc)
        ts = flows["start_time"].values
        out_dir = args.out_dir / sc
        out_dir.mkdir(parents=True, exist_ok=True)
        n_old = n_new = n_nodes = n_mismatch = 0
        src_ips: set[str] = set()
        for fp in files:
            g = torch.load(fp, weights_only=False)
            t0 = pd.Timestamp(g.window_start)
            if t0.tzinfo is None:
                t0 = t0.tz_localize("UTC")
            t1 = t0 + pd.Timedelta(seconds=args.window_s)
            lo = ts.searchsorted(t0.to_datetime64(), side="left")
            hi = ts.searchsorted(t1.to_datetime64(), side="left")
            sub = flows.iloc[lo:hi]
            nodes = list(g.node_ips)
            node_set = set(nodes)
            between = sub[sub["src_ip"].isin(node_set) & sub["dst_ip"].isin(node_set)]
            any_set = set(between["src_ip"]) | set(between["dst_ip"])
            src_set = set((sub if args.any_dst else between)["src_ip"])
            if args.medbiot_devices_only and sc.startswith("medbiot"):
                src_set = {ip for ip in src_set
                           if ip.startswith("192.168.10.") and ip not in MEDBIOT_CC}
            y_any = np.array([ip in any_set for ip in nodes], dtype=np.int64)
            y_src = np.array([ip in src_set for ip in nodes], dtype=np.int64)
            if not np.array_equal(y_any, g.y.numpy()):
                n_mismatch += 1
            src_ips |= {ip for ip in nodes if ip in src_set}
            n_old += int(g.y.sum())
            n_new += int(y_src.sum())
            n_nodes += len(nodes)
            g.y = torch.from_numpy(y_src).long()
            g.graph_y = torch.tensor([int(y_src.sum() > 0)], dtype=torch.long)
            torch.save(g, out_dir / fp.name)
        marker = sc_dir / ".complete"
        if marker.exists():
            (out_dir / ".complete").write_text(marker.read_text())
        summary[sc] = {
            "windows": len(files), "nodes": n_nodes,
            "bot_any_endpoint": n_old, "bot_infected_src": n_new,
            "label_mismatch_windows": n_mismatch,
            "distinct_src_bot_ips": sorted(src_ips)[:50],
            "n_distinct_src_bot_ips": len(src_ips),
        }
        print(f"{sc:34s} win={len(files):5d} nodes={n_nodes:7d} "
              f"bot_any={n_old:6d} bot_src={n_new:6d} ips={len(src_ips):4d} "
              f"mismatch={n_mismatch}", flush=True)
    args.out_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
