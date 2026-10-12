"""Recompute node features from ALL of a host's flows in the window (v3).

`src/data/graph.py::build_graph` computes `x` from flows *between selected
nodes* only, so a scanner whose targets were pruned (< 3 flows each, or beyond
the 400-node cap) gets x ~ 0 and looks idle (fable_beating_trees_2026-10-11.md:
39 of XGBoost's 62 test FNs). This script copies the graphs of --src (default
the v2 labels), keeps nodes, edges, edge features and labels, and recomputes x
from every flow in the window that touches a node (the same rows build_graph
saw before its node restriction: label in {bot, benign}, start_time in
[window_start, window_start + W)).

Check: recomputing x from the between-node flows must reproduce the stored x;
windows where it does not are counted and reported.

    uv run python scripts/refeature_graphs.py            # data/graphs_src2 -> data/graphs_v3
Train on the result with HIGT_GRAPHS_DIR=data/graphs_v3.
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
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.data.build_graphs_streaming import FLOW_COLUMNS, _window_filter
from src.data.features import compute_node_features

PROCESSED = Path("data/processed")


class _SequentialWindows:
    """One sequential pass over a time-sorted parquet: windows must be requested in
    increasing time order. Same rows as the filtered scan (label in {bot, benign},
    t0 <= start_time < t1), without re-scanning the file for every window."""

    def __init__(self, pf: pq.ParquetFile) -> None:
        self.pf, self.next_rg = pf, 0
        self.buf = pd.DataFrame(columns=FLOW_COLUMNS)

    def _more(self) -> bool:
        if self.next_rg >= self.pf.num_row_groups:
            return False
        df = self.pf.read_row_group(self.next_rg, columns=FLOW_COLUMNS).to_pandas()
        self.next_rg += 1
        df = df[df["label"].isin(["bot", "benign"])]
        self.buf = pd.concat([self.buf, df], ignore_index=True) if len(self.buf) else df.reset_index(drop=True)
        return True

    def window(self, t0, t1) -> pd.DataFrame:
        while (not len(self.buf) or self.buf["start_time"].max() < t1) and self._more():
            pass
        st = self.buf["start_time"]
        out = self.buf[(st >= t0) & (st < t1)].reset_index(drop=True)
        self.buf = self.buf[st >= t1].reset_index(drop=True)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/graphs_src2"))
    ap.add_argument("--dst", type=Path, default=Path("data/graphs_v3"))
    ap.add_argument("--window-s", type=int, default=300)
    ap.add_argument("--fast", action="store_true",
                    help="single sequential pass per (time-sorted) parquet instead of one filtered scan per window")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these scenarios")
    ap.add_argument("--out-json", type=Path, default=Path("data/inspection_logs/refeature_v3.json"))
    args = ap.parse_args()

    summary = json.loads(args.out_json.read_text()) if args.out_json.exists() else {}
    for sc_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        sc = sc_dir.name
        files = sorted(sc_dir.glob("window_*.pt"))
        if not files or (args.only and sc not in args.only):
            continue
        out_dir = args.dst / sc
        if (out_dir / ".complete").exists():
            print(f"{sc:34s} already done", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset = ds.dataset(PROCESSED / f"{sc}.parquet", format="parquet")
        pf = pq.ParquetFile(PROCESSED / f"{sc}.parquet")
        reader = _SequentialWindows(pf) if args.fast else None
        n_mismatch = n_changed_nodes = n_nodes = 0
        zero_before = zero_after = 0
        for fp in files:
            g = torch.load(fp, weights_only=False)
            t0 = pd.Timestamp(g.window_start)
            if t0.tzinfo is None:
                t0 = t0.tz_localize("UTC")
            t1 = t0 + pd.Timedelta(seconds=args.window_s)
            if reader is not None:
                flows = reader.window(t0, t1)
            else:
                flows = dataset.to_table(filter=_window_filter(t0, t1, drop_background=True),
                                         columns=FLOW_COLUMNS).to_pandas()
            nodes = list(g.node_ips)
            node_set = set(nodes)
            touching = flows[flows["src_ip"].isin(node_set) | flows["dst_ip"].isin(node_set)]
            between = touching[touching["src_ip"].isin(node_set) & touching["dst_ip"].isin(node_set)]
            x_old = compute_node_features(between.reset_index(drop=True), nodes, args.window_s)
            if not np.allclose(x_old, g.x.numpy(), rtol=1e-4, atol=1e-4):
                n_mismatch += 1
            x_new = compute_node_features(touching.reset_index(drop=True), nodes, args.window_s)
            n_changed_nodes += int((~np.isclose(x_new, x_old, rtol=1e-4, atol=1e-4)).any(axis=1).sum())
            n_nodes += len(nodes)
            y = g.y.numpy().astype(bool)
            zero_before += int((g.x.numpy()[y, 0] == 0).sum())  # bots with fan_out == 0
            zero_after += int((x_new[y, 0] == 0).sum())
            g.x = torch.from_numpy(x_new).float()
            torch.save(g, out_dir / fp.name)
        (out_dir / ".complete").write_text(f"graphs={len(files)}\nrefeatured=1\n", encoding="utf-8")
        summary[sc] = {"windows": len(files), "nodes": n_nodes, "nodes_changed": n_changed_nodes,
                       "reproduce_mismatch_windows": n_mismatch,
                       "bots_fan_out_zero_before": zero_before, "bots_fan_out_zero_after": zero_after}
        args.out_json.write_text(json.dumps(summary, indent=2))
        print(f"{sc:34s} win={len(files):5d} nodes={n_nodes:7d} changed={n_changed_nodes:6d} "
              f"mismatch={n_mismatch} bots_fanout0 {zero_before}->{zero_after}", flush=True)


if __name__ == "__main__":
    main()
