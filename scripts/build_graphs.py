"""Build PyG graphs from canonical-schema parquet files.

Reads data/processed/*.parquet, applies time-window splitting + graph
construction, writes data/graphs/<scenario>/window_<NNNN>.pt.

Examples
--------
# One parquet
uv run python scripts/build_graphs.py --parquet data/processed/ctu13-10.parquet

# Everything in data/processed/
uv run python scripts/build_graphs.py --all

# Smaller windows for ablation
uv run python scripts/build_graphs.py --parquet data/processed/ctu13-10.parquet --window-s 60
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from src.data.graph import GraphConfig, build_graph, graph_summary
from src.data.window import WindowConfig, iter_windows

PROCESSED = Path("data/processed")
GRAPHS = Path("data/graphs")


def build_for_parquet(
    parquet: Path,
    win_cfg: WindowConfig,
    graph_cfg: GraphConfig,
    *,
    drop_background: bool = True,
) -> dict:
    df = pd.read_parquet(parquet)
    scenario = parquet.stem
    n_in = len(df)
    if drop_background:
        df = df[df["label"].isin(["bot", "benign"])].reset_index(drop=True)
    n_used = len(df)

    out_dir = GRAPHS / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    n_built = 0
    n_bot_graphs = 0
    total_bot_nodes = 0
    t0 = time.perf_counter()
    for idx, t_start, sub in iter_windows(df, win_cfg):
        g = build_graph(
            sub, scenario=scenario, window_idx=idx,
            window_start=t_start, window_seconds=win_cfg.window_s,
            cfg=graph_cfg,
        )
        if g is None:
            continue
        torch.save(g, out_dir / f"window_{idx:05d}.pt")
        n_built += 1
        if int(g.graph_y) == 1:
            n_bot_graphs += 1
            total_bot_nodes += int(g.y.sum())
    dt = time.perf_counter() - t0

    return {
        "scenario": scenario,
        "flows_in": n_in,
        "flows_used": n_used,
        "graphs": n_built,
        "bot_graphs": n_bot_graphs,
        "total_bot_nodes": total_bot_nodes,
        "time_s": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, help="A single parquet file")
    ap.add_argument("--all", action="store_true", help="Process every parquet under data/processed/")
    ap.add_argument("--window-s", type=int, default=300)
    ap.add_argument("--max-nodes", type=int, default=400)
    ap.add_argument("--min-flows-per-node", type=int, default=3)
    ap.add_argument("--keep-background", action="store_true",
                    help="By default, drop 'background' flows (CTU-13 only has them). "
                         "Set this to include them.")
    args = ap.parse_args()

    win = WindowConfig(window_s=args.window_s)
    gcfg = GraphConfig(min_flows_per_node=args.min_flows_per_node, max_nodes=args.max_nodes)

    if args.all:
        files = sorted(PROCESSED.glob("*.parquet"))
    elif args.parquet:
        files = [args.parquet]
    else:
        ap.error("provide --parquet PATH or --all")

    print(f"window={args.window_s}s  max_nodes={args.max_nodes}  files={len(files)}")
    print()
    for p in files:
        res = build_for_parquet(p, win, gcfg, drop_background=not args.keep_background)
        print(f"[{res['scenario']:<32s}]  flows {res['flows_used']:>10,}/{res['flows_in']:>10,}"
              f"  graphs {res['graphs']:>4}  bot {res['bot_graphs']:>3}"
              f"  bot-nodes {res['total_bot_nodes']:>5}"
              f"  {res['time_s']:>5.1f}s")


if __name__ == "__main__":
    main()
