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

# Windows: import torch before pandas/other native libs to avoid intermittent
# WinError 1114 on c10.dll (RTX laptops under load).
import os

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
import torch  # noqa: E402

import argparse
import time
from pathlib import Path

import pandas as pd

from src.data.build_graphs_streaming import (
    build_for_parquet_streaming,
    should_stream_parquet,
)
from src.data.graph import GraphConfig, build_graph, graph_summary
from src.data.window import WindowConfig, iter_windows

PROCESSED = Path("data/processed")
GRAPHS = Path("data/graphs")
COMPLETE_MARKER = ".complete"


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
    (out_dir / COMPLETE_MARKER).write_text(
        f"graphs={n_built}\nflows_used={n_used}\n", encoding="utf-8"
    )

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
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip scenarios that already have window_*.pt files.")
    ap.add_argument(
        "--streaming",
        action="store_true",
        help="Per-window PyArrow reads (auto for parquet >= 100 MB). "
             "Use for IoT-23 17-1 / 33-1 / 39-1 to avoid 15+ GB RAM spikes.",
    )
    ap.add_argument(
        "--no-streaming",
        action="store_true",
        help="Force full pd.read_parquet even for large files.",
    )
    args = ap.parse_args()

    win = WindowConfig(window_s=args.window_s)
    gcfg = GraphConfig(min_flows_per_node=args.min_flows_per_node, max_nodes=args.max_nodes)

    if args.all:
        files = sorted(PROCESSED.glob("*.parquet"))
    elif args.parquet:
        files = [args.parquet]
    else:
        ap.error("provide --parquet PATH or --all")

    use_stream = args.streaming and not args.no_streaming
    print(
        f"window={args.window_s}s  max_nodes={args.max_nodes}  files={len(files)}"
        f"  streaming={'on' if use_stream else 'auto>=100MB'}",
        flush=True,
    )
    print(flush=True)
    for p in files:
        scenario = p.stem
        out_dir = GRAPHS / scenario
        if args.skip_existing and (out_dir / COMPLETE_MARKER).is_file():
            n = len(list(out_dir.glob("window_*.pt")))
            print(f"[skip] {scenario:<32s}  ({n} graphs, complete)", flush=True)
            continue
        stream = use_stream or (
            not args.no_streaming and should_stream_parquet(p, force=False)
        )
        if stream:
            print(f"\n[{scenario}] streaming per-window reads ...", flush=True)
            res = build_for_parquet_streaming(
                p, win, gcfg, drop_background=not args.keep_background
            )
        else:
            res = build_for_parquet(p, win, gcfg, drop_background=not args.keep_background)
        tag = " stream" if res.get("streaming") else ""
        print(
            f"[{res['scenario']:<32s}]  flows {res['flows_used']:>10,}/{res['flows_in']:>10,}"
            f"  graphs {res['graphs']:>4}  bot {res['bot_graphs']:>3}"
            f"  bot-nodes {res['total_bot_nodes']:>5}"
            f"  {res['time_s']:>5.1f}s{tag}",
            flush=True,
        )


if __name__ == "__main__":
    main()
