"""Precompute per-(scenario, window, node) flow sequences and persist them
under data/flow_seqs/<scenario>/window_<NNNN>.pt.

Mirrors scripts/build_graphs.py — same scenario list driven by
configs/split.yaml, same window config defaults, same drop_background=True
filter. Output is keyed by (scenario, window_idx) and aligned to the
node_ips already saved by Phase 3.

Run after Phase 3 graphs exist; before scripts/train_phase5.py.

Examples
--------
uv run python scripts/build_flow_sequences.py --all
uv run python scripts/build_flow_sequences.py --scenario ctu13-10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from src.data.dataset import SplitSpec
from src.data.flow_seq import build_node_sequences
from src.data.flow_seq_dataset import save_flow_sequences
from src.data.window import WindowConfig, iter_windows

PROCESSED = Path("data/processed")
GRAPHS = Path("data/graphs")
FLOW_SEQS = Path("data/flow_seqs")


def all_scenarios_from_split() -> list[str]:
    spec = SplitSpec.load()
    return list(spec.train_scenarios) + list(spec.holdout_test_scenarios)


def build_for_scenario(
    scenario: str, *, window_s: int, max_flows: int, seed: int,
    drop_background: bool = True,
) -> dict:
    parquet = PROCESSED / f"{scenario}.parquet"
    if not parquet.exists():
        return {"scenario": scenario, "graphs": 0, "skipped": "no parquet"}
    graph_dir = GRAPHS / scenario
    graph_files = sorted(graph_dir.glob("window_*.pt"))
    if not graph_files:
        return {"scenario": scenario, "graphs": 0, "skipped": "no graphs"}

    df = pd.read_parquet(parquet)
    if drop_background:
        df = df[df["label"].isin(["bot", "benign"])].reset_index(drop=True)

    wcfg = WindowConfig(window_s=window_s)

    # Build a lookup graph_window_idx -> (window_start, node_ips, graph_path)
    by_idx: dict[int, tuple[pd.Timestamp, list[str], Path]] = {}
    for gp in graph_files:
        g = torch.load(gp, weights_only=False)
        by_idx[int(g.window_idx)] = (pd.Timestamp(g.window_start), list(g.node_ips), gp)

    t0 = time.perf_counter()
    n_done = 0
    for idx, ts, sub in iter_windows(df, wcfg):
        if idx not in by_idx:
            continue   # window dropped at graph construction → skip here too
        _, node_ips, gp = by_idx[idx]
        flows_arr, mask_arr = build_node_sequences(
            sub, node_ips=node_ips, window_start=ts,
            window_seconds=float(window_s),
            max_flows=max_flows, seed=seed, eval_mode=False,
        )
        # Build a minimal object compatible with save_flow_sequences().
        class _S:
            pass
        s = _S()
        s.flows = torch.from_numpy(flows_arr)
        s.flow_mask = torch.from_numpy(mask_arr)
        s.node_ips = node_ips
        s.scenario = scenario
        s.window_idx = idx
        save_flow_sequences(s, root=FLOW_SEQS)
        n_done += 1
    dt = time.perf_counter() - t0
    return {"scenario": scenario, "graphs": n_done, "seconds": dt}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", default=None,
                    help="Repeatable. If omitted with --all, all train+holdout scenarios.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--window-s", type=int, default=300)
    ap.add_argument("--max-flows", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.all:
        scenarios = all_scenarios_from_split()
    elif args.scenario:
        scenarios = list(args.scenario)
    else:
        ap.error("provide --scenario NAME (repeatable) or --all")

    print(f"window={args.window_s}s  max_flows={args.max_flows}  "
          f"scenarios={len(scenarios)}\n")
    for sc in scenarios:
        res = build_for_scenario(sc, window_s=args.window_s,
                                  max_flows=args.max_flows, seed=args.seed)
        skip = res.get("skipped")
        if skip:
            print(f"[{sc:<32s}]  SKIP ({skip})")
            continue
        print(f"[{sc:<32s}]  windows {res['graphs']:>4}  {res['seconds']:>6.1f}s")


if __name__ == "__main__":
    main()
