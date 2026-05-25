"""Precompute per-(scenario, window, node) flow sequences and persist them
under data/flow_seqs/<scenario>/window_<NNNN>.pt.

Mirrors scripts/build_graphs.py — same scenario list driven by
configs/split.yaml, same window config defaults, same drop_background=True
filter. Output is keyed by (scenario, window_idx) and aligned to the
node_ips already saved by Phase 3.

Large parquets (>= 100 MB) use per-window PyArrow filters so RAM stays
bounded (~1–3 GB per window instead of loading 54M rows).

Run after Phase 3 graphs exist; before scripts/train_phase5.py.

Examples
--------
uv run python scripts/build_flow_sequences.py --all --skip-existing
uv run python scripts/build_flow_sequences.py --scenario ctu13-10
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch  # before pandas on Windows (c10.dll load order)
import pandas as pd

from src.data.build_flow_seq_streaming import build_flow_seqs_streaming
from src.data.build_graphs_streaming import should_stream_parquet
from src.data.dataset import SplitSpec
from src.data.flow_seq import build_node_sequences
from src.data.flow_seq_dataset import cache_path, save_flow_sequences
from src.data.window import WindowConfig

PROCESSED = Path("data/processed")
GRAPHS = Path("data/graphs")
FLOW_SEQS = Path("data/flow_seqs")


def all_scenarios_from_split() -> list[str]:
    spec = SplitSpec.load()
    return list(spec.train_scenarios) + list(spec.holdout_test_scenarios)


def _load_graph_index(scenario: str) -> dict[int, tuple[pd.Timestamp, list[str], Path]]:
    graph_dir = GRAPHS / scenario
    by_idx: dict[int, tuple[pd.Timestamp, list[str], Path]] = {}
    for gp in sorted(graph_dir.glob("window_*.pt")):
        g = torch.load(gp, weights_only=False)
        by_idx[int(g.window_idx)] = (
            pd.Timestamp(g.window_start), list(g.node_ips), gp,
        )
    return by_idx


def build_for_scenario(
    scenario: str,
    *,
    window_s: int,
    max_flows: int,
    seed: int,
    drop_background: bool = True,
    skip_existing: bool = False,
    force_streaming: bool | None = None,
) -> dict:
    parquet = PROCESSED / f"{scenario}.parquet"
    if not parquet.exists():
        return {"scenario": scenario, "graphs": 0, "skipped": "no parquet"}
    graph_dir = GRAPHS / scenario
    if not graph_dir.is_dir():
        return {"scenario": scenario, "graphs": 0, "skipped": "no graphs"}

    by_idx = _load_graph_index(scenario)
    if not by_idx:
        return {"scenario": scenario, "graphs": 0, "skipped": "no graphs"}

    use_streaming = (
        force_streaming if force_streaming is not None
        else should_stream_parquet(parquet)
    )

    if use_streaming:
        res = build_flow_seqs_streaming(
            scenario, parquet, by_idx,
            window_s=window_s, max_flows=max_flows, seed=seed,
            drop_background=drop_background, skip_existing=skip_existing,
            root=FLOW_SEQS,
        )
        tag = "stream"
        n_done = res["graphs"]
        n_skip = res.get("skipped", 0)
        dt = res["seconds"]
    else:
        df = pd.read_parquet(parquet)
        if drop_background:
            df = df[df["label"].isin(["bot", "benign"])].reset_index(drop=True)
        win_delta = pd.Timedelta(seconds=window_s)
        t0 = time.perf_counter()
        n_done = n_skip = 0
        for idx in sorted(by_idx):
            if skip_existing and cache_path(scenario, idx, FLOW_SEQS).exists():
                n_skip += 1
                continue
            ws, node_ips, _ = by_idx[idx]
            t_end = ws + win_delta
            sub = df[(df["start_time"] >= ws) & (df["start_time"] < t_end)]
            flows_arr, mask_arr = build_node_sequences(
                sub, node_ips=node_ips, window_start=ws,
                window_seconds=float(window_s),
                max_flows=max_flows, seed=seed, eval_mode=False,
            )
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
        tag = "mem"
        del df

    gc.collect()
    out = {"scenario": scenario, "graphs": n_done, "seconds": dt, "mode": tag}
    if n_skip:
        out["skipped_existing"] = n_skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", action="append", default=None,
                    help="Repeatable. If omitted with --all, all train+holdout scenarios.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--window-s", type=int, default=300)
    ap.add_argument("--max-flows", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip windows whose flow_seq .pt already exists.")
    ap.add_argument("--streaming", action="store_true",
                    help="Force per-window PyArrow reads.")
    ap.add_argument("--no-streaming", action="store_true",
                    help="Force full parquet load (small files only).")
    args = ap.parse_args()
    if args.streaming and args.no_streaming:
        ap.error("use at most one of --streaming / --no-streaming")

    force_streaming: bool | None = None
    if args.streaming:
        force_streaming = True
    elif args.no_streaming:
        force_streaming = False

    if args.all:
        scenarios = all_scenarios_from_split()
    elif args.scenario:
        scenarios = list(args.scenario)
    else:
        ap.error("provide --scenario NAME (repeatable) or --all")

    print(f"window={args.window_s}s  max_flows={args.max_flows}  "
          f"skip_existing={args.skip_existing}  scenarios={len(scenarios)}\n",
          flush=True)
    for sc in scenarios:
        res = build_for_scenario(
            sc, window_s=args.window_s, max_flows=args.max_flows,
            seed=args.seed, skip_existing=args.skip_existing,
            force_streaming=force_streaming,
        )
        skip = res.get("skipped")
        if skip:
            print(f"[{sc:<32s}]  SKIP ({skip})", flush=True)
            continue
        extra = ""
        if res.get("skipped_existing"):
            extra = f"  skip={res['skipped_existing']}"
        print(f"[{sc:<32s}]  windows {res['graphs']:>4}  "
              f"{res['seconds']:>6.1f}s  [{res.get('mode', '?')}]{extra}",
              flush=True)


if __name__ == "__main__":
    main()
