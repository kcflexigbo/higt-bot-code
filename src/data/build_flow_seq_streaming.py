"""Memory-bounded flow-sequence precompute for large parquets.

Only builds windows that already have Phase-3 graphs (``by_idx``), loading
each window's flows via PyArrow filters — same idea as ``build_graphs_streaming``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import torch

from src.data.build_graphs_streaming import _window_filter
from src.data.flow_seq import build_node_sequences
from src.data.flow_seq_dataset import cache_path, save_flow_sequences
from src.data.schema import FLOW_COLUMNS


def build_flow_seqs_streaming(
    scenario: str,
    parquet: Path,
    by_idx: dict[int, tuple[pd.Timestamp, list[str], Path]],
    *,
    window_s: int,
    max_flows: int,
    seed: int,
    drop_background: bool = True,
    skip_existing: bool = False,
    root: Path,
    log_every: int = 50,
) -> dict:
    """Write ``data/flow_seqs/<scenario>/window_*.pt`` window-by-window."""
    dataset = ds.dataset(parquet, format="parquet")
    win_delta = pd.Timedelta(seconds=window_s)
    t0 = time.perf_counter()
    n_done = n_skip = 0

    for idx in sorted(by_idx):
        out = cache_path(scenario, idx, root)
        if skip_existing and out.exists():
            n_skip += 1
            continue

        ws, node_ips, _ = by_idx[idx]
        t_end = ws + win_delta
        table = dataset.to_table(
            filter=_window_filter(ws, t_end, drop_background=drop_background),
            columns=FLOW_COLUMNS,
        )
        if table.num_rows == 0:
            continue
        sub = table.to_pandas()
        flows_arr, mask_arr = build_node_sequences(
            sub,
            node_ips=node_ips,
            window_start=ws,
            window_seconds=float(window_s),
            max_flows=max_flows,
            seed=seed,
            eval_mode=False,
        )

        class _S:
            pass

        s = _S()
        s.flows = torch.from_numpy(flows_arr)
        s.flow_mask = torch.from_numpy(mask_arr)
        s.node_ips = node_ips
        s.scenario = scenario
        s.window_idx = idx
        save_flow_sequences(s, root=root)
        n_done += 1
        if log_every and n_done % log_every == 0:
            print(f"    ...{n_done} flow_seqs saved (window {idx})", flush=True)

    dt = time.perf_counter() - t0
    return {
        "scenario": scenario,
        "graphs": n_done,
        "skipped": n_skip,
        "seconds": dt,
        "streaming": True,
    }
