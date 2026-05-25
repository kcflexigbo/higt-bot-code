"""Memory-bounded graph construction for large parquet files.

Instead of ``pd.read_parquet`` (loads all rows — 54M IoT-23 flows can exceed 15 GB RAM),
this module:

1. Reads parquet statistics (or only ``start_time``) for [t_min, t_max].
2. Walks window boundaries.
3. Loads rows per window via PyArrow dataset filters.

Phase 2 IoT-23 **parsing** used ``--streaming``; this is the Phase 3 analogue.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
import torch  # noqa: E402

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.data.graph import GraphConfig, build_graph
from src.data.schema import FLOW_COLUMNS
from src.data.window import WindowConfig, iter_window_starts

# Auto-enable streaming above this parquet size (bytes).
STREAMING_SIZE_THRESHOLD = 100 * 1024 * 1024  # 100 MB


def should_stream_parquet(path: Path, force: bool = False) -> bool:
    return force or path.stat().st_size >= STREAMING_SIZE_THRESHOLD


def parquet_time_bounds(path: Path) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """Return (t_min, t_max, row_count) without loading the full table."""
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    col_idx = pf.schema_arrow.get_field_index("start_time")

    t_min = t_max = None
    for rg in range(pf.metadata.num_row_groups):
        stats = pf.metadata.row_group(rg).column(col_idx).statistics
        if stats is None or not stats.has_min_max:
            continue
        lo = pd.Timestamp(stats.min)
        hi = pd.Timestamp(stats.max)
        if lo.tzinfo is None:
            lo = lo.tz_localize("UTC")
        else:
            lo = lo.tz_convert("UTC")
        if hi.tzinfo is None:
            hi = hi.tz_localize("UTC")
        else:
            hi = hi.tz_convert("UTC")
        t_min = lo if t_min is None else min(t_min, lo)
        t_max = hi if t_max is None else max(t_max, hi)

    if t_min is not None and t_max is not None:
        return t_min, t_max, n_rows

    table = pf.read(columns=["start_time"])
    col = table.column("start_time")
    return (
        pd.Timestamp(pc.min(col).as_py(), tz="UTC"),
        pd.Timestamp(pc.max(col).as_py(), tz="UTC"),
        n_rows,
    )


def _window_filter(
    t0: pd.Timestamp, t1: pd.Timestamp, *, drop_background: bool
) -> pc.Expression:
    t0s = pa.scalar(t0.to_pydatetime())
    t1s = pa.scalar(t1.to_pydatetime())
    filt = (pc.field("start_time") >= t0s) & (pc.field("start_time") < t1s)
    if drop_background:
        filt = filt & pc.field("label").isin(["bot", "benign"])
    return filt


def build_for_parquet_streaming(
    parquet: Path,
    win_cfg: WindowConfig,
    graph_cfg: GraphConfig,
    *,
    drop_background: bool = True,
    log_every: int = 25,
) -> dict:
    """Build graphs window-by-window with bounded RAM."""
    scenario = parquet.stem
    t_min, t_max, n_in = parquet_time_bounds(parquet)
    dataset = ds.dataset(parquet, format="parquet")

    out_dir = Path("data/graphs") / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    win_delta = pd.Timedelta(seconds=win_cfg.window_s)
    n_built = n_bot_graphs = total_bot_nodes = n_used = 0
    t0_wall = time.perf_counter()
    idx = 0

    for _, t_start in iter_window_starts(t_min, t_max, win_cfg):
        t_end = t_start + win_delta
        table = dataset.to_table(
            filter=_window_filter(t_start, t_end, drop_background=drop_background),
            columns=FLOW_COLUMNS,
        )
        if table.num_rows < win_cfg.min_flows:
            continue
        sub = table.to_pandas()
        n_used += len(sub)
        n_nodes = pd.concat([sub["src_ip"], sub["dst_ip"]]).nunique()
        if n_nodes < win_cfg.min_nodes:
            continue

        g = build_graph(
            sub,
            scenario=scenario,
            window_idx=idx,
            window_start=t_start,
            window_seconds=win_cfg.window_s,
            cfg=graph_cfg,
        )
        if g is None:
            continue
        torch.save(g, out_dir / f"window_{idx:05d}.pt")
        n_built += 1
        if int(g.graph_y) == 1:
            n_bot_graphs += 1
            total_bot_nodes += int(g.y.sum())
        if log_every and n_built % log_every == 0:
            print(
                f"    ...{n_built} graphs saved  (last window {table.num_rows:,} flows)",
                flush=True,
            )
        idx += 1

    dt = time.perf_counter() - t0_wall
    (out_dir / ".complete").write_text(
        f"graphs={n_built}\nflows_used={n_used}\nstreaming=1\n", encoding="utf-8"
    )
    return {
        "scenario": scenario,
        "flows_in": n_in,
        "flows_used": n_used,
        "graphs": n_built,
        "bot_graphs": n_bot_graphs,
        "total_bot_nodes": total_bot_nodes,
        "time_s": dt,
        "streaming": True,
    }
