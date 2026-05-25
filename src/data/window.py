"""Time-window iteration over a canonical-schema flow DataFrame.

A "window" is a non-overlapping interval [t0, t0 + W) anchored to the
scenario's first flow. Every flow whose start_time falls in that interval
belongs to that window.

The iterator yields (window_idx, window_start_utc, sub_df) tuples. Windows
with too few flows or too few distinct nodes are skipped — they have nothing
to learn.

The choice of W = 300s (5 min) is the plan's default; ablation later sweeps
{60, 180, 300, 600}.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

DEFAULT_WINDOW_S = 300


@dataclass
class WindowConfig:
    """Parameters for time-window slicing."""

    window_s: int = DEFAULT_WINDOW_S
    stride_s: int | None = None          # default = window_s (non-overlapping)
    min_flows: int = 20                  # skip windows with fewer than this many flows
    min_nodes: int = 5                   # skip windows with fewer than this many distinct IPs

    def stride(self) -> int:
        return self.stride_s if self.stride_s is not None else self.window_s


def iter_window_starts(
    t_min: pd.Timestamp, t_max: pd.Timestamp, cfg: WindowConfig | None = None
) -> Iterator[tuple[int, pd.Timestamp]]:
    """Yield (window_idx, window_start) without materializing flow rows."""
    if cfg is None:
        cfg = WindowConfig()
    start = t_min.floor(f"{cfg.window_s}s")
    win_delta = pd.Timedelta(seconds=cfg.window_s)
    stride_delta = pd.Timedelta(seconds=cfg.stride())

    idx = 0
    t0 = start
    while t0 <= t_max:
        yield idx, t0
        idx += 1
        t0 = t0 + stride_delta


def iter_windows(
    flows: pd.DataFrame, cfg: WindowConfig | None = None
) -> Iterator[tuple[int, pd.Timestamp, pd.DataFrame]]:
    """Yield (idx, window_start, sub_df) for each non-empty window.

    `flows` must be in canonical schema (see src/data/schema.py).
    """
    if cfg is None:
        cfg = WindowConfig()
    if flows.empty:
        return

    start = flows["start_time"].min().floor(f"{cfg.window_s}s")
    end = flows["start_time"].max()
    win_delta = pd.Timedelta(seconds=cfg.window_s)
    stride_delta = pd.Timedelta(seconds=cfg.stride())

    flows_sorted = flows.sort_values("start_time").reset_index(drop=True)
    # Use searchsorted on the underlying numpy timestamps for O(log n) slicing.
    ts_values = flows_sorted["start_time"].values

    idx = 0
    t0 = start
    while t0 <= end:
        t1 = t0 + win_delta
        lo = ts_values.searchsorted(t0.to_numpy(), side="left")
        hi = ts_values.searchsorted(t1.to_numpy(), side="left")
        if hi - lo >= cfg.min_flows:
            sub = flows_sorted.iloc[lo:hi]
            n_nodes = pd.concat([sub["src_ip"], sub["dst_ip"]]).nunique()
            if n_nodes >= cfg.min_nodes:
                yield idx, t0, sub.copy()
                idx += 1
        t0 = t0 + stride_delta
