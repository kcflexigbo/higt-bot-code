"""How many bot hosts never reach the model? (review item C12)

Graph construction keeps only hosts with >= 3 flows in the window and at most
the 400 most active hosts, and skips windows with < 20 flows or < 5 hosts.
For every scenario in the split this counts, per (window, host) pair, the bot
hosts present in the bot flows of a window and how many of them are nodes of
the built graph, under both labelling rules (any endpoint / infected source).

Output: data/inspection_logs/blind_spot_counts.{md,json}
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
import torch  # noqa: E402

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.data.dataset import SplitSpec

GRAPHS = Path("data/graphs")
PROCESSED = Path("data/processed")
OUT = Path("data/inspection_logs/blind_spot_counts")
W = pd.Timedelta(seconds=300)


def main():
    spec = SplitSpec.load()
    rows = {}
    for sc in spec.train_scenarios + spec.holdout_test_scenarios:
        flows = pq.read_table(PROCESSED / f"{sc}.parquet", columns=["start_time", "src_ip", "dst_ip"],
                              filters=[("label", "==", "bot")]).to_pandas()
        built = {}
        for fp in sorted((GRAPHS / sc).glob("window_*.pt")):
            g = torch.load(fp, weights_only=False)
            t0 = pd.Timestamp(g.window_start)
            built[t0 if t0.tzinfo else t0.tz_localize("UTC")] = set(g.node_ips)
        t_min = flows["start_time"].min() if len(flows) else None
        # Window starts for bot flows (non-overlapping 300 s windows anchored at the
        # scenario's first flow, floored like src/data/window.py).
        all_min = pq.read_table(PROCESSED / f"{sc}.parquet", columns=["start_time"]).column(0).to_pandas().min()
        origin = all_min.floor("300s")
        flows["w"] = origin + ((flows["start_time"] - origin) // W) * W
        c = {"any_pairs": 0, "any_kept": 0, "src_pairs": 0, "src_kept": 0,
             "src_pairs_in_unbuilt_windows": 0, "bot_flows": int(len(flows))}
        for w, sub in flows.groupby("w"):
            any_ips = set(sub["src_ip"]) | set(sub["dst_ip"])
            src_ips = set(sub["src_ip"])
            nodes = built.get(w)
            c["any_pairs"] += len(any_ips); c["src_pairs"] += len(src_ips)
            if nodes is None:
                c["src_pairs_in_unbuilt_windows"] += len(src_ips)
                continue
            c["any_kept"] += len(any_ips & nodes); c["src_kept"] += len(src_ips & nodes)
        rows[sc] = c
        print(sc, c, flush=True)
    OUT.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    tot = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))}
    md = ["# Bot hosts removed by graph-construction filters", "",
          "Counts are (window, host) pairs over all windows of every split scenario. "
          "'kept' = the host is a node of the built window graph.", "",
          "| Scenario | bot-source pairs | kept | in skipped windows | any-endpoint pairs | kept |",
          "|---|---|---|---|---|---|"]
    for sc, r in list(rows.items()) + [("**total**", tot)]:
        md.append(f"| {sc} | {r['src_pairs']} | {r['src_kept']} ({r['src_kept'] / max(r['src_pairs'], 1):.1%}) | "
                  f"{r['src_pairs_in_unbuilt_windows']} | {r['any_pairs']} | "
                  f"{r['any_kept']} ({r['any_kept'] / max(r['any_pairs'], 1):.1%}) |")
    OUT.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT.with_suffix(".md"))


if __name__ == "__main__":
    main()
