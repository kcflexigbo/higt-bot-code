"""Per-scenario graph statistics relevant to P2P vs centralised botnets.

For every scenario under a graphs dir: windows, windows with a bot, bot host-windows,
edges, bot-bot edges, fraction of bot nodes with at least one bot neighbour (either
direction), edge homophily, and the share of a bot's edges that go to other bots.

    uv run python -m scripts.p2p_graph_stats --graphs data/graphs_v3 --out data/inspection_logs/p2p/graph_stats.json
"""
from __future__ import annotations

import torch  # noqa: F401  (import before pandas on Windows)

import argparse
import json
from pathlib import Path

import numpy as np


def stats(sc_dir: Path) -> dict:
    n_win = n_botwin = n_bot = n_nodes = n_edge = n_bb = n_same = 0
    bot_with_bot_nb = 0
    bot_edge_tot = bot_edge_tobot = 0
    for f in sorted(sc_dir.glob("window_*.pt")):
        g = torch.load(f, weights_only=False)
        y = g.y.numpy().astype(int); ei = g.edge_index.numpy()
        n_win += 1; n_nodes += len(y); nb = int(y.sum()); n_bot += nb; n_botwin += nb > 0
        if ei.shape[1] == 0:
            continue
        ys, yd = y[ei[0]], y[ei[1]]
        n_edge += ei.shape[1]; n_bb += int((ys & yd).sum()); n_same += int((ys == yd).sum())
        bb = (ys == 1) & (yd == 1)
        hit = set(ei[0][bb].tolist()) | set(ei[1][bb].tolist())
        bot_with_bot_nb += len(hit)
        inc = (ys == 1) | (yd == 1)
        bot_edge_tot += int(inc.sum()); bot_edge_tobot += int(bb.sum())
    return {"windows": n_win, "windows_with_bot": n_botwin, "bot_host_windows": n_bot, "nodes": n_nodes,
            "edges": n_edge, "bot_bot_edges": n_bb,
            "frac_bots_with_bot_neighbour": bot_with_bot_nb / max(n_bot, 1),
            "edge_homophily": n_same / max(n_edge, 1),
            "share_of_bot_edges_bot_bot": bot_edge_tobot / max(bot_edge_tot, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=Path, default=Path("data/graphs_v3"))
    ap.add_argument("--out", type=Path, default=Path("data/inspection_logs/p2p/graph_stats.json"))
    a = ap.parse_args()
    res = {d.name: stats(d) for d in sorted(p for p in a.graphs.iterdir() if p.is_dir())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k:34s} win {v['windows']:5d} botwin {v['windows_with_bot']:5d} bots {v['bot_host_windows']:6d} "
              f"bb_edges {v['bot_bot_edges']:6d} bots_w_bot_nb {v['frac_bots_with_bot_neighbour']:.3f} "
              f"hom {v['edge_homophily']:.3f} bb_share {v['share_of_bot_edges_bot_bot']:.3f}")


if __name__ == "__main__":
    main()
