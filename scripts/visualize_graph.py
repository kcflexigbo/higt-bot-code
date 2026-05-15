"""Visualize one constructed graph with NetworkX.

Plan's Phase 3 gate: visualize one window — bot nodes should appear densely
interconnected, benign nodes mostly peripheral. If they don't, DiffPool will
have nothing useful to cluster later.

By default, picks the bot-positive window with the most bot nodes (best
visual evidence).

Usage:
    uv run python scripts/visualize_graph.py --scenario ctu13-10
    uv run python scripts/visualize_graph.py --scenario ctu13-10 --window 17
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

GRAPHS = Path("data/graphs")


def pick_window(scenario_dir: Path, window: int | None) -> Path:
    """If --window given, use that. Else pick the window with the most bot nodes."""
    if window is not None:
        return scenario_dir / f"window_{window:05d}.pt"
    best_path, best_bot = None, -1
    for fp in sorted(scenario_dir.glob("window_*.pt")):
        g = torch.load(fp, weights_only=False)
        n_bot = int(g.y.sum())
        if n_bot > best_bot:
            best_bot, best_path = n_bot, fp
    if best_path is None:
        raise FileNotFoundError(f"No graphs in {scenario_dir}")
    print(f"  auto-picked {best_path.name} (n_bot_nodes={best_bot})")
    return best_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scenario_dir = GRAPHS / args.scenario
    fp = pick_window(scenario_dir, args.window)
    g = torch.load(fp, weights_only=False)

    n_nodes = int(g.num_nodes)
    n_edges = int(g.edge_index.size(1))
    n_bot = int(g.y.sum())
    print(f"  {fp.name}: nodes={n_nodes} edges={n_edges} bot_nodes={n_bot}")

    # Build a NetworkX DiGraph
    G = nx.DiGraph()
    ips = getattr(g, "node_ips", [str(i) for i in range(n_nodes)])
    for i, ip in enumerate(ips):
        G.add_node(i, ip=ip, bot=int(g.y[i].item()))
    ei = g.edge_index.cpu().numpy()
    for s, d in ei.T:
        G.add_edge(int(s), int(d))

    pos = nx.spring_layout(G, seed=42, k=1.2)

    bot_nodes = [i for i in G.nodes if G.nodes[i]["bot"] == 1]
    benign_nodes = [i for i in G.nodes if G.nodes[i]["bot"] == 0]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.25, ax=ax, arrows=False, width=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=benign_nodes, node_color="#3b82f6",
                            node_size=120, label=f"benign ({len(benign_nodes)})", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=bot_nodes, node_color="#ef4444",
                            node_size=320, label=f"bot ({len(bot_nodes)})", edgecolors="black",
                            linewidths=1.2, ax=ax)

    # Label only bots — labels on every node would be unreadable.
    bot_labels = {i: ips[i] for i in bot_nodes}
    nx.draw_networkx_labels(G, pos, labels=bot_labels, font_size=8, ax=ax)

    title = (f"{args.scenario} — {fp.stem}  "
             f"({n_nodes} nodes, {n_edges} edges, {n_bot} bot)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.axis("off")
    fig.tight_layout()

    out = args.out or (Path("data/inspection_logs/figures") / f"graph_{args.scenario}_{fp.stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved → {out}")


if __name__ == "__main__":
    main()
