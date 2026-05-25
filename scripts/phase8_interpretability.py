"""Phase 8.5 — interpretability artefacts for the final HiGT-Bot model.

For each of a few representative test windows (chosen to span easy, mid, and
hard scenarios), extract from the final GT-edge model:
  1. The SAGPool per-node *score* — which nodes the pooling layer ranks
     as most discriminative. Compared to the ground-truth label, this tells
     us whether the pooling layer is finding the bots or losing them.
  2. The pre-pool GINE embedding norm per node — coarse "model attention" proxy.
  3. Histograms of pool score split by true label.

Outputs:
  data/inspection_logs/figures/phase8_sagpool_scores.png   (one panel/scenario)
  data/inspection_logs/phase8_interpretability.md
  data/inspection_logs/phase8_interpretability.json        (raw arrays per window)

Picks one window per chosen scenario from the test split.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from src.data.dataset import (
    SplitSpec, fit_edge_scaler, fit_feature_scaler, load_graphs, load_split_files,
)
from src.data.embedding_dataset import EmbeddingGraphDataset
from src.models.higt_bot import HiGTBot
from src.utils.seeding import pick_device, set_seed

LOG_DIR = Path("data/inspection_logs")
FIG_DIR = LOG_DIR / "figures"
CKPT = Path("experiments/phase7/higt_bot_edge.pt")
TARGET_SCENARIOS = [
    "iot23-35-1",        # hardest (31/9905)
    "ctu13-10",          # hard
    "iot23-7-1",         # mid
    "ctu13-9",           # easy (high bot rate)
    "medbiot-bashlite_mal_spread_all",  # near-100% bots
]


def find_one_window(files: list[Path], scenario: str) -> Path | None:
    """Pick a test window for `scenario` that has at least one positive node."""
    for f in files:
        g = torch.load(f, weights_only=False)
        if g.scenario != scenario:
            continue
        if int(g.y.sum().item()) >= 1:
            return f
        # If no positive window, just take the first scenario window
    for f in files:
        g = torch.load(f, weights_only=False)
        if g.scenario == scenario:
            return f
    return None


@torch.no_grad()
def extract(model, batch) -> dict:
    """Run a single-batch forward and pull out SAGPool score + embed norm."""
    model.eval()
    # Mirror HiGTBot.forward up to the post-pool stage.
    x = torch.cat([batch.node_emb, batch.x], dim=-1)
    z = model.embed_gnn(x, batch.edge_index, batch.edge_attr)
    z = model.embed_proj(z)
    z_pool, ei_pool, ea_pool, b_pool, perm, score = model.pool(
        z, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch,
    )
    out = model(batch)
    pred = out["logits"].argmax(dim=-1)
    return {
        "score": score.detach().cpu().numpy(),
        "perm": perm.detach().cpu().numpy(),
        "kept_y": batch.y[perm].detach().cpu().numpy(),
        "z_norm": z.norm(dim=-1).detach().cpu().numpy(),
        "y": batch.y.cpu().numpy(),
        "pred": pred.cpu().numpy(),
        "node_ips": list(batch.node_ips) if isinstance(batch.node_ips, list)
                                          else None,
        "scenario": batch.scenario,
    }


def plot_scenario(info: dict, scenario: str, out_path: Path) -> None:
    score = info["score"]
    y = info["y"]
    perm = info["perm"]
    kept_y = info["kept_y"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: pool score histogram split by label.
    score_pos = score[(kept_y == 1)]
    score_neg = score[(kept_y == 0)]
    bins = 30
    axes[0].hist(score_neg, bins=bins, alpha=0.5, label="benign (kept)",
                  color="tab:blue")
    axes[0].hist(score_pos, bins=bins, alpha=0.7, label="bot (kept)",
                  color="tab:red")
    axes[0].set_title(f"SAGPool retained-node scores — {scenario}")
    axes[0].set_xlabel("score")
    axes[0].set_ylabel("nodes")
    axes[0].legend()

    # Panel 2: keep-rate per class (how many bots survived?).
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_pos_kept = int((kept_y == 1).sum())
    n_neg_kept = int((kept_y == 0).sum())
    cats = ["benign", "bot"]
    kept_rate = [n_neg_kept / max(n_neg, 1), n_pos_kept / max(n_pos, 1)]
    axes[1].bar(cats, kept_rate, color=["tab:blue", "tab:red"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("fraction kept after SAGPool")
    axes[1].set_title(f"Per-class survival — {scenario}\n"
                       f"benign {n_neg_kept}/{n_neg}, bot {n_pos_kept}/{n_pos}")
    for i, v in enumerate(kept_rate):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(42)
    device = pick_device()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    blob = torch.load(CKPT, map_location=device, weights_only=False)
    args = blob.get("args", {})

    spec = SplitSpec.load()
    tr_files = load_split_files("train", spec)
    te_files = load_split_files("test", spec)
    tr_graphs = load_graphs(tr_files)
    e_mean, e_std = fit_edge_scaler(tr_graphs)
    n_mean, n_std = fit_feature_scaler(tr_graphs)
    raw_feat_dim = int(tr_graphs[0].x.size(1))
    edge_dim = int(tr_graphs[0].edge_attr.size(1))
    del tr_graphs
    gc.collect()

    model = HiGTBot(
        d_model=64, raw_feat_dim=raw_feat_dim, edge_dim=edge_dim,
        hidden=int(args.get("hidden", 128)),
        pool_ratio=float(args.get("pool_ratio", 0.5)),
        gt_variant=str(args.get("gt_variant", "edge")),
        gt_layers=int(args.get("gt_layers", 2)),
        gt_heads=int(args.get("gt_heads", 4)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(blob["state_dict"])

    summary = []
    for scenario in TARGET_SCENARIOS:
        gf = find_one_window(te_files, scenario)
        if gf is None:
            print(f"skip — no test window for {scenario}")
            continue
        ds = EmbeddingGraphDataset([gf],
                                     emb_root=Path("data/flow_embeddings"),
                                     edge_mean=e_mean, edge_std=e_std,
                                     node_mean=n_mean, node_std=n_std)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        batch = next(iter(loader)).to(device)
        info = extract(model, batch)
        out_png = FIG_DIR / f"phase8_sagpool_{scenario.replace('/', '_')}.png"
        plot_scenario(info, scenario, out_png)
        n = int(info["y"].size)
        n_pos = int(info["y"].sum())
        n_pos_kept = int((info["kept_y"] == 1).sum())
        n_kept_total = int(info["perm"].size)
        f1_tp = int(((info["y"] == 1) & (info["pred"] == 1)).sum())
        f1_fn = int(((info["y"] == 1) & (info["pred"] == 0)).sum())
        f1_fp = int(((info["y"] == 0) & (info["pred"] == 1)).sum())
        summary.append({
            "scenario": scenario, "window": gf.name,
            "n": n, "n_pos": n_pos,
            "n_kept": n_kept_total, "n_pos_kept": n_pos_kept,
            "bot_keep_rate": n_pos_kept / max(n_pos, 1),
            "benign_keep_rate": (n_kept_total - n_pos_kept) / max(n - n_pos, 1),
            "pred_tp": f1_tp, "pred_fn": f1_fn, "pred_fp": f1_fp,
            "figure": str(out_png),
        })
        print(f"  {scenario:<35} n={n:>5} pos={n_pos:>4}  bot-kept={n_pos_kept}/{n_pos}  "
              f"pred TP={f1_tp} FN={f1_fn} FP={f1_fp}")

    md = ["# Phase 8.5 — Interpretability Artefacts", "",
          "Per-scenario inspection of the final GT-edge model:",
          "  - what fraction of bots survive SAGPool coarsening?",
          "  - does the SAGPool score actually separate bots from benigns?", ""]
    md.append("| Scenario | n | n_pos | bot survival | benign survival | TP | FN | FP |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in summary:
        md.append(
            f"| {r['scenario']} | {r['n']} | {r['n_pos']} | "
            f"{r['bot_keep_rate']:.2f} | {r['benign_keep_rate']:.2f} | "
            f"{r['pred_tp']} | {r['pred_fn']} | {r['pred_fp']} |"
        )
    md.append("")
    md.append("See `figures/phase8_sagpool_<scenario>.png` for histograms.")
    (LOG_DIR / "phase8_interpretability.md").write_text(
        "\n".join(md), encoding="utf-8")
    (LOG_DIR / "phase8_interpretability.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {LOG_DIR / 'phase8_interpretability.md'}")


if __name__ == "__main__":
    main()
