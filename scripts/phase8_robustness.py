"""Phase 8.4 — robustness: random edge drop at test time on the final model.

For each perturbation rate ρ ∈ {0, 0.05, 0.10, 0.20, 0.30}, drop ρ fraction
of edges (uniformly at random) at test time and re-evaluate. No retraining.

Robust models degrade gracefully; brittle ones cliff. Plot F1 vs ρ.

Output: data/inspection_logs/phase8_robustness.md + .json
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

from src.data.dataset import (
    SplitSpec, fit_edge_scaler, fit_feature_scaler, load_graphs, load_split_files,
)
from src.data.embedding_dataset import EmbeddingGraphDataset
from src.models.higt_bot import HiGTBot
from src.utils.seeding import pick_device, set_seed

LOG_DIR = Path("data/inspection_logs")
CKPT = Path("experiments/phase7/higt_bot_edge.pt")
PERTURBATION_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]


def drop_edges(batch, rate: float, generator: torch.Generator | None = None):
    """Drop `rate` fraction of edges from batch (in-place clone).

    Returns a new Data-like batch with edge_index/edge_attr filtered.
    """
    if rate <= 0:
        return batch
    ei, ea = batch.edge_index, batch.edge_attr
    E = ei.size(1)
    if E == 0:
        return batch
    keep_mask = torch.rand(E, generator=generator, device=ei.device) >= rate
    batch.edge_index = ei[:, keep_mask]
    batch.edge_attr = ea[keep_mask]
    return batch


@torch.no_grad()
def evaluate_at_rate(model, loader, device, rate: float, seed: int = 42) -> dict:
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    y_t, y_p = [], []
    for batch in loader:
        batch = batch.to(device)
        batch = drop_edges(batch, rate, generator=gen)
        out = model(batch)
        pred = out["logits"].argmax(dim=-1).cpu().numpy()
        y_t.append(batch.y.cpu().numpy())
        y_p.append(pred)
    y_t = np.concatenate(y_t)
    y_p = np.concatenate(y_p)
    return {
        "rate": rate,
        "f1": float(f1_score(y_t, y_p, zero_division=0)),
        "n": int(len(y_t)),
        "n_pos": int(y_t.sum()),
        "tp": int(((y_t == 1) & (y_p == 1)).sum()),
        "fp": int(((y_t == 0) & (y_p == 1)).sum()),
        "fn": int(((y_t == 1) & (y_p == 0)).sum()),
        "tn": int(((y_t == 0) & (y_p == 0)).sum()),
    }


def main() -> None:
    set_seed(42)
    device = pick_device()
    print(f"device: {device}")

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

    te_ds = EmbeddingGraphDataset(te_files,
                                    emb_root=Path("data/flow_embeddings"),
                                    edge_mean=e_mean, edge_std=e_std,
                                    node_mean=n_mean, node_std=n_std)
    loader = DataLoader(te_ds, batch_size=4, shuffle=False, num_workers=0)

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

    rows = []
    for rate in PERTURBATION_RATES:
        r = evaluate_at_rate(model, loader, device, rate=rate)
        print(f"  rate={rate:.2f}  F1={r['f1']:.4f}  tp={r['tp']} fp={r['fp']} "
              f"fn={r['fn']}", flush=True)
        rows.append(r)

    f1_base = rows[0]["f1"]
    md = ["# Phase 8.4 — Test-Time Edge-Drop Robustness", "",
          "Final model: Phase 7 GT-edge.  Test set.",
          "Edges dropped uniformly at random at inference time (no retraining).",
          ""]
    md.append("| Edge drop rate | Test F1 | Δ vs baseline | TP | FP | FN |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        delta = r["f1"] - f1_base
        md.append(
            f"| {int(r['rate']*100)}% | {r['f1']:.4f} | "
            f"{delta:+.4f} | {r['tp']} | {r['fp']} | {r['fn']} |"
        )
    md.append("")
    md.append(f"**Baseline (0% drop) F1**: {f1_base:.4f}")
    (LOG_DIR / "phase8_robustness.json").write_text(json.dumps(rows, indent=2),
                                                       encoding="utf-8")
    (LOG_DIR / "phase8_robustness.md").write_text("\n".join(md),
                                                     encoding="utf-8")
    print(f"\nwrote {LOG_DIR / 'phase8_robustness.md'}")


if __name__ == "__main__":
    main()
