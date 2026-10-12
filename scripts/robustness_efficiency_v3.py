"""Edge-drop robustness and inference cost on the corrected data (graphs_v3).

Same perturbation protocol as scripts/phase8_robustness.py (drop a fraction of
edges uniformly at random at test time, no retraining), applied to
  - HiGT-Bot v2 (XGBoost on x + grp, seed 42, threshold tuned on validation):
    grp is recomputed from the perturbed graph; x comes from flows and is unchanged
  - directed HiGT-Bot (experiments/v3/phase7/higt_bot_edge.pt, HIGT_DIRECTED=1),
    argmax decision as in phase8_robustness.py
Cost: median / p95 per-window latency over the test windows (v2: grp block + tree
prediction on CPU; HiGT-Bot: forward pass on the GPU with cached embeddings, batch 1).

    HIGT_DIRECTED=1 uv run python scripts/robustness_efficiency_v3.py
Output: data/inspection_logs/v3/robustness_efficiency_v3.{md,json}
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
os.environ.setdefault("HIGT_DIRECTED", "1")
import torch  # noqa: E402  (import before pandas on Windows)

import gc
import json
import time
from pathlib import Path

import numpy as np
from torch_geometric.loader import DataLoader
from xgboost import XGBClassifier

from scripts.phase8_robustness import PERTURBATION_RATES, drop_edges, evaluate_at_rate
from src.data.dataset import SplitSpec, fit_edge_scaler, fit_feature_scaler, load_graphs, load_split_files
from src.data.embedding_dataset import EmbeddingGraphDataset
from src.features.host_blocks import fit_group_scaler, grp_block
from src.models.higt_bot import HiGTBot
from src.utils.seeding import pick_device, set_seed

OUT = Path("data/inspection_logs/v3")
CACHE = Path("data/inspection_logs/v3_emb3")
CKPT = Path("experiments/v3/phase7/higt_bot_edge.pt")
EMB = Path("data/flow_embeddings_v3")
GRID = np.linspace(0.01, 0.99, 99)


def conf(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn}


def v2_part(spec):
    tr = np.load(CACHE / "cache_train.npz"); va = np.load(CACHE / "cache_val.npz")
    Xtr = np.hstack([tr["x"], tr["grp"]]); ytr = tr["y"]
    clf = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=(len(ytr) - ytr.sum()) / max(ytr.sum(), 1), tree_method="hist",
                        random_state=42, n_jobs=-1, eval_metric="aucpr").fit(Xtr, ytr)
    pva = clf.predict_proba(np.hstack([va["x"], va["grp"]]))[:, 1]
    thr = float(GRID[int(np.argmax([conf(va["y"], (pva >= g).astype(int))["f1"] for g in GRID]))])
    qt = fit_group_scaler(tr["x"])
    graphs = [torch.load(fp, weights_only=False) for fp in load_split_files("test", spec)]
    y = np.concatenate([g.y.numpy() for g in graphs]).astype(int)
    rows = []
    for rate in PERTURBATION_RATES:
        rng = np.random.default_rng(42)
        feats = []
        for g in graphs:
            ei = g.edge_index.numpy()
            if rate > 0 and ei.shape[1]:
                ei = ei[:, rng.random(ei.shape[1]) >= rate]
            x = g.x.numpy().astype(np.float32)
            feats.append(np.hstack([x, grp_block(x, ei, qt)]))
        p = clf.predict_proba(np.vstack(feats))[:, 1]
        rows.append({"rate": rate, **conf(y, (p >= thr).astype(int))})
        print(f"[v2] rate {rate:.2f} F1 {rows[-1]['f1']:.4f} fn {rows[-1]['fn']} fp {rows[-1]['fp']}", flush=True)
    # latency per window: grp block + tree prediction (x is shared preprocessing for every model)
    clf.set_params(n_jobs=1)
    lat = []
    for g in graphs:
        t0 = time.perf_counter()
        x = g.x.numpy().astype(np.float32)
        clf.predict_proba(np.hstack([x, grp_block(x, g.edge_index.numpy(), qt)]))
        lat.append((time.perf_counter() - t0) * 1e3)
    n_nodes = int(clf.get_booster().trees_to_dataframe()["Tree"].nunique())
    return rows, {"median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
                  "trees": n_nodes, "threshold": thr, "device": "CPU, 1 thread"}


def higt_part(spec):
    set_seed(42)
    device = pick_device()
    blob = torch.load(CKPT, map_location=device, weights_only=False)
    args = blob.get("args", {})
    tr_graphs = load_graphs(load_split_files("train", spec))
    e_mean, e_std = fit_edge_scaler(tr_graphs); n_mean, n_std = fit_feature_scaler(tr_graphs)
    raw_feat_dim = int(tr_graphs[0].x.size(1)); edge_dim = int(tr_graphs[0].edge_attr.size(1))
    del tr_graphs; gc.collect()
    ds = EmbeddingGraphDataset(load_split_files("test", spec), emb_root=EMB, edge_mean=e_mean, edge_std=e_std,
                               node_mean=n_mean, node_std=n_std)
    model = HiGTBot(d_model=64, raw_feat_dim=raw_feat_dim, edge_dim=edge_dim, hidden=int(args.get("hidden", 128)),
                    pool_ratio=float(args.get("pool_ratio", 0.5)), gt_variant=str(args.get("gt_variant", "edge")),
                    gt_layers=int(args.get("gt_layers", 2)), gt_heads=int(args.get("gt_heads", 4)), dropout=0.0).to(device)
    model.load_state_dict(blob["state_dict"])
    rows = [evaluate_at_rate(model, DataLoader(ds, batch_size=4, shuffle=False), device, rate=r) for r in PERTURBATION_RATES]
    for r in rows:
        print(f"[HiGT dir] rate {r['rate']:.2f} F1 {r['f1']:.4f} fn {r['fn']} fp {r['fp']}", flush=True)
    model.eval(); lat = []
    with torch.no_grad():
        for i, b in enumerate(DataLoader(ds, batch_size=1, shuffle=False)):
            b = b.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter(); model(b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 20:  # warm-up excluded
                lat.append((time.perf_counter() - t0) * 1e3)
            if i >= 1020:
                break
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return rows, {"median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)), "params": n_params,
                  "device": str(device)}


def main():
    spec = SplitSpec.load()
    v2_rows, v2_cost = v2_part(spec)
    h_rows, h_cost = higt_part(spec)
    (OUT / "robustness_efficiency_v3.json").write_text(json.dumps(
        {"v2": v2_rows, "v2_cost": v2_cost, "higt_directed": h_rows, "higt_cost": h_cost}, indent=2))
    md = ["# Edge-drop robustness and inference cost (corrected labels, v3 features)", "",
          "Edges dropped uniformly at random from the test graphs at inference (seed 42), no retraining. HiGT-Bot v2: "
          "grp recomputed from the perturbed graph, x unchanged (computed from flows). Directed HiGT-Bot: argmax "
          "decision as in phase8_robustness.py.", "",
          "| Edge drop | v2 F1 | v2 FN / FP | directed HiGT-Bot F1 | HiGT-Bot FN / FP |", "|---|---|---|---|---|"]
    for a, b in zip(v2_rows, h_rows):
        md.append(f"| {int(a['rate'] * 100)}% | {a['f1']:.4f} | {a['fn']} / {a['fp']} | {b['f1']:.4f} | {b['fn']} / {b['fp']} |")
    md += ["", "| Model | Size | Median ms / window | p95 ms / window | Device |", "|---|---|---|---|---|",
           f"| HiGT-Bot v2 (grp block + trees) | {v2_cost['trees']} trees | {v2_cost['median_ms']:.2f} | {v2_cost['p95_ms']:.2f} | {v2_cost['device']} |",
           f"| HiGT-Bot, directed (cached encoder) | {h_cost['params']:,} params | {h_cost['median_ms']:.2f} | {h_cost['p95_ms']:.2f} | {h_cost['device']} |",
           "", "Both exclude the shared preprocessing (flow parsing, host features x; the encoder for HiGT-Bot)."]
    (OUT / "robustness_efficiency_v3.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
