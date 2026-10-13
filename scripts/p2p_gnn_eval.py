"""Directed T-GINE and directed HiGT-Bot (v3 checkpoints, trained on the thesis split) on
P2P target scenarios they never saw. Inference only (no training); FP32.

Thresholds: re-tuned on the main-split validation predictions saved by
scripts/reeval_mixed_pool_fp32.py (v3_gnn/predictions/*_val.npz), as in
scripts/bootstrap_significance.py. Embeddings for the targets must exist in
data/flow_embeddings_v3 (scripts/cache_embeddings.py --split-config configs/split_p2p_transfer.yaml).
Idempotent: a target/model pair whose prediction file exists is skipped.

    HIGT_DIRECTED=1 uv run python -m scripts.p2p_gnn_eval --targets ctu13-12 isot-2010 [--device cpu]
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
os.environ.setdefault("HIGT_DIRECTED", "1")
import torch  # noqa: E402

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from torch_geometric.loader import DataLoader

from scripts.higt_v2_trees import metrics, tune
from scripts.reeval_mixed_pool_fp32 import _predict_fp32
from src.data.dataset import SplitSpec, fit_edge_scaler, fit_feature_scaler, load_graphs, load_split_files
from src.data.embedding_dataset import EmbeddingGraphDataset
from src.data.flow_seq_dataset import FlowSeqGraphDataset
from src.models.higt_bot import HiGTBot
from src.models.hybrid import TemporalGINE

OUT = Path("data/inspection_logs/p2p")
G3 = Path("data/inspection_logs/v3_gnn/predictions")
EMB = Path("data/flow_embeddings_v3")
TGINE_CKPT = Path("experiments/v3/phase5/temporal_gine_skip.pt")
HIGT_CKPT = Path("experiments/v3/phase7/higt_bot_edge.pt")


def val_threshold(name):
    v = np.load(G3 / f"{name}_val.npz", allow_pickle=True)
    return tune(v["y"].astype(int), v["proba"].astype(float))


def tgine(files, device):
    blob = torch.load(TGINE_CKPT, map_location="cpu", weights_only=False); cfg = blob["cfg"]
    e_mean, e_std = blob["edge_scaler"]; n_mean, n_std = blob["node_scaler"]
    kw = dict(edge_mean=e_mean.cpu(), edge_std=e_std.cpu(), node_mean=n_mean.cpu(), node_std=n_std.cpu())
    model = TemporalGINE(flow_feat_dim=cfg["encoder"]["flow_feat_dim"],
                         edge_dim=int(torch.load(files[0], weights_only=False).edge_attr.size(1)),
                         d_model=cfg["encoder"]["d_model"], nhead=cfg["encoder"]["nhead"],
                         num_layers=cfg["encoder"]["num_layers"], max_flows=cfg["encoder"]["max_flows"],
                         encoder_dropout=0.0, gin_hidden=cfg["gin"]["hidden"], gin_layers=cfg["gin"]["num_layers"],
                         dropout=0.0, out_dim=2, raw_feat_dim=blob.get("raw_feat_dim")).to(device)
    model.load_state_dict(blob["state_dict"])
    with torch.no_grad():
        y, _, p, _ = _predict_fp32(model, DataLoader(FlowSeqGraphDataset(files, **kw), batch_size=int(cfg["train"]["batch_size"])), device)
    return y.astype(int), p, val_threshold("t_gine_skip")


def higt(files, device, spec):
    blob = torch.load(HIGT_CKPT, map_location=device, weights_only=False); args = blob.get("args", {})
    tr = load_graphs(load_split_files("train", spec))
    e_mean, e_std = fit_edge_scaler(tr); n_mean, n_std = fit_feature_scaler(tr)
    raw, ed = int(tr[0].x.size(1)), int(tr[0].edge_attr.size(1))
    del tr; gc.collect()
    ds = EmbeddingGraphDataset(files, emb_root=EMB, edge_mean=e_mean, edge_std=e_std, node_mean=n_mean, node_std=n_std)
    model = HiGTBot(d_model=64, raw_feat_dim=raw, edge_dim=ed, hidden=int(args.get("hidden", 128)),
                    pool_ratio=float(args.get("pool_ratio", 0.5)), gt_variant=str(args.get("gt_variant", "edge")),
                    gt_layers=int(args.get("gt_layers", 2)), gt_heads=int(args.get("gt_heads", 4)), dropout=0.0).to(device)
    model.load_state_dict(blob["state_dict"])
    with torch.no_grad():
        y, _, p, _ = _predict_fp32(model, DataLoader(ds, batch_size=4), device)
    return y.astype(int), p, val_threshold("higt_bot_edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--graphs", type=Path, default=Path(os.environ["HIGT_GRAPHS_DIR"]))
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    spec = SplitSpec.load()
    path = OUT / "gnn_ckpt_results.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for sc in a.targets:
        files = sorted((a.graphs / sc).glob("window_*.pt"))
        win = np.concatenate([np.full(int(torch.load(f, weights_only=False).num_nodes), i) for i, f in enumerate(files)])
        for name, fn in (("Directed T-GINE", lambda: tgine(files, device)), ("HiGT-Bot directed", lambda: higt(files, device, spec))):
            pf = OUT / f"pred_transfer_{sc}_{name.replace(' ', '_')}.npz"
            if pf.exists():
                d = np.load(pf); y, p, t = d["y"], d["proba"], float(d["threshold"])
            else:
                y, p, t = fn()
                np.savez(pf, y=y, proba=p, pred=(p >= t).astype(int), threshold=t, win=win)
            r = metrics(y, (p >= t).astype(int), p) | {"threshold": t}
            res.setdefault(sc, {})[name] = r
            print(f"[transfer {sc}] {name:20s} F1 {r['f1']:.3f} PR {r['pr_auc']:.3f} FN {r['fn']} FP {r['fp']}", flush=True)
            path.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
