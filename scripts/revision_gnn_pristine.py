"""Frozen graph models on the never-used captures (review item C1), inference only.

Models (checkpoints from queue_E_v3.sh, graphs_v3, HIGT_DIRECTED=1):
  directed T-GINE   experiments/v3/phase5/temporal_gine_skip{,_seed1}.pt
  directed HiGT-Bot experiments/v3/phase7/higt_bot_edge{,_seed1,_seed2}.pt (cached encoder embeddings)
Thresholds: tuned on the validation predictions saved with each model
(data/inspection_logs/v3_gnn/predictions/*_val.npz), as in bootstrap_significance.py.
Needs data/flow_seqs and data/flow_embeddings_v3 for the configs/split_pristine.yaml scenarios
(scripts/build_flow_sequences.py, scripts/cache_embeddings.py).

    HIGT_DIRECTED=1 uv run python -m scripts.revision_gnn_pristine
Output: data/inspection_logs/revision/gnn_pristine/<model>.npz (skipped if present)
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
os.environ.setdefault("HIGT_DIRECTED", "1")
import torch  # noqa: E402

import gc
from pathlib import Path

import numpy as np
from torch_geometric.loader import DataLoader

from scripts.reeval_mixed_pool_fp32 import _predict_fp32
from scripts.train_phase7 import predict as phase7_predict
from src.data.dataset import SplitSpec, fit_edge_scaler, fit_feature_scaler, load_graphs, load_split_files
from src.data.embedding_dataset import EmbeddingGraphDataset
from src.data.flow_seq_dataset import FlowSeqGraphDataset
from src.models.higt_bot import HiGTBot
from src.models.hybrid import TemporalGINE
from src.utils.seeding import pick_device, set_seed

OUT = Path("data/inspection_logs/revision/gnn_pristine")
PRED = Path("data/inspection_logs/v3_gnn/predictions")
GRID = np.linspace(0.01, 0.99, 99)
PRISTINE = SplitSpec.load(Path("configs/split_pristine.yaml"))


def f1(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def val_threshold(stem):
    v = np.load(PRED / f"{stem}_val.npz", allow_pickle=True)
    return float(GRID[int(np.argmax([f1(v["y"].astype(int), (v["proba"] >= g).astype(int)) for g in GRID]))])


def tgine(device, files):
    for ck, stem in (("temporal_gine_skip", "t_gine_skip"), ("temporal_gine_skip_seed1", None)):
        out = OUT / f"tgine_{ck}.npz"
        p = Path("experiments/v3/phase5") / f"{ck}.pt"
        if out.exists() or not p.exists():
            continue
        blob = torch.load(p, map_location="cpu", weights_only=False); cfg = blob["cfg"]
        e_mean, e_std = blob["edge_scaler"]; n_mean, n_std = blob["node_scaler"]
        kw = dict(edge_mean=e_mean.cpu(), edge_std=e_std.cpu(), node_mean=n_mean.cpu(), node_std=n_std.cpu())
        model = TemporalGINE(flow_feat_dim=cfg["encoder"]["flow_feat_dim"],
                             edge_dim=int(torch.load(files[0], weights_only=False).edge_attr.size(1)),
                             d_model=cfg["encoder"]["d_model"], nhead=cfg["encoder"]["nhead"],
                             num_layers=cfg["encoder"]["num_layers"], max_flows=cfg["encoder"]["max_flows"],
                             encoder_dropout=0.0, gin_hidden=cfg["gin"]["hidden"], gin_layers=cfg["gin"]["num_layers"],
                             dropout=0.0, out_dim=2, raw_feat_dim=blob.get("raw_feat_dim")).to(device)
        model.load_state_dict(blob["state_dict"])
        bs = int(cfg["train"]["batch_size"])
        y, _, pr, sc = _predict_fp32(model, DataLoader(FlowSeqGraphDataset(files, **kw), batch_size=bs), device)
        thr = val_threshold(stem) if stem else float("nan")
        np.savez(out, y=y, proba=pr, scenario=sc.astype(str), threshold=thr)
        print(f"T-GINE {ck}: F1 {f1(y.astype(int), (pr >= thr).astype(int)) if thr == thr else float('nan'):.4f}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()


def higt(device, files):
    spec = SplitSpec.load()
    tr = load_graphs(load_split_files("train", spec))
    e_mean, e_std = fit_edge_scaler(tr); n_mean, n_std = fit_feature_scaler(tr)
    raw_feat_dim = int(tr[0].x.size(1)); edge_dim = int(tr[0].edge_attr.size(1))
    del tr; gc.collect()
    ds = EmbeddingGraphDataset(files, emb_root=Path("data/flow_embeddings_v3"), edge_mean=e_mean, edge_std=e_std,
                               node_mean=n_mean, node_std=n_std)
    for ck, stem in (("higt_bot_edge", "higt_bot_edge"), ("higt_bot_edge_seed1", "higt_bot_edge_seed1"),
                     ("higt_bot_edge_seed2", "higt_bot_edge_seed2")):
        out = OUT / f"{ck}.npz"
        if out.exists():
            continue
        blob = torch.load(Path("experiments/v3/phase7") / f"{ck}.pt", map_location=device, weights_only=False)
        a = blob.get("args", {})
        model = HiGTBot(d_model=64, raw_feat_dim=raw_feat_dim, edge_dim=edge_dim, hidden=int(a.get("hidden", 128)),
                        pool_ratio=float(a.get("pool_ratio", 0.5)), gt_variant=str(a.get("gt_variant", "edge")),
                        gt_layers=int(a.get("gt_layers", 2)), gt_heads=int(a.get("gt_heads", 4)), dropout=0.0).to(device)
        model.load_state_dict(blob["state_dict"]); model.eval()
        with torch.no_grad():
            y, _, pr, ss = phase7_predict(model, DataLoader(ds, batch_size=4, shuffle=False), device)
        thr = val_threshold(stem)
        np.savez(out, y=y, proba=pr, scenario=ss.astype(str), threshold=thr)
        print(f"HiGT-Bot {ck}: F1 {f1(y.astype(int), (pr >= thr).astype(int)):.4f}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    device = pick_device()
    files = load_split_files("test", PRISTINE)
    tgine(device, files)
    higt(device, files)


if __name__ == "__main__":
    main()
