"""E-GraphSAGE baseline (Lo et al., NOMS 2022) for host-level bot detection.

Two node-classification variants (see src/models/egraphsage.py):
  --variant node       classify host v from its E-GraphSAGE embedding z_v
  --variant edge2node  train the paper's edge classifier on [z_u || z_v] with
                       edge label = node label of the edge's SOURCE host, then
                       score each host by max P(malicious) over its outgoing
                       edges (0 for hosts with no outgoing edge)

Follows run_gnn.py conventions: train-set feature scaling (edges; node input is
all-ones per the paper), class-weighted CE or focal loss, Adam + ReduceLROnPlateau,
grad clipping, early stopping on validation node-F1 (threshold 0.5).
Additionally reports metrics at a validation-tuned threshold (same grid as
run_rf.py --tune-threshold). Label version is chosen by HIGT_GRAPHS_DIR.

Usage:
    HIGT_GRAPHS_DIR=data/graphs_src2 uv run python scripts/baselines/run_egraphsage.py \
        --variant node --seed 42 --out data/inspection_logs/egraphsage/v2/node_seed42.json \
        --save-preds data/inspection_logs/egraphsage/v2/node_seed42_preds.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch  # noqa: I001  (import torch before numpy-heavy libs on Windows)
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge

from src.data.dataset import (
    GRAPHS_DIR,
    SplitSpec,
    apply_edge_scaler,
    fit_edge_scaler,
    load_split,
)
from src.models.egraphsage import EGraphSAGE, edge_labels_from_nodes, edge_to_node_scores
from src.training.evaluate import evaluate
from src.training.loop import FocalLoss, TrainConfig, predict, train_one_model
from src.utils.seeding import pick_device, set_seed


def tune_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    """Same grid/criterion as run_rf.py --tune-threshold."""
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [2 * np.sum((proba >= t) & (y == 1)) / max(np.sum(proba >= t) + np.sum(y == 1), 1)
           for t in grid]
    return float(grid[int(np.argmax(f1s))])


@torch.no_grad()
def predict_edge2node(model: nn.Module, loader: DataLoader, device: torch.device):
    """Node-level (y, pred@0.5, proba, scenario) from edge scores."""
    model.eval()
    y_t, y_pr, scen = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        e_proba = torch.softmax(logits, dim=-1)[:, 1]
        n_proba = edge_to_node_scores(e_proba, batch.edge_index, batch.num_nodes)
        y_t.append(batch.y.cpu().numpy())
        y_pr.append(n_proba.cpu().numpy())
        bi = batch.batch.cpu().numpy()
        per_graph = batch.scenario if isinstance(batch.scenario, list) else [batch.scenario]
        scen.append(np.array(per_graph, dtype=object)[bi])
    y_t = np.concatenate(y_t); y_pr = np.concatenate(y_pr)
    return y_t, (y_pr >= 0.5).astype(np.int64), y_pr, np.concatenate(scen)


def train_edge2node(model: nn.Module, tr, va, cfg: TrainConfig, device: torch.device,
                    log_prefix: str = "") -> dict:
    """Mirror of training.loop.train_one_model, but the loss is on edges
    (label = source-host label) and early stopping uses validation NODE F1."""
    model = model.to(device)
    ye = torch.cat([edge_labels_from_nodes(g.y, g.edge_index) for g in tr]).numpy()
    n_pos = max(int(ye.sum()), 1); n_neg = max(len(ye) - n_pos, 1)
    cw = (len(ye) / (2 * n_neg), len(ye) / (2 * n_pos))
    print(f"{log_prefix}train edges={len(ye):,}  malicious={int(ye.sum()):,}  "
          f"class weights (benign, bot) = {cw}")
    if cfg.use_focal:
        alpha = torch.tensor(cw if cfg.focal_alpha is None
                             else (1.0 - cfg.focal_alpha, cfg.focal_alpha), dtype=torch.float32)
        loss_fn = FocalLoss(gamma=cfg.focal_gamma, alpha=alpha)
        print(f"{log_prefix}using focal loss gamma={cfg.focal_gamma}  alpha={alpha.tolist()}")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))

    train_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=cfg.batch_size, shuffle=False)
    optim = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=5)
    best_f1, bad = -1.0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train(); tot = 0.0; nb = 0; t0 = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            ei, ea = batch.edge_index, batch.edge_attr
            if cfg.drop_edge_p > 0:
                ei, em = dropout_edge(ei, p=cfg.drop_edge_p, training=True)
                ea = ea[em]
            if ei.size(1) == 0:
                continue
            optim.zero_grad()
            logits = model(batch.x, ei, edge_attr=ea)
            loss = loss_fn(logits, edge_labels_from_nodes(batch.y, ei))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optim.step()
            tot += float(loss.item()); nb += 1
        yv, pv, _, _ = predict_edge2node(model, val_loader, device)
        val_f1 = float(f1_score(yv, pv, zero_division=0))
        sched.step(val_f1)
        improved = val_f1 > best_f1
        if improved:
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        lr_now = optim.param_groups[0]["lr"]
        print(f"{log_prefix}epoch {epoch:>3d}  train_loss {tot / max(nb, 1):.4f}  "
              f"val_nodeF1 {val_f1:.4f}  best {best_f1:.4f}  lr {lr_now:.1e}  "
              f"{'*' if improved else ' '}  {time.perf_counter() - t0:.1f}s", flush=True)
        history.append({"epoch": epoch, "train_loss": tot / max(nb, 1), "val_f1": val_f1, "lr": lr_now})
        if bad >= cfg.patience:
            print(f"{log_prefix}early stopping at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    return {"history": history, "best_val_f1": best_f1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["node", "edge2node"], required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--focal-gamma", type=float, default=None)
    ap.add_argument("--focal-alpha", type=float, default=None)
    ap.add_argument("--drop-edge-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-preds", type=Path, default=None,
                    help=".npz of test node predictions (y, pred@0.5, proba, scenario, pred_tuned); "
                         "validation preds go to <stem>_val.npz")
    ap.add_argument("--split-config", type=Path, default=Path("configs/split.yaml"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}   graphs dir: {GRAPHS_DIR} (HIGT_GRAPHS_DIR={os.environ.get('HIGT_GRAPHS_DIR')})")

    spec = SplitSpec.load(args.split_config)
    tr = load_split("train", spec); va = load_split("val", spec); te = load_split("test", spec)
    print(f"  train graphs={len(tr)}  val graphs={len(va)}  test graphs={len(te)}")
    edge_dim = int(tr[0].edge_attr.size(1))

    # Node input is all-ones (paper); only edge features are scaled.
    e_mean, e_std = fit_edge_scaler(tr)
    for gs in (tr, va, te):
        apply_edge_scaler(gs, e_mean, e_std)

    # Structural coverage: bots with no outgoing edge can never be detected by edge2node.
    def _no_out(gs):
        n_bot = n_bot_noout = 0
        for g in gs:
            has_out = torch.zeros(g.num_nodes, dtype=torch.bool)
            has_out[g.edge_index[0]] = True
            n_bot += int(g.y.sum()); n_bot_noout += int(((g.y == 1) & ~has_out).sum())
        return n_bot, n_bot_noout
    cov = {k: _no_out(v) for k, v in (("train", tr), ("val", va), ("test", te))}
    for k, (nb, nn_) in cov.items():
        print(f"  {k}: bot nodes={nb:,}  bot nodes without outgoing edge={nn_:,}")

    model = EGraphSAGE(edge_dim, hidden=args.hidden, num_layers=args.layers,
                       dropout=args.dropout, variant=args.variant)
    print(f"  model = E-GraphSAGE[{args.variant}]  params={sum(p.numel() for p in model.parameters()):,}")
    cfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                      max_epochs=args.epochs, patience=args.patience,
                      use_focal=(args.focal_gamma is not None), focal_gamma=args.focal_gamma or 2.0,
                      focal_alpha=args.focal_alpha, drop_edge_p=args.drop_edge_p)

    t0 = time.perf_counter()
    prefix = f"[egs-{args.variant}] "
    if args.variant == "node":
        info = train_one_model(model, tr, va, cfg=cfg, device=device, log_prefix=prefix)
    else:
        info = train_edge2node(model, tr, va, cfg, device, log_prefix=prefix)
    fit_s = time.perf_counter() - t0
    print(f"\ntotal fit time {fit_s:.1f}s, best val F1 {info['best_val_f1']:.4f}")

    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False)
    tl = DataLoader(te, batch_size=args.batch_size, shuffle=False)
    pf = predict if args.variant == "node" else predict_edge2node
    yv_t, yv_p, yv_pr, scen_v = pf(model, vl, device)
    yt_t, yt_p, yt_pr, scen_t = pf(model, tl, device)

    thr = tune_threshold(yv_t, yv_pr)
    yv_pt = (yv_pr >= thr).astype(np.int64); yt_pt = (yt_pr >= thr).astype(np.int64)
    val_res = evaluate(yv_t, yv_p, yv_pr, scenarios=scen_v)
    test_res = evaluate(yt_t, yt_p, yt_pr, scenarios=scen_t)
    val_tuned = evaluate(yv_t, yv_pt, yv_pr)
    test_tuned = evaluate(yt_t, yt_pt, yt_pr, scenarios=scen_t)
    print(); print(val_res.pretty(f"val (E-GraphSAGE {args.variant}, t=0.5)"))
    print(); print(test_res.pretty(f"test (E-GraphSAGE {args.variant}, t=0.5)"))
    print(f"\nval-tuned threshold {thr:.2f}: test F1 {test_tuned.f1:.4f}  "
          f"fn={test_tuned.confusion[1][0]} fp={test_tuned.confusion[0][1]}")

    out = args.out or Path(f"data/inspection_logs/egraphsage/egraphsage_{args.variant}_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": f"E-GraphSAGE-{args.variant}",
        "variant": args.variant,
        "graphs_dir": str(GRAPHS_DIR),
        "edge_label_rule": ("edge label = y[source host] (edge2node only)"
                            if args.variant == "edge2node" else None),
        "hidden": args.hidden, "num_layers": args.layers, "dropout": args.dropout,
        "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "patience": args.patience, "seed": args.seed, "device": str(device),
        "use_focal": cfg.use_focal, "focal_gamma": cfg.focal_gamma if cfg.use_focal else None,
        "focal_alpha": cfg.focal_alpha, "drop_edge_p": cfg.drop_edge_p,
        "fit_seconds": fit_s, "best_val_f1": info["best_val_f1"],
        "epochs_run": len(info["history"]),
        "bot_nodes_without_outgoing_edge": {k: {"bot": a, "no_out": b} for k, (a, b) in cov.items()},
        "val": val_res.as_row(),
        "test": test_res.as_row(),
        "test_per_scenario": test_res.per_scenario,
        "threshold_tuned": thr,
        "val_tuned": val_tuned.as_row(),
        "test_tuned": test_tuned.as_row(),
        "test_tuned_per_scenario": test_tuned.per_scenario,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    if args.save_preds is not None:
        args.save_preds.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_preds, y=yt_t, pred=yt_p, proba=yt_pr,
                 scenario=np.asarray(scen_t).astype(str), pred_tuned=yt_pt, threshold=thr)
        vpath = args.save_preds.with_name(args.save_preds.stem + "_val.npz")
        np.savez(vpath, y=yv_t, pred=yv_p, proba=yv_pr,
                 scenario=np.asarray(scen_v).astype(str), pred_tuned=yv_pt, threshold=thr)
        print(f"wrote {args.save_preds}\nwrote {vpath}")


if __name__ == "__main__":
    main()
