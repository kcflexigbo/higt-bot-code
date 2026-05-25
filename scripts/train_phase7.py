"""Phase 7 — train full HiGT-Bot (encoder → SAGPool → Graph Transformer → head).

Selects the GT variant via `--gt-variant {edge,global,none}`:
  edge   — PyG TransformerConv (edge-aware), sparse on coarsened graph.
  global — nn.MultiheadAttention over padded super-node sequence per graph.
  none   — bypass GT (= Phase 6.4 SAGPool baseline, kept for ablation).

Mirrors scripts/train_phase6_sparse.py for the data, loss, and loop. Uses the
same EmbeddingGraphDataset / focal loss / AMP / grad-accum machinery; only
the model class differs.

Usage
-----
uv run python scripts/train_phase7.py --gt-variant edge \
    --emb-dir data/flow_embeddings --results-suffix _edge
uv run python scripts/train_phase7.py --gt-variant global \
    --emb-dir data/flow_embeddings --results-suffix _global
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch  # before yaml/pandas on Windows
import yaml
import numpy as np
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from src.data.dataset import (
    SplitSpec, fit_edge_scaler, fit_feature_scaler, load_graphs, load_split_files,
)
from src.data.embedding_dataset import EmbeddingGraphDataset
from src.models.higt_bot import HiGTBot
from src.training.evaluate import evaluate
from src.training.loop import FocalLoss, _class_weights_from_train
from src.utils.seeding import pick_device, set_seed

CKPT_DIR = Path("experiments/phase7")
RESULTS = Path("data/inspection_logs/phase7_higt_bot.json")


def _amp_ctx(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


@torch.no_grad()
def predict(model, loader, device, *, use_amp: bool = False):
    model.eval()
    y_t, y_p, y_pr, scen = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        with _amp_ctx(device, use_amp):
            out = model(batch)
        logits = out["logits"]
        proba = torch.softmax(logits.float(), dim=-1)[:, 1]
        pred = logits.argmax(dim=-1)
        y_t.append(batch.y.cpu().numpy())
        y_p.append(pred.cpu().numpy())
        y_pr.append(proba.cpu().numpy())
        bi = batch.batch.cpu().numpy()
        per_graph_scenarios = (batch.scenario if isinstance(batch.scenario, list)
                                else [batch.scenario])
        scen.append(np.array(per_graph_scenarios, dtype=object)[bi])
    return (np.concatenate(y_t), np.concatenate(y_p),
            np.concatenate(y_pr), np.concatenate(scen))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/phase5.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--pool-ratio", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--gt-variant",
                    choices=["edge", "global", "hybrid", "none"], default="edge")
    ap.add_argument("--gt-layers", type=int, default=2)
    ap.add_argument("--gt-heads", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--results-suffix", type=str, default="")
    ap.add_argument("--emb-dir", type=Path, default=Path("data/flow_embeddings"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"device: {device}  amp={use_amp}  emb_dir={args.emb_dir}  "
          f"gt={args.gt_variant} layers={args.gt_layers} heads={args.gt_heads}",
          flush=True)

    spec = SplitSpec.load()
    tr_files = load_split_files("train", spec)
    va_files = load_split_files("val", spec)
    te_files = load_split_files("test", spec)
    print(f"train={len(tr_files)} val={len(va_files)} test={len(te_files)}", flush=True)

    print("fitting scalers + class weights ...", flush=True)
    tr_graphs = load_graphs(tr_files)
    e_mean, e_std = fit_edge_scaler(tr_graphs)
    n_mean, n_std = fit_feature_scaler(tr_graphs)
    class_weight = _class_weights_from_train(tr_graphs)
    raw_feat_dim = int(tr_graphs[0].x.size(1))
    edge_dim = int(tr_graphs[0].edge_attr.size(1))
    del tr_graphs
    gc.collect()

    ds_kw = dict(emb_root=args.emb_dir, edge_mean=e_mean, edge_std=e_std,
                  node_mean=n_mean, node_std=n_std)
    tr_ds = EmbeddingGraphDataset(tr_files, **ds_kw)
    va_ds = EmbeddingGraphDataset(va_files, **ds_kw)
    te_ds = EmbeddingGraphDataset(te_files, **ds_kw)

    probe = tr_ds[0]
    d_model = int(probe.node_emb.size(1))

    model = HiGTBot(
        d_model=d_model, raw_feat_dim=raw_feat_dim, edge_dim=edge_dim,
        hidden=args.hidden, pool_ratio=args.pool_ratio,
        gt_variant=args.gt_variant, gt_layers=args.gt_layers, gt_heads=args.gt_heads,
        num_classes=2, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HiGTBot params: {n_params:,}", flush=True)

    train_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    alpha = torch.tensor(class_weight, dtype=torch.float32)
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
    print(f"focal loss gamma=2.0 alpha={alpha.tolist()}", flush=True)

    optim = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
    t_total = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_b = 0
        t0 = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=True)
            with _amp_ctx(device, use_amp):
                out = model(batch)
                ce = loss_fn(out["logits"], batch.y)
                loss = ce / args.grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            ep_loss += float(ce.item())
            n_b += 1
            if step % args.grad_accum == 0 or step == len(train_loader):
                if use_amp:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                optim.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        yv_t, yv_p, _, _ = predict(model, val_loader, device, use_amp=use_amp)
        val_f1 = float(f1_score(yv_t, yv_p, zero_division=0))
        sched.step(val_f1)
        imp = val_f1 > best_f1
        if imp:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        lr_now = optim.param_groups[0]["lr"]
        print(f"[higt-{args.gt_variant}] epoch {epoch:>3d}  ce {ep_loss/max(n_b,1):.4f}  "
              f"val_F1 {val_f1:.4f}  best {best_f1:.4f}  lr {lr_now:.1e}  "
              f"{'*' if imp else ' '}  {dt:.1f}s", flush=True)
        if bad >= args.patience:
            print(f"[higt-{args.gt_variant}] early stop at epoch {epoch}")
            break

    fit_s = time.perf_counter() - t_total
    print(f"\ntotal fit {fit_s:.1f}s  best val F1 {best_f1:.4f}", flush=True)

    model.load_state_dict(best_state)
    yv_t, yv_p, yv_pr, scen_v = predict(model, val_loader, device, use_amp=use_amp)
    yt_t, yt_p, yt_pr, scen_t = predict(model, test_loader, device, use_amp=use_amp)
    val_res = evaluate(yv_t, yv_p, yv_pr, scenarios=scen_v)
    test_res = evaluate(yt_t, yt_p, yt_pr, scenarios=scen_t)
    print(); print(val_res.pretty(f"val (higt-{args.gt_variant})"))
    print(); print(test_res.pretty(f"test (higt-{args.gt_variant})"))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"higt_bot{args.results_suffix}.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                 "args": vars(args), "edge_scaler": (e_mean, e_std),
                 "node_scaler": (n_mean, n_std)}, ckpt)
    print(f"saved {ckpt}")

    results_path = RESULTS.with_name(f"phase7_higt_bot{args.results_suffix}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({
        "fit_seconds": fit_s, "best_val_f1": best_f1, "params": n_params,
        "val": val_res.as_row(), "test": test_res.as_row(),
        "test_per_scenario": test_res.per_scenario,
        "args": vars(args),
    }, indent=2, default=str))
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
