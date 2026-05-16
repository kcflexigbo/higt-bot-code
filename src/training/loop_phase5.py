"""Phase 5 training loop. Almost identical to src/training/loop.py but the
model is called as `model(batch)` instead of
`model(batch.x, batch.edge_index, edge_attr=...)` — the hybrid model unpacks
its own inputs (flows, flow_mask, edge_index, edge_attr) from the batch.

All other behaviour (Adam + ReduceLROnPlateau, focal-loss option, early
stopping on val F1, grad clipping, W&B logging) is reused from Phase 4."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.training.loop import FocalLoss, TrainConfig, _class_weights_from_train


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_t, y_p, y_pr, scen = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        proba = torch.softmax(logits, dim=-1)[:, 1]
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


def train_one_model(
    model: nn.Module,
    train_graphs: list[Data],
    val_graphs: list[Data],
    *,
    cfg: TrainConfig | None = None,
    device: torch.device | str = "cpu",
    wandb_run=None,
    log_prefix: str = "",
) -> dict:
    if cfg is None:
        cfg = TrainConfig()
    device = torch.device(device)
    model = model.to(device)

    if cfg.class_weight is None:
        cfg.class_weight = _class_weights_from_train(train_graphs)
    print(f"{log_prefix}class weights (benign, bot) = {cfg.class_weight}")

    train_loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg.batch_size, shuffle=False)

    weight = torch.tensor(cfg.class_weight, dtype=torch.float32, device=device)
    if cfg.use_focal:
        alpha = torch.tensor(
            cfg.class_weight if cfg.focal_alpha is None
            else (1.0 - cfg.focal_alpha, cfg.focal_alpha),
            dtype=torch.float32,
        )
        loss_fn = FocalLoss(gamma=cfg.focal_gamma, alpha=alpha)
        print(f"{log_prefix}using focal loss γ={cfg.focal_gamma}  α={alpha.tolist()}")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weight)

    optim = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=5)

    best_f1 = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        dt = time.perf_counter() - t0

        yv_t, yv_p, _, _ = predict(model, val_loader, device)
        val_f1 = float(f1_score(yv_t, yv_p, zero_division=0))
        sched.step(val_f1)

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        lr_now = optim.param_groups[0]["lr"]
        print(f"{log_prefix}epoch {epoch:>3d}  train_loss {train_loss:.4f}  "
              f"val_F1 {val_f1:.4f}  best {best_f1:.4f}  lr {lr_now:.1e}  "
              f"{'*' if improved else ' '}  {dt:.1f}s", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_f1": val_f1, "lr": lr_now})

        if wandb_run is not None:
            wandb_run.log({
                f"{log_prefix.strip()}/train_loss": train_loss,
                f"{log_prefix.strip()}/val_f1": val_f1,
                f"{log_prefix.strip()}/lr": lr_now,
                "epoch": epoch,
            })

        if bad_epochs >= cfg.patience:
            print(f"{log_prefix}early stopping at epoch {epoch} "
                  f"(no val_F1 improvement for {cfg.patience} epochs)")
            break

    model.load_state_dict(best_state)
    return {"history": history, "best_val_f1": best_f1}
