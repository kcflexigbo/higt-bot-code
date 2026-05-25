"""Phase 5 training loop. Almost identical to src/training/loop.py but the
model is called as `model(batch)` instead of
`model(batch.x, batch.edge_index, edge_attr=...)` — the hybrid model unpacks
its own inputs (flows, flow_mask, edge_index, edge_attr) from the batch.

Supports mixed precision and gradient accumulation to stay within 16 GB RAM /
VRAM on laptop GPUs."""

from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.training.loop import FocalLoss, TrainConfig, _class_weights_from_train


def _amp_ctx(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, *, use_amp: bool = False):
    model.eval()
    y_t, y_p, y_pr, scen = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        with _amp_ctx(device, use_amp):
            logits = model(batch)
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


def train_one_model(
    model: nn.Module,
    train_graphs: list[Data] | torch.utils.data.Dataset,
    val_graphs: list[Data] | torch.utils.data.Dataset,
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
    use_amp = cfg.use_amp and device.type == "cuda"
    accum = max(int(cfg.grad_accum_steps), 1)

    if cfg.class_weight is None:
        if isinstance(train_graphs, list):
            cfg.class_weight = _class_weights_from_train(train_graphs)
        else:
            raise ValueError("class_weight must be set when using a Dataset")
    print(f"{log_prefix}class weights (benign, bot) = {cfg.class_weight}")
    if use_amp:
        print(f"{log_prefix}AMP enabled", flush=True)
    if accum > 1:
        print(f"{log_prefix}grad_accum_steps={accum}  "
              f"effective_batch={cfg.batch_size * accum}", flush=True)

    train_loader = DataLoader(
        train_graphs, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_graphs, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
    )

    weight = torch.tensor(cfg.class_weight, dtype=torch.float32, device=device)
    if cfg.use_focal:
        alpha = torch.tensor(
            cfg.class_weight if cfg.focal_alpha is None
            else (1.0 - cfg.focal_alpha, cfg.focal_alpha),
            dtype=torch.float32,
        )
        loss_fn = FocalLoss(gamma=cfg.focal_gamma, alpha=alpha)
        print(f"{log_prefix}using focal loss gamma={cfg.focal_gamma}  alpha={alpha.tolist()}")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weight)

    optim = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=True)
            with _amp_ctx(device, use_amp):
                logits = model(batch)
                loss = loss_fn(logits, batch.y) / accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            epoch_loss += float(loss.item()) * accum
            n_batches += 1
            if step % accum == 0 or step == len(train_loader):
                if use_amp:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                    scaler.step(optim)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                    optim.step()
                optim.zero_grad(set_to_none=True)
        train_loss = epoch_loss / max(n_batches, 1)
        dt = time.perf_counter() - t0

        yv_t, yv_p, _, _ = predict(model, val_loader, device, use_amp=use_amp)
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
