"""Shared training loop for Phase 4+ node-classification baselines.

Features per the plan:
- Adam(lr=1e-3, weight_decay=1e-5)
- ReduceLROnPlateau on val F1
- Gradient clipping (max_norm=1.0)
- Early stopping on val F1 (patience=20)
- W&B logging (optional, falls back to stdout)
- Optional focal loss (γ, α) — fixes per-scenario weak spots where
  class-weighted CE overcorrects under 70%-positive global imbalance
- Optional DropEdge during training (PyG `dropout_edge`) — cheap
  regularization, stabilizes training on dense windows

Models output per-node logits [N, 2]. Default loss is class-weighted CE;
set `cfg.use_focal=True` to switch to focal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    use_amp: bool = False
    class_weight: tuple[float, float] | None = None  # (w_benign, w_bot); None => uniform
    # Loss
    use_focal: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float | None = None   # None → use class_weight as α per class
    # DropEdge
    drop_edge_p: float = 0.0           # 0.0 = disabled


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017). FL = -α (1 - p_t)^γ log(p_t).

    `alpha` is a per-class tensor (length C); if None, weighting matches CE.
    This shifts gradient mass to hard examples and fixes the per-scenario
    "predict the majority class" collapse seen with class-weighted CE on
    rare-positive scenarios (e.g. ctu13-3, iot23-35-1).
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha   # tensor of shape [C] or None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        # Per-sample log p_t
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        # Focal modulation
        focal_term = (1.0 - p_t).pow(self.gamma)
        loss = -focal_term * logp_t
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[target]
            loss = alpha_t * loss
        return loss.mean()


def _class_weights_from_train(train_graphs: list[Data]) -> tuple[float, float]:
    """Inverse-frequency weights, normalized so the mean is 1."""
    y = torch.cat([g.y for g in train_graphs]).cpu().numpy()
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - n_pos, 1)
    # Inverse-frequency
    w_pos = len(y) / (2 * n_pos)
    w_neg = len(y) / (2 * n_neg)
    return float(w_neg), float(w_pos)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Run inference; return (y_true, y_pred, y_proba, scenarios)."""
    model.eval()
    y_t, y_p, y_pr, scen = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, edge_attr=getattr(batch, "edge_attr", None))
        proba = torch.softmax(logits, dim=-1)[:, 1]
        pred = logits.argmax(dim=-1)
        y_t.append(batch.y.cpu().numpy())
        y_p.append(pred.cpu().numpy())
        y_pr.append(proba.cpu().numpy())
        # Per-batch scenarios — PyG keeps per-graph metadata; reconstruct per-node
        # via batch.batch which maps node→graph index within the batch.
        bi = batch.batch.cpu().numpy()
        # batch.scenario is a list[str] of length num_graphs in the batch
        per_graph_scenarios = (batch.scenario if isinstance(batch.scenario, list)
                                else [batch.scenario])
        scen.append(np.array(per_graph_scenarios, dtype=object)[bi])
    return (
        np.concatenate(y_t),
        np.concatenate(y_p),
        np.concatenate(y_pr),
        np.concatenate(scen),
    )


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
    """Run the canonical training loop. Returns history dict with per-epoch stats
    and the best model weights (model is restored to best-val-F1 state in-place)."""
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
        alpha = torch.tensor(cfg.class_weight if cfg.focal_alpha is None
                              else (1.0 - cfg.focal_alpha, cfg.focal_alpha),
                              dtype=torch.float32)
        loss_fn = FocalLoss(gamma=cfg.focal_gamma, alpha=alpha)
        print(f"{log_prefix}using focal loss gamma={cfg.focal_gamma}  alpha={alpha.tolist()}")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weight)
    if cfg.drop_edge_p > 0:
        print(f"{log_prefix}DropEdge enabled, p={cfg.drop_edge_p}")
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
            ei = batch.edge_index
            ea = getattr(batch, "edge_attr", None)
            if cfg.drop_edge_p > 0:
                # PyG dropout_edge returns (edge_index, edge_mask). Apply same
                # mask to edge_attr so GINE still sees matching pairs.
                ei, em = dropout_edge(ei, p=cfg.drop_edge_p, training=True)
                if ea is not None:
                    ea = ea[em]
            logits = model(batch.x, ei, edge_attr=ea)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        dt = time.perf_counter() - t0

        # Val F1
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
        msg = (f"{log_prefix}epoch {epoch:>3d}  train_loss {train_loss:.4f}  "
               f"val_F1 {val_f1:.4f}  best {best_f1:.4f}  lr {lr_now:.1e}  "
               f"{'*' if improved else ' '}  {dt:.1f}s")
        print(msg, flush=True)
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

    # Restore best weights
    model.load_state_dict(best_state)
    return {"history": history, "best_val_f1": best_f1}
