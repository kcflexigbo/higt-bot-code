"""Topology-Aware Margin (TAM) loss for class-imbalanced node classification.

Reference: Song, Park, Yi, "TAM: Topology-Aware Margin Loss for
Class-Imbalanced Node Classification" (ICML 2022).

Core idea: a node's predicted-class logit gets a *negative* margin proportional
to how over-represented that class is in the node's local neighborhood
relative to the global prior. Effect: the model can't shortcut by predicting
"whatever my neighbors look like" — minority-class nodes whose neighbors are
mostly majority get a free push toward their true class.

Two margin terms (both implemented):
  - ACM (Anomaly Class Margin): local pos rate minus global pos rate;
    penalizes the class that is locally over-represented.
  - CCM (Class-Conditional Margin, optional): per-class connectivity prior;
    scales the margin so the rare class gets a bigger push.

Plug-in style: pass through a base loss (focal or CE) and apply margin to the
logits first.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter as _pyg_scatter


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: int | None = None) -> torch.Tensor:
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


def topology_margin(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    *,
    pi_global: torch.Tensor,
    alpha: float = 1.5,
    use_ccm: bool = True,
    ccm_beta: float = 0.5,
) -> torch.Tensor:
    """Compute TAM-adjusted logits.

    Args:
        logits: [N, C] raw model logits.
        edge_index: [2, E] graph edges. Direction (src, dst) — neighbors of
            node v are nodes u such that (u, v) ∈ edge_index.
        y: [N] long, ground-truth labels in [0, C).
        pi_global: [C] global class prior (e.g. training-set frequencies).
        alpha: ACM strength; 0 disables.
        use_ccm: if True, scale margin per-class by inverse global frequency,
            putting more push on the rare class.
        ccm_beta: CCM strength when use_ccm.

    Returns:
        [N, C] logits with TAM margin added.
    """
    N, C = logits.shape
    device = logits.device
    src, dst = edge_index
    ones = torch.ones(src.size(0), device=device, dtype=torch.float)
    deg = scatter_add(ones, dst, dim=0, dim_size=N).clamp(min=1.0)  # [N]

    # Local per-class neighbor frequency: [N, C]
    pi_local = torch.zeros(N, C, device=device)
    for c in range(C):
        is_c = (y[src] == c).float()
        pi_local[:, c] = scatter_add(is_c, dst, dim=0, dim_size=N) / deg

    pi_g = pi_global.to(device).view(1, C)
    margin = -alpha * (pi_local - pi_g)               # [N, C]; penalizes locally over-represented class
    if use_ccm:
        inv_freq = (1.0 / pi_g.clamp(min=1e-6))
        inv_freq = inv_freq / inv_freq.mean()         # normalize to mean 1
        margin = margin * (1.0 + ccm_beta * (inv_freq - 1.0))
    return logits + margin


class TopologyAwareFocalLoss(nn.Module):
    """Focal loss with TAM logit adjustment.

    Wraps the existing focal-loss formulation; TAM is applied before the
    softmax so it shifts the cross-entropy term directly.
    """
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha_focal: torch.Tensor | None = None,
        pi_global: torch.Tensor,
        tam_alpha: float = 1.5,
        use_ccm: bool = True,
        ccm_beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha_focal = alpha_focal           # per-class focal weight, [C]
        self.register_buffer("pi_global", pi_global.float())
        self.tam_alpha = tam_alpha
        self.use_ccm = use_ccm
        self.ccm_beta = ccm_beta

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        adj_logits = topology_margin(
            logits, edge_index, target,
            pi_global=self.pi_global, alpha=self.tam_alpha,
            use_ccm=self.use_ccm, ccm_beta=self.ccm_beta,
        )
        logp = torch.log_softmax(adj_logits, dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        focal = (1.0 - p_t).pow(self.gamma)
        loss = -focal * logp_t
        if self.alpha_focal is not None:
            a = self.alpha_focal.to(logits.device)[target]
            loss = a * loss
        return loss.mean()
