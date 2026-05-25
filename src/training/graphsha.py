"""GraphSHA-lite: intra-graph manifold mixup on minority-class node features.

Reference: Li, Bian, Mou et al. "GraphSHA: Synthesizing Harder Samples for
Class-Imbalanced Node Classification" (KDD 2023).

This is a simplified variant that captures the core idea (intra-class
interpolation of minority nodes) without the graph-rewiring + edge-predictor
machinery. We mix the minority class's *embedding-space* representation, not
the raw graph topology — so the model continues to see the original graph and
the minority node count stays constant, but each minority node's features get
randomly interpolated with another minority node from the same graph.

Why this works for our setup:
- It operates per-graph, so high-bot scenarios (medbiot-bashlite-spread, ~97%
  positives) and ultra-low-bot scenarios (iot23-35-1, 0.3% positives) both
  benefit without a single global prior controlling the augmentation strength.
- Both endpoints have y=1, so the resulting label is still 1 — no soft-label
  bookkeeping in the loss.
- Applied only at training time; val/test are unmodified.
- It augments BOTH the cached encoder embedding `node_emb` and the raw scaled
  feature `x` with the same λ, keeping the two channels consistent.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def minority_manifold_mixup(
    node_emb: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    pos_class: int = 1,
    beta_alpha: float = 1.0,
    mix_prob: float = 1.0,
    max_local_pos_rate: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (node_emb', x') with minority embeddings interpolated within graph.

    For each graph in the batch:
      - Find all positive-class nodes belonging to that graph.
      - If <2 such nodes, leave the graph unchanged.
      - Otherwise, for each positive node v, pick a *different* positive node u
        from the same graph (uniformly), draw λ ~ Beta(β, β), and replace
        node_emb[v] ← λ·node_emb[v] + (1-λ)·node_emb[u] (same for x).

    Args:
        node_emb: [N, d_model] cached encoder output.
        x:        [N, raw_feat_dim] scaled raw node features.
        y:        [N] long labels in {0, 1}.
        batch_idx:[N] long batch-membership indices.
        pos_class: which class to mix (default 1 — the bot/minority class).
        beta_alpha: Beta distribution shape parameter. β=1 is uniform on [0,1];
            larger β concentrates λ near 0.5 (more mixing); smaller β pushes
            λ to the endpoints (less mixing).
        mix_prob: per-node Bernoulli probability of applying mixup; lower
            values keep some original minority nodes untouched.
        generator: optional torch.Generator for reproducible sampling.

    Returns:
        (node_emb', x') — new tensors (originals not modified).
    """
    device = node_emb.device
    out_emb = node_emb.clone()
    out_x = x.clone()
    pos = (y == pos_class).nonzero(as_tuple=True)[0]
    if pos.numel() < 2:
        return out_emb, out_x

    pos_batch = batch_idx[pos]
    for g in pos_batch.unique():
        g_mask = pos_batch == g
        g_pos = pos[g_mask]
        n = g_pos.numel()
        if n < 2:
            continue
        # Skip graphs where pos_class is already locally majority — augmenting
        # the bulk of the graph's nodes destroys signal for high-bot scenarios
        # like medbiot-bashlite_mal_spread_all (~97% positives).
        graph_size = int((batch_idx == g).sum().item())
        if n / graph_size > max_local_pos_rate:
            continue
        # Permutation that maps each pos to a *different* pos in the same graph.
        # Random derangement is fiddly; a single random shift is sufficient
        # in practice (each node pairs with a deterministic shifted neighbor,
        # which is non-self for n >= 2).
        shift = int(torch.randint(1, n, (1,), generator=generator, device=device).item())
        perm_idx = (torch.arange(n, device=device) + shift) % n
        partner = g_pos[perm_idx]

        lam = torch.distributions.Beta(beta_alpha, beta_alpha).sample((n,)).to(device)
        if mix_prob < 1.0:
            apply = torch.rand(n, generator=generator, device=device) < mix_prob
            lam = torch.where(apply, lam, torch.ones_like(lam))
        lam_e = lam.unsqueeze(-1)

        out_emb[g_pos] = lam_e * node_emb[g_pos] + (1.0 - lam_e) * node_emb[partner]
        out_x[g_pos] = lam_e * x[g_pos] + (1.0 - lam_e) * x[partner]

    return out_emb, out_x
