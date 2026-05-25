"""Self-supervised pretraining for the TemporalFlowEncoder via BERT-style
masked-flow reconstruction.

Motivation (Phase 6.3): TAM and GraphSHA both failed because iot23-35-1's
training distribution has too few positive examples for supervised
imbalance-handling to fire. SSL exploits the ~40k *unlabeled* benign flow
sequences to teach the encoder "what normal looks like" — bots then become
anomalies-from-baseline at downstream classification time.

Architecture:
  TemporalFlowEncoder (frozen layout) + learned [MASK] token + linear
  reconstruction head (d_model → flow_feat_dim). Pretrain objective is MSE
  between predicted and original flow features on the masked positions.

Usage pattern:
  ssl = MaskedFlowSSL(encoder, flow_feat_dim=13)
  for batch in pretrain_loader:
      flows, pad_mask = batch         # [B, L, F], [B, L]
      loss, _ = ssl(flows, pad_mask)
      loss.backward()
      opt.step()

After pretraining, `ssl.encoder.state_dict()` holds the warmed-up encoder
weights to load into a downstream supervised model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.temporal import TemporalFlowEncoder


class MaskedFlowSSL(nn.Module):
    """Wraps a TemporalFlowEncoder for masked-flow reconstruction.

    Masking strategy: per non-padded position, draw Bernoulli(mask_ratio); if
    True, replace the *projected* flow embedding with a learned mask token
    before the Transformer sees it. Reconstruct the original flow feature
    vector from the corresponding token's final hidden state.
    """

    def __init__(
        self,
        encoder: TemporalFlowEncoder,
        *,
        flow_feat_dim: int,
        mask_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder.d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.recon_head = nn.Linear(encoder.d_model, flow_feat_dim)

    def forward(
        self, flows: torch.Tensor, pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss and return (loss, mask).

        Args:
            flows: [B, L, F]
            pad_mask: [B, L]  True at padded positions

        Returns:
            loss: scalar tensor (MSE averaged over masked positions)
            mask: [B, L] boolean indicating which positions were masked
        """
        B, L, F = flows.shape
        device = flows.device

        # Project flows into d_model space (re-use the encoder's input proj).
        x = self.encoder.input_proj(flows)                              # [B, L, d]

        # Build per-position mask: only mask non-padded slots, then sample
        # Bernoulli(mask_ratio). Force at least one masked slot per row so
        # the loss is always defined.
        valid = ~pad_mask                                               # [B, L]
        bern = torch.rand(B, L, device=device) < self.mask_ratio
        mask = bern & valid                                             # [B, L]
        # Per-row safety: rows with zero masked positions get position 0 masked
        # if position 0 is valid, otherwise skip the row's loss contribution.
        no_mask = mask.sum(dim=1) == 0                                  # [B]
        if no_mask.any():
            rows = no_mask.nonzero(as_tuple=True)[0]
            for r in rows.tolist():
                first_valid = valid[r].nonzero(as_tuple=True)[0]
                if first_valid.numel() > 0:
                    mask[r, int(first_valid[0].item())] = True

        # Replace masked positions with the learned mask token.
        mt = self.mask_token.expand(B, L, -1)                           # [B, L, d]
        x = torch.where(mask.unsqueeze(-1), mt, x)

        # Prepend CLS, add positional encoding, run encoder.
        cls = self.encoder.cls_token.expand(B, -1, -1)                  # [B, 1, d]
        x = torch.cat([cls, x], dim=1)                                  # [B, L+1, d]
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_pad, pad_mask], dim=1)
        x = self.encoder.pos_enc(x)
        h = self.encoder.encoder(x, src_key_padding_mask=full_mask)     # [B, L+1, d]

        # Reconstruction: predict masked flow features.
        token_h = h[:, 1:]                                              # [B, L, d]
        recon = self.recon_head(token_h)                                # [B, L, F]
        per_pos = ((recon - flows) ** 2).mean(dim=-1)                   # [B, L]
        m = mask.float()
        denom = m.sum().clamp(min=1.0)
        loss = (per_pos * m).sum() / denom
        return loss, mask
