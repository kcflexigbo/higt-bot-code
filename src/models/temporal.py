"""Phase 5 — per-node temporal Transformer over flow sequences.

`TemporalFlowEncoder` consumes [B, L, F] flow features plus a [B, L] padding
mask (True = padded) and returns [B, d_model] CLS-token embeddings, one per
node. Batches are constructed per (scenario, window) graph by stacking all
nodes' sequences along the batch dim.

Implementation choices (matching the plan, §"Phase 5 — Temporal Transformer"):
- batch_first=True throughout.
- Learned CLS token; only its final state is returned.
- Sinusoidal positional encoding (not learned) so the encoder generalizes
  to nodes with very different sequence lengths in eval.
- GELU activation, pre-norm via PyTorch's `norm_first=True`.
- An all-padded sequence is handled by zeroing the row of the padding mask
  for the CLS position, so attention always has at least one valid key.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                              * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [1, max_len, d_model] for broadcast addition.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TemporalFlowEncoder(nn.Module):
    """CLS-token Transformer over per-node flow sequences.

    Input:
        flows:    [B, L, F]   flow feature vectors
        pad_mask: [B, L]      True at padded positions

    Output:
        [B, d_model]   per-node embedding (CLS final state)
    """

    def __init__(
        self,
        flow_feat_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_flows: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(flow_feat_dim, d_model)
        # +1 for the CLS slot prepended in forward().
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_flows + 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, flows: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # flows: [B, L, F]  pad_mask: [B, L] True=pad
        B, L, _ = flows.shape
        x = self.input_proj(flows)                        # [B, L, d]
        cls = self.cls_token.expand(B, -1, -1)             # [B, 1, d]
        x = torch.cat([cls, x], dim=1)                     # [B, L+1, d]
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        full_mask = torch.cat([cls_pad, pad_mask], dim=1)  # [B, L+1]
        # If every real position is padded, force CLS to attend to itself only by
        # leaving CLS unmasked (already done) — PyTorch handles all-padded keys
        # outside CLS without NaN as long as one key (CLS) is valid.
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=full_mask)
        return self.out_norm(x[:, 0])                      # [B, d]
