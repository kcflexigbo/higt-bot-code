"""Unit tests for src/models/temporal.py — the per-node Transformer that turns
a flow sequence into a node embedding."""

from __future__ import annotations

import torch

from src.models.temporal import (
    SinusoidalPositionalEncoding,
    TemporalFlowEncoder,
)


def test_pos_enc_shape_and_zero_pos() -> None:
    pe = SinusoidalPositionalEncoding(d_model=16, max_len=32)
    x = torch.zeros(2, 5, 16)
    y = pe(x)
    assert y.shape == x.shape
    # pos=0 → sin(0)=0, cos(0)=1 on alternating dims
    assert torch.allclose(y[0, 0, 0::2], torch.zeros_like(y[0, 0, 0::2]), atol=1e-6)
    assert torch.allclose(y[0, 0, 1::2], torch.ones_like(y[0, 0, 1::2]), atol=1e-6)


def test_temporal_encoder_output_shape() -> None:
    enc = TemporalFlowEncoder(flow_feat_dim=13, d_model=64, nhead=4,
                               num_layers=2, max_flows=256)
    flows = torch.randn(7, 32, 13)
    mask = torch.zeros(7, 32, dtype=torch.bool)
    out = enc(flows, mask)
    assert out.shape == (7, 64)
    assert torch.isfinite(out).all()


def test_temporal_encoder_pad_mask_ignored_positions_do_not_change_output() -> None:
    """If we zero-pad an unmasked position vs mask it, embedding for fully-padded
    positions must not leak into the CLS output.

    Specifically: two sequences that agree on real positions but differ on padded
    positions (with matching masks) must produce the same CLS embedding.
    """
    torch.manual_seed(0)
    enc = TemporalFlowEncoder(flow_feat_dim=13, d_model=32, nhead=4,
                               num_layers=2, max_flows=8, dropout=0.0).eval()
    flows_a = torch.randn(1, 8, 13)
    flows_b = flows_a.clone()
    # Replace positions 5,6,7 with random garbage in B; mark them padded in both.
    flows_b[0, 5:] = torch.randn(3, 13) * 50.0
    mask = torch.tensor([[False, False, False, False, False, True, True, True]])
    with torch.no_grad():
        out_a = enc(flows_a, mask)
        out_b = enc(flows_b, mask)
    assert torch.allclose(out_a, out_b, atol=1e-5), (
        "padded positions are leaking into CLS — mask is wrong"
    )


def test_temporal_encoder_handles_all_padded_node() -> None:
    """A node with zero real flows (all positions padded) must not produce NaN."""
    enc = TemporalFlowEncoder(flow_feat_dim=13, d_model=32, nhead=4,
                               num_layers=2, max_flows=8, dropout=0.0).eval()
    flows = torch.zeros(1, 8, 13)
    mask = torch.ones(1, 8, dtype=torch.bool)   # everything padded
    with torch.no_grad():
        out = enc(flows, mask)
    assert out.shape == (1, 32)
    assert torch.isfinite(out).all()
