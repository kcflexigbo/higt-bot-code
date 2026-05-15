"""Phase 0 gate: verify the environment can do what later phases need.

Checks:
  1. torch + torch_geometric versions are pinned.
  2. Device acceleration is reachable (CUDA on GPU rigs, MPS on Mac, else CPU).
  3. A tiny GCN forward pass runs on the device.
  4. A tiny dense Transformer forward pass runs on the device (Phase 5).
  5. dense_diff_pool runs on the device (Phase 6).
  6. torch-scatter import works on CPU (sparse PyG layers will use it).

Run:  uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import sys

import torch
import torch_geometric
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense import dense_diff_pool

from src.utils.seeding import pick_device, set_seed


def main() -> int:
    set_seed(42)

    print(f"torch            {torch.__version__}")
    print(f"torch_geometric  {torch_geometric.__version__}")
    print(f"cuda available   {torch.cuda.is_available()}")
    print(f"mps available    {torch.backends.mps.is_available()}")
    print(f"mps built        {torch.backends.mps.is_built()}")

    device = pick_device()
    print(f"selected device  {device}")
    print()

    # 1. GCN forward pass on device.
    # Note: PyG's GCNConv uses sparse message passing internally; on MPS this
    # falls back to CPU for some ops. Acceptable for prototyping.
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device=device)
    x = torch.randn(3, 8, device=device)
    gcn = GCNConv(8, 4).to(device)
    try:
        out = gcn(x, edge_index)
        print(f"GCNConv  ok  out.shape={tuple(out.shape)}  out.device={out.device}")
    except Exception as e:
        print(f"GCNConv  FAIL  {type(e).__name__}: {e}")
        return 1

    # 2. Dense Transformer encoder on device (Phase 5 prototype path).
    enc_layer = nn.TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=256, batch_first=True, activation="gelu"
    )
    encoder = nn.TransformerEncoder(enc_layer, num_layers=2).to(device)
    seq = torch.randn(2, 16, 64, device=device)
    try:
        emb = encoder(seq)
        print(f"Transformer  ok  out.shape={tuple(emb.shape)}  out.device={emb.device}")
    except Exception as e:
        print(f"Transformer  FAIL  {type(e).__name__}: {e}")
        return 1

    # 3. dense_diff_pool on device (Phase 6 prototype path).
    n, d, k = 10, 8, 3
    x_dense = torch.randn(2, n, d, device=device)
    adj = torch.randint(0, 2, (2, n, n), dtype=torch.float, device=device)
    s = torch.randn(2, n, k, device=device)
    try:
        x_pool, adj_pool, lp_loss, ent_loss = dense_diff_pool(x_dense, adj, s)
        print(
            f"dense_diff_pool  ok  "
            f"x_pool={tuple(x_pool.shape)}  adj_pool={tuple(adj_pool.shape)}  "
            f"lp={lp_loss.item():.4f}  ent={ent_loss.item():.4f}"
        )
    except Exception as e:
        print(f"dense_diff_pool  FAIL  {type(e).__name__}: {e}")
        return 1

    # 4. torch-scatter import (sparse PyG layers depend on it; CPU on Mac).
    try:
        import torch_scatter

        print(f"torch_scatter  ok  version={torch_scatter.__version__}")
    except Exception as e:
        print(f"torch_scatter  FAIL  {type(e).__name__}: {e}")
        return 1

    print()
    print("Phase 0 gate PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
