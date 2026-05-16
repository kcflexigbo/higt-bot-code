"""Freeze the trained TemporalFlowEncoder and dump per-(scenario, window, node)
embeddings to parquet so Phase 6 (DiffPool) loads `[N, d_model]` tensors
without re-running the encoder. Per the plan, this trades ~3× wall-clock and
~2× peak VRAM in Phase 6 for one-time disk I/O now.

Output layout: data/flow_embeddings/<scenario>/window_<NNNN>.parquet with
columns: node_idx (int32), node_ip (string), dim_000 ... dim_<D-1> (float32).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import SplitSpec, load_split
from src.data.flow_seq_dataset import load_flow_sequences_into
from src.models.hybrid import TemporalGINE
from src.utils.seeding import pick_device, set_seed

EMB_DIR = Path("data/flow_embeddings")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("experiments/phase5/temporal_gine.pt"))
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}")

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = blob["cfg"]
    spec = SplitSpec.load()

    # We need every (scenario, window) — load every split.
    all_graphs = (load_split("train", spec)
                   + load_split("val", spec)
                   + load_split("test", spec))
    print(f"total graphs: {len(all_graphs)}")
    load_flow_sequences_into(all_graphs)

    edge_dim = int(all_graphs[0].edge_attr.size(1))
    model = TemporalGINE(
        flow_feat_dim=cfg["encoder"]["flow_feat_dim"],
        edge_dim=edge_dim,
        d_model=cfg["encoder"]["d_model"],
        nhead=cfg["encoder"]["nhead"],
        num_layers=cfg["encoder"]["num_layers"],
        max_flows=cfg["encoder"]["max_flows"],
        encoder_dropout=0.0,
        gin_hidden=cfg["gin"]["hidden"],
        gin_layers=cfg["gin"]["num_layers"],
        dropout=0.0,
        out_dim=2,
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()

    loader = DataLoader(all_graphs, batch_size=8, shuffle=False)
    written = 0
    with torch.no_grad():
        cursor = 0
        for batch in loader:
            batch = batch.to(device)
            emb = model.encoder(batch.flows, batch.flow_mask).cpu().numpy()
            bi = batch.batch.cpu().numpy()
            for g_in_batch in range(int(bi.max()) + 1):
                node_rows = (bi == g_in_batch).nonzero()[0]
                g = all_graphs[cursor]
                cursor += 1
                node_emb = emb[node_rows]
                D = node_emb.shape[1]
                df = pd.DataFrame(node_emb.astype("float32"),
                                   columns=[f"dim_{d:03d}" for d in range(D)])
                df.insert(0, "node_ip", list(g.node_ips))
                df.insert(0, "node_idx", range(len(g.node_ips)))
                out = EMB_DIR / g.scenario / f"window_{int(g.window_idx):05d}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out, index=False)
                written += 1
    print(f"wrote {written} embedding files under {EMB_DIR}")


if __name__ == "__main__":
    main()
