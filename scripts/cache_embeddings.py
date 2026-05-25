"""Freeze the trained TemporalFlowEncoder and dump per-(scenario, window, node)
embeddings to parquet so Phase 6 (DiffPool) loads `[N, d_model]` tensors
without re-running the encoder.

Streams one window at a time — loads graph + flow_seq from disk, runs the
encoder, writes parquet, frees the tensors. Skips windows whose parquet
already exists so it is safely resumable. Avoids the ~10 GB RAM spike that
happens when every graph + flow tensor is pinned in memory at once.

Output layout: data/flow_embeddings/<scenario>/window_<NNNN>.parquet with
columns: node_idx (int32), node_ip (string), dim_000 ... dim_<D-1> (float32).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch  # MUST come before pandas on Windows to avoid DLL ordering issue
import pandas as pd

from src.data.dataset import SplitSpec, load_split_files
from src.data.flow_seq_dataset import cache_path as flow_cache_path
from src.models.hybrid import TemporalGINE
from src.utils.seeding import pick_device, set_seed

EMB_DIR = Path("data/flow_embeddings")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("experiments/phase5/temporal_gine_skip.pt"))
    ap.add_argument("--out-dir", type=Path, default=EMB_DIR)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-every", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}", flush=True)

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = blob["cfg"]
    raw_feat_dim = blob.get("raw_feat_dim")

    model = TemporalGINE(
        flow_feat_dim=cfg["encoder"]["flow_feat_dim"],
        edge_dim=10,
        d_model=cfg["encoder"]["d_model"],
        nhead=cfg["encoder"]["nhead"],
        num_layers=cfg["encoder"]["num_layers"],
        max_flows=cfg["encoder"]["max_flows"],
        encoder_dropout=0.0,
        gin_hidden=cfg["gin"]["hidden"],
        gin_layers=cfg["gin"]["num_layers"],
        dropout=0.0,
        out_dim=2,
        raw_feat_dim=raw_feat_dim,
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()

    spec = SplitSpec.load()
    files = (load_split_files("train", spec)
              + load_split_files("val", spec)
              + load_split_files("test", spec))
    print(f"total graphs: {len(files)}", flush=True)

    written = 0
    skipped = 0
    with torch.no_grad():
        for i, gf in enumerate(files, start=1):
            g = torch.load(gf, weights_only=False)
            out = args.out_dir / g.scenario / f"window_{int(g.window_idx):05d}.parquet"
            if out.exists():
                skipped += 1
                del g
                continue
            fs_path = flow_cache_path(g.scenario, int(g.window_idx))
            fs_blob = torch.load(fs_path, weights_only=False)
            if list(fs_blob["node_ips"]) != list(g.node_ips):
                raise ValueError(
                    f"flow_seq/graph node_ips mismatch for {g.scenario} w{g.window_idx}"
                )
            flows = fs_blob["flows"].to(device)
            flow_mask = fs_blob["flow_mask"].to(device)
            emb = model.encoder(flows, flow_mask).cpu().numpy()
            D = emb.shape[1]
            df = pd.DataFrame(emb.astype("float32"),
                               columns=[f"dim_{d:03d}" for d in range(D)])
            df.insert(0, "node_ip", list(g.node_ips))
            df.insert(0, "node_idx", range(len(g.node_ips)))
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            written += 1
            del flows, flow_mask, emb, df, fs_blob, g

            if i % args.print_every == 0:
                print(f"  [{i}/{len(files)}] written={written} skipped={skipped}",
                      flush=True)

    print(f"done -- written={written} skipped={skipped} dir={args.out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
