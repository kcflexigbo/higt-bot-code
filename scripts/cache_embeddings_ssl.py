"""Stage 2 of Phase 6.3 — re-cache encoder embeddings using the SSL-pretrained
encoder (no supervised fine-tune).

Mirrors scripts/cache_embeddings.py but loads weights from
`experiments/phase6/encoder_ssl_pretrain.pt` (the masked-flow pretraining
output) into a fresh TemporalFlowEncoder, then writes embeddings to a
separate directory so the original (T-GINE-skip) cache is preserved for
comparison.

Streams windows one at a time. Resumable — skips parquet files that already
exist in the target dir.

Output: data/flow_embeddings_ssl/<scenario>/window_<NNNN>.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch  # before pandas on Windows
import pandas as pd

from src.data.dataset import SplitSpec, load_split_files
from src.data.flow_seq_dataset import cache_path as flow_cache_path
from src.models.temporal import TemporalFlowEncoder
from src.utils.seeding import pick_device, set_seed

DEFAULT_EMB_DIR = Path("data/flow_embeddings_ssl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("experiments/phase6/encoder_ssl_pretrain.pt"))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_EMB_DIR)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-every", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}", flush=True)

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    enc_cfg = blob["encoder_cfg"]
    encoder = TemporalFlowEncoder(
        flow_feat_dim=enc_cfg["flow_feat_dim"],
        d_model=enc_cfg["d_model"],
        nhead=enc_cfg["nhead"],
        num_layers=enc_cfg["num_layers"],
        max_flows=enc_cfg["max_flows"],
        dropout=0.0,                       # inference-only
    )
    encoder.load_state_dict(blob["encoder_state"])
    encoder.to(device).eval()
    print(f"loaded SSL encoder from {args.checkpoint} (best_loss={blob.get('best_loss', '?')})",
          flush=True)

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
            emb = encoder(flows, flow_mask).cpu().numpy()
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
