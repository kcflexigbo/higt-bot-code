"""Stage 1 of Phase 6.3 — self-supervised pretrain the TemporalFlowEncoder.

Trains `src.training.pretrain_ssl.MaskedFlowSSL` over every cached flow
sequence in `data/flow_seqs/` (no labels used). After training, saves the
encoder's `state_dict` to `experiments/phase6/encoder_ssl_pretrain.pt`.

Streams windows from disk one at a time (same pattern as the streaming
cache_embeddings script) so RAM stays flat.

Usage
-----
uv run python scripts/pretrain_encoder.py --epochs 20 --batch-size 4
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch  # before yaml/pandas on Windows
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from src.models.temporal import TemporalFlowEncoder
from src.training.pretrain_ssl import MaskedFlowSSL
from src.utils.seeding import pick_device, set_seed

FLOW_SEQS_DIR = Path("data/flow_seqs")
CKPT_DIR = Path("experiments/phase6")


class FlowSeqWindowDataset(Dataset):
    """Yields one window's stacked flow tensor at a time, no labels.

    Each item is a tuple (flows: [N, L, F], pad_mask: [N, L]) for one
    (scenario, window). The training loop flattens these along the node dim
    to form a batch of node-level sequences.
    """
    def __init__(self, files: list[Path]) -> None:
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        blob = torch.load(self.files[idx], weights_only=False)
        return blob["flows"], blob["flow_mask"]


def pretrain_collate(items):
    """Concatenate windows along the node axis → one big [sumN, L, F] batch."""
    flows = torch.cat([x for x, _ in items], dim=0)
    masks = torch.cat([m for _, m in items], dim=0)
    return flows, masks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/phase5.yaml"))
    ap.add_argument("--out", type=Path,
                    default=CKPT_DIR / "encoder_ssl_pretrain.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="number of windows per optimizer step (each contributes "
                         "all of its nodes as parallel sequences)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="optional cap (smoke testing)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}", flush=True)

    files = sorted(FLOW_SEQS_DIR.rglob("window_*.pt"))
    if args.max_windows:
        files = files[: args.max_windows]
    print(f"flow_seq windows: {len(files)}", flush=True)

    ds = FlowSeqWindowDataset(files)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, collate_fn=pretrain_collate)

    encoder = TemporalFlowEncoder(
        flow_feat_dim=cfg["encoder"]["flow_feat_dim"],
        d_model=cfg["encoder"]["d_model"],
        nhead=cfg["encoder"]["nhead"],
        num_layers=cfg["encoder"]["num_layers"],
        max_flows=cfg["encoder"]["max_flows"],
        dropout=cfg["encoder"]["dropout"],
    ).to(device)
    ssl = MaskedFlowSSL(
        encoder, flow_feat_dim=cfg["encoder"]["flow_feat_dim"],
        mask_ratio=args.mask_ratio,
    ).to(device)
    n_params = sum(p.numel() for p in ssl.parameters())
    print(f"MaskedFlowSSL params: {n_params:,}  mask_ratio={args.mask_ratio}", flush=True)

    optim = Adam(ssl.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    best_encoder_state = None
    t_total = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        ssl.train()
        ep_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()
        for flows, pad_mask in loader:
            flows = flows.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss, _ = ssl(flows, pad_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl.parameters(), 1.0)
            optim.step()
            ep_loss += float(loss.item())
            n_batches += 1
        dt = time.perf_counter() - t0
        avg = ep_loss / max(n_batches, 1)
        better = avg < best_loss
        if better:
            best_loss = avg
            best_encoder_state = {k: v.detach().cpu().clone()
                                    for k, v in ssl.encoder.state_dict().items()}
        print(f"[ssl] epoch {epoch:>3d}  loss {avg:.4f}  best {best_loss:.4f}  "
              f"{'*' if better else ' '}  {dt:.1f}s", flush=True)

    total_s = time.perf_counter() - t_total
    print(f"\ntotal pretrain {total_s:.1f}s  best loss {best_loss:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": best_encoder_state,
        "encoder_cfg": cfg["encoder"],
        "mask_ratio": args.mask_ratio,
        "best_loss": best_loss,
        "epochs": args.epochs,
    }, args.out)
    print(f"saved {args.out}", flush=True)
    del ssl, encoder, best_encoder_state
    gc.collect()


if __name__ == "__main__":
    main()
