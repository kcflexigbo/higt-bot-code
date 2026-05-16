"""Phase 5 entry point: train TemporalGINE end-to-end.

Mirrors scripts/baselines/run_gnn.py but uses:
  - src/data/flow_seq_dataset.py to attach cached flow sequences,
  - src/models/hybrid.TemporalGINE,
  - src/training/loop_phase5 (model(batch) call shape).

Usage
-----
uv run python scripts/build_flow_sequences.py --all   # one-time
uv run python scripts/train_phase5.py                  # train

The script writes data/inspection_logs/phase5_temporal_gine.json and saves
the best checkpoint to experiments/phase5/temporal_gine.pt.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from src.data.dataset import (
    SplitSpec,
    apply_edge_scaler,
    fit_edge_scaler,
    load_split,
)
from src.data.flow_seq_dataset import load_flow_sequences_into
from src.models.hybrid import TemporalGINE
from src.training.evaluate import evaluate
from src.training.loop import TrainConfig
from src.training.loop_phase5 import predict, train_one_model
from src.utils.seeding import pick_device, set_seed

CONFIG_PATH = Path("configs/phase5.yaml")
CKPT_DIR = Path("experiments/phase5")
RESULTS = Path("data/inspection_logs/phase5_temporal_gine.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config max_epochs (useful for smoke tests)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}")

    spec = SplitSpec.load()
    tr = load_split("train", spec)
    va = load_split("val", spec)
    te = load_split("test", spec)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    print("loading flow_seq cache ...")
    load_flow_sequences_into(tr)
    load_flow_sequences_into(va)
    load_flow_sequences_into(te)

    # Edge scaler from train (node scaler is no longer needed — features come
    # from the temporal encoder, which sees its own log1p-transformed inputs).
    e_mean, e_std = fit_edge_scaler(tr)
    apply_edge_scaler(tr, e_mean, e_std)
    apply_edge_scaler(va, e_mean, e_std)
    apply_edge_scaler(te, e_mean, e_std)

    edge_dim = int(tr[0].edge_attr.size(1))
    model = TemporalGINE(
        flow_feat_dim=cfg["encoder"]["flow_feat_dim"],
        edge_dim=edge_dim,
        d_model=cfg["encoder"]["d_model"],
        nhead=cfg["encoder"]["nhead"],
        num_layers=cfg["encoder"]["num_layers"],
        max_flows=cfg["encoder"]["max_flows"],
        encoder_dropout=cfg["encoder"]["dropout"],
        gin_hidden=cfg["gin"]["hidden"],
        gin_layers=cfg["gin"]["num_layers"],
        dropout=cfg["gin"]["dropout"],
        out_dim=2,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TemporalGINE params: {n_params:,}")

    train_cfg = TrainConfig(
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
        batch_size=cfg["train"]["batch_size"],
        max_epochs=args.epochs if args.epochs is not None else cfg["train"]["max_epochs"],
        patience=cfg["train"]["patience"], grad_clip=cfg["train"]["grad_clip"],
        use_focal=cfg["train"]["use_focal"], focal_gamma=cfg["train"]["focal_gamma"],
        focal_alpha=cfg["train"]["focal_alpha"],
    )

    t0 = time.perf_counter()
    info = train_one_model(model, tr, va, cfg=train_cfg, device=device,
                            log_prefix="[t-gine] ")
    fit_s = time.perf_counter() - t0
    print(f"\ntotal fit {fit_s:.1f}s  best val F1 {info['best_val_f1']:.4f}")

    val_loader = DataLoader(va, batch_size=train_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(te, batch_size=train_cfg.batch_size, shuffle=False)
    yv_t, yv_p, yv_pr, scen_v = predict(model, val_loader, device)
    yt_t, yt_p, yt_pr, scen_t = predict(model, test_loader, device)
    val_res = evaluate(yv_t, yv_p, yv_pr, scenarios=scen_v)
    test_res = evaluate(yt_t, yt_p, yt_pr, scenarios=scen_t)
    print(); print(val_res.pretty("val (temporal-gine)"))
    print(); print(test_res.pretty("test (temporal-gine)"))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / "temporal_gine.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                 "edge_scaler": (e_mean, e_std)}, ckpt)
    print(f"saved {ckpt}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "fit_seconds": fit_s,
        "best_val_f1": info["best_val_f1"],
        "params": n_params,
        "val": val_res.as_row(),
        "test": test_res.as_row(),
        "test_per_scenario": test_res.per_scenario,
        "config": cfg,
    }, indent=2))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
