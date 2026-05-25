"""Phase 5 entry point: train TemporalGINE end-to-end.

Mirrors scripts/baselines/run_gnn.py but uses:
  - src/data/flow_seq_dataset.FlowSeqGraphDataset (lazy flow_seq load),
  - src/models/hybrid.TemporalGINE,
  - src/training/loop_phase5 (model(batch) call shape).

Usage
-----
uv run python scripts/build_flow_sequences.py --all --skip-existing
uv run python scripts/train_phase5.py

The script writes data/inspection_logs/phase5_temporal_gine.json and saves
the best checkpoint to experiments/phase5/temporal_gine.pt.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch  # before yaml/pandas paths on Windows
import yaml
from torch_geometric.loader import DataLoader

from src.data.dataset import (
    SplitSpec,
    fit_edge_scaler,
    fit_feature_scaler,
    load_graphs,
    load_split_files,
)
from src.data.flow_seq_dataset import FlowSeqGraphDataset
from src.models.hybrid import TemporalGINE
from src.training.evaluate import evaluate
from src.training.loop import TrainConfig, _class_weights_from_train
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
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override config batch_size")
    ap.add_argument("--raw-feature-skip", action="store_true",
                    help="concat raw scaled node features into GIN input")
    ap.add_argument("--results-suffix", type=str, default="",
                    help="suffix for output files (e.g. '_skip')")
    ap.add_argument("--init-from-ssl", type=Path, default=None,
                    help="path to SSL-pretrained encoder checkpoint "
                         "(from scripts/pretrain_encoder.py); loads encoder_state "
                         "into model.encoder before training")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}", flush=True)

    spec = SplitSpec.load()
    tr_files = load_split_files("train", spec)
    va_files = load_split_files("val", spec)
    te_files = load_split_files("test", spec)
    print(f"train={len(tr_files)} val={len(va_files)} test={len(te_files)}", flush=True)

    # Edge + node scalers + class weights from graphs only (no flow tensors in RAM).
    print("fitting edge/node scalers + class weights (graphs only) ...", flush=True)
    tr_graphs = load_graphs(tr_files)
    e_mean, e_std = fit_edge_scaler(tr_graphs)
    n_mean, n_std = fit_feature_scaler(tr_graphs)
    class_weight = _class_weights_from_train(tr_graphs)
    raw_feat_dim = int(tr_graphs[0].x.size(1)) if args.raw_feature_skip else None
    del tr_graphs
    gc.collect()

    ds_kw = dict(edge_mean=e_mean, edge_std=e_std)
    if args.raw_feature_skip:
        ds_kw.update(node_mean=n_mean, node_std=n_std)
    tr_ds = FlowSeqGraphDataset(tr_files, **ds_kw)
    va_ds = FlowSeqGraphDataset(va_files, **ds_kw)
    te_ds = FlowSeqGraphDataset(te_files, **ds_kw)
    print(f"lazy flow_seq datasets ready  raw_feat_dim={raw_feat_dim}", flush=True)

    edge_dim = int(torch.load(tr_files[0], weights_only=False).edge_attr.size(1))
    enc_chunk = int(cfg["train"].get("encoder_chunk_size", 64))
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
        encoder_chunk_size=enc_chunk,
        raw_feat_dim=raw_feat_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TemporalGINE params: {n_params:,}  encoder_chunk={enc_chunk}", flush=True)

    if args.init_from_ssl is not None:
        ssl_blob = torch.load(args.init_from_ssl, map_location="cpu",
                                weights_only=False)
        missing, unexpected = model.encoder.load_state_dict(
            ssl_blob["encoder_state"], strict=True,
        )
        print(f"loaded SSL encoder init from {args.init_from_ssl}  "
              f"best_pretrain_loss={ssl_blob.get('best_loss', '?'):.4f}",
              flush=True)

    batch_size = args.batch_size if args.batch_size is not None else cfg["train"]["batch_size"]
    train_cfg = TrainConfig(
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
        batch_size=batch_size,
        max_epochs=args.epochs if args.epochs is not None else cfg["train"]["max_epochs"],
        patience=cfg["train"]["patience"], grad_clip=cfg["train"]["grad_clip"],
        use_focal=cfg["train"]["use_focal"], focal_gamma=cfg["train"]["focal_gamma"],
        focal_alpha=cfg["train"]["focal_alpha"],
        grad_accum_steps=int(cfg["train"].get("grad_accum_steps", 1)),
        use_amp=bool(cfg["train"].get("amp", False)),
        class_weight=class_weight,
    )

    t0 = time.perf_counter()
    info = train_one_model(model, tr_ds, va_ds, cfg=train_cfg, device=device,
                            log_prefix="[t-gine] ")
    fit_s = time.perf_counter() - t0
    print(f"\ntotal fit {fit_s:.1f}s  best val F1 {info['best_val_f1']:.4f}", flush=True)

    val_loader = DataLoader(va_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(te_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0)
    yv_t, yv_p, yv_pr, scen_v = predict(model, val_loader, device, use_amp=train_cfg.use_amp)
    yt_t, yt_p, yt_pr, scen_t = predict(model, test_loader, device, use_amp=train_cfg.use_amp)
    val_res = evaluate(yv_t, yv_p, yv_pr, scenarios=scen_v)
    test_res = evaluate(yt_t, yt_p, yt_pr, scenarios=scen_t)
    print(); print(val_res.pretty("val (temporal-gine)"))
    print(); print(test_res.pretty("test (temporal-gine)"))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"temporal_gine{args.results_suffix}.pt"
    save_blob = {"state_dict": model.state_dict(), "cfg": cfg,
                  "edge_scaler": (e_mean, e_std),
                  "raw_feat_dim": raw_feat_dim}
    if args.raw_feature_skip:
        save_blob["node_scaler"] = (n_mean, n_std)
    torch.save(save_blob, ckpt)
    print(f"saved {ckpt}")

    results_path = RESULTS.with_name(f"phase5_temporal_gine{args.results_suffix}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({
        "fit_seconds": fit_s,
        "best_val_f1": info["best_val_f1"],
        "params": n_params,
        "val": val_res.as_row(),
        "test": test_res.as_row(),
        "test_per_scenario": test_res.per_scenario,
        "config": cfg,
        "raw_feature_skip": args.raw_feature_skip,
    }, indent=2))
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
