"""Phase 4 baselines: GAT and GIN node classifiers.

One script, two models — pick via --model gat|gin. Trains on the canonical
split, evaluates with the shared metric bundle, writes a JSON results blob.

Usage:
    uv run python scripts/baselines/run_gnn.py --model gin
    uv run python scripts/baselines/run_gnn.py --model gat --epochs 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import SplitSpec, load_split
from src.models.gnn_baselines import build_model
from src.training.evaluate import evaluate
from src.training.loop import TrainConfig, predict, train_one_model
from src.utils.seeding import pick_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["gin", "gat"], required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4, help="GAT only")
    ap.add_argument("--layers", type=int, default=3, help="GIN only")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None, help="auto if unset (CUDA>MPS>CPU)")
    ap.add_argument("--out", type=Path, default=None,
                    help="results json path; default data/inspection_logs/baseline_<model>.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}")

    spec = SplitSpec.load()
    print(f"Loading split v{spec.version} ...")
    tr = load_split("train", spec)
    va = load_split("val", spec)
    te = load_split("test", spec)
    print(f"  train graphs={len(tr)}  val graphs={len(va)}  test graphs={len(te)}")
    in_dim = int(tr[0].x.size(1))
    print(f"  node feature dim = {in_dim}")

    model_kw: dict = {"hidden": args.hidden, "dropout": args.dropout}
    if args.model == "gat":
        model_kw["heads"] = args.heads
    else:
        model_kw["num_layers"] = args.layers
    model = build_model(args.model, in_dim, **model_kw)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model = {args.model.upper()}  hidden={args.hidden}  "
          f"params={n_params:,}")

    cfg = TrainConfig(
        lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, max_epochs=args.epochs,
        patience=args.patience,
    )

    t0 = time.perf_counter()
    info = train_one_model(model, tr, va, cfg=cfg, device=device,
                            log_prefix=f"[{args.model}] ")
    fit_s = time.perf_counter() - t0
    print(f"\ntotal fit time {fit_s:.1f}s, best val F1 {info['best_val_f1']:.4f}")

    # Final eval on val and test
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False)
    yv_t, yv_p, yv_pr, scen_v = predict(model, val_loader, device)
    yt_t, yt_p, yt_pr, scen_t = predict(model, test_loader, device)

    val_res = evaluate(yv_t, yv_p, yv_pr, scenarios=scen_v)
    test_res = evaluate(yt_t, yt_p, yt_pr, scenarios=scen_t)

    print(); print(val_res.pretty(f"val ({args.model})"))
    print(); print(test_res.pretty(f"test ({args.model})"))

    out = args.out or Path(f"data/inspection_logs/baseline_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "hidden": args.hidden,
        "heads": args.heads if args.model == "gat" else None,
        "num_layers": args.layers if args.model == "gin" else 2,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "fit_seconds": fit_s,
        "best_val_f1": info["best_val_f1"],
        "val": val_res.as_row(),
        "test": test_res.as_row(),
        "test_per_scenario": test_res.per_scenario,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
