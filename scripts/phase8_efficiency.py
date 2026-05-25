"""Phase 8.3 — efficiency report.

For each saved checkpoint (Phase 4 GIN-baseline through Phase 7 HiGT-Bot),
measure:
  - Total parameter count.
  - Peak GPU memory at inference (single forward pass on a representative
    batch).
  - Inference time per graph, median over 1000 test graphs.

Output: data/inspection_logs/phase8_efficiency.md + .json

Skips models we no longer have checkpoints for. Designed to run after the
final pipeline is locked.
"""
from __future__ import annotations

import gc
import json
import statistics
import time
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from src.data.dataset import (
    SplitSpec, fit_edge_scaler, fit_feature_scaler, load_graphs, load_split_files,
)
from src.data.embedding_dataset import EmbeddingGraphDataset
from src.data.flow_seq_dataset import FlowSeqGraphDataset
from src.models.diffpool import HiGTBotDiffPool
from src.models.higt_bot import HiGTBot
from src.models.hybrid import TemporalGINE
from src.models.sparse_pool import HiGTBotSparsePool
from src.utils.seeding import pick_device, set_seed

LOG_DIR = Path("data/inspection_logs")

# (label, builder_fn, checkpoint_path, dataset_factory)
EXPERIMENTS = [
    ("Phase 5 T-GINE+skip", "tgine",
     Path("experiments/phase5/temporal_gine_skip.pt"), "flow_seq"),
    ("Phase 6 DiffPool", "diffpool",
     Path("experiments/phase6/hgt_diffpool.pt"), "embedding"),
    ("Phase 6.4 SAGPool", "sparse",
     Path("experiments/phase6/hgt_sparse_baseline.pt"), "embedding"),
    ("Phase 7 GT-edge", "higt",
     Path("experiments/phase7/higt_bot_edge.pt"), "embedding"),
    ("Phase 7 GT-global", "higt",
     Path("experiments/phase7/higt_bot_global.pt"), "embedding"),
    ("Phase 7 GT-hybrid 2L", "higt",
     Path("experiments/phase7/higt_bot_hybrid.pt"), "embedding"),
]


def build_model(kind: str, blob: dict, *, edge_dim: int, raw_feat_dim: int) -> torch.nn.Module:
    if kind == "tgine":
        cfg = blob["cfg"]
        return TemporalGINE(
            flow_feat_dim=cfg["encoder"]["flow_feat_dim"], edge_dim=edge_dim,
            d_model=cfg["encoder"]["d_model"], nhead=cfg["encoder"]["nhead"],
            num_layers=cfg["encoder"]["num_layers"],
            max_flows=cfg["encoder"]["max_flows"], encoder_dropout=0.0,
            gin_hidden=cfg["gin"]["hidden"], gin_layers=cfg["gin"]["num_layers"],
            dropout=0.0, out_dim=2, raw_feat_dim=blob.get("raw_feat_dim"),
        )
    if kind == "diffpool":
        mcfg = blob["cfg"]["model"] if "model" in blob.get("cfg", {}) else None
        # Phase 6 was trained via CLI args; fall back to defaults.
        return HiGTBotDiffPool(
            d_model=64, raw_feat_dim=raw_feat_dim, edge_dim=edge_dim,
            hidden=128, max_nodes=400, pool_ratio=0.25, dropout=0.0,
        )
    if kind == "sparse":
        return HiGTBotSparsePool(
            d_model=64, raw_feat_dim=raw_feat_dim, edge_dim=edge_dim,
            hidden=128, pool_ratio=0.5, dropout=0.0,
        )
    if kind == "higt":
        args = blob.get("args", {})
        return HiGTBot(
            d_model=64, raw_feat_dim=raw_feat_dim, edge_dim=edge_dim,
            hidden=int(args.get("hidden", 128)),
            pool_ratio=float(args.get("pool_ratio", 0.5)),
            gt_variant=str(args.get("gt_variant", "edge")),
            gt_layers=int(args.get("gt_layers", 2)),
            gt_heads=int(args.get("gt_heads", 4)),
            dropout=0.0,
        )
    raise ValueError(f"unknown kind {kind!r}")


def make_dataset(kind: str, files: list[Path], *, e_mean, e_std, n_mean, n_std):
    if kind == "flow_seq":
        return FlowSeqGraphDataset(files, edge_mean=e_mean, edge_std=e_std,
                                     node_mean=n_mean, node_std=n_std)
    return EmbeddingGraphDataset(files, edge_mean=e_mean, edge_std=e_std,
                                   node_mean=n_mean, node_std=n_std)


def measure(model, loader, device, *, n_graphs: int) -> dict:
    """Returns dict of params, peak_vram_mb, median_ms_per_graph, n_graphs."""
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    times_per_graph: list[float] = []
    peak_vram = 0
    seen = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        # warmup
        for batch in loader:
            batch = batch.to(device)
            try:
                _ = model(batch)
            except Exception:
                _ = model.forward(batch)
            break
        if device.type == "cuda":
            torch.cuda.synchronize()
        for batch in loader:
            batch = batch.to(device)
            B = int(batch.batch.max().item()) + 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                out = model(batch)
            except Exception:
                out = model.forward(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0  # ms per batch
            per_graph_ms = dt / max(B, 1)
            for _ in range(B):
                times_per_graph.append(per_graph_ms)
            seen += B
            if seen >= n_graphs:
                break
        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "params": n_params,
        "peak_vram_mb": round(peak_vram, 2),
        "median_ms_per_graph": round(statistics.median(times_per_graph), 3),
        "p95_ms_per_graph": round(
            statistics.quantiles(times_per_graph, n=20)[-1], 3),
        "graphs_measured": len(times_per_graph),
    }


def main() -> None:
    set_seed(42)
    device = pick_device()
    print(f"device: {device}")

    cfg5 = yaml.safe_load(Path("configs/phase5.yaml").read_text())
    spec = SplitSpec.load()
    te_files = load_split_files("test", spec)[:200]  # cap for speed; 200×batch=8 ≈ 1600 graphs >> 1000

    tr_graphs = load_graphs(load_split_files("train", spec)[:200])
    e_mean, e_std = fit_edge_scaler(tr_graphs)
    n_mean, n_std = fit_feature_scaler(tr_graphs)
    raw_feat_dim = int(tr_graphs[0].x.size(1))
    edge_dim = int(tr_graphs[0].edge_attr.size(1))
    del tr_graphs
    gc.collect()

    rows = []
    for label, kind, ckpt_path, ds_kind in EXPERIMENTS:
        if not ckpt_path.exists():
            print(f"skip (no ckpt): {label}  {ckpt_path}")
            rows.append({"label": label, "ckpt": str(ckpt_path), "missing": True})
            continue
        print(f"\n== {label} ==", flush=True)
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(kind, blob, edge_dim=edge_dim,
                             raw_feat_dim=raw_feat_dim)
        try:
            model.load_state_dict(blob["state_dict"])
        except RuntimeError as e:
            print(f"  state_dict load failed: {e}")
            rows.append({"label": label, "ckpt": str(ckpt_path),
                          "missing": False, "load_error": str(e)})
            continue
        model = model.to(device).eval()

        ds = make_dataset(ds_kind, te_files, e_mean=e_mean, e_std=e_std,
                            n_mean=n_mean, n_std=n_std)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        m = measure(model, loader, device, n_graphs=1000)
        m["label"] = label
        m["ckpt"] = str(ckpt_path)
        print(f"  params={m['params']:,}  "
              f"peak_vram={m['peak_vram_mb']} MB  "
              f"median_ms={m['median_ms_per_graph']}  "
              f"p95_ms={m['p95_ms_per_graph']}")
        rows.append(m)
        del model, blob
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Write outputs
    out_json = LOG_DIR / "phase8_efficiency.json"
    out_md = LOG_DIR / "phase8_efficiency.md"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = ["# Phase 8.3 — Efficiency Report", "",
             "Measured on the held-out test graphs (median over 1000 graphs).",
             ""]
    lines.append("| Model | Params | Peak VRAM (MB) | Median ms/graph | p95 ms/graph |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        if r.get("missing"):
            lines.append(f"| {r['label']} | (no ckpt) | | | |")
            continue
        if "load_error" in r:
            lines.append(f"| {r['label']} | (load error) | | | |")
            continue
        lines.append(
            f"| {r['label']} | {r['params']:,} | {r['peak_vram_mb']} | "
            f"{r['median_ms_per_graph']} | {r['p95_ms_per_graph']} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_md}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
