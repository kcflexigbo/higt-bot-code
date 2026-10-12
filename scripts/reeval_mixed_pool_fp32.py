"""Sanity-check the canonical (mixed-pool) Phase 5/6/7 results in FP32.

The AMP-vs-FP32 bug surfaced under LODO-MedBIoT, where heavy distribution
shift pushed the model into a regime where FP16 softmax saturated. The
canonical mixed-pool test set is largely in-distribution, but it does
contain three held-out scenarios (ctu13-10, iot23-8-1, medbiot-torii_mal_all)
that are family/scenario shifts from training. This script re-evaluates the
canonical checkpoints in FP32 and prints the delta against the stored
result JSONs so we can tell whether any AMP-correctable degradation
hides in the headline mixed-pool numbers.

Read-only: does NOT modify any stored result file. Output to stdout +
data/inspection_logs/mixed_pool_fp32_audit.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader

from src.data.dataset import SplitSpec, load_split_files
from src.data.embedding_dataset import EmbeddingGraphDataset
from src.data.flow_seq_dataset import FlowSeqGraphDataset
from src.models.higt_bot import HiGTBot
from src.models.hybrid import TemporalGINE
from src.models.sparse_pool import HiGTBotSparsePool
from src.training.evaluate import evaluate
from src.utils.seeding import pick_device, set_seed

INSPECTION = Path("data/inspection_logs")
CANONICAL_SPLIT = Path("configs/split.yaml")
EXP = Path("experiments")
EMB = Path("data/flow_embeddings")


@torch.no_grad()
def _predict_fp32(model, loader, device):
    model.eval()
    ys, ps, prs, scens = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        logits = (out["logits"] if isinstance(out, dict) else out).float()
        proba = torch.softmax(logits, dim=-1)[:, 1]
        pred = logits.argmax(dim=-1)
        ys.append(batch.y.cpu().numpy())
        ps.append(pred.cpu().numpy())
        prs.append(proba.cpu().numpy())
        bi = batch.batch.cpu().numpy()
        per_graph_scenarios = (
            batch.scenario if isinstance(batch.scenario, list) else [batch.scenario]
        )
        scens.append(np.array(per_graph_scenarios, dtype=object)[bi])
    return (
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(prs),
        np.concatenate(scens),
    )


def _save_preds(path, yt, pt, prt, scen_t):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, y=yt, pred=pt, proba=prt, scenario=scen_t.astype(str))
    print(f"  wrote {path}")


def _phase5(device, te_files, preds_out=None):
    ckpt = EXP / "phase5/temporal_gine_skip.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    raw_feat_dim = blob.get("raw_feat_dim")
    e_mean, e_std = blob["edge_scaler"]
    e_mean, e_std = e_mean.cpu(), e_std.cpu()
    n_scaler = blob.get("node_scaler")
    ds_kw = dict(edge_mean=e_mean, edge_std=e_std)
    if n_scaler is not None:
        n_mean, n_std = n_scaler
        ds_kw.update(node_mean=n_mean.cpu(), node_std=n_std.cpu())
    te_ds = FlowSeqGraphDataset(te_files, **ds_kw)
    sample = torch.load(te_files[0], weights_only=False)
    edge_dim = int(sample.edge_attr.size(1))
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
        raw_feat_dim=raw_feat_dim,
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    bs = int(cfg["train"]["batch_size"])
    loader = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)
    yt, pt, prt, scen_t = _predict_fp32(model, loader, device)
    _save_preds(preds_out, yt, pt, prt, scen_t)
    return evaluate(yt, pt, prt, scenarios=scen_t)


def _phase6(device, te_files, preds_out=None):
    ckpt = EXP / "phase6/hgt_sparse_baseline.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = blob["args"]
    e_mean, e_std = blob["edge_scaler"]
    n_mean, n_std = blob["node_scaler"]
    ds_kw = dict(
        emb_root=EMB,
        edge_mean=e_mean.cpu(),
        edge_std=e_std.cpu(),
        node_mean=n_mean.cpu(),
        node_std=n_std.cpu(),
    )
    te_ds = EmbeddingGraphDataset(te_files, **ds_kw)
    probe = te_ds[0]
    d_model = int(probe.node_emb.size(1))
    raw_feat_dim = int(probe.x.size(1))
    edge_dim = int(probe.edge_attr.size(1))
    model = HiGTBotSparsePool(
        d_model=d_model,
        raw_feat_dim=raw_feat_dim,
        edge_dim=edge_dim,
        hidden=a["hidden"],
        pool_ratio=a["pool_ratio"],
        num_classes=2,
        dropout=a["dropout"],
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    bs = int(a["batch_size"])
    loader = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)
    yt, pt, prt, scen_t = _predict_fp32(model, loader, device)
    _save_preds(preds_out, yt, pt, prt, scen_t)
    return evaluate(yt, pt, prt, scenarios=scen_t)


def _phase7(device, te_files, ckpt_path, preds_out=None):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = blob["args"]
    e_mean, e_std = blob["edge_scaler"]
    n_mean, n_std = blob["node_scaler"]
    ds_kw = dict(
        emb_root=EMB,
        edge_mean=e_mean.cpu(),
        edge_std=e_std.cpu(),
        node_mean=n_mean.cpu(),
        node_std=n_std.cpu(),
    )
    te_ds = EmbeddingGraphDataset(te_files, **ds_kw)
    probe = te_ds[0]
    d_model = int(probe.node_emb.size(1))
    raw_feat_dim = int(probe.x.size(1))
    edge_dim = int(probe.edge_attr.size(1))
    model = HiGTBot(
        d_model=d_model,
        raw_feat_dim=raw_feat_dim,
        edge_dim=edge_dim,
        hidden=a["hidden"],
        pool_ratio=a["pool_ratio"],
        gt_variant=a["gt_variant"],
        gt_layers=a["gt_layers"],
        gt_heads=a["gt_heads"],
        num_classes=2,
        dropout=a["dropout"],
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    bs = int(a["batch_size"])
    loader = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)
    yt, pt, prt, scen_t = _predict_fp32(model, loader, device)
    _save_preds(preds_out, yt, pt, prt, scen_t)
    return evaluate(yt, pt, prt, scenarios=scen_t)


def _stored_f1(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())["test"]


def _print_row(label, fp32_res, stored, key="f1"):
    fp32_val = getattr(fp32_res, key)
    stored_val = stored.get(key) if stored else None
    if stored_val is None:
        delta = "(no stored)"
    elif isinstance(stored_val, (int, float)):
        delta = f"delta={fp32_val - stored_val:+.4f}"
    else:
        delta = f"stored={stored_val}"
    print(
        f"  {label:32s}  FP32 {key}={fp32_val:.4f}  "
        f"stored={stored_val if stored_val is None else f'{stored_val:.4f}'}  {delta}"
    )


def main():
    global EXP, EMB, INSPECTION
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-preds-dir", type=Path, default=None,
                    help="optional dir for per-node test predictions (<model>.npz), used by phase8_significance.py")
    ap.add_argument("--also-val", action="store_true",
                    help="also write validation predictions (<model>_val.npz) for threshold tuning")
    ap.add_argument("--exp-root", type=Path, default=Path("experiments"),
                    help="checkpoint root (phase5/, phase6/, phase7/ below it)")
    ap.add_argument("--emb-dir", type=Path, default=Path("data/flow_embeddings"))
    ap.add_argument("--results-root", type=Path, default=INSPECTION,
                    help="where stored phase*_*.json live and the audit is written")
    args = ap.parse_args()
    EXP, EMB, INSPECTION = args.exp_root, args.emb_dir, args.results_root
    pd = args.save_preds_dir
    set_seed(42)
    device = pick_device()
    print(f"device: {device}", flush=True)
    spec = SplitSpec.load(CANONICAL_SPLIT)
    te_files = load_split_files("test", spec)
    print(f"canonical test set: {len(te_files)} graphs", flush=True)

    audit: dict = {}

    print("\n=== Phase 5: TemporalGINE +skip ===")
    fp32 = _phase5(device, te_files, pd / "t_gine_skip.npz" if pd else None)
    stored = _stored_f1(INSPECTION / "phase5_temporal_gine_skip.json")
    for k in ("f1", "pr_auc", "roc_auc", "accuracy", "precision", "recall"):
        _print_row("mixed-pool", fp32, stored, k)
    audit["phase5"] = {"fp32": fp32.as_row(), "stored": stored}

    print("\n=== Phase 6: SAGPool (hgt_sparse_baseline) ===")
    fp32 = _phase6(device, te_files, pd / "sagpool.npz" if pd else None)
    stored = _stored_f1(INSPECTION / "phase6_sparse_baseline.json")
    for k in ("f1", "pr_auc", "roc_auc", "accuracy", "precision", "recall"):
        _print_row("mixed-pool", fp32, stored, k)
    audit["phase6"] = {"fp32": fp32.as_row(), "stored": stored}

    for ckpt_name in ("higt_bot_edge.pt", "higt_bot_edge_seed1.pt", "higt_bot_edge_seed2.pt"):
        ckpt = EXP / "phase7" / ckpt_name
        if not ckpt.exists():
            continue
        print(f"\n=== Phase 7: {ckpt_name} ===")
        fp32 = _phase7(device, te_files, ckpt, pd / f"{ckpt.stem}.npz" if pd else None)
        # Match the stored result filename (it uses suffix _edge or _edge_seedN)
        stem = ckpt.stem  # higt_bot_edge or higt_bot_edge_seed1 etc.
        suffix = stem[len("higt_bot"):]  # _edge or _edge_seed1
        stored = _stored_f1(INSPECTION / f"phase7_higt_bot{suffix}.json")
        for k in ("f1", "pr_auc", "roc_auc", "accuracy", "precision", "recall"):
            _print_row(f"mixed-pool {ckpt_name}", fp32, stored, k)
        audit[f"phase7_{ckpt_name}"] = {"fp32": fp32.as_row(), "stored": stored}

    if args.also_val and pd:
        va_files = load_split_files("val", spec)
        print(f"validation set: {len(va_files)} graphs", flush=True)
        _phase5(device, va_files, pd / "t_gine_skip_val.npz")
        _phase6(device, va_files, pd / "sagpool_val.npz")
        for ckpt_name in ("higt_bot_edge.pt", "higt_bot_edge_seed1.pt", "higt_bot_edge_seed2.pt"):
            ckpt = EXP / "phase7" / ckpt_name
            if ckpt.exists():
                _phase7(device, va_files, ckpt, pd / f"{ckpt.stem}_val.npz")

    out = INSPECTION / "mixed_pool_fp32_audit.json"
    out.write_text(json.dumps(audit, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
