"""Re-evaluate every Phase 5 / Phase 6 / Phase 7 LODO checkpoint in FP32.

Why: the trainers wrap their final predict() call in torch.autocast (AMP). For
LODO-MedBIoT and LODO-IoT23, the test-distribution shift drives the model into
a regime where FP16 softmax underflows and argmax flips, producing nonsense
predictions (F1=0 on MedBIoT, NaN PR-AUC across both directions). LODO-CTU13
was unaffected because its logits stay in a non-saturating range.

This script reloads each checkpoint, runs FP32 inference on val + test, and
rewrites the result JSON with the corrected numbers. The original AMP-tainted
files are kept under a `*_amp_backup.json` name so the bug stays auditable.

Usage:
    uv run python scripts/reeval_lodo_fp32.py --holdout medbiot
    uv run python scripts/reeval_lodo_fp32.py --all
"""
from __future__ import annotations

import argparse
import json
import shutil
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

HOLDOUTS = ("ctu13", "iot23", "medbiot")
INSPECTION = Path("data/inspection_logs")


@torch.no_grad()
def _predict_fp32(model, loader, device):
    """FP32 inference. NO autocast wrapping. Handles both dict-returning
    models (HiGTBot, HiGTBotSparsePool) and tensor-returning ones
    (TemporalGINE)."""
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


def _val_tuned_threshold(yv, prv) -> tuple[float, float]:
    valid = ~np.isnan(prv)
    if valid.sum() == 0:
        return 0.5, float("nan")
    thresholds = np.linspace(0.001, 0.999, 200)
    f1s = [
        f1_score(yv[valid], (prv[valid] >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    i = int(np.argmax(f1s))
    return float(thresholds[i]), float(f1s[i])


def _maybe_backup(result_path: Path):
    if not result_path.exists():
        return
    bak = result_path.with_name(result_path.stem + "_amp_backup.json")
    if not bak.exists():
        shutil.copy2(result_path, bak)


def _rewrite_result(
    result_path: Path,
    val_res,
    test_res,
    val_thr: float,
    val_thr_f1: float,
    test_f1_tuned: float,
    test_p_tuned: float,
    test_r_tuned: float,
):
    blob = json.loads(result_path.read_text())
    blob["val"] = val_res.as_row()
    blob["test"] = test_res.as_row()
    blob["test_per_scenario"] = test_res.per_scenario
    blob["fp32_reeval"] = {
        "val_threshold_optimal": val_thr,
        "val_f1_at_optimal": val_thr_f1,
        "test_f1_at_val_optimal_threshold": test_f1_tuned,
        "test_precision_at_val_optimal": test_p_tuned,
        "test_recall_at_val_optimal": test_r_tuned,
    }
    result_path.write_text(json.dumps(blob, indent=2, default=str))


def _ckpt_path_phase5(holdout: str) -> Path:
    return (
        Path("experiments/lodo")
        / holdout
        / "phase5"
        / f"temporal_gine_lodo_{holdout}_skip.pt"
    )


def _ckpt_path_phase6(holdout: str) -> Path:
    return (
        Path("experiments/lodo")
        / holdout
        / "phase6"
        / f"hgt_sparse_lodo_{holdout}_sparse.pt"
    )


def _ckpt_path_phase7(holdout: str, seed: int) -> Path:
    return (
        Path("experiments/lodo")
        / holdout
        / "phase7"
        / f"higt_bot_lodo_{holdout}_edge_seed{seed}.pt"
    )


def _result_path_phase5(holdout: str) -> Path:
    return INSPECTION / f"phase5_temporal_gine_lodo_{holdout}_skip.json"


def _result_path_phase6(holdout: str) -> Path:
    return INSPECTION / f"phase6_sparse_lodo_{holdout}_sparse.json"


def _result_path_phase7(holdout: str, seed: int) -> Path:
    return INSPECTION / f"phase7_higt_bot_lodo_{holdout}_edge_seed{seed}.json"


def _reeval_phase5(holdout: str, device):
    ckpt_path = _ckpt_path_phase5(holdout)
    result_path = _result_path_phase5(holdout)
    print(f"== Phase 5 {holdout} ==  {ckpt_path}")

    spec = SplitSpec.load(Path(f"configs/split_lodo_{holdout}.yaml"))
    va_files = load_split_files("val", spec)
    te_files = load_split_files("test", spec)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    raw_feat_dim = blob.get("raw_feat_dim")
    e_mean, e_std = blob["edge_scaler"]
    if "node_scaler" in blob:
        n_mean, n_std = blob["node_scaler"]
    else:
        n_mean = n_std = None

    ds_kw = dict(edge_mean=e_mean.cpu(), edge_std=e_std.cpu())
    if n_mean is not None:
        ds_kw.update(node_mean=n_mean.cpu(), node_std=n_std.cpu())
    va_ds = FlowSeqGraphDataset(va_files, **ds_kw)
    te_ds = FlowSeqGraphDataset(te_files, **ds_kw)

    sample = torch.load(va_files[0], weights_only=False)
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
    model.eval()

    bs = int(cfg["train"]["batch_size"])
    val_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)

    yv, pv, prv, scen_v = _predict_fp32(model, val_loader, device)
    yt, pt, prt, scen_t = _predict_fp32(model, test_loader, device)

    val_res = evaluate(yv, pv, prv, scenarios=scen_v)
    test_res = evaluate(yt, pt, prt, scenarios=scen_t)

    thr, val_f1_opt = _val_tuned_threshold(yv, prv)
    valid_t = ~np.isnan(prt)
    test_pred_tuned = (prt[valid_t] >= thr).astype(int)
    test_f1_tuned = float(f1_score(yt[valid_t], test_pred_tuned, zero_division=0))
    test_p_tuned = float(precision_score(yt[valid_t], test_pred_tuned, zero_division=0))
    test_r_tuned = float(recall_score(yt[valid_t], test_pred_tuned, zero_division=0))

    _maybe_backup(result_path)
    _rewrite_result(
        result_path, val_res, test_res, thr, val_f1_opt,
        test_f1_tuned, test_p_tuned, test_r_tuned,
    )
    print(
        f"   F1@0.5={test_res.f1:.4f}  PR={test_res.pr_auc:.4f}  ROC={test_res.roc_auc:.4f}"
        f"  F1@val-tuned(thr={thr:.3f})={test_f1_tuned:.4f}"
    )


def _reeval_phase6(holdout: str, device):
    ckpt_path = _ckpt_path_phase6(holdout)
    result_path = _result_path_phase6(holdout)
    print(f"== Phase 6 {holdout} ==  {ckpt_path}")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    args_saved = blob["args"]
    e_mean, e_std = blob["edge_scaler"]
    n_mean, n_std = blob["node_scaler"]

    spec = SplitSpec.load(Path(f"configs/split_lodo_{holdout}.yaml"))
    va_files = load_split_files("val", spec)
    te_files = load_split_files("test", spec)
    emb_root = Path(f"data/flow_embeddings_lodo/{holdout}")

    ds_kw = dict(
        emb_root=emb_root,
        edge_mean=e_mean.cpu(),
        edge_std=e_std.cpu(),
        node_mean=n_mean.cpu(),
        node_std=n_std.cpu(),
    )
    va_ds = EmbeddingGraphDataset(va_files, **ds_kw)
    te_ds = EmbeddingGraphDataset(te_files, **ds_kw)

    probe = va_ds[0]
    d_model = int(probe.node_emb.size(1))
    raw_feat_dim = int(probe.x.size(1))
    edge_dim = int(probe.edge_attr.size(1))

    model = HiGTBotSparsePool(
        d_model=d_model,
        raw_feat_dim=raw_feat_dim,
        edge_dim=edge_dim,
        hidden=args_saved["hidden"],
        pool_ratio=args_saved["pool_ratio"],
        num_classes=2,
        dropout=args_saved["dropout"],
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    bs = int(args_saved["batch_size"])
    val_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)

    yv, pv, prv, scen_v = _predict_fp32(model, val_loader, device)
    yt, pt, prt, scen_t = _predict_fp32(model, test_loader, device)

    val_res = evaluate(yv, pv, prv, scenarios=scen_v)
    test_res = evaluate(yt, pt, prt, scenarios=scen_t)

    thr, val_f1_opt = _val_tuned_threshold(yv, prv)
    valid_t = ~np.isnan(prt)
    test_pred_tuned = (prt[valid_t] >= thr).astype(int)
    test_f1_tuned = float(f1_score(yt[valid_t], test_pred_tuned, zero_division=0))
    test_p_tuned = float(precision_score(yt[valid_t], test_pred_tuned, zero_division=0))
    test_r_tuned = float(recall_score(yt[valid_t], test_pred_tuned, zero_division=0))

    _maybe_backup(result_path)
    _rewrite_result(
        result_path, val_res, test_res, thr, val_f1_opt,
        test_f1_tuned, test_p_tuned, test_r_tuned,
    )
    print(
        f"   F1@0.5={test_res.f1:.4f}  PR={test_res.pr_auc:.4f}  ROC={test_res.roc_auc:.4f}"
        f"  F1@val-tuned(thr={thr:.3f})={test_f1_tuned:.4f}"
    )


def _reeval_phase7(holdout: str, seed: int, device):
    ckpt_path = _ckpt_path_phase7(holdout, seed)
    result_path = _result_path_phase7(holdout, seed)
    print(f"== Phase 7 {holdout} seed={seed} ==  {ckpt_path}")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_saved = blob["args"]
    e_mean, e_std = blob["edge_scaler"]
    n_mean, n_std = blob["node_scaler"]

    spec = SplitSpec.load(Path(f"configs/split_lodo_{holdout}.yaml"))
    va_files = load_split_files("val", spec)
    te_files = load_split_files("test", spec)
    emb_root = Path(f"data/flow_embeddings_lodo/{holdout}")

    ds_kw = dict(
        emb_root=emb_root,
        edge_mean=e_mean.cpu(),
        edge_std=e_std.cpu(),
        node_mean=n_mean.cpu(),
        node_std=n_std.cpu(),
    )
    va_ds = EmbeddingGraphDataset(va_files, **ds_kw)
    te_ds = EmbeddingGraphDataset(te_files, **ds_kw)

    probe = va_ds[0]
    d_model = int(probe.node_emb.size(1))
    raw_feat_dim = int(probe.x.size(1))
    edge_dim = int(probe.edge_attr.size(1))

    model = HiGTBot(
        d_model=d_model,
        raw_feat_dim=raw_feat_dim,
        edge_dim=edge_dim,
        hidden=args_saved["hidden"],
        pool_ratio=args_saved["pool_ratio"],
        gt_variant=args_saved["gt_variant"],
        gt_layers=args_saved["gt_layers"],
        gt_heads=args_saved["gt_heads"],
        num_classes=2,
        dropout=args_saved["dropout"],
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    bs = int(args_saved["batch_size"])
    val_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)

    yv, pv, prv, scen_v = _predict_fp32(model, val_loader, device)
    yt, pt, prt, scen_t = _predict_fp32(model, test_loader, device)

    val_res = evaluate(yv, pv, prv, scenarios=scen_v)
    test_res = evaluate(yt, pt, prt, scenarios=scen_t)

    thr, val_f1_opt = _val_tuned_threshold(yv, prv)
    valid_t = ~np.isnan(prt)
    test_pred_tuned = (prt[valid_t] >= thr).astype(int)
    test_f1_tuned = float(f1_score(yt[valid_t], test_pred_tuned, zero_division=0))
    test_p_tuned = float(precision_score(yt[valid_t], test_pred_tuned, zero_division=0))
    test_r_tuned = float(recall_score(yt[valid_t], test_pred_tuned, zero_division=0))

    _maybe_backup(result_path)
    _rewrite_result(
        result_path, val_res, test_res, thr, val_f1_opt,
        test_f1_tuned, test_p_tuned, test_r_tuned,
    )
    print(
        f"   F1@0.5={test_res.f1:.4f}  PR={test_res.pr_auc:.4f}  ROC={test_res.roc_auc:.4f}"
        f"  F1@val-tuned(thr={thr:.3f})={test_f1_tuned:.4f}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", choices=HOLDOUTS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    args = ap.parse_args()
    if not args.all and args.holdout is None:
        ap.error("specify --holdout or --all")

    set_seed(42)
    device = pick_device()
    print(f"device: {device}", flush=True)

    targets = HOLDOUTS if args.all else [args.holdout]
    for h in targets:
        _reeval_phase5(h, device)
        _reeval_phase6(h, device)
        for s in args.seeds:
            _reeval_phase7(h, s, device)


if __name__ == "__main__":
    main()
