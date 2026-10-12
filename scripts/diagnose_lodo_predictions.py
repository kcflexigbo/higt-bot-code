"""Diagnose LODO Phase 7 predictions: dump proba stats, threshold sweeps, and
val-tuned operating points so we can tell prevalence artifacts from real
degradation.

For each LODO direction:
  1. Load the GT-edge seed=42 checkpoint.
  2. Run inference on val + test of that LODO split.
  3. Print proba distribution (overall + per-class).
  4. Pick threshold that maximises val F1 -> apply to test.
  5. Compute PR-AUC and ROC-AUC manually with extra logging.

Writes data/inspection_logs/lodo_diagnose_<holdout>.json per direction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

from src.data.dataset import SplitSpec, load_split_files
from src.data.embedding_dataset import EmbeddingGraphDataset
from src.models.higt_bot import HiGTBot
from src.utils.seeding import pick_device, set_seed


def _safe(fn, *a, **kw):
    try:
        return float(fn(*a, **kw))
    except Exception as e:
        return f"ERR: {type(e).__name__}: {e}"


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, prs, gids, scens = [], [], [], [], []
    g_offset = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        logits = out["logits"].float()
        proba = torch.softmax(logits, dim=-1)[:, 1]
        pred = logits.argmax(dim=-1)
        ys.append(batch.y.cpu().numpy())
        ps.append(pred.cpu().numpy())
        prs.append(proba.cpu().numpy())
        bi = batch.batch.cpu().numpy()
        gids.append(bi + g_offset)
        g_offset += int(batch.num_graphs)
        sc = batch.scenario if isinstance(batch.scenario, list) else [batch.scenario]
        scens.append(np.array(sc, dtype=object)[bi])
    predict.last_graph_ids = np.concatenate(gids)
    predict.last_scenarios = np.concatenate(scens).astype(str)
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(prs)


def proba_stats(y, pr):
    nan_count = int(np.isnan(pr).sum())
    pr_clean = pr[~np.isnan(pr)]
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
        "nan_proba_count": nan_count,
        "proba_min": float(pr_clean.min()) if len(pr_clean) else None,
        "proba_max": float(pr_clean.max()) if len(pr_clean) else None,
        "proba_mean": float(pr_clean.mean()) if len(pr_clean) else None,
        "proba_std": float(pr_clean.std()) if len(pr_clean) else None,
        "proba_median": float(np.median(pr_clean)) if len(pr_clean) else None,
        "proba_unique_values_6dp": int(len(np.unique(np.round(pr_clean, 6)))),
    }
    if len(pr_clean):
        bins = [0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0001]
        hist, _ = np.histogram(pr_clean, bins=bins)
        out["hist_bins"] = bins
        out["hist_counts"] = hist.tolist()
        for cls, name_cls in [(1, "bot"), (0, "benign")]:
            m = (y == cls)
            if m.sum() > 0:
                pc = pr[m]
                pc = pc[~np.isnan(pc)]
                if len(pc):
                    out[f"proba_{name_cls}_mean"] = float(pc.mean())
                    out[f"proba_{name_cls}_median"] = float(np.median(pc))
                    out[f"proba_{name_cls}_std"] = float(pc.std())
                    out[f"proba_{name_cls}_min"] = float(pc.min())
                    out[f"proba_{name_cls}_max"] = float(pc.max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True, choices=["ctu13", "iot23", "medbiot"])
    ap.add_argument("--run", default=None,
                    help="LODO run dir under experiments/lodo (default: holdout; e.g. iot23_fp32)")
    ap.add_argument("--seed", type=int, default=42, help="which GT-edge LODO checkpoint seed to load")
    ap.add_argument("--save-preds", type=Path, default=None,
                    help="optional .npz for val+test probabilities, labels, graph ids, scenarios")
    args = ap.parse_args()
    set_seed(42)
    device = pick_device()

    split_cfg = Path(f"configs/split_lodo_{args.holdout}.yaml")
    run = args.run or args.holdout
    ckpt_path = Path(
        f"experiments/lodo/{run}/phase7/"
        f"higt_bot_lodo_{run}_edge_seed{args.seed}.pt"
    )
    emb_root = Path(f"data/flow_embeddings_lodo/{run}")

    spec = SplitSpec.load(split_cfg)
    va_files = load_split_files("val", spec)
    te_files = load_split_files("test", spec)
    print(f"splits  val={len(va_files)}  test={len(te_files)}", flush=True)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_saved = blob.get("args", {})
    e_mean, e_std = blob["edge_scaler"]
    n_mean, n_std = blob["node_scaler"]
    e_mean, e_std = e_mean.cpu(), e_std.cpu()
    n_mean, n_std = n_mean.cpu(), n_std.cpu()

    ds_kw = dict(
        emb_root=emb_root,
        edge_mean=e_mean,
        edge_std=e_std,
        node_mean=n_mean,
        node_std=n_std,
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
        hidden=args_saved.get("hidden", 128),
        pool_ratio=args_saved.get("pool_ratio", 0.5),
        gt_variant=args_saved.get("gt_variant", "edge"),
        gt_layers=args_saved.get("gt_layers", 2),
        gt_heads=args_saved.get("gt_heads", 4),
        num_classes=2,
        dropout=args_saved.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(blob["state_dict"])

    val_loader = DataLoader(va_ds, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(te_ds, batch_size=4, shuffle=False, num_workers=0)

    print("inferring val ...", flush=True)
    yv, pv, prv = predict(model, val_loader, device)
    gv, sv = predict.last_graph_ids, predict.last_scenarios
    print("inferring test ...", flush=True)
    yt, pt, prt = predict(model, test_loader, device)
    gt, st = predict.last_graph_ids, predict.last_scenarios
    if args.save_preds is not None:
        args.save_preds.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_preds, val_y=yv, val_proba=prv, val_graph=gv, val_scenario=sv,
                 test_y=yt, test_proba=prt, test_graph=gt, test_scenario=st)
        print(f"wrote {args.save_preds}", flush=True)

    val_stats = proba_stats(yv, prv)
    test_stats = proba_stats(yt, prt)

    val_metrics = {
        "pr_auc": _safe(average_precision_score, yv, prv),
        "roc_auc": _safe(roc_auc_score, yv, prv),
        "f1_at_0.5": _safe(f1_score, yv, (prv >= 0.5).astype(int)),
    }
    test_metrics = {
        "pr_auc": _safe(average_precision_score, yt, prt),
        "roc_auc": _safe(roc_auc_score, yt, prt),
        "f1_at_0.5": _safe(f1_score, yt, (prt >= 0.5).astype(int)),
    }

    valid_v = ~np.isnan(prv)
    if valid_v.sum() > 0:
        thresholds = np.linspace(0.001, 0.999, 200)
        val_f1s = [
            f1_score(yv[valid_v], (prv[valid_v] >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        best_idx = int(np.argmax(val_f1s))
        best_t = float(thresholds[best_idx])
        best_val_f1 = float(val_f1s[best_idx])
        valid_t = ~np.isnan(prt)
        if valid_t.sum() > 0:
            test_pred_tuned = (prt[valid_t] >= best_t).astype(int)
            test_f1_tuned = float(f1_score(yt[valid_t], test_pred_tuned, zero_division=0))
            test_p_tuned = float(precision_score(yt[valid_t], test_pred_tuned, zero_division=0))
            test_r_tuned = float(recall_score(yt[valid_t], test_pred_tuned, zero_division=0))
        else:
            test_f1_tuned = test_p_tuned = test_r_tuned = float("nan")
        tune = {
            "val_f1_optimal_threshold": best_t,
            "val_f1_at_optimal": best_val_f1,
            "test_f1_at_val_optimal_threshold": test_f1_tuned,
            "test_precision_at_val_optimal": test_p_tuned,
            "test_recall_at_val_optimal": test_r_tuned,
        }
    else:
        tune = {"note": "val proba all NaN; cannot tune"}

    payload = {
        "holdout": args.holdout,
        "checkpoint": str(ckpt_path),
        "val": {**val_stats, **val_metrics},
        "test": {**test_stats, **test_metrics},
        "threshold_tuning": tune,
    }
    suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    out = Path(f"data/inspection_logs/lodo_diagnose_{run}{suffix}.json")
    out.write_text(json.dumps(payload, indent=2, default=str))
    print("=== SUMMARY ===")
    print(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
