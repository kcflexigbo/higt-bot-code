"""Threshold calibration for the trained Phase 5 T-GINE checkpoint.

Loads experiments/phase5/temporal_gine.pt, runs inference on val and test,
then sweeps thresholds two ways:
  1) Global: one threshold tuned to maximize val F1.
  2) Per-scenario: per-scenario threshold tuned on val, applied at test.

No retraining. Writes data/inspection_logs/phase5_calibrated.json.
"""
from __future__ import annotations

import json
import gc
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from src.data.dataset import (
    SplitSpec, fit_edge_scaler, load_graphs, load_split_files,
)
from src.data.flow_seq_dataset import FlowSeqGraphDataset
from src.models.hybrid import TemporalGINE
from src.training.evaluate import evaluate
from src.training.loop_phase5 import predict
from src.utils.seeding import pick_device, set_seed

import argparse
DEFAULT_CKPT = Path("experiments/phase5/temporal_gine.pt")
DEFAULT_OUT = Path("data/inspection_logs/phase5_calibrated.json")


def best_threshold(y_true: np.ndarray, proba: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    """Return (threshold, F1) maximizing F1 on (y_true, proba). Handles zero-pos: returns (0.5, 0.0)."""
    if y_true.sum() == 0:
        return 0.5, 0.0
    best_t, best_f = 0.5, -1.0
    for t in grid:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f = 2 * prec * rec / (prec + rec)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    set_seed(42)
    device = pick_device()
    print(f"device: {device}  ckpt={args.ckpt}", flush=True)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    e_mean, e_std = blob["edge_scaler"]
    raw_feat_dim = blob.get("raw_feat_dim")
    n_scaler = blob.get("node_scaler")

    spec = SplitSpec.load()
    tr_files = load_split_files("train", spec)
    va_files = load_split_files("val", spec)
    te_files = load_split_files("test", spec)
    print(f"val={len(va_files)} test={len(te_files)}", flush=True)

    # Need edge_dim from one graph
    edge_dim = int(torch.load(tr_files[0], weights_only=False).edge_attr.size(1))

    ds_kw = dict(edge_mean=e_mean, edge_std=e_std)
    if n_scaler is not None:
        ds_kw.update(node_mean=n_scaler[0], node_std=n_scaler[1])
    va_ds = FlowSeqGraphDataset(va_files, **ds_kw)
    te_ds = FlowSeqGraphDataset(te_files, **ds_kw)

    enc_chunk = int(cfg["train"].get("encoder_chunk_size", 64))
    model = TemporalGINE(
        flow_feat_dim=cfg["encoder"]["flow_feat_dim"], edge_dim=edge_dim,
        d_model=cfg["encoder"]["d_model"], nhead=cfg["encoder"]["nhead"],
        num_layers=cfg["encoder"]["num_layers"], max_flows=cfg["encoder"]["max_flows"],
        encoder_dropout=cfg["encoder"]["dropout"],
        gin_hidden=cfg["gin"]["hidden"], gin_layers=cfg["gin"]["num_layers"],
        dropout=cfg["gin"]["dropout"], out_dim=2, encoder_chunk_size=enc_chunk,
        raw_feat_dim=raw_feat_dim,
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()

    bs = int(cfg["train"]["batch_size"])
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"

    print("running inference on val and test ...", flush=True)
    yv_t, _, yv_pr, scen_v = predict(model, DataLoader(va_ds, batch_size=bs, shuffle=False), device, use_amp=use_amp)
    yt_t, _, yt_pr, scen_t = predict(model, DataLoader(te_ds, batch_size=bs, shuffle=False), device, use_amp=use_amp)

    grid = np.linspace(0.05, 0.95, 91)  # 0.01 steps

    # ---- Strategy 1: argmax baseline (threshold=0.5) ----
    base_val = evaluate(yv_t, (yv_pr >= 0.5).astype(int), yv_pr, scenarios=scen_v)
    base_test = evaluate(yt_t, (yt_pr >= 0.5).astype(int), yt_pr, scenarios=scen_t)

    # ---- Strategy 2: global threshold tuned on val ----
    g_thr, g_val_f1 = best_threshold(yv_t, yv_pr, grid)
    print(f"[global] val best tau={g_thr:.2f}  val_F1={g_val_f1:.4f}", flush=True)
    g_test = evaluate(yt_t, (yt_pr >= g_thr).astype(int), yt_pr, scenarios=scen_t)

    # ---- Strategy 3: per-scenario threshold tuned on val ----
    scen_thresholds: dict[str, float] = {}
    for sc in np.unique(scen_v):
        m = scen_v == sc
        if yv_t[m].sum() == 0:
            scen_thresholds[str(sc)] = g_thr   # fallback if no positives in val
            continue
        t, _ = best_threshold(yv_t[m], yv_pr[m], grid)
        scen_thresholds[str(sc)] = t

    # Apply per-scenario at test (fallback to global threshold if scenario unseen in val)
    yt_pred_ps = np.zeros_like(yt_t)
    for i, sc in enumerate(scen_t):
        t = scen_thresholds.get(str(sc), g_thr)
        yt_pred_ps[i] = 1 if yt_pr[i] >= t else 0
    ps_test = evaluate(yt_t, yt_pred_ps, yt_pr, scenarios=scen_t)

    # ---- Report ----
    print()
    print(f"{'strategy':<22s} {'val_F1':>8s} {'test_F1':>9s} {'test_P':>8s} {'test_R':>8s}")
    print(f"{'argmax (tau=0.5)':<22s} {base_val.f1:>8.4f} {base_test.f1:>9.4f} {base_test.precision:>8.4f} {base_test.recall:>8.4f}")
    print(f"{'global tau=' + f'{g_thr:.2f}':<22s} {g_val_f1:>8.4f} {g_test.f1:>9.4f} {g_test.precision:>8.4f} {g_test.recall:>8.4f}")
    print(f"{'per-scenario tau':<22s} {'-':>8s} {ps_test.f1:>9.4f} {ps_test.precision:>8.4f} {ps_test.recall:>8.4f}")

    print("\nper-scenario test F1 (argmax vs per-scenario tau):")
    print(f"  {'scenario':<32s}  {'tau':>5s}  {'argmax':>7s}  {'tuned':>7s}  {'delta':>7s}")
    for sc in sorted(set(base_test.per_scenario) | set(ps_test.per_scenario)):
        a = base_test.per_scenario.get(sc, {}).get("f1", float("nan"))
        b = ps_test.per_scenario.get(sc, {}).get("f1", float("nan"))
        t = scen_thresholds.get(sc, g_thr)
        d = b - a
        print(f"  {sc:<32s}  {t:>5.2f}  {a:>7.4f}  {b:>7.4f}  {d:>+7.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "global_threshold": g_thr,
        "global_val_f1": g_val_f1,
        "argmax": {"val": base_val.as_row(), "test": base_test.as_row(),
                    "test_per_scenario": base_test.per_scenario},
        "global_tau": {"test": g_test.as_row(), "test_per_scenario": g_test.per_scenario},
        "per_scenario_tau": {"test": ps_test.as_row(),
                              "test_per_scenario": ps_test.per_scenario,
                              "thresholds": scen_thresholds},
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
