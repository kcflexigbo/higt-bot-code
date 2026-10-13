"""Tree models on P2P botnet scenarios (corrected labels and features, data/graphs_v3).

Two protocols:
  transfer  train/validate on the thesis split (cached tables in data/inspection_logs/v3_emb3),
            test on ALL windows of each target scenario (never seen in training);
  within    chronological 60/20/20 split inside each target scenario (train/val/test
            by window order, as configs/split.yaml does), for scenarios with enough windows.
Models: XGB[x] and HiGT-Bot v2 = XGB[x+grp] (seeds 42, 1, 2), RF (500 trees, balanced, seed 42);
threshold tuned on validation F1. Also saves seed-42 predictions for the bootstrap.

    uv run python -m scripts.p2p_transfer_trees --targets ctu13-12 isot-2010
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
import torch  # noqa: E402  (import before pandas on Windows)

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from scripts.higt_v2_trees import fit_predict, metrics, tune
from src.features.host_blocks import fit_group_scaler, grp_block

CACHE = Path("data/inspection_logs/v3_emb3")
OUT = Path("data/inspection_logs/p2p")
SEEDS = (42, 1, 2)


def tables(files, qt):
    X, G, Y, W = [], [], [], []
    for i, fp in enumerate(files):
        g = torch.load(fp, weights_only=False)
        x = g.x.numpy().astype(np.float32)
        X.append(x); G.append(grp_block(x, g.edge_index.numpy(), qt)); Y.append(g.y.numpy().astype(int))
        W.append(np.full(len(x), i))
    return {"x": np.vstack(X), "grp": np.vstack(G), "y": np.concatenate(Y), "win": np.concatenate(W)}


def run_models(tr, va, te, tag, out_dir):
    res = {}
    for name, bl in (("XGB[x]", ["x"]), ("HiGT-Bot v2 (x+grp)", ["x", "grp"])):
        rows = []
        for seed in SEEDS:
            S = lambda T: np.hstack([T[b] for b in bl])  # noqa: E731
            pva, pte = fit_predict(S(tr), tr["y"], S(va), S(te), seed)
            t = tune(va["y"], pva); pred = (pte >= t).astype(int)
            rows.append(metrics(te["y"], pred, pte) | {"threshold": t})
            if seed == 42:
                np.savez(out_dir / f"pred_{tag}_{name.split()[0].replace('[', '_').replace(']', '')}.npz",
                         y=te["y"], pred=pred, proba=pte, win=te["win"])
        res[name] = rows
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1).fit(tr["x"], tr["y"])
    pva, pte = rf.predict_proba(va["x"])[:, 1], rf.predict_proba(te["x"])[:, 1]
    t = tune(va["y"], pva); pred = (pte >= t).astype(int)
    res["RF"] = [metrics(te["y"], pred, pte) | {"threshold": t}]
    np.savez(out_dir / f"pred_{tag}_RF.npz", y=te["y"], pred=pred, proba=pte, win=te["win"])
    for k, rows in res.items():
        f1 = [r["f1"] for r in rows]; pr = [r["pr_auc"] for r in rows]
        print(f"[{tag}] {k:22s} F1 {np.mean(f1):.3f}±{np.std(f1):.3f} PR {np.nanmean(pr):.3f} "
              f"FN {np.mean([r['fn'] for r in rows]):.1f} FP {np.mean([r['fp'] for r in rows]):.1f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--graphs", type=Path, default=Path(os.environ["HIGT_GRAPHS_DIR"]))
    ap.add_argument("--min-windows-within", type=int, default=30)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tr = dict(np.load(CACHE / "cache_train.npz", allow_pickle=True)); va = dict(np.load(CACHE / "cache_val.npz", allow_pickle=True))
    qt = fit_group_scaler(tr["x"])
    path = OUT / "trees_results.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    for sc in a.targets:
        files = sorted((a.graphs / sc).glob("window_*.pt"))
        te = tables(files, qt)
        print(f"{sc}: {len(files)} windows, {len(te['y'])} host-windows, {int(te['y'].sum())} bot", flush=True)
        r = {"windows": len(files), "host_windows": int(len(te["y"])), "bots": int(te["y"].sum())}
        r["transfer"] = run_models(tr, va, te, f"transfer_{sc}", OUT)
        if len(files) >= a.min_windows_within:
            n = len(files); i1, i2 = int(0.6 * n), int(0.8 * n)
            sub = lambda lo, hi: {k: v[(te["win"] >= lo) & (te["win"] < hi)] for k, v in te.items()}  # noqa: E731
            wtr, wva, wte = sub(0, i1), sub(i1, i2), sub(i2, n)
            # group scaler refit on the target's own training part for the within protocol
            qt_w = fit_group_scaler(wtr["x"])
            fl = lambda lo, hi: tables(files[lo:hi], qt_w)  # noqa: E731
            wtr, wva, wte = fl(0, i1), fl(i1, i2), fl(i2, n)
            r["within_split_bots"] = [int(wtr["y"].sum()), int(wva["y"].sum()), int(wte["y"].sum())]
            if min(r["within_split_bots"]) > 0:
                r["within"] = run_models(wtr, wva, wte, f"within_{sc}", OUT)
        allres[sc] = r
        path.write_text(json.dumps(allres, indent=2))


if __name__ == "__main__":
    main()
