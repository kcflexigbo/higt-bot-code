"""Paired window-level bootstrap of test-set differences between models (v2 labels).

Resamples test *windows* (all nodes of a window together, stratified by scenario)
and recomputes F1 / PR-AUC for each model on the same resample, giving a
percentile CI and one-sided p-value for the difference. Models whose predictions
come with a *_val.npz get their threshold re-tuned on validation F1 here (same
grid as scripts/higt_v2_trees.py); tree-head predictions are already val-tuned.

    uv run python scripts/bootstrap_significance.py
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
import torch  # noqa: E402  (import before pandas on Windows)

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from src.data.dataset import SplitSpec, load_split_files

V2, V3 = Path("data/inspection_logs/v2/predictions"), Path("data/inspection_logs/v3")
G3 = Path("data/inspection_logs/v3_gnn/predictions")  # graph models retrained on graphs_v3 (queue_E_v3.sh)
MODELS = {
    "HiGT-Bot v2 (x+grp)": V3 / "pred_x_grp.npz",
    "HiGT-Bot v2 (x+nbr+seq+grp)": V3 / "pred_x_nbr_seq_grp.npz",
    "XGB[x] (v3 features)": V3 / "pred_x.npz",
    "RF tuned (v2 features)": V2 / "rf_tuned.npz",
    "Directed T-GINE (v2)": V2 / "t_gine_skip.npz",
    "HiGT-Bot directed (v2)": V2 / "higt_bot_edge.npz",
    "RF tuned (v3 features)": G3 / "rf_tuned.npz",
    "Directed GINE (v3)": G3 / "dirgine_seed42.npz",
    "Directed T-GINE (v3)": G3 / "t_gine_skip.npz",
    "HiGT-Bot directed (v3)": G3 / "higt_bot_edge.npz",
}
PAIRS = [("HiGT-Bot v2 (x+grp)", "XGB[x] (v3 features)"),
         ("HiGT-Bot v2 (x+nbr+seq+grp)", "XGB[x] (v3 features)"),
         ("HiGT-Bot v2 (x+grp)", "RF tuned (v2 features)"),
         ("HiGT-Bot v2 (x+grp)", "Directed T-GINE (v2)"),
         ("HiGT-Bot v2 (x+grp)", "HiGT-Bot directed (v2)"),
         ("XGB[x] (v3 features)", "RF tuned (v2 features)"),
         ("HiGT-Bot v2 (x+grp)", "RF tuned (v3 features)"),
         ("HiGT-Bot v2 (x+grp)", "Directed GINE (v3)"),
         ("HiGT-Bot v2 (x+grp)", "Directed T-GINE (v3)"),
         ("HiGT-Bot v2 (x+grp)", "HiGT-Bot directed (v3)")]


def f1(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def load(path: Path):
    d = np.load(path, allow_pickle=True)
    y, proba, pred = d["y"].astype(int), d["proba"].astype(float), d["pred"].astype(int)
    vp = path.with_name(path.stem + "_val.npz")
    if vp.exists():
        v = np.load(vp, allow_pickle=True)
        grid = np.linspace(0.01, 0.99, 99)
        t = grid[int(np.argmax([f1(v["y"].astype(int), (v["proba"] >= g).astype(int)) for g in grid]))]
        pred = (proba >= t).astype(int)
    return y, pred, proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("data/inspection_logs/v3/bootstrap_significance"))
    args = ap.parse_args()
    spec = SplitSpec.load()
    win = np.concatenate([np.full(int(torch.load(fp, weights_only=False).num_nodes), i)
                          for i, fp in enumerate(load_split_files("test", spec))])
    M = {k: load(p) for k, p in MODELS.items() if p.exists()}
    y = next(iter(M.values()))[0]
    assert len(win) == len(y) and all((m[0] == y).all() for m in M.values()), "row alignment"
    sc = np.load(MODELS["XGB[x] (v3 features)"], allow_pickle=True)["scenario"].astype(str)
    subsets = {"all": np.ones(len(y), bool), "held-out": np.isin(sc, spec.holdout_test_scenarios),
               "CTU-13": np.char.startswith(sc, "ctu13"), "ctu13-10": sc == "ctu13-10"}
    rng = np.random.default_rng(0)
    res = {}
    for sub, m in subsets.items():
        w_ids = np.unique(win[m]); w_sc = {w: sc[win == w][0] for w in w_ids}
        by_sc = {}
        for w in w_ids:
            by_sc.setdefault(w_sc[w], []).append(w)
        rows_of = {w: np.flatnonzero((win == w) & m) for w in w_ids}
        point = {k: (f1(y[m], v[1][m]), average_precision_score(y[m], v[2][m])) for k, v in M.items()}
        boot = {k: [] for k in M}
        for _ in range(args.reps):
            pick = np.concatenate([rng.choice(ws, len(ws)) for ws in by_sc.values()])
            idx = np.concatenate([rows_of[w] for w in pick])
            yy = y[idx]
            if yy.sum() == 0:
                continue
            for k, v in M.items():
                boot[k].append((f1(yy, v[1][idx]), average_precision_score(yy, v[2][idx])))
        boot = {k: np.array(v) for k, v in boot.items()}
        res[sub] = {"point": point, "pairs": {}}
        for a, b in PAIRS:
            if a not in M or b not in M:
                continue
            d = boot[a] - boot[b]
            res[sub]["pairs"][f"{a} - {b}"] = {
                met: {"delta": point[a][j] - point[b][j], "ci95": [float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))],
                      "p_one_sided": float((d[:, j] <= 0).mean())} for j, met in enumerate(("f1", "pr_auc"))}
        print(sub, "done", flush=True)
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    md = [f"# Paired window-level bootstrap ({args.reps} resamples, stratified by scenario)", "",
          "Delta = first minus second model; p = share of resamples where the delta is <= 0.", ""]
    for sub, r in res.items():
        md += [f"## {sub}", "", "| Comparison | dF1 [95% CI] | p | dPR-AUC [95% CI] | p |", "|---|---|---|---|---|"]
        for name, v in r["pairs"].items():
            md.append(f"| {name} | {v['f1']['delta']:+.3f} [{v['f1']['ci95'][0]:+.3f}, {v['f1']['ci95'][1]:+.3f}] | {v['f1']['p_one_sided']:.3f} "
                      f"| {v['pr_auc']['delta']:+.3f} [{v['pr_auc']['ci95'][0]:+.3f}, {v['pr_auc']['ci95'][1]:+.3f}] | {v['pr_auc']['p_one_sided']:.3f} |")
        md.append("")
    args.out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", args.out.with_suffix(".md"))


if __name__ == "__main__":
    main()
