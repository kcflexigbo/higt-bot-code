"""Hybrid: gradient-boosted trees / random forest on host features + graph context.

Does graph information add anything on top of strong per-host features? Feature
blocks per node (all from the relabelled graphs, HIGT_GRAPHS_DIR):
  x      9 raw host features (what the RF baseline uses)
  nbr    out-neighbour and in-neighbour mean/max of x, out/in degree,
         sum of edge features over out-edges and in-edges (direction-aware,
         ego kept separate - no mixing of the node's own features)
  emb    64-d temporal encoder embedding (data/flow_embeddings_relabel)
Each block combination is trained with RF and XGBoost; the decision threshold
is tuned on validation F1. Output: data/inspection_logs/relabel/hybrid_trees.{md,json}
and predictions/hybrid_<name>.npz.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
import torch  # noqa: E402

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.data.dataset import SplitSpec, load_split_files

EMB = Path(os.environ.get("HYBRID_EMB_DIR", "data/flow_embeddings_relabel"))
OUT = Path(os.environ.get("HYBRID_OUT_DIR", "data/inspection_logs/relabel"))


def node_blocks(g):
    x = g.x.numpy().astype(np.float32)
    n = x.shape[0]
    ei = g.edge_index.numpy()
    ea = g.edge_attr.numpy().astype(np.float32)
    src, dst = ei[0], ei[1]
    feats = {"x": x}
    blk = []
    for a, b in ((src, dst), (dst, src)):  # out-neighbours of a, then in-neighbours
        deg = np.bincount(a, minlength=n).astype(np.float32)
        s = np.zeros_like(x); np.add.at(s, a, x[b])
        mx = np.full_like(x, -np.inf); np.maximum.at(mx, a, x[b])
        mx[np.isinf(mx)] = 0
        es = np.zeros((n, ea.shape[1]), np.float32); np.add.at(es, a, ea)
        blk += [deg[:, None], s / np.maximum(deg, 1)[:, None], mx, np.log1p(np.abs(es)) * np.sign(es)]
    feats["nbr"] = np.hstack(blk)
    df = pd.read_parquet(EMB / g.scenario / f"window_{int(g.window_idx):05d}.parquet")
    assert list(df["node_ip"]) == list(g.node_ips)
    feats["emb"] = df[[c for c in df.columns if c.startswith("dim_")]].to_numpy(np.float32)
    return feats


def load(name, spec):
    parts = {"x": [], "nbr": [], "emb": []}; ys = []; sc = []
    for fp in load_split_files(name, spec):
        g = torch.load(fp, weights_only=False)
        f = node_blocks(g)
        for k in parts:
            parts[k].append(f[k])
        ys.append(g.y.numpy()); sc += [g.scenario] * int(g.num_nodes)
    return {k: np.vstack(v) for k, v in parts.items()}, np.concatenate(ys), np.array(sc)


def metrics(y, p, pr):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "mcc": (tp * tn - fp * fn) / d if d else 0.0,
            "fn": fn, "fp": fp, "pr_auc": float(average_precision_score(y, pr))}


def tune(y, pr):
    grid = np.linspace(0.01, 0.99, 99)
    f = [metrics(y, (pr >= t).astype(int), pr)["f1"] for t in grid]
    return float(grid[int(np.argmax(f))])


def main():
    spec = SplitSpec.load()
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    tr, ytr, _ = load("train", spec); va, yva, _ = load("val", spec); te, yte, ste = load("test", spec)
    print("nodes", len(ytr), len(yva), len(yte), "bots", ytr.sum(), yva.sum(), yte.sum(), flush=True)
    combos = {"x": ["x"], "x+nbr": ["x", "nbr"], "x+emb": ["x", "emb"], "x+nbr+emb": ["x", "nbr", "emb"]}
    spw = (len(ytr) - ytr.sum()) / ytr.sum()
    rows = {}
    masks = {"all": np.ones_like(yte, bool), "excl. bashlite_spread": ste != "medbiot-bashlite_mal_spread_all",
             "CTU-13": np.char.startswith(ste.astype(str), "ctu13"),
             "held-out": np.isin(ste, spec.holdout_test_scenarios)}
    for cname, blocks in combos.items():
        Xtr, Xva, Xte = (np.hstack([d[b] for b in blocks]) for d in (tr, va, te))
        for mname in ("RF", "XGB"):
            res = []
            for seed in (42, 1, 2):
                if mname == "RF":
                    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=seed)
                else:
                    clf = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                                        colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                                        random_state=seed, n_jobs=-1, eval_metric="aucpr")
                clf.fit(Xtr, ytr)
                pva = clf.predict_proba(Xva)[:, 1]; pte = clf.predict_proba(Xte)[:, 1]
                t = tune(yva, pva); pred = (pte >= t).astype(int)
                res.append({k: metrics(yte[m], pred[m], pte[m]) for k, m in masks.items()} | {"threshold": t})
                if seed == 42:
                    np.savez(OUT / "predictions" / f"hybrid_{mname}_{cname.replace('+', '_')}.npz",
                             y=yte, pred=pred, proba=pte, scenario=ste)
            rows[f"{mname} [{cname}]"] = res
            r = res
            print(f"{mname:4s} {cname:10s} " + "  ".join(
                f"{k}: F1 {np.mean([x[k]['f1'] for x in r]):.3f} PR {np.mean([x[k]['pr_auc'] for x in r]):.3f} "
                f"FN {np.mean([x[k]['fn'] for x in r]):.0f} FP {np.mean([x[k]['fp'] for x in r]):.0f}"
                for k in ("all", "excl. bashlite_spread")), flush=True)
    (OUT / "hybrid_trees.json").write_text(json.dumps(rows, indent=2))
    md = ["# Hybrid tree models on relabelled data (infected-source labels)", "",
          "Mean ± SD over 3 seeds; threshold tuned on validation F1. Blocks: x = 9 host features; "
          "nbr = direction-aware neighbour aggregates (ego kept separate); emb = temporal encoder embedding.", ""]
    for sub in masks:
        md += [f"## {sub}", "", "| Model | F1 | MCC | FN | FP | PR-AUC |", "|---|---|---|---|---|---|"]
        for name, r in rows.items():
            f = lambda k, d=3: f"{np.mean([x[sub][k] for x in r]):.{d}f} ± {np.std([x[sub][k] for x in r], ddof=1):.{d}f}"
            md.append(f"| {name} | {f('f1')} | {f('mcc')} | {f('fn', 0)} | {f('fp', 0)} | {f('pr_auc')} |")
        md.append("")
    (OUT / "hybrid_trees.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "hybrid_trees.md")


if __name__ == "__main__":
    main()
