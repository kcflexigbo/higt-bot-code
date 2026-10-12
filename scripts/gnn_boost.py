"""GNN embedding + gradient-boosted trees: does a direction-aware GNN add to XGBoost?

For each seed: train a flat GINE with direction-aware layers (HIGT_DIRECTED=1) and
one with the original layers, export each node's pre-head embedding, and train
XGBoost on combinations of
  x       9 host features
  nbr     direction-aware neighbour aggregates (scripts/hybrid_trees.py::node_blocks)
  dirgnn  directed-GINE embedding
  gnn     original (undirected-aggregation) GINE embedding, as the control
Thresholds are tuned on validation F1. Run with HIGT_GRAPHS_DIR set to the label
version to test. Output: <out>/gnn_boost.{md,json}.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
# hybrid_trees reads an embedding parquet per window; point it at an existing
# cache (node order is identical across label versions; that block is unused here).
os.environ.setdefault("HYBRID_EMB_DIR", "data/flow_embeddings_relabel")
import torch  # noqa: E402,F401

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_trees import load, metrics, tune  # noqa: E402

from src.data.dataset import SplitSpec  # noqa: E402


def gnn_embeddings(out: Path, seed: int, directed: bool) -> dict:
    name = f"{'dirgine' if directed else 'gine'}_seed{seed}"
    path = out / "emb" / f"{name}.npz"
    if not path.exists():
        env = dict(os.environ, HIGT_DIRECTED="1" if directed else "0")
        cmd = [sys.executable, "scripts/baselines/run_gnn.py", "--model", "gine", "--hidden", "64",
               "--layers", "2", "--focal-gamma", "2.0", "--drop-edge-p", "0.2", "--seed", str(seed),
               "--out", str(out / f"gb_{name}.json"), "--save-emb", str(path)]
        with open(out / f"gb_{name}.log", "w") as log:
            subprocess.run(cmd, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    z = np.load(path)
    return {k: z[k] for k in ("train", "val", "test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = SplitSpec.load()
    tr, ytr, _ = load("train", spec); va, yva, _ = load("val", spec); te, yte, ste = load("test", spec)
    print("nodes", len(ytr), len(yva), len(yte), "bots", ytr.sum(), yva.sum(), yte.sum(), flush=True)
    masks = {"all": np.ones_like(yte, bool),
             "CTU-13": np.char.startswith(ste.astype(str), "ctu13"),
             "IoT-23": np.char.startswith(ste.astype(str), "iot23"),
             "held-out": np.isin(ste, spec.holdout_test_scenarios)}
    combos = {"x": ["x"], "x+nbr": ["x", "nbr"], "dirgnn": ["dirgnn"], "x+dirgnn": ["x", "dirgnn"],
              "x+gnn (undirected ctrl)": ["x", "gnn"], "x+nbr+dirgnn": ["x", "nbr", "dirgnn"]}
    spw = (len(ytr) - ytr.sum()) / ytr.sum()
    rows: dict = {c: [] for c in combos}
    for seed in args.seeds:
        d = gnn_embeddings(args.out, seed, directed=True)
        u = gnn_embeddings(args.out, seed, directed=False)
        blocks = {s: {"x": dd["x"], "nbr": dd["nbr"], "dirgnn": d[s], "gnn": u[s]}
                  for s, dd in (("train", tr), ("val", va), ("test", te))}
        for cname, bl in combos.items():
            Xtr, Xva, Xte = (np.hstack([blocks[s][b] for b in bl]) for s in ("train", "val", "test"))
            clf = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                                random_state=seed, n_jobs=-1, eval_metric="aucpr")
            clf.fit(Xtr, ytr)
            pva = clf.predict_proba(Xva)[:, 1]; pte = clf.predict_proba(Xte)[:, 1]
            t = tune(yva, pva); pred = (pte >= t).astype(int)
            rows[cname].append({k: metrics(yte[m], pred[m], pte[m]) for k, m in masks.items()} | {"threshold": t})
            r = rows[cname][-1]["all"]
            print(f"seed {seed} XGB [{cname}]  F1 {r['f1']:.3f} PR {r['pr_auc']:.3f} FN {r['fn']} FP {r['fp']}", flush=True)
            if seed == args.seeds[0]:
                np.savez(args.out / f"pred_xgb_{cname.split()[0].replace('+', '_')}.npz",
                         y=yte, pred=pred, proba=pte, scenario=ste)
    (args.out / "gnn_boost.json").write_text(json.dumps(rows, indent=2))
    md = ["# GNN embedding + XGBoost", "",
          f"Labels: {os.environ.get('HIGT_GRAPHS_DIR', 'data/graphs')}. Mean ± SD over seeds {args.seeds}; "
          "threshold tuned on validation F1. dirgnn = direction-aware GINE embedding; gnn = original GINE embedding.", ""]
    for sub in masks:
        md += [f"## {sub}", "", "| XGBoost on | F1 | MCC | FN | FP | PR-AUC |", "|---|---|---|---|---|---|"]
        for cname, r in rows.items():
            f = lambda k, dd=3: (f"{np.mean([x[sub][k] for x in r]):.{dd}f} ± "
                                 f"{np.std([x[sub][k] for x in r], ddof=1) if len(r) > 1 else 0:.{dd}f}")
            md.append(f"| {cname} | {f('f1')} | {f('mcc')} | {f('fn', 0)} | {f('fp', 0)} | {f('pr_auc')} |")
        md.append("")
    (args.out / "gnn_boost.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", args.out / "gnn_boost.md")


if __name__ == "__main__":
    main()
