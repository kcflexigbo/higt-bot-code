"""HiGT-Bot v2: temporal host encoder + cohort features + gradient-boosted head.

Builds per-node feature tables for train / val / test from the graphs in
HIGT_GRAPHS_DIR (default data/graphs_v3: v2 labels + node features from all
window flows), then trains XGBoost on block combinations:
  x    9 host features            seq  flow-sequence statistics
  grp  in-window group similarity  nbr  direction-aware neighbour aggregates
  emb  temporal-encoder embedding (directed T-GINE encoder, --emb-dir)
(see src/features/host_blocks.py), plus an ensemble that averages the x model and
the x+seq+grp model. Two protocols:
  split  the thesis split (per-scenario chronological 60/20/20 + held-out), 3 seeds,
         threshold tuned on validation F1
  loso   leave-one-scenario-out over CTU-13: train/val on every other scenario,
         test on all windows of the left-out one (1 seed)
Output: <out>/higt_v2_trees.{md,json}, <out>/pred_<model>.npz (seed 42, split).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
import torch  # noqa: E402

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.data.dataset import SplitSpec, load_split_files
from src.features.host_blocks import fit_group_scaler, grp_block, nbr_block, seq_block

FLOW_SEQS = Path("data/flow_seqs")
COMBOS = {
    "x": ["x"],
    "x+seq": ["x", "seq"],
    "x+grp": ["x", "grp"],
    "x+seq+grp": ["x", "seq", "grp"],
    "x+emb": ["x", "emb"],
    "x+grp+emb": ["x", "grp", "emb"],
    "x+seq+grp+emb": ["x", "seq", "grp", "emb"],
    "x+nbr+seq+grp": ["x", "nbr", "seq", "grp"],
}
ENSEMBLE = ("x", "x+seq+grp")


def build_tables(spec, emb_dir: Path, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {s: load_split_files(s, spec) for s in ("train", "val", "test")}
    paths = {s: cache_dir / f"cache_{s}.npz" for s in files}
    if all(p.exists() for p in paths.values()):
        return {s: dict(np.load(p, allow_pickle=True)) for s, p in paths.items()}
    x_train = np.vstack([torch.load(fp, weights_only=False).x.numpy() for fp in files["train"]])
    qt = fit_group_scaler(x_train)
    out = {}
    for s, fl in files.items():
        t0 = time.time()
        B = {k: [] for k in ("x", "nbr", "seq", "grp", "emb")}
        y, sc = [], []
        for fp in fl:
            g = torch.load(fp, weights_only=False)
            x = g.x.numpy().astype(np.float32)
            ei = g.edge_index.numpy(); ea = g.edge_attr.numpy().astype(np.float32)
            blob = torch.load(FLOW_SEQS / g.scenario / f"window_{int(g.window_idx):05d}.pt", weights_only=False)
            assert list(blob["node_ips"]) == list(g.node_ips), f"flow-seq node mismatch {fp}"
            df = pd.read_parquet(emb_dir / g.scenario / f"window_{int(g.window_idx):05d}.parquet")
            assert list(df["node_ip"]) == list(g.node_ips), f"embedding node mismatch {fp}"
            B["x"].append(x); B["nbr"].append(nbr_block(x, ei, ea))
            B["seq"].append(seq_block(blob["flows"].numpy(), blob["flow_mask"].numpy()))
            B["grp"].append(grp_block(x, ei, qt))
            B["emb"].append(df[[c for c in df.columns if c.startswith("dim_")]].to_numpy(np.float32))
            y.append(g.y.numpy()); sc += [g.scenario] * int(g.num_nodes)
        out[s] = {k: np.vstack(v) for k, v in B.items()} | {"y": np.concatenate(y), "scenario": np.array(sc)}
        np.savez(paths[s], **out[s])
        print(f"built {s}: {len(sc)} nodes in {time.time() - t0:.0f}s", flush=True)
    return out


def metrics(y, pred, proba):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pr = float(average_precision_score(y, proba)) if 0 < y.sum() < len(y) else float("nan")
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "fn": fn, "fp": fp, "pr_auc": pr}


def tune(y, p):
    grid = np.linspace(0.01, 0.99, 99)
    return float(grid[int(np.argmax([metrics(y, (p >= t).astype(int), p)["f1"] for t in grid]))])


def fit_predict(Xtr, ytr, Xva, Xte, seed):
    clf = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=(len(ytr) - ytr.sum()) / max(ytr.sum(), 1),
                        tree_method="hist", random_state=seed, n_jobs=-1, eval_metric="aucpr")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]


def stack(T, blocks, mask=None):
    X = np.hstack([T[b] for b in blocks])
    return X if mask is None else X[mask]


def run_split(T, spec, seeds, out_dir):
    te = T["test"]; ste = te["scenario"].astype(str)
    masks = {"all": np.ones(len(te["y"]), bool), "CTU-13": np.char.startswith(ste, "ctu13"),
             "IoT-23": np.char.startswith(ste, "iot23"), "held-out": np.isin(ste, spec.holdout_test_scenarios),
             "ctu13-10": ste == "ctu13-10"}
    rows = {c: [] for c in list(COMBOS) + ["ensemble(x, x+seq+grp)"]}
    for seed in seeds:
        P = {}
        for c, bl in COMBOS.items():
            pva, pte = fit_predict(stack(T["train"], bl), T["train"]["y"], stack(T["val"], bl), stack(te, bl), seed)
            P[c] = (pva, pte)
        P["ensemble(x, x+seq+grp)"] = tuple((P[ENSEMBLE[0]][i] + P[ENSEMBLE[1]][i]) / 2 for i in (0, 1))
        for c, (pva, pte) in P.items():
            t = tune(T["val"]["y"], pva); pred = (pte >= t).astype(int)
            rows[c].append({k: metrics(te["y"][m], pred[m], pte[m]) for k, m in masks.items()} | {"threshold": t})
            if seed == seeds[0]:
                np.savez(out_dir / f"pred_{c.replace('+', '_').replace(', ', '_').replace('(', '').replace(')', '')}.npz",
                         y=te["y"], pred=pred, proba=pte, scenario=ste)
            r = rows[c][-1]
            print(f"[split seed {seed}] {c:24s} all PR {r['all']['pr_auc']:.3f} F1 {r['all']['f1']:.3f} | "
                  f"CTU PR {r['CTU-13']['pr_auc']:.3f} | held-out PR {r['held-out']['pr_auc']:.3f} F1 {r['held-out']['f1']:.3f}",
                  flush=True)
    return rows, list(masks)


def run_loso(T, seed=42):
    allx = {k: np.concatenate([T[s][k] for s in ("train", "val", "test")]) for k in T["train"]}
    part = np.concatenate([np.full(len(T[s]["y"]), s) for s in ("train", "val", "test")])
    sc = allx["scenario"].astype(str)
    scen = sorted(s for s in np.unique(sc) if s.startswith("ctu13"))
    res = {c: {} for c in list(COMBOS) + ["ensemble(x, x+seq+grp)"]}
    for s in scen:
        tr = (part == "train") & (sc != s); va = (part == "val") & (sc != s); te = sc == s
        if allx["y"][te].sum() == 0:
            continue
        P = {}
        for c, bl in COMBOS.items():
            P[c] = fit_predict(stack(allx, bl, tr), allx["y"][tr], stack(allx, bl, va), stack(allx, bl, te), seed)
        P["ensemble(x, x+seq+grp)"] = tuple((P[ENSEMBLE[0]][i] + P[ENSEMBLE[1]][i]) / 2 for i in (0, 1))
        for c, (pva, pte) in P.items():
            t = tune(allx["y"][va], pva)
            res[c][s] = metrics(allx["y"][te], (pte >= t).astype(int), pte)
        print(f"[loso] {s}: " + "  ".join(f"{c} {res[c][s]['pr_auc']:.3f}" for c in ("x", "x+seq+grp", "ensemble(x, x+seq+grp)")),
              flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/inspection_logs/v3"))
    ap.add_argument("--emb-dir", type=Path, default=Path("data/flow_embeddings_v2"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--skip-loso", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = SplitSpec.load()
    T = build_tables(spec, args.emb_dir, args.out)
    split_rows, subsets = run_split(T, spec, args.seeds, args.out)
    loso = {} if args.skip_loso else run_loso(T)
    (args.out / "higt_v2_trees.json").write_text(json.dumps({"split": split_rows, "loso": loso}, indent=2))

    f = lambda r, sub, k, d=3: (f"{np.nanmean([x[sub][k] for x in r]):.{d}f}"
                                + (f" ± {np.nanstd([x[sub][k] for x in r], ddof=1):.{d}f}" if len(r) > 1 else ""))
    md = ["# HiGT-Bot v2 tree head on graphs_v3 (v2 labels, features from all window flows)", "",
          f"XGBoost, seeds {args.seeds}, threshold tuned on validation F1. Blocks: x host features, seq flow-sequence "
          "statistics, grp in-window group similarity, nbr direction-aware neighbour aggregates, emb directed temporal "
          "encoder embedding.", ""]
    for sub in subsets:
        md += [f"## Thesis split: {sub}", "", "| Model | PR-AUC | F1 | FN | FP |", "|---|---|---|---|---|"]
        for c, r in split_rows.items():
            md.append(f"| {c} | {f(r, sub, 'pr_auc')} | {f(r, sub, 'f1')} | {f(r, sub, 'fn', 0)} | {f(r, sub, 'fp', 0)} |")
        md.append("")
    if loso:
        sc = sorted(next(iter(loso.values())))
        md += ["## Leave-one-scenario-out over CTU-13 (PR-AUC on the left-out scenario)", "",
               "| Model | " + " | ".join(sc) + " | mean |", "|---|" + "---|" * (len(sc) + 1)]
        for c, r in loso.items():
            v = [r[s]["pr_auc"] for s in sc]
            md.append(f"| {c} | " + " | ".join(f"{x:.3f}" for x in v) + f" | **{np.nanmean(v):.3f}** |")
        md += ["", "F1 on the left-out scenario:", "", "| Model | " + " | ".join(sc) + " | mean |",
               "|---|" + "---|" * (len(sc) + 1)]
        for c, r in loso.items():
            v = [r[s]["f1"] for s in sc]
            md.append(f"| {c} | " + " | ".join(f"{x:.3f}" for x in v) + f" | **{np.mean(v):.3f}** |")
    (args.out / "higt_v2_trees.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", args.out / "higt_v2_trees.md")


if __name__ == "__main__":
    main()
