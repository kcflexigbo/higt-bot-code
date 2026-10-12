"""Two-level (seed x window) paired bootstrap with Holm correction (review item C2).

Each resample draws test windows with replacement, stratified by scenario, and, for every
model, one of its training seeds at random, so the intervals include training variance.
Point estimates are means over seeds. Two-sided p = 2 x min(P(d <= 0), P(d >= 0)); Holm
correction over all comparisons x metrics within a subset. With R resamples the smallest
resolvable two-sided p is 2/R; such values are reported as "< 2/R".

Tree models: data/inspection_logs/revision/seed_preds (5 seeds, scripts/revision_trees.py).
Graph models: data/inspection_logs/v3_gnn/predictions (thresholds re-tuned on validation).

    uv run python -m scripts.bootstrap_seeds_holm --reps 2000
Output: data/inspection_logs/v3/bootstrap_seeds_holm.{md,json}
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
import torch  # noqa: E402,F401  (import before pandas on Windows)

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from src.data.dataset import SplitSpec

R = Path("data/inspection_logs/revision")
G3 = Path("data/inspection_logs/v3_gnn/predictions")
GRID = np.linspace(0.01, 0.99, 99)
REF = "HiGT-Bot v2 (XGB x+grp)"


def f1(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def tree_runs(stem):
    out = []
    for s in (42, 1, 2, 3, 4):
        d = np.load(R / "seed_preds" / f"{stem}_s{s}.npz")
        out.append(((d["p_test"] >= d["threshold"]).astype(int), d["p_test"].astype(float)))
    return out


def gnn_runs(stems):
    out = []
    for st in stems:
        d = np.load(G3 / f"{st}.npz", allow_pickle=True); v = np.load(G3 / f"{st}_val.npz", allow_pickle=True)
        t = GRID[int(np.argmax([f1(v["y"].astype(int), (v["proba"] >= g).astype(int)) for g in GRID]))]
        out.append(((d["proba"] >= t).astype(int), d["proba"].astype(float)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("data/inspection_logs/v3/bootstrap_seeds_holm"))
    a = ap.parse_args()
    spec = SplitSpec.load()
    tab = np.load(R / "tables.npz", allow_pickle=True)
    y = tab["test_y"].astype(int); win = tab["test_win"]; sc = tab["test_scenario"].astype(str)
    M = {REF: tree_runs("v2_XGB_x_grp"), "XGB[x]": tree_runs("XGB_x"), "RF tuned": tree_runs("RF_tuned"),
         "Directed GINE": gnn_runs(["dirgine_seed42", "dirgine_seed1", "dirgine_seed2"]),
         "HiGT-Bot directed": gnn_runs(["higt_bot_edge", "higt_bot_edge_seed1", "higt_bot_edge_seed2"]),
         "Directed T-GINE": gnn_runs(["t_gine_skip"]), "SAGPool": gnn_runs(["sagpool"])}
    for k in ("Directed GINE", "HiGT-Bot directed", "Directed T-GINE", "SAGPool"):
        assert all(len(r[1]) == len(y) for r in M[k]), k
    subsets = {"all": np.ones(len(y), bool), "held-out": np.isin(sc, spec.holdout_test_scenarios),
               "ctu13-10": sc == "ctu13-10"}
    rng = np.random.default_rng(0)
    res = {}
    for sub, m in subsets.items():
        w_ids = np.unique(win[m])
        rows_of = {w: np.flatnonzero((win == w) & m) for w in w_ids}
        by_sc = {}
        for w in w_ids:
            by_sc.setdefault(sc[rows_of[w][0]], []).append(w)
        point = {k: {"f1": float(np.mean([f1(y[m], r[0][m]) for r in v])),
                     "f1_sd": float(np.std([f1(y[m], r[0][m]) for r in v], ddof=1)) if len(v) > 1 else None,
                     "pr_auc": float(np.mean([average_precision_score(y[m], r[1][m]) for r in v])),
                     "seeds": len(v)} for k, v in M.items()}
        boot = {k: [] for k in M}
        for _ in range(a.reps):
            pick = np.concatenate([rng.choice(ws, len(ws)) for ws in by_sc.values()])
            idx = np.concatenate([rows_of[w] for w in pick]); yy = y[idx]
            for k, v in M.items():
                r = v[rng.integers(len(v))]
                boot[k].append((f1(yy, r[0][idx]), average_precision_score(yy, r[1][idx])))
        boot = {k: np.array(v) for k, v in boot.items()}
        comps = []
        for b in M:
            if b == REF:
                continue
            d = boot[REF] - boot[b]
            for j, met in enumerate(("f1", "pr_auc")):
                p = min(1.0, 2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean()))
                comps.append({"comparator": b, "metric": met, "delta": point[REF][met] - point[b][met],
                              "ci95": [float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))],
                              "p_two_sided": float(p)})
        order = sorted(range(len(comps)), key=lambda i: comps[i]["p_two_sided"])  # Holm step-down
        n = len(comps); running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (n - rank) * comps[i]["p_two_sided"]))
            comps[i]["p_holm"] = running
        res[sub] = {"point": point, "comparisons": comps}
        print(sub, "done", flush=True)
    a.out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    lo = 2 / a.reps
    fp = lambda p: f"< {lo:.4f}" if p < lo else f"{p:.4f}"
    md = [f"# Seed x window paired bootstrap with Holm correction ({a.reps} resamples)", "",
          "Windows resampled with replacement, stratified by scenario; each model's training seed drawn at random per "
          "resample (trees: 5 seeds; directed GINE and directed HiGT-Bot: 3 seeds; T-GINE and SAGPool: 1 seed, so "
          "their training variance is not included). Point values are means over seeds. Delta = HiGT-Bot v2 minus "
          "comparator; two-sided p; Holm over all comparisons x metrics within each subset.", ""]
    for sub, r in res.items():
        md += [f"## {sub}", "", "| Model | seeds | F1 (mean ± sd) | PR-AUC |", "|---|---|---|---|"]
        for k, v in r["point"].items():
            sd = f" ± {v['f1_sd']:.3f}" if v["f1_sd"] is not None else ""
            md.append(f"| {k} | {v['seeds']} | {v['f1']:.3f}{sd} | {v['pr_auc']:.3f} |")
        md += ["", "| Comparison | metric | Delta [95% CI] | p (two-sided) | p (Holm) |", "|---|---|---|---|---|"]
        for c in r["comparisons"]:
            md.append(f"| v2 - {c['comparator']} | {c['metric']} | {c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
                      f"| {fp(c['p_two_sided'])} | {fp(c['p_holm'])} |")
        md.append("")
    a.out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", a.out.with_suffix(".md"))


if __name__ == "__main__":
    main()
