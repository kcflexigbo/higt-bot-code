"""One table of every v3 model on the test split (v2 labels, graphs_v3 features).

Graph-model thresholds are re-tuned on validation F1 from their *_val.npz (same
grid as scripts/higt_v2_trees.py); tree-head predictions are already val-tuned.
    uv run python scripts/summarise_v3.py   -> data/inspection_logs/v3/summary_v3.md
"""
from __future__ import annotations

import torch  # noqa: F401  (import before pandas on Windows)

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from src.data.dataset import SplitSpec

G3, T3 = Path("data/inspection_logs/v3_gnn/predictions"), Path("data/inspection_logs/v3")
GRID = np.linspace(0.01, 0.99, 99)


def f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1), int(fn), int(fp)


def load(path):
    d = np.load(path, allow_pickle=True); y, pr, pred = d["y"].astype(int), d["proba"], d["pred"].astype(int)
    vp = path.with_name(path.stem + "_val.npz")
    if vp.exists():
        v = np.load(vp, allow_pickle=True)
        t = GRID[int(np.argmax([f1(v["y"].astype(int), (v["proba"] >= g).astype(int))[0] for g in GRID]))]
        pred = (pr >= t).astype(int)
    return y, pred, pr, d["scenario"].astype(str)


def main():
    spec = SplitSpec.load()
    files = sorted(p for p in G3.glob("*.npz") if not p.stem.endswith("_val")) + \
        [T3 / f"pred_{k}.npz" for k in ("x", "x_grp", "x_seq_grp", "x_nbr_seq_grp")]
    md = ["# v3 test results (v2 labels, node features from all window flows)", "",
          "Thresholds tuned on validation F1. held-out = scenarios never used in training.", "",
          "| Model | F1 | PR-AUC | FN | FP | held-out F1 | held-out PR-AUC | ctu13-10 F1 | ctu13-10 PR-AUC |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in files:
        y, pred, pr, sc = load(p)
        h = np.isin(sc, spec.holdout_test_scenarios); c = sc == "ctu13-10"
        a, fn, fp = f1(y, pred)
        md.append(f"| {p.stem} | {a:.3f} | {average_precision_score(y, pr):.3f} | {fn} | {fp} | {f1(y[h], pred[h])[0]:.3f} | "
                  f"{average_precision_score(y[h], pr[h]):.3f} | {f1(y[c], pred[c])[0]:.3f} | {average_precision_score(y[c], pr[c]):.3f} |")
    (T3 / "summary_v3.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))


if __name__ == "__main__":
    main()
