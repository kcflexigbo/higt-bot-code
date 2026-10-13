"""Redraw the two figures that still showed earlier runs (revision, 2026-10-13).

fig5_1_errors_v3.png: missed bots / false alarms per model on the v3 test set; tree models
  now use the five-seed runs at 8 threads (data/inspection_logs/revision/seed_preds), graph
  models the runs of scripts/make_v3_tables_figures.py (three seeds where available).
fig6_loso_heatmap.png: CTU-13 leave-one-scenario-out with the group scaler refitted per fold,
  mean over three seeds (revision/loso_refit.json): PR-AUC and F1 at the validation threshold.
fig6_bootstrap_forest_holm.png: ΔF1 of HiGT-Bot v2 over each comparator with 95% intervals from the
  seed x window bootstrap (v3/bootstrap_seeds_holm.json); green = Holm-corrected p < 0.05.

    uv run python -m scripts.make_revision_figures
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.make_v3_tables_figures import LOGS, OUT, model_rows, scores  # noqa: E402
from src.data.dataset import SplitSpec  # noqa: E402

SEEDS = (42, 1, 2, 3, 4)
SP = LOGS / "revision/seed_preds"


def tree_runs(stem, y):
    out = []
    for s in SEEDS:
        d = np.load(SP / f"{stem}_s{s}.npz", allow_pickle=True)
        pred = (d["p_test"] >= float(d["threshold"])).astype(int)
        out.append(scores(y, pred, d["p_test"]))
    return out


def fig_errors():
    M = model_rows(SplitSpec.load())
    y = np.load(LOGS / "v3/pred_x.npz", allow_pickle=True)["y"].astype(int)
    trees = {"Random forest (tuned)": "RF_tuned", "XGBoost, host features (x)": "XGB_x",
             "HiGT-Bot v2 (x + grp)": "v2_XGB_x_grp"}
    names, fn, fp, fne, fpe = [], [], [], [], []
    for name, runs in M.items():
        S = tree_runs(trees[name], y) if name in trees else [scores(*r[:3]) for r in runs]
        a = [s["fn"] for s in S]; b = [s["fp"] for s in S]
        print(f"{name:32s} n={len(S)} F1 {np.mean([s['f1'] for s in S]):.3f} FN {np.mean(a):.0f} FP {np.mean(b):.0f}")
        names.append(name.replace(", ", "\n") + f"\n({len(S)} seed{'s' if len(S) > 1 else ''})")
        fn.append(np.mean(a)); fp.append(np.mean(b))
        fne.append(np.std(a, ddof=1) if len(a) > 1 else np.nan); fpe.append(np.std(b, ddof=1) if len(b) > 1 else np.nan)
    fig, ax = plt.subplots(figsize=(10, 3.8)); xs = np.arange(len(names)); w = 0.38
    ax.bar(xs - w / 2, fn, w, yerr=fne, capsize=3, label="missed bots (FN)", color="#c0504d")
    ax.bar(xs + w / 2, fp, w, yerr=fpe, capsize=3, label="false alarms (FP)", color="#f79646")
    ax.set_xticks(xs, names, fontsize=7); ax.set_ylabel("count on the test set")
    ax.set_title("Test set: 1,607 bots among 59,210 host-windows (mean ± sd over seeds)", fontsize=9)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "fig5_1_errors_v3.png"); plt.close(fig)


def fig_loso():
    d = json.loads((LOGS / "revision/loso_refit.json").read_text())
    sc = sorted(d, key=lambda s: int(s.split("-")[1]))
    models = [("XGB[x]", "x"), ("v2 XGB[x+grp]", "x + grp (v2)")]
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.4), constrained_layout=True)
    for ax, (key, title) in zip(axes, [("pr_auc", "PR-AUC"), ("f1_val_thr", "F1 at validation threshold")]):
        A = np.array([[np.mean([d[s][seed][m][key] for seed in d[s]]) for s in sc] for m, _ in models])
        im = ax.imshow(A, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(sc)), sc, fontsize=8)
        ax.set_yticks(range(len(models)), [lab for _, lab in models], fontsize=8)
        ax.set_title(title, fontsize=9)
        for i in range(len(models)):
            for j in range(len(sc)):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7)
        print(title, {m: round(float(A[i].mean()), 3) for i, (m, _) in enumerate(models)})
    fig.colorbar(im, ax=axes, label="mean of 3 seeds", shrink=0.8)
    fig.savefig(OUT / "fig6_loso_heatmap.png"); plt.close(fig)


def fig_forest():
    d = json.loads((LOGS / "v3/bootstrap_seeds_holm.json").read_text())
    subsets = [("all", "All test windows"), ("held-out", "Held-out scenarios")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, subsets):
        C = [c for c in d[key]["comparisons"] if c["metric"] == "f1"]
        for i, c in enumerate(C):
            sig = c["p_holm"] < 0.05
            lo, hi = c["ci95"]
            ax.errorbar(c["delta"], i, xerr=[[c["delta"] - lo], [hi - c["delta"]]], fmt="o", capsize=3,
                        color="#2e7d32" if sig else "#888888")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(C)), [c["comparator"] for c in C], fontsize=8)
        ax.set_xlabel("ΔF1 (HiGT-Bot v2 − comparator)", fontsize=8)
        ax.set_title(title, fontsize=9); ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "fig6_bootstrap_forest_holm.png"); plt.close(fig)


if __name__ == "__main__":
    fig_errors()
    fig_loso()
    fig_forest()
