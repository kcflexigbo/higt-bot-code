"""Phase 8.6 — Statistical significance of the final model against its ablations and baselines.

Reads per-node test predictions written by
  scripts/reeval_mixed_pool_fp32.py --save-preds-dir data/inspection_logs/predictions
  scripts/baselines/run_gnn.py --model gine --save-preds .../gine.npz
  scripts/baselines/run_rf.py --save-preds .../rf.npz
and reports, for the reference model (final HiGT-Bot, GT-edge) against each comparator:

* exact McNemar test on all test nodes and on true-bot nodes only (missed-bot comparison);
* window-level cluster bootstrap (resampling whole test graphs, since hosts in one
  window are not independent) of the difference in F1, PR-AUC and false negatives,
  with percentile 95% confidence intervals.

Read-only with respect to stored results. Output:
data/inspection_logs/phase8_significance.{md,json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score

from src.data.dataset import SplitSpec, load_split

INSPECTION = Path("data/inspection_logs")
LABELS = {
    "higt_bot_edge": "HiGT-Bot (GT-edge, seed 42)",
    "higt_bot_edge_seed1": "HiGT-Bot (GT-edge, seed 1)",
    "higt_bot_edge_seed2": "HiGT-Bot (GT-edge, seed 2)",
    "sagpool": "SAGPool, no GT (Phase 6.4)",
    "t_gine_skip": "T-GINE + skip, no hierarchy (Phase 5)",
    "gine": "GINE, flat (Phase 4, rerun)",
    "rf": "Random Forest (Phase 4, rerun)",
}


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def _weighted_stats(y, pred, proba, w):
    tp = float(np.sum(w * ((pred == 1) & (y == 1))))
    fp = float(np.sum(w * ((pred == 1) & (y == 0))))
    fn = float(np.sum(w * ((pred == 0) & (y == 1))))
    return _f1(tp, fp, fn), average_precision_score(y, proba, sample_weight=w), fn


def _mcnemar(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))  # only A right
    c = int(np.sum(~correct_a & correct_b))  # only B right
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"only_ref_correct": b, "only_other_correct": c, "p_value": p}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds-dir", type=Path, default=INSPECTION / "predictions")
    ap.add_argument("--ref", default="higt_bot_edge")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-config", type=Path, default=Path("configs/split.yaml"))
    args = ap.parse_args()

    runs = {}
    for name in LABELS:
        f = args.preds_dir / f"{name}.npz"
        if f.exists():
            d = np.load(f, allow_pickle=False)
            runs[name] = {k: d[k] for k in ("y", "pred", "proba", "scenario")}
    if args.ref not in runs:
        raise SystemExit(f"reference predictions missing: {args.ref}")
    ref = runs[args.ref]
    for name, r in runs.items():  # all models must score the same nodes in the same order
        if not (np.array_equal(r["y"], ref["y"]) and np.array_equal(r["scenario"], ref["scenario"])):
            raise SystemExit(f"node order mismatch between {name} and {args.ref}")
    y = ref["y"].astype(int)

    # graph id per node, in the canonical test order shared by every loader above
    te = load_split("test", SplitSpec.load(args.split_config))
    gid = np.concatenate([np.full(int(g.num_nodes), i) for i, g in enumerate(te)])
    if len(gid) != len(y):
        raise SystemExit(f"graph-id length {len(gid)} != prediction length {len(y)}")
    n_graphs = len(te)

    rng = np.random.default_rng(args.seed)
    boot_w = []
    for _ in range(args.n_boot):
        counts = np.bincount(rng.integers(0, n_graphs, n_graphs), minlength=n_graphs)
        boot_w.append(counts[gid].astype(float))

    point, boots = {}, {}
    for name, r in runs.items():
        point[name] = _weighted_stats(y, r["pred"], r["proba"], np.ones_like(y, float))
        boots[name] = np.array([_weighted_stats(y, r["pred"], r["proba"], w) for w in boot_w])
        print(f"bootstrapped {name}", flush=True)

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    out = {"ref": args.ref, "n_nodes": int(len(y)), "n_graphs": int(n_graphs), "n_boot": args.n_boot, "models": {}, "vs_ref": {}}
    for name in runs:
        f1, ap_, fn = point[name]
        out["models"][name] = {"f1": f1, "f1_ci": ci(boots[name][:, 0]), "pr_auc": ap_,
                               "pr_auc_ci": ci(boots[name][:, 1]), "fn": int(fn), "fn_ci": ci(boots[name][:, 2])}
    ref_ok = ref["pred"] == y
    bot = y == 1
    for name, r in runs.items():
        if name == args.ref:
            continue
        ok = r["pred"] == y
        d = boots[args.ref] - boots[name]
        out["vs_ref"][name] = {
            "mcnemar_all": _mcnemar(ref_ok, ok),
            "mcnemar_bots": _mcnemar(ref_ok[bot], ok[bot]),
            "delta_f1": point[args.ref][0] - point[name][0], "delta_f1_ci": ci(d[:, 0]),
            "delta_pr_auc": point[args.ref][1] - point[name][1], "delta_pr_auc_ci": ci(d[:, 1]),
            "delta_fn": point[args.ref][2] - point[name][2], "delta_fn_ci": ci(d[:, 2]),
        }

    (INSPECTION / "phase8_significance.json").write_text(json.dumps(out, indent=2))

    L = [f"# Phase 8.6 — Significance of {LABELS[args.ref]}", "",
         f"Test set: {out['n_nodes']:,} host-window nodes in {n_graphs:,} windows. "
         f"Bootstrap: {args.n_boot} resamples of whole windows (cluster bootstrap), percentile 95% CIs. "
         "McNemar: exact two-sided binomial test on discordant nodes.", "",
         "## Per-model metrics (95% CI)", "",
         "| Model | F1 | PR-AUC | FN |", "|---|---|---|---|"]
    for name, m in out["models"].items():
        L.append(f"| {LABELS[name]} | {m['f1']:.4f} [{m['f1_ci'][0]:.4f}, {m['f1_ci'][1]:.4f}] | "
                 f"{m['pr_auc']:.4f} [{m['pr_auc_ci'][0]:.4f}, {m['pr_auc_ci'][1]:.4f}] | "
                 f"{m['fn']} [{m['fn_ci'][0]:.0f}, {m['fn_ci'][1]:.0f}] |")
    L += ["", f"## {LABELS[args.ref]} minus comparator", "",
          "| Comparator | ΔF1 [95% CI] | ΔPR-AUC [95% CI] | ΔFN [95% CI] | McNemar p (all) | McNemar p (bots only; ref-only / other-only detections) |",
          "|---|---|---|---|---|---|"]
    for name, v in out["vs_ref"].items():
        mb = v["mcnemar_bots"]
        L.append(f"| {LABELS[name]} | {v['delta_f1']:+.4f} [{v['delta_f1_ci'][0]:+.4f}, {v['delta_f1_ci'][1]:+.4f}] | "
                 f"{v['delta_pr_auc']:+.4f} [{v['delta_pr_auc_ci'][0]:+.4f}, {v['delta_pr_auc_ci'][1]:+.4f}] | "
                 f"{v['delta_fn']:+.0f} [{v['delta_fn_ci'][0]:+.0f}, {v['delta_fn_ci'][1]:+.0f}] | "
                 f"{v['mcnemar_all']['p_value']:.2e} | {mb['p_value']:.2e} ({mb['only_ref_correct']} / {mb['only_other_correct']}) |")
    (INSPECTION / "phase8_significance.md").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
