"""Phase 4 baseline (optional in the plan, free here): XGBoost on flat node features.

Same plumbing as the RF baseline. XGBoost often slightly beats RF on tabular
features, so this gives a tighter "non-graph" upper bound for the GNNs to clear.

Usage:
    uv run python scripts/baselines/run_xgb.py
    uv run python scripts/baselines/run_xgb.py --n-estimators 1000 --max-depth 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from src.data.dataset import SplitSpec, load_split
from src.training.evaluate import evaluate


def pool_nodes(graphs):
    X_parts, y_parts, scen_parts = [], [], []
    for g in graphs:
        X_parts.append(g.x.cpu().numpy())
        y_parts.append(g.y.cpu().numpy())
        scen_parts.append(np.full(int(g.num_nodes), g.scenario, dtype=object))
    return (
        np.vstack(X_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.int64),
        np.concatenate(scen_parts),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.1, dest="learning_rate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/inspection_logs/baseline_xgb.json"))
    args = ap.parse_args()

    spec = SplitSpec.load()
    print(f"Loading split v{spec.version} ...")
    tr = load_split("train", spec)
    va = load_split("val", spec)
    te = load_split("test", spec)
    print(f"  train graphs={len(tr)}  val graphs={len(va)}  test graphs={len(te)}")

    X_tr, y_tr, _ = pool_nodes(tr)
    X_va, y_va, _ = pool_nodes(va)
    X_te, y_te, scen_te = pool_nodes(te)
    print(f"  nodes:  train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")
    print(f"  bot frac train={y_tr.mean():.4f}  val={y_va.mean():.4f}  test={y_te.mean():.4f}")

    # scale_pos_weight balances the loss when positives are rare. Here the
    # data is positive-heavy (~70%), so this number is < 1 and de-emphasizes
    # the positive class slightly.
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"  scale_pos_weight = {spw:.4f}")

    print(f"\nFitting XGBClassifier(n={args.n_estimators}, depth={args.max_depth}, "
          f"lr={args.learning_rate}, early_stopping_rounds=20) ...")
    t0 = time.perf_counter()
    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=20,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    fit_s = time.perf_counter() - t0
    best_iter = clf.best_iteration if hasattr(clf, "best_iteration") else None
    print(f"  fit in {fit_s:.1f}s, best_iteration={best_iter}")

    p_va = clf.predict(X_va); pp_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict(X_te); pp_te = clf.predict_proba(X_te)[:, 1]

    val_res = evaluate(y_va, p_va, pp_va)
    test_res = evaluate(y_te, p_te, pp_te, scenarios=scen_te)

    print(); print(val_res.pretty("val"))
    print(); print(test_res.pretty("test"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "XGBoost",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "scale_pos_weight": spw,
        "best_iteration": best_iter,
        "seed": args.seed,
        "fit_seconds": fit_s,
        "val": val_res.as_row(),
        "test": test_res.as_row(),
        "test_per_scenario": test_res.per_scenario,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
