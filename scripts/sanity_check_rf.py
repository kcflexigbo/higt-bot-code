"""Phase 3 sanity gate: Random Forest on flat node features.

The plan: after graph construction, train a plain sklearn RandomForest on
the flat node features (ignoring graph structure) for node classification.
Expected: F1 >= 0.85 on a simple scenario. If you cannot reach that, the
*labels* are wrong — fix before moving on.

This deliberately ignores edges/structure. The point isn't to compete with
GNNs; it's to confirm that the node features alone carry signal, which
implies the labelling and feature extraction are sound.

Usage:
    uv run python scripts/sanity_check_rf.py --scenario ctu13-10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.features import NODE_FEATURE_NAMES

GRAPHS = Path("data/graphs")


def load_node_table(scenario_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Pool node features and labels across all windows of one scenario."""
    X_parts = []
    y_parts = []
    for fp in sorted(scenario_dir.glob("window_*.pt")):
        g = torch.load(fp, weights_only=False)
        X_parts.append(g.x.cpu().numpy())
        y_parts.append(g.y.cpu().numpy())
    if not X_parts:
        raise FileNotFoundError(f"No graphs under {scenario_dir}")
    return np.vstack(X_parts), np.concatenate(y_parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, help="e.g. ctu13-10")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=500)
    args = ap.parse_args()

    scenario_dir = GRAPHS / args.scenario
    X, y = load_node_table(scenario_dir)

    print(f"scenario:   {args.scenario}")
    print(f"nodes:      {len(X):,}")
    print(f"features:   {X.shape[1]} ({list(NODE_FEATURE_NAMES)})")
    print(f"label dist: bot={int(y.sum()):,}  benign={int((y == 0).sum()):,}  bot-frac={y.mean():.4f}")
    print()

    if y.sum() == 0 or y.sum() == len(y):
        print("Skipping — only one class present.")
        return

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_frac, random_state=args.seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    p_te = clf.predict(X_te)
    proba_te = clf.predict_proba(X_te)[:, 1]

    f1 = f1_score(y_te, p_te)
    prec, rec, _, _ = precision_recall_fscore_support(y_te, p_te, average="binary")
    try:
        roc = roc_auc_score(y_te, proba_te)
    except ValueError:
        roc = float("nan")

    print("=== Random Forest sanity check (test set) ===")
    print(f"  precision  {prec:.4f}")
    print(f"  recall     {rec:.4f}")
    print(f"  F1         {f1:.4f}")
    print(f"  ROC-AUC    {roc:.4f}")
    print()
    print(classification_report(y_te, p_te, target_names=["benign", "bot"], digits=4))
    print("confusion matrix [rows: true, cols: pred]")
    print(confusion_matrix(y_te, p_te))
    print()

    # Feature importances — useful diagnostic when F1 disappoints.
    imp = clf.feature_importances_
    order = np.argsort(-imp)
    print("Feature importance:")
    for k in order:
        print(f"  {NODE_FEATURE_NAMES[k]:<22s} {imp[k]:.4f}")

    gate = 0.85
    if f1 >= gate:
        print(f"\nPhase 3 RF sanity gate PASSED  (F1 {f1:.4f} >= {gate})")
    else:
        print(f"\nPhase 3 RF sanity gate FAILED  (F1 {f1:.4f} < {gate}) — labels or features may be wrong")


if __name__ == "__main__":
    main()
