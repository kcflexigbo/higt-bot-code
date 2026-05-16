"""Shared evaluation function for Phase 4+ baselines.

Every model — RF, GAT, GIN, the final HiGT-Bot — calls `evaluate()` so the
results table is directly comparable. Includes overall metrics and a
per-scenario breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    confusion: list[list[int]]   # [[tn, fp], [fn, tp]]
    per_scenario: dict[str, dict[str, float]]
    n: int
    n_pos: int

    def as_row(self) -> dict[str, float | int]:
        """Flat dict suitable for W&B / a results table."""
        return {
            "n": self.n,
            "n_pos": self.n_pos,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "tn": self.confusion[0][0],
            "fp": self.confusion[0][1],
            "fn": self.confusion[1][0],
            "tp": self.confusion[1][1],
        }

    def pretty(self, title: str = "results") -> str:
        head = (f"=== {title} ===\n"
                f"  n         {self.n:>8,}    n_pos {self.n_pos:>8,}\n"
                f"  accuracy  {self.accuracy:.4f}\n"
                f"  precision {self.precision:.4f}\n"
                f"  recall    {self.recall:.4f}\n"
                f"  F1        {self.f1:.4f}\n"
                f"  PR-AUC    {self.pr_auc:.4f}\n"
                f"  ROC-AUC   {self.roc_auc:.4f}\n"
                f"  confusion [[tn={self.confusion[0][0]} fp={self.confusion[0][1]}]\n"
                f"             [fn={self.confusion[1][0]} tp={self.confusion[1][1]}]]")
        if not self.per_scenario:
            return head
        rows = ["", "per-scenario:"]
        rows.append(f"  {'scenario':<32s}  {'n':>6s}  {'pos':>5s}  {'F1':>6s}  {'PR-AUC':>7s}")
        for sc, m in sorted(self.per_scenario.items()):
            rows.append(
                f"  {sc:<32s}  {m['n']:>6,.0f}  {m['n_pos']:>5,.0f}  "
                f"{m['f1']:>6.4f}  {m['pr_auc']:>7.4f}"
            )
        return head + "\n" + "\n".join(rows)


def _safe_roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(average_precision_score(y, p))
    except ValueError:
        return float("nan")


def evaluate(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_proba: Sequence[float] | np.ndarray | None = None,
    scenarios: Sequence[str] | None = None,
) -> EvalResult:
    """Compute the canonical metrics bundle.

    Args:
        y_true: binary node labels (0/1)
        y_pred: binary predictions (0/1)
        y_proba: P(class=1). Required for PR-AUC / ROC-AUC.
        scenarios: per-sample scenario tags. If given, breakdown is computed.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    proba = (np.asarray(y_proba) if y_proba is not None else y_pred.astype(float))

    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    per_scenario: dict[str, dict[str, float]] = {}
    if scenarios is not None:
        scen = np.asarray(scenarios)
        for sc in np.unique(scen):
            mask = scen == sc
            if mask.sum() == 0:
                continue
            yt, yp, pp = y_true[mask], y_pred[mask], proba[mask]
            per_scenario[str(sc)] = {
                "n": float(mask.sum()),
                "n_pos": float(yt.sum()),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "pr_auc": _safe_pr_auc(yt, pp) if yt.sum() else float("nan"),
            }

    return EvalResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        pr_auc=_safe_pr_auc(y_true, proba),
        roc_auc=_safe_roc_auc(y_true, proba),
        confusion=cm,
        per_scenario=per_scenario,
        n=int(len(y_true)),
        n_pos=int(y_true.sum()),
    )
