"""Unit tests for src/training/evaluate.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.training.evaluate import EvalResult, evaluate


def test_evaluate_perfect_predictions() -> None:
    y = np.array([0, 0, 1, 1, 1, 0, 1])
    res = evaluate(y, y, y.astype(float))
    assert res.accuracy == 1.0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0
    assert res.pr_auc == 1.0
    assert res.roc_auc == 1.0
    assert res.confusion == [[3, 0], [0, 4]]


def test_evaluate_all_negative_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    res = evaluate(y_true, y_pred)
    # No positive predictions ⇒ precision and recall are 0; f1 = 0
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1 == 0.0
    # tn=2, fp=0, fn=2, tp=0
    assert res.confusion == [[2, 0], [2, 0]]


def test_evaluate_single_class_roc_is_nan() -> None:
    """ROC-AUC is undefined when only one class is present."""
    y = np.zeros(10, dtype=int)
    p = np.zeros(10, dtype=int)
    res = evaluate(y, p, p.astype(float))
    assert math.isnan(res.roc_auc)


def test_evaluate_per_scenario_breakdown() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.9, 0.4, 0.3, 0.8, 0.95, 0.7])
    scen = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    res = evaluate(y_true, y_pred, proba, scenarios=scen)
    assert set(res.per_scenario.keys()) == {"A", "B"}
    # A: y=[0,0,1,1], p=[0,0,1,0] → F1 = 2*1*0.5/(1+0.5)= 0.6667
    assert res.per_scenario["A"]["n"] == 4
    assert res.per_scenario["A"]["n_pos"] == 2
    assert abs(res.per_scenario["A"]["f1"] - 0.6667) < 1e-3
    # B: y=[0,1,1,1], p=[0,1,1,1] → F1 = 1.0
    assert res.per_scenario["B"]["f1"] == 1.0


def test_eval_result_as_row_keys() -> None:
    y = np.array([0, 0, 1, 1])
    res = evaluate(y, y, y.astype(float))
    row = res.as_row()
    for k in ("n", "n_pos", "accuracy", "precision", "recall",
              "f1", "pr_auc", "roc_auc", "tn", "fp", "fn", "tp"):
        assert k in row, f"missing key {k}"


def test_evaluate_pretty_includes_per_scenario() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0, 1, 0, 1])
    s = np.array(["A", "A", "B", "B"])
    text = evaluate(y, p, p.astype(float), scenarios=s).pretty("test")
    assert "per-scenario:" in text
    assert "A" in text and "B" in text
