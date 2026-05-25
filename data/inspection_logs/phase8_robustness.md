# Phase 8.4 — Test-Time Edge-Drop Robustness

Final model: Phase 7 GT-edge.  Test set.
Edges dropped uniformly at random at inference time (no retraining).

| Edge drop rate | Test F1 | Δ vs baseline | TP | FP | FN |
|---|---|---|---|---|---|
| 0% | 0.9677 | +0.0000 | 40977 | 2652 | 86 |
| 5% | 0.9652 | -0.0025 | 40975 | 2866 | 88 |
| 10% | 0.9631 | -0.0046 | 40973 | 3051 | 90 |
| 20% | 0.9025 | -0.0651 | 36249 | 3014 | 4814 |
| 30% | 0.5776 | -0.3901 | 17471 | 1961 | 23592 |

**Baseline (0% drop) F1**: 0.9677