# LODO results — prevalence-adjusted (FP32-corrected)

F1@0.5 is confounded by prevalence shift across directions (CTU-13 ~0.36 positive, IoT-23 ~0.74, MedBIoT ~0.99). Read **MCC**, **balanced accuracy**, and **PR-AUC lift** (PR-AUC minus prevalence) as the prior-robust metrics. *F1 vs no-skill* is F1 minus the predict-all-bot baseline; negative means the model scores below trivially predicting the majority class.

## LODO-CTU13  (test prevalence = 0.368, no-skill F1 = 0.538, no-skill MCC/BalAcc/ROC = 0 / 0.5 / 0.5)

| Model | F1@0.5 | F1 vs no-skill | Bal.Acc | MCC | PR-AUC | PR-AUC lift | ROC-AUC |
|---|---|---|---|---|---|---|---|
| RandomForest | 0.368 | -0.170 | 0.415 | -0.167 | 0.309 | -0.059 | 0.357 |
| XGBoost | 0.280 | -0.258 | 0.333 | -0.324 | 0.294 | -0.074 | 0.316 |
| GAT (flat) | 0.683 | 0.145 | 0.756 | 0.576 | 0.874 | 0.506 | 0.930 |
| GIN (flat) | 0.473 | -0.065 | 0.570 | 0.138 | 0.554 | 0.186 | 0.570 |
| GINE (flat) | 0.482 | -0.056 | 0.459 | -0.109 | 0.636 | 0.268 | 0.622 |
| T-GINE + skip | 0.387 | -0.151 | 0.408 | -0.188 | 0.304 | -0.064 | 0.377 |
| SAGPool (sparse) | 0.380 | -0.158 | 0.467 | -0.064 | 0.315 | -0.053 | 0.416 |
| GT-edge (final) | 0.439 ± 0.036 | -0.099 | 0.505 | 0.010 | 0.363 | -0.005 | 0.465 |

## LODO-IOT23  (test prevalence = 0.742, no-skill F1 = 0.852, no-skill MCC/BalAcc/ROC = 0 / 0.5 / 0.5)

| Model | F1@0.5 | F1 vs no-skill | Bal.Acc | MCC | PR-AUC | PR-AUC lift | ROC-AUC |
|---|---|---|---|---|---|---|---|
| RandomForest | 0.850 | -0.002 | 0.501 | 0.015 | 0.766 | 0.025 | 0.559 |
| XGBoost | 0.852 | 0.000 | 0.503 | 0.048 | 0.775 | 0.034 | 0.265 |
| GAT (flat) | 0.847 | -0.004 | 0.507 | 0.043 | 0.605 | -0.137 | 0.278 |
| GIN (flat) | 0.849 | -0.002 | 0.503 | 0.024 | 0.686 | -0.056 | 0.401 |
| GINE (flat) | 0.852 | 0.000 | 0.508 | 0.077 | 0.592 | -0.149 | 0.238 |
| T-GINE + skip | 0.852 | 0.001 | 0.503 | 0.062 | 0.734 | -0.008 | 0.542 |
| SAGPool (sparse) | 0.853 | 0.001 | 0.505 | 0.084 | 0.837 | 0.095 | 0.712 |
| GT-edge (final) | 0.850 ± 0.001 | -0.002 | 0.501 | 0.014 | 0.849 | 0.108 | 0.729 |

## LODO-MEDBIOT  (test prevalence = 0.987, no-skill F1 = 0.994, no-skill MCC/BalAcc/ROC = 0 / 0.5 / 0.5)

| Model | F1@0.5 | F1 vs no-skill | Bal.Acc | MCC | PR-AUC | PR-AUC lift | ROC-AUC |
|---|---|---|---|---|---|---|---|
| RandomForest | 0.856 | -0.138 | 0.874 | 0.191 | 1.000 | 0.013 | 1.000 |
| XGBoost | 0.907 | -0.086 | 0.915 | 0.242 | 1.000 | 0.013 | 0.998 |
| GAT (flat) | 0.851 | -0.142 | 0.871 | 0.188 | 1.000 | 0.013 | 1.000 |
| GIN (flat) | 0.824 | -0.170 | 0.850 | 0.170 | 0.997 | 0.010 | 0.776 |
| GINE (flat) | 0.948 | -0.045 | 0.951 | 0.323 | 1.000 | 0.012 | 0.975 |
| T-GINE + skip | 0.957 | -0.037 | 0.959 | 0.352 | 1.000 | 0.012 | 0.976 |
| SAGPool (sparse) | 0.915 | -0.079 | 0.803 | 0.184 | 0.998 | 0.011 | 0.886 |
| GT-edge (final) | 0.908 ± 0.056 | -0.085 | 0.534 | 0.021 | 0.997 | 0.009 | 0.762 |

## GT-edge (final model) across directions

| Metric | LODO-CTU13 | LODO-IoT23 | LODO-MedBIoT |
|---|---|---|---|
| Test prevalence | 0.368 | 0.742 | 0.987 |
| F1@0.5 | 0.439 | 0.850 | 0.908 |
| No-skill F1 (all-bot) | 0.538 | 0.852 | 0.994 |
| F1 vs no-skill | -0.099 | -0.002 | -0.085 |
| Balanced accuracy | 0.505 | 0.501 | 0.534 |
| MCC | 0.010 | 0.014 | 0.021 |
| PR-AUC | 0.363 | 0.849 | 0.997 |
| PR-AUC lift | -0.005 | 0.108 | 0.009 |
| ROC-AUC | 0.465 | 0.729 | 0.762 |

## Reading

- **CTU-13 (enterprise, held out of an IoT-only training pool):** genuine failure — MCC ~0, ROC-AUC ~0.5, PR-AUC at the prevalence line. The model has no discriminative signal in this direction.
- **IoT-23:** F1@0.5 sits on the no-skill line, so F1 is uninformative here; the real evidence of transfer is the positive **PR-AUC lift** and **MCC > 0**.
- **MedBIoT:** high-prevalence regime. F1@0.5 looks high but is *below* the all-bot baseline (negative F1-vs-no-skill); PR-AUC lift is small. Use this direction only to show the FP32 result did not collapse, not as strong transfer.
