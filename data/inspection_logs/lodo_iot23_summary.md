# LODO-IOT23 summary

Held-out dataset: **iot23**. All models trained from scratch on the two remaining datasets using `configs/split_lodo_iot23.yaml`.

| Model | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| RandomForest | 0.8499 | 0.7663 | 0.5587 |
| XGBoost | 0.8519 | 0.7754 | 0.2650 |
| GAT (flat) | 0.8474 | 0.6047 | 0.2779 |
| GIN (flat) | 0.8494 | 0.6858 | 0.4008 |
| GINE (flat) | 0.8522 | 0.5923 | 0.2378 |
| T-GINE + skip | 0.8524 | 0.7341 | 0.5420 |
| SAGPool (sparse) | 0.8530 | 0.8367 | 0.7119 |
| GT-edge (final) | 0.8500 | 0.8493 | 0.7285 |
