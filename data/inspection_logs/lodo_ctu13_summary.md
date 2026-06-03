# LODO-CTU13 summary

Held-out dataset: **ctu13**. All models trained from scratch on the two remaining datasets using `configs/split_lodo_ctu13.yaml`.

| Model | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| RandomForest | 0.3682 | 0.3093 | 0.3575 |
| XGBoost | 0.2803 | 0.2941 | 0.3160 |
| GAT (flat) | 0.6831 | 0.8737 | 0.9304 |
| GIN (flat) | 0.4729 | 0.5539 | 0.5695 |
| GINE (flat) | 0.4820 | 0.6363 | 0.6219 |
| T-GINE + skip | 0.3868 | 0.3038 | 0.3768 |
| SAGPool (sparse) | 0.3794 | 0.3148 | 0.4164 |
| GT-edge (final) | 0.4352 | 0.3626 | 0.4654 |
