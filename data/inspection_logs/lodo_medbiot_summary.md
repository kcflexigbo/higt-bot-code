# LODO-MEDBIOT summary

Held-out dataset: **medbiot**. All models trained from scratch on the two remaining datasets using `configs/split_lodo_medbiot.yaml`.

| Model | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| RandomForest | 0.8558 | 1.0000 | 1.0000 |
| XGBoost | 0.9074 | 1.0000 | 0.9979 |
| GAT (flat) | 0.8514 | 1.0000 | 1.0000 |
| GIN (flat) | 0.8239 | 0.9968 | 0.7763 |
| GINE (flat) | 0.9482 | 0.9997 | 0.9747 |
| T-GINE + skip | 0.0868 | -- | -- |
| SAGPool (sparse) | 0.0000 | -- | -- |
| GT-edge (final) | 0.0000 | -- | -- |
