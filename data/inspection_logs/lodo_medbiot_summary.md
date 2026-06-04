# LODO-MEDBIOT summary

Held-out dataset: **medbiot**. All models trained from scratch on the two remaining datasets using `configs/split_lodo_medbiot.yaml`.

| Model | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| RandomForest | 0.8558 | 1.0000 | 1.0000 |
| XGBoost | 0.9074 | 1.0000 | 0.9979 |
| GAT (flat) | 0.8514 | 1.0000 | 1.0000 |
| GIN (flat) | 0.8239 | 0.9968 | 0.7763 |
| GINE (flat) | 0.9482 | 0.9997 | 0.9747 |
| T-GINE + skip | 0.9570 | 0.9997 | 0.9759 |
| SAGPool (sparse) | 0.9150 | 0.9985 | 0.8863 |
| GT-edge (final) | 0.9064 | 0.9965 | 0.7619 |
