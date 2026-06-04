# LODO comparison table

Test F1 (mean +/- std across seeds where applicable). Mixed-pool column from existing `phase{4..7}_*.json`.

| Model | LODO-CTU13 | LODO-IoT23 | LODO-MedBIoT |
|---|---|---|---|
| RandomForest | 0.3682 | 0.8499 | 0.8558 |
| XGBoost | 0.2803 | 0.8519 | 0.9074 |
| GAT (flat) | 0.6831 | 0.8474 | 0.8514 |
| GIN (flat) | 0.4729 | 0.8494 | 0.8239 |
| GINE (flat) | 0.4820 | 0.8522 | 0.9482 |
| T-GINE + skip | 0.3871 | 0.8524 | 0.9570 |
| SAGPool (sparse) | 0.3795 | 0.8530 | 0.9150 |
| GT-edge (final) | 0.4353 +/- 0.0365 | 0.8500 +/- 0.0009 | 0.9064 +/- 0.0558 |
