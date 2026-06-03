# LODO comparison table

Test F1 (mean +/- std across seeds where applicable). Mixed-pool column from existing `phase{4..7}_*.json`.

| Model | LODO-CTU13 | LODO-IoT23 | LODO-MedBIoT |
|---|---|---|---|
| RandomForest | -- | -- | 0.8558 |
| XGBoost | -- | -- | 0.9074 |
| GAT (flat) | -- | -- | 0.8514 |
| GIN (flat) | -- | -- | 0.8239 |
| GINE (flat) | -- | -- | 0.9482 |
| T-GINE + skip | -- | -- | 0.0868 |
| SAGPool (sparse) | -- | -- | 0.0000 |
| GT-edge (final) | -- | -- | 0.0000 +/- 0.0000 |
