# Phase 8.2 — Final Ablation Table

Auto-compiled from `data/inspection_logs/phase{4-7}_*.json`.
All metrics on the held-out test set (n = 59,210; n_pos = 41,063).

## Phase 4 baseline

| Model | F1 | PR-AUC | Recall | FN | iot23-35-1 | ctu13-10 | Params |
|---|---|---|---|---|---|---|---|
| GAT (flat) | 0.9115 | 0.9220 | 0.9207 | 3255 | 0.0246 | 0.1732 | — |
| GIN (flat) | 0.9443 | 0.9490 | 0.9685 | 1294 | 0.0276 | 0.5901 | — |
| GINE (flat) | 0.9595 | 0.9610 | 0.9906 | 384 | 0.0671 | 0.6667 | — |
| GINE matched-alpha | 0.9593 | 0.9550 | 0.9904 | 393 | 0.0670 | 0.6288 | — |

## Phase 4 baseline (tabular)

| Model | F1 | PR-AUC | Recall | FN | iot23-35-1 | ctu13-10 | Params |
|---|---|---|---|---|---|---|---|
| RandomForest | 0.9597 | 0.9600 | 0.9839 | 663 | 0.6333 | 0.7236 | — |
| XGBoost | 0.9595 | 0.9629 | 0.9821 | 733 | 0.4706 | 0.4632 | — |

## Phase 5 (temporal encoder)

| Model | F1 | PR-AUC | Recall | FN | iot23-35-1 | ctu13-10 | Params |
|---|---|---|---|---|---|---|---|
| T-GINE | 0.9578 | 0.9628 | 0.9917 | 342 | 0.0641 | 0.3429 | 143749 |
| T-GINE + raw-skip | 0.9661 | 0.9669 | 0.9973 | 111 | 0.2778 | 0.6019 | 144325 |
| T-GINE + SSL-init | 0.9601 | 0.9646 | 0.9824 | 721 | 0.6230 | 0.5870 | 144325 |

## Phase 6 (hierarchical, dense pool)

| Model | F1 | PR-AUC | Recall | FN | iot23-35-1 | ctu13-10 | Params |
|---|---|---|---|---|---|---|---|
| HiGT-Bot DiffPool | 0.9673 | 0.9691 | 0.9971 | 118 | 0.5600 | 0.6355 | 225818 |

## Phase 6 ablations

| Model | F1 | PR-AUC | Recall | FN | iot23-35-1 | ctu13-10 | Params |
|---|---|---|---|---|---|---|---|
| DiffPool + TAM v1 | 0.9577 | 0.9647 | 0.9773 | 934 | 0.6038 | 0.5882 | 225818 |
| DiffPool + TAM v2 | 0.9647 | 0.9692 | 0.9922 | 320 | 0.1765 | 0.6007 | 225818 |
| DiffPool + GraphSHA | 0.9629 | 0.9685 | 0.9881 | 487 | 0.0625 | 0.6118 | 225818 |
| DiffPool + SSL alone | 0.9643 | 0.9619 | 0.9956 | 179 | 0.3077 | 0.6228 | 225818 |
| DiffPool + SSL→FT | 0.9664 | 0.9702 | 0.9953 | 194 | 0.2703 | 0.4724 | 225818 |
| SAGPool × SSL-FT | 0.9659 | 0.9682 | 0.9963 | 150 | 0.5882 | 0.4868 | 211335 |

## Phase 6 (hierarchical, sparse pool)

| Model | F1 | PR-AUC | Recall | FN | iot23-35-1 | ctu13-10 | Params |
|---|---|---|---|---|---|---|---|
| HiGT-Bot SAGPool | 0.9674 | 0.9710 | 0.9968 | 132 | 0.5532 | 0.5275 | 211335 |

## Phase 7 (full HiGT-Bot)

| Model | F1 | PR-AUC | Recall | FN | iot23-35-1 | ctu13-10 | Params |
|---|---|---|---|---|---|---|---|
| HiGT-Bot full (GT-edge) | 0.9677 | 0.9700 | 0.9979 | 86 | 0.5957 | 0.6957 | 1355399 |
| HiGT-Bot full (GT-global) | 0.9673 | 0.9673 | 0.9981 | 80 | 0.6250 | 0.5954 | 607879 |
| HiGT-Bot full (hybrid 2L) | 0.9634 | 0.9638 | 0.9900 | 411 | 0.7143 | 0.5519 | 981639 |
| HiGT-Bot full (hybrid 4L) | 0.9653 | 0.9693 | 0.9965 | 143 | 0.2778 | 0.5620 | 1751943 |

## Headline

- **Final model**: HiGT-Bot full (GT-edge)
- **Test F1**: 0.9677
- **PR-AUC**: 0.9700
- **Recall**: 0.9979
- **FN**: 86
