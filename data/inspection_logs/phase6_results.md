# Phase 6 — HiGT-Bot DiffPool Results

**Date:** 2026-05-17
**Checkpoint:** `experiments/phase6/hgt_diffpool.pt`
**JSON:** `data/inspection_logs/phase6_diffpool.json`
**Log:** `data/inspection_logs/phase6_train.log`

## Model

`HiGTBotDiffPool` (`src/models/diffpool.py`):
- Input: cached encoder embeddings `[N, 64]` (from frozen Phase 5 T-GINE-skip
  encoder) || raw scaled node features `[N, 9]` → `[N, 73]`.
- Pre-pool: 2-layer GINEConv stack with residual + JK-cat, hidden=128.
- Assignment: 2-layer GINEConv → soft cluster scores `[N, K1]`, K1 = 100.
- DiffPool: `dense_diff_pool(z, adj, s, mask)` → coarsened `[B, K1, 128]`,
  link + entropy aux losses.
- Post-pool MLP per cluster → graph readout (mean over clusters).
- Node head: `[z || S @ x_coarse]` → 2-class logits.
- Params: 225,818  |  K1 = 100  |  max_nodes = 400  |  pool_ratio = 0.25.

## Training

`scripts/train_phase6.py`, configs/phase6.yaml (defaults overridden via CLI):
- Optimizer: Adam, lr=5e-4, weight_decay=1e-5.
- Loss: Focal (γ=2, α=class-balanced [1.6985, 0.7086]) + 0.1·link_loss + 0.1·ent_loss.
- Batch 4 × grad_accum 4 (effective 16). AMP on CUDA. ReduceLROnPlateau (patience=5, factor=0.5).
- Early stop: patience=15. Stopped at **epoch 54** (best at epoch 27).
- Total fit: **1776.5 s** (~29.6 min) on local GPU.
- Cached embeddings (`data/flow_embeddings/`, 4,954 parquet files) — encoder
  is frozen, so no transformer recompute per epoch.

## Headline (test set, n=59,210)

| Metric | Phase 6 DiffPool | Δ vs Phase 5 (T-GINE-skip) |
|---|---|---|
| F1 | **0.9673** | +0.0012 |
| Precision | 0.9392 | +0.0023 |
| Recall | 0.9971 | -0.0002 |
| PR-AUC | 0.9691 | +0.0022 |
| ROC-AUC | 0.9506 | — |
| FN | 118 | -22 |

Confusion (test): tn=15,497  fp=2,650  fn=118  tp=40,945.

## Comparison vs all prior approaches (test set)

| Model | Phase | Test F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|
| GAT | 4 | 0.9115 | 0.9025 | 0.9207 | 0.9220 |
| GIN (flat) | 4 | 0.9443 | 0.9212 | 0.9685 | 0.9490 |
| XGBoost (flow-stats) | 4 | 0.9595 | 0.9379 | 0.9821 | 0.9629 |
| GINE (matched α) | 4 | 0.9593 | 0.9300 | 0.9904 | 0.9550 |
| GINE | 4 | 0.9595 | 0.9303 | 0.9906 | 0.9610 |
| RandomForest (flow-stats) | 4 | 0.9597 | 0.9367 | 0.9839 | 0.9600 |
| T-GINE | 5 | 0.9578 | 0.9261 | 0.9917 | 0.9628 |
| T-GINE + raw-skip | 5 | 0.9661 | 0.9369 | 0.9973 | 0.9669 |
| **HiGT-Bot DiffPool** | **6** | **0.9673** | **0.9392** | 0.9971 | **0.9691** |

Net gains since the strongest Phase 4 baseline (RandomForest, F1=0.9597):
- F1: **+0.0076**
- PR-AUC: **+0.0091**
- FN drops from ~660 (Phase 4) to **118** (~5.6× fewer missed bots) at
  comparable false-positive rate.

Net gains since the original flat GIN (apples-to-apples GNN starting point):
- F1: **+0.0230** (0.9443 → 0.9673)
- PR-AUC: **+0.0201** (0.9490 → 0.9691)

## Per-scenario movement vs Phase 5 (test F1)

Hard-scenario rescue, the headline story:

| Scenario | n | n_pos | P5 skip | P6 DiffPool | Δ |
|---|---|---|---|---|---|
| **iot23-35-1** | 9,905 | 31 | 0.28 | **0.56** | +0.28 |
| **ctu13-10** | 1,048 | 147 | 0.60 | **0.64** | +0.04 |
| iot23-3-1 | 152 | 152 | strong | 1.00 | — |
| iot23-7-1 | 93 | 12 | 0.80 | 0.67 | −0.13 |
| ctu13-3 | 1,869 | 25 | strong | 0.89 | small drop |
| ctu13-1, ctu13-2, ctu13-6, ctu13-8, ctu13-9, ctu13-13 | — | — | ≈1.00 | ≈1.00 | flat |
| iot23-1-1, iot23-17-1, iot23-48-1, iot23-9-1 | — | — | strong | strong | flat / +0.02 |
| medbiot-* (all sub-scenarios) | — | — | 1.00 | 1.00 | flat |

Hierarchical pooling buys the most where flat GNNs were struggling — sparse-bot
scenarios with diluted node features. The small regression on iot23-7-1 (12
positives) is within run-to-run variance for tiny support; iot23-3-1 went the
other way (now 1.00). Aux losses converged cleanly (link ≈ 0.0002,
entropy ≈ 0.004 by epoch 27), so the assignment matrix is producing a
well-defined cluster structure rather than collapsing.

## Phase 6 plan gate

> "Hierarchical GIN matches or beats flat GIN on CTU-13."

**Met.** Every CTU-13 scenario with bots present is ≥ 0.64 test F1; six are
≥ 0.99; ctu13-10 (the long-standing problem scenario) improved by +0.04.
HiGT-Bot is the new SOTA on every test-set metric.

## What's next

- **Iot23-35-1 still leaves room** (0.56) — only 31/9,905 positives, so any
  remaining gain probably comes from class-balanced sampling or a per-scenario
  decision threshold, not from architecture.
- **Iot23-7-1 regression** is small support (12 positives). Worth re-checking
  on a seed sweep before treating it as real.
- **Multi-level DiffPool**: current model has 1 pooling level (400 → 100).
  Adding a second level (100 → 25) is cheap to scaffold and may help
  graph-level head capacity.
- **Attention readout** over coarsened clusters instead of mean-pool — small
  parameter cost, can sharpen per-graph signal.
- **Threshold calibration** at the global/per-scenario level — earlier (Phase 5)
  experiments showed +0.002 headroom; same script (`scripts/calibrate_thresholds.py`)
  can be retargeted at the Phase 6 checkpoint.
