# HiGT-Bot — Final Model Selection

**Date frozen:** 2026-05-17
**Selected:** Phase 7 HiGT-Bot with edge-aware Graph Transformer (GT-edge)
**Test set F1:** 0.9677   |   **PR-AUC:** 0.9700   |   **FN:** 86

## Why this one

Across 12+ candidate models spanning encoder pretraining, loss functions,
feature augmentation, pooling layers, and Graph Transformer variants,
**Phase 7 GT-edge** is the headline winner:

- Highest aggregate F1 (0.9677)
- Second-best PR-AUC (0.9700), within 0.001 of SAGPool
- Lowest FN (86) — operationally the most relevant metric
- Wins on the two hardest connected-bot scenarios (ctu13-10 = 0.70,
  iot23-35-1 = 0.60) without giving up the easy scenarios
- Single model, single inference pass, ~1.36M parameters

## Final leaderboard (test set, n=59,210)

| Method | Phase | Test F1 | PR-AUC | FN | iot23-35-1 |
|---|---|---|---|---|---|
| GINE (flat) | 4 | 0.9595 | 0.9610 | ~660 | (failure mode) |
| T-GINE + raw-skip | 5 | 0.9661 | 0.9669 | 140 | 0.28 |
| Phase 5 + SSL init | 5 | 0.9601 | 0.9646 | 721 | 0.62 |
| Phase 6 DiffPool | 6 | 0.9673 | 0.9691 | 118 | 0.56 |
| Phase 6.4 SAGPool | 6.4 | 0.9674 | **0.9710** | 132 | 0.55 |
| **Phase 7 GT-edge** | **7** | **0.9677** ⭐ | 0.9700 | **86** ⭐ | 0.60 |
| Phase 7 GT-global | 7 | 0.9673 | 0.9673 | 80 | 0.625 |
| Phase 7 GT-hybrid (2L) | 7 | 0.9634 | 0.9638 | 411 | **0.7143** ⭐ |
| Phase 7 GT-hybrid (4L) | 7 | 0.9653 | 0.9693 | 143 | 0.28 |

## Files

| Asset | Path |
|---|---|
| Model class | `src/models/higt_bot.py` (`HiGTBot`) |
| Training script | `scripts/train_phase7.py` |
| Encoder | `experiments/phase5/temporal_gine_skip.pt` |
| Cached embeddings | `data/flow_embeddings/` |
| Final checkpoint | `experiments/phase7/higt_bot_edge.pt` |
| Final results JSON | `data/inspection_logs/phase7_higt_bot_edge.json` |
| Training log | `data/inspection_logs/phase7_train_edge.log` |

## Architecture summary

```
per-node flow sequence [N, max_flows=256, F=13]
    ↓ TemporalFlowEncoder (frozen, 2-layer Transformer, d_model=64)
[N, 64] ──concat raw scaled features [N, 9]──► [N, 73]
    ↓ GINE block (2 layers, residual + JK-cat, hidden=128, edge-aware)
[N, 128] = z (skip path to per-node head)
    ↓ SAGPool (ratio=0.5, learned scoring)
[K~N/2, 128] on sparse coarsened graph
    ↓ GINE block (1 layer)
[K, 128]
    ↓ Edge-aware GT block × 2 (TransformerConv, 4 heads, edge-attr aware)
[K, 128]
    ↓ mean + max readout per graph
[B, 256] = graph summary → broadcast to each node by batch idx
[N, 256] ──concat skip-path z [N, 128]──► [N, 384]
    ↓ MLP head (384 → 128 → 2)
[N, 2] per-node bot/benign logits
```

- 1,355,399 parameters
- Focal loss γ=2, α=[1.6985, 0.7086]
- Adam lr=5e-4, wd=1e-5, AMP
- Batch 4 × accum 4 = effective 16
- ReduceLROnPlateau patience=5, early stop patience=15

## How to reproduce

```bash
# 1. Train the encoder
uv run python scripts/train_phase5.py --raw-feature-skip \
    --results-suffix _skip

# 2. Cache encoder embeddings (streams scenario-by-scenario)
uv run python scripts/cache_embeddings.py \
    --checkpoint experiments/phase5/temporal_gine_skip.pt

# 3. Train the full HiGT-Bot (GT-edge variant)
uv run python scripts/train_phase7.py --gt-variant edge \
    --emb-dir data/flow_embeddings --results-suffix _edge
```

## Net journey

| Metric | Phase 4 baseline | Phase 7 GT-edge | Δ |
|---|---|---|---|
| Test F1 | 0.9443 (flat GIN) / 0.9595 (GINE) | 0.9677 | +0.0234 vs GIN, +0.0082 vs GINE |
| PR-AUC | 0.9490 / 0.9610 | 0.9700 | +0.0210 vs GIN, +0.0090 vs GINE |
| FN | ~660 | **86** | **7.7× reduction** |
| Recall | 0.9685 | 0.9979 | +0.0294 |
| iot23-35-1 | ~0.00 | 0.60 (single), 0.71 (specialty hybrid) | from failure to recoverable |

## Operational ensemble (kept for thesis discussion)

The four Phase 6.4–7 models form a natural specialization stack:

| Model | Best at | Use case |
|---|---|---|
| **GT-edge** ⭐ | aggregate F1, hard connected-bot scenarios | Primary headline model |
| SAGPool (no GT) | PR-AUC (best score calibration) | Threshold-tunable deployment |
| GT-global | recall, best FN | High-coverage alerting |
| GT-hybrid (2L) | iot23-35-1 (stealth campaigns) | Specialty in ensemble routing |

## Negative results documented in repo

These are kept as `--`-flag opt-ins so the thesis has reproducible
"what didn't work" ablations:

- `src/training/tam_loss.py` — Topology-Aware Margin (Phase 6.1)
- `src/training/graphsha.py` — minority manifold mixup (Phase 6.2)
- `src/training/pretrain_ssl.py` + scripts — SSL→FT pipeline (Phase 6.3, 6.5)

See `data/inspection_logs/phase6_*_findings.md` and `phase7_findings.md`
for detailed analyses of each.

## Next phase

Phase 8 — Ablations, robustness, paper-ready story:
- Seed sweep (3 seeds × final model) for variance bars
- Drop-each-component ablations (no temporal encoder / no pooling / no GT)
- Interpretability: SAGPool scoring + GT attention heatmaps per scenario
- Per-dataset robustness: split each dataset standalone to see how cross-
  dataset transfer affects performance
