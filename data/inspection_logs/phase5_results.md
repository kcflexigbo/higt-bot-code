# Phase 5 Results — TemporalGINE with Raw-Feature Skip

Recorded 2026-05-16. All metrics on the canonical chronological split (train=2796 / val=926 / test=1232 windows).

## Headline test-set comparison

| Model | Val F1 | Test F1 | Test P | Test R | FP | FN | params | fit (s) |
|---|---|---|---|---|---|---|---|---|
| RandomForest | 0.961 | 0.9597 | 0.937 | 0.984 | 2,730 | 663 | — | 7 |
| XGBoost | 0.960 | 0.9595 | 0.938 | 0.982 | 2,672 | 733 | — | 2 |
| GIN (plain, no edge feats) | 0.950 | 0.9443 | 0.921 | 0.968 | 3,400 | 1,294 | — | 105 |
| GAT | 0.906 | 0.9115 | 0.902 | 0.921 | 4,086 | 3,255 | — | 33 |
| GINE (2L, DropEdge=0.2) | 0.963 | 0.9595 | 0.930 | 0.991 | 3,047 | 384 | 26,628 | 106 |
| GINE-matched (3L, no DropEdge) | 0.963 | 0.9593 | 0.930 | 0.990 | 3,061 | 393 | 39,173 | 97 |
| T-GINE (original) | 0.966 | 0.9578 | 0.926 | 0.992 | 3,248 | 342 | 143,749 | 2,143 |
| **T-GINE + raw skip** | **0.968** | **0.9661** | **0.937** | **0.997** | **2,759** | **111** | 144,325 | 1,612 |
| T-GINE + raw skip + per-scenario τ | — | 0.9664 | 0.937 | 0.998 | — | — | — | — |

T-GINE + raw skip beats every prior baseline by ~+0.006 test F1 and cuts FN 3× vs the original T-GINE.

## Per-scenario test F1 (selected — regressions fixed)

| Scenario | n / pos | RF | Old T-GINE | **+raw skip** | Δ from old |
|---|---|---|---|---|---|
| ctu13-10 | 1048/147 | 0.72 | 0.34 | **0.60** | +0.26 |
| iot23-35-1 | 9905/31 | 0.63 | 0.06 | **0.28** | +0.22 |
| iot23-7-1 | 93/12 | 0.96 | 0.17 | **0.80** | +0.63 |
| iot23-9-1 | 153/21 | n/a | 0.37 | **0.95** | +0.58 |
| ctu13-3 | 1869/25 | 0.13 | 0.62 | **0.83** | +0.21 |
| iot23-33-1 | 142/120 | 0.98 | 0.99 | 0.91 | −0.08 |

All other scenarios in 0.95–1.00 range (no regressions of note).

## Methodology notes

- **Apples-to-apples gate.** GINE baseline already used identical focal-α (1.6985, 0.7086) via the `class_weight` fallback in `loop.py:147`. The "matched" GINE row uses 3 layers (T-GINE's GIN depth) and DropEdge=0 to isolate the temporal encoder's contribution. Encoder alone: +0.0026 val F1, −0.0015 test F1 vs matched GINE — noise. Encoder + raw skip: +0.0050 val F1, +0.0068 test F1 — real.
- **Raw skip implementation.** `TemporalGINE.forward` now does `torch.cat([encoder_out, batch.x], dim=-1)` before feeding GIN. GIN's `in_dim` grew 64 → 73. Param overhead: +576 (negligible).
- **Node feature scaling** added to `FlowSeqGraphDataset` (was previously only edge-scaled). Same mean/std-by-train as the Phase 4 baselines.
- **Threshold calibration.** Per-scenario τ tuned on val gives +0.0003 on the raw-skip model (was +0.002 on the original). Most of the headroom is already absorbed by the architecture. Recommend reporting argmax (τ=0.5) for simplicity.

## Outputs

- Checkpoint: `experiments/phase5/temporal_gine_skip.pt`
- Metrics JSON: `data/inspection_logs/phase5_temporal_gine_skip.json`
- Training log: `data/inspection_logs/phase5_train_skip.log`
- Calibration: `data/inspection_logs/phase5_skip_calibrated.json`

## Open issues for Phase 6+

1. **iot23-35-1 still at F1=0.28** (31 positives in 9,905). Extreme prevalence: ~1:319. RF reaches 0.63 because its decision rule stays conservative at low prevalence. Candidate fixes:
   - BAT topological augmentation (Liu et al., ICML 2024) — plug-and-play with focal loss.
   - GraphSMOTE / GraphENS minority oversampling in embedding space.
   - Recall-precision-balanced threshold scoped to ultra-imbalanced scenarios.
2. **ctu13-10 at 0.60** (RF: 0.72). Mean 18 nodes/window — message passing has too little structure. Phase 6 DiffPool should help here by reducing the effective node count to clusters.
3. **Variance across runs not measured.** Single seed (42). Worth running 3–5 seeds before final Phase 7 write-up.
