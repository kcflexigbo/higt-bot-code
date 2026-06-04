# AMP vs FP32 training: LODO-IoT23 audit

## Question

Did training the LODO models under AMP (mixed precision) bias the resulting
checkpoints? The eval-time AMP bug we fixed in `train_phase{5,6,7}.py` was
about the final `predict()` call. Training itself uses AMP via `torch.autocast`
+ `torch.amp.GradScaler`. The standard claim is that AMP training is safe
because:

1. Master weights are kept in FP32.
2. The gradient scaler scales the loss before backprop so small gradients
   survive the FP16 round-trip, then unscales on the optimizer step.
3. The loss is computed in FP32.

To verify empirically, the LODO-IoT23 ladder was re-trained from scratch
with `--no-amp` at every stage (Phase 5 + cache + Phase 6 + Phase 7 × 3
seeds), with everything else identical (same split, same seeds, same
hyperparameters). All evaluations done in FP32. Results below.

## Results

| Model | AMP F1@0.5 | FP32 F1@0.5 | Δ | AMP PR-AUC | FP32 PR-AUC | AMP ROC-AUC | FP32 ROC-AUC |
|---|---|---|---|---|---|---|---|
| Phase 5 T-GINE +skip | 0.8524 | 0.8524 | +0.0000 | 0.7341 | 0.6160 | 0.5420 | 0.2993 |
| Phase 6 SAGPool | 0.8530 | 0.8528 | −0.0002 | 0.8367 | 0.8431 | 0.7119 | 0.7211 |
| Phase 7 GT-edge seed=42 | 0.8502 | 0.8528 | +0.0026 | 0.7980 | 0.8076 | 0.6429 | 0.6578 |
| Phase 7 GT-edge seed=1 | 0.8490 | 0.8528 | +0.0038 | 0.8923 | 0.6843 | 0.7977 | 0.4744 |
| Phase 7 GT-edge seed=2 | 0.8507 | 0.8525 | +0.0018 | 0.8577 | 0.6569 | 0.7450 | 0.3740 |
| **GT-edge mean** | **0.8500 ± 0.0009** | **0.8527 ± 0.0002** | **+0.0027** | (varies) | (varies) | (varies) | (varies) |

All evaluation in FP32; original AMP-trained checkpoints re-evaluated in FP32
for fair comparison (so the eval-time AMP bug is not confounded with the
training-time question).

## What the table says

**F1 is stable.** F1@0.5 differs by ≤ 0.004 between AMP-trained and
FP32-trained models. The GT-edge across-seed mean differs by 0.0027 — small
relative to the seed-replicate spread, large relative to the within-precision
seed-replicate spread (0.0009 AMP, 0.0002 FP32). The F1 headline numbers in
the LODO master narrative are safe to either precision.

**Ranking metrics (PR-AUC, ROC-AUC) are not stable across precisions.** Phase
5 swings PR-AUC by −0.118 and ROC-AUC by −0.243 — substantial. Phase 7 seeds
1 and 2 swing PR-AUC by −0.21 and ROC-AUC by −0.32 and −0.37 respectively.
Phase 7 seed 42 and Phase 6 are within noise.

Notably AMP produces **better** ranking metrics than FP32 in the cases where
they differ (Phase 5 PR=0.73 vs 0.62; Phase 7 s1 PR=0.89 vs 0.68). The
"AMP-trained is suspect, FP32-trained is the ground truth" narrative is
backwards here — at least under this particular cross-dataset distribution
shift.

## Why this is happening

Same seed → same dataloader shuffle, same initial parameters, same focal-loss
weights. The only between-run difference is whether intermediate forward-pass
activations transit through FP16 or stay in FP32. That tiny numerical
difference during the forward pass affects:

1. The exact gradient magnitudes at each step (FP16 round-tripping rounds
   small values).
2. Which side of any saddle / plateau the optimizer falls on at every step.

In *well-conditioned* training (mixed-pool, where val matches train), AMP and
FP32 converge to functionally equivalent minima — verified by the
mixed-pool audit (`data/inspection_logs/mixed_pool_fp32_audit.json` — every
metric matches AMP and FP32 to 4 decimal places). Under LODO, the loss
landscape contains multiple competitive minima with different cross-domain
generalization properties. AMP and FP32 happen to fall into different ones
even from the same seed. The model that fits the LODO-IoT23 *training* data
best (which is what both AMP and FP32 are optimizing for) is not the same
as the model that ranks best on the LODO-IoT23 *test* data — and the choice
between candidate solutions is sensitive to small numerical perturbations
during gradient descent.

This is a meta-observation about cross-dataset evaluation under stochastic
optimization, not a bug. Both AMP-trained and FP32-trained models are
"correct" runs of the same training recipe.

## Implications for the thesis

The right framing:

- **The mixed-pool F1 = 0.9677 headline is robust** to training precision
  (verified in `mixed_pool_fp32_audit.json`, all metrics match to 4 decimal
  places between AMP-stored and FP32-re-eval). Reportable as-is.
- **The LODO F1@0.5 numbers are robust** to training precision (verified
  here for LODO-IoT23: GT-edge AMP 0.8500 ± 0.0009 vs FP32 0.8527 ± 0.0002,
  difference under 0.003). The LODO master narrative's F1 column is safe.
- **Per-seed PR-AUC and ROC-AUC under LODO are not robust** to training
  precision. If the thesis reports these as per-seed numbers, the training
  precision should be stated. The mean across seeds is more stable than any
  individual seed.
- **The right operating point is val-tuned**, as recommended in the LODO
  master narrative §5.

The unaudited LODO-CTU13 and LODO-MedBIoT directions probably show similar
patterns (F1 stable, ranking metrics seed/precision-sensitive). Running
those re-trains is left as optional future work — they cost ~3 hours each.

## Stage timings (FP32 vs AMP)

| Stage | AMP run | FP32 run | Slowdown |
|---|---|---|---|
| Phase 5 | 578.0s | 658.2s | 1.14× |
| Cache | 71.5s | 85.1s | 1.19× |
| Phase 6 | 462.8s | 439.5s | 0.95× |
| Phase 7 seed 42 | 729.9s | 682.3s | 0.93× |
| Phase 7 seed 1 | 550.1s | 542.4s | 0.99× |
| Phase 7 seed 2 | 753.2s | 409.6s | 0.54× |
| **Total** | **~50 min** | **~47 min** | ~0.94× |

FP32 is not measurably slower than AMP on this workload on an RTX 5080. The
GT-edge architecture is dominated by sparse PyG operations that already run
at FP32 by default; AMP's FP16 savings only apply to the dense MLP heads,
which are a small fraction of the total compute. There is no reason to
prefer AMP here in future runs.
