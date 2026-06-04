# LODO master narrative (FP32-corrected)

Leave-one-dataset-out generalization for HiGT-Bot. Three directions, full Phase 4–7 ladder per direction, GT-edge headline replicated across three seeds. **The original (AMP-tainted) results were retracted on 2026-06-04** after a reviewer pointed out that the MedBIoT direction's PR-AUC was NaN and the IoT-23 direction's ROC-AUC was sub-random. Re-evaluation in FP32 produced materially different numbers; the AMP-vs-FP32 contrast is itself a finding worth reporting (see §6).

All numbers in this file come from FP32 inference. The original AMP results are preserved at `data/inspection_logs/*_amp_backup.json` for audit. Source data: `data/inspection_logs/*_lodo_*.json` and `experiments/lodo/<holdout>/phase4/*.json`. Pipeline orchestrated by `scripts/run_lodo_ladder.py`; re-evaluation by `scripts/reeval_lodo_fp32.py`; aggregated into `lodo_comparison_table.md` and the three per-holdout summaries.

## 1. Headline (corrected)

GT-edge on the canonical mixed pool: **F1 = 0.9677** (Phase 7 final).

Under leave-one-dataset-out, GT-edge test metrics (FP32, fixed threshold 0.5, mean ± std across seeds {42, 1, 2}):

| Direction | F1 | PR-AUC | ROC-AUC | F1 @ val-tuned threshold |
|---|---|---|---|---|
| LODO-MedBIoT | **0.906 ± 0.056** | 0.997 / 0.996 / 0.997 | 0.778 / 0.710 / 0.798 | 0.914 ± 0.052 |
| LODO-IoT23 | **0.850 ± 0.001** | 0.798 / 0.892 / 0.858 | 0.643 / 0.798 / 0.745 | 0.850 ± 0.001 |
| LODO-CTU13 | **0.435 ± 0.037** | 0.385 / 0.314 / 0.389 | 0.506 / 0.407 / 0.484 | 0.471 ± 0.018 |

The headline is **two-out-of-three transfers**, not three-out-of-three collapses:

- **LODO-MedBIoT transfers well.** PR-AUC ≈ 1.0 says ranking is essentially perfect; F1@0.5 = 0.906 is only modest because the prevalence-shifted prior makes 0.5 a slightly sub-optimal threshold; val-tuned F1 = 0.914.
- **LODO-IoT23 is moderate.** F1@0.5 = 0.85 is partially a prevalence artifact (IoT-23 test is 74% bot, so an all-bot predictor would land near 0.85), but the underlying ranking has real signal: PR-AUC = 0.80–0.89 vs the prevalence-line ≈ 0.74, and ROC-AUC = 0.64–0.80 (well above random).
- **LODO-CTU13 genuinely fails.** PR-AUC ≈ 0.37 ≈ prevalence; ROC-AUC ≈ 0.50; F1 stays at ~0.44 even after val-tuning. The model has no real signal on enterprise traces after training only on IoT data.

The thesis claim **does** generalize to cross-dataset for two of three directions. The one direction that fails is the one where the test family (enterprise networks) is structurally distinct from every training scenario (all IoT). Joint training mostly helps, but it's not unconditionally required.

## 2. Full comparison vs mixed-pool baseline

Test F1 at threshold = 0.5 (mean ± std across seeds where applicable):

| Model | Mixed-pool | LODO-CTU13 | LODO-IoT23 | LODO-MedBIoT |
|---|---|---|---|---|
| RandomForest | 0.9597 | 0.3682 | 0.8499 | 0.8558 |
| XGBoost | 0.9595 | 0.2803 | 0.8519 | 0.9074 |
| GAT (flat) | 0.9115 | **0.6831** | 0.8474 | 0.8514 |
| GIN (flat) | 0.9443 | 0.4729 | 0.8494 | 0.8239 |
| GINE (flat) | 0.9595 | 0.4820 | 0.8522 | 0.9482 |
| T-GINE + skip | 0.9661 | 0.3871 | 0.8524 | **0.9570** |
| SAGPool (sparse) | 0.9674 | 0.3795 | 0.8530 | 0.9150 |
| **GT-edge (final)** | **0.9677** | **0.4353 ± 0.0365** | **0.8500 ± 0.0009** | **0.9064 ± 0.0558** |

The previous (AMP-tainted) version of this table had MedBIoT GT-edge at 0.0000 ± 0.0000. The actual value is 0.906 ± 0.056. The narrative based on the bug — "deeper architectures catastrophically fail to transfer" — was wrong.

## 3. Per-direction analysis

### LODO-MedBIoT (train on CTU-13 + IoT-23, test on MedBIoT)

Test set = 1,965 nodes (43 windows) across `medbiot-{bashlite,mirai,torii}_mal_*`. Test prevalence = 98.7% bot.

Deep models match or beat every flat baseline:

- T-GINE + skip: F1@0.5 = 0.957, PR-AUC = 0.9997, val-tuned F1 = 0.959.
- SAGPool: F1@0.5 = 0.915, PR-AUC = 0.9985.
- GT-edge (mean across seeds): F1@0.5 = 0.906, PR-AUC = 0.997, ROC-AUC = 0.762.

The encoder genuinely transfers — bot patterns learned from CTU-13 and IoT-23 fire correctly on MedBIoT's bashlite, mirai, and torii malware. The PR-AUCs near 1.0 mean the *ranking* is essentially perfect; F1 is only modestly below the ranking would suggest because 0.5 is not the F1-optimal threshold on the prior-shifted test distribution. After val-tuning the threshold per direction (best F1 on val), GT-edge improves to 0.914 ± 0.052.

This is the most interesting direction because of §6 below: the AMP-vs-FP32 gap is enormous here, and the previous "catastrophic collapse" headline was an artifact of saturated FP16 softmax. With FP32 the model demonstrably transfers.

### LODO-IoT23 (train on CTU-13 + MedBIoT, test on IoT-23)

Test set = 255,352 nodes (3,351 windows). Test prevalence = 74.2% bot.

Every model lands at F1@0.5 ≈ 0.85 ± 0.005. That coincidence makes the F1 number suspect — predicting "all bot" yields F1 = 2·0.742·1 / (0.742+1) = 0.852. So F1 = 0.85 is consistent with everything from "the model has zero discrimination, only learned the prior" to "the model has moderate discrimination, but the 0.5 threshold lands near the all-bot trivial solution."

PR-AUC and ROC-AUC separate the two cases:

- Phase 6 SAGPool: PR-AUC = 0.837 (vs prevalence-line 0.742), ROC-AUC = 0.712. Real signal.
- Phase 7 GT-edge: PR-AUC = 0.80–0.89, ROC-AUC = 0.64–0.80 across seeds. Real signal, varying by seed.
- Flat baselines: PR-AUC = 0.59–0.78. Less ranking quality than the deep models.
- RF: PR-AUC = 0.77, ROC-AUC = 0.56. Tabular features carry some IoT-domain transferability.

So the deep models DO have ranking quality above the trivial all-bot predictor on IoT-23. F1@0.5 = 0.85 hides this; the per-model PR-AUC differences (0.59 for GINE-flat vs 0.89 for Phase 7 seed 1) reveal it. The val-tuned threshold doesn't help because GT-edge probabilities are saturated near 1.0 — almost any threshold below ~0.98 produces the same all-bot-ish prediction. This is the direction most contaminated by prevalence; the prior dominates over the (real) signal.

### LODO-CTU13 (train on IoT-23 + MedBIoT, test on CTU-13)

Test set = 29,316 nodes (1,560 windows) across all 9 CTU-13 scenarios. Test prevalence = 36.8% bot.

This is the **genuine cross-dataset failure direction.** None of the deep models reach F1 = 0.5; the seed-replicate spread is wide for an architecture whose mixed-pool spread is ±0.001:

- Phase 5 T-GINE: F1@0.5 = 0.387, PR-AUC = 0.304 (below prevalence-line 0.368), ROC-AUC = 0.377 (below random).
- Phase 6 SAGPool: F1@0.5 = 0.380, PR-AUC = 0.315, ROC-AUC = 0.416.
- Phase 7 GT-edge: F1@0.5 = 0.435 ± 0.037, PR-AUC = 0.36, ROC-AUC = 0.47.

ROC-AUC below 0.5 in two of the three deep models means the *ranking is wrong-way* — the model assigns higher bot probability to actual benign nodes than to actual bot nodes. Predicting the inverse of its proba would do better than predicting straight. Val-tuning the threshold lifts F1 only slightly (0.45 → 0.47) because there's no operating point that recovers a discriminative classifier from inverted ranking.

The standout positive transfer in this direction is **GAT (flat), F1 = 0.6831, PR-AUC = 0.8737, ROC-AUC = 0.9304** — far ahead of every other model. GAT's attention mechanism is the only Phase-4 model that uses learnable per-edge weighting without per-feature normalization. Topology features that distinguish CTU-13 botnets generalize from IoT topology better than the per-flow temporal features the deep models build.

## 4. Why the asymmetry?

Three test-set characteristics correlate with the success direction:

1. **Test bot prevalence.** LODO-CTU13 test is 37% bot, LODO-IoT23 is 74% bot, LODO-MedBIoT is 99% bot. The two directions where the deep models work best are also the high-prevalence ones, but the high-prevalence cases are also the ones where the prior dominates the F1 metric. PR-AUC controls for this: MedBIoT PR-AUC near 1.0 confirms genuine discrimination, IoT-23 PR-AUC 0.8 confirms moderate discrimination, CTU-13 PR-AUC 0.31–0.39 ≈ prevalence confirms no discrimination.

2. **Domain similarity.** MedBIoT is lab-generated IoT malware; IoT-23 is real-world IoT malware. They are *both* IoT-domain. CTU-13 is enterprise traffic with classic P2P bots (Neris, Rbot, Virut, Murlo). When CTU-13 is the holdout, the training pool is IoT + IoT — no exposure to enterprise topology. When MedBIoT or IoT-23 is the holdout, the training pool always contains the other IoT dataset, so IoT topology features remain in train.

3. **Per-flow feature availability.** Of four NaN-able schema fields (`mean_iat_ms`, `std_iat_ms`, `min_pkt_size`, `max_pkt_size`):

   | Dataset | mean_iat_ms | std_iat_ms | min_pkt_size | max_pkt_size |
   |---|---|---|---|---|
   | CTU-13 | 100% NaN | 100% NaN | 100% NaN | 100% NaN |
   | IoT-23 | 100% NaN | 100% NaN | 100% NaN | 100% NaN |
   | MedBIoT | ~1.6% NaN | ~1.6% NaN | 0% NaN | 0% NaN |

   So MedBIoT is the **only** dataset that supplies these per-packet stats. In LODO-MedBIoT, the model never sees real values for these features in training. In LODO-CTU13 and LODO-IoT23, the model sees real values from MedBIoT in training and zero-filled values from the other training dataset, then sees zero-filled at test.

   Naively this should hurt LODO-MedBIoT more than the other two. Empirically it doesn't (F1 = 0.96 / 0.91 / 0.91 across the deep models). The encoder is robust to "feature was zero in training, suddenly real at test" — it can ignore those features and rely on the others. The breakage direction (CTU-13) is *not* the one where the feature distribution shifts most; it's the one where the **domain** shifts most.

Combining (2) and (3): the dominant axis is domain (IoT vs enterprise), not feature availability or label distribution. The model learns features that are useful across IoT subtypes (Mirai-vs-bashlite-vs-Kenjiro on real and lab capture environments) but those features don't transfer to a fundamentally different topology (enterprise P2P).

## 5. Operating-point analysis

F1 at threshold 0.5 is the default in the trainer scripts and the original results. It is not the right operating point for class-imbalanced cross-dataset evaluation. Val-tuned threshold (select on val, apply to test):

| Direction | GT-edge F1@0.5 | val_thr | GT-edge F1@val-tuned |
|---|---|---|---|
| LODO-CTU13 | 0.435 ± 0.037 | 0.36–0.44 | 0.471 ± 0.018 |
| LODO-IoT23 | 0.850 ± 0.001 | 0.48–0.59 | 0.850 ± 0.001 |
| LODO-MedBIoT | 0.906 ± 0.056 | 0.42–0.48 | 0.914 ± 0.052 |

Val-tuning helps CTU-13 modestly (+0.04) and MedBIoT marginally (+0.01). IoT-23 is unaffected because the model's probabilities are saturated near 1.0 — almost any threshold below ~0.98 yields the same predictions. None of this changes the qualitative conclusions of §3; it confirms them. The val-tuned numbers should be the primary headline in the thesis writeup.

## 6. AMP vs FP32 — a finding in itself

The original results were generated with `predict(..., use_amp=True)` (PyTorch autocast wrapping inference). For LODO-MedBIoT this caused argmax to flip on most test nodes:

| | Stored (AMP eval) | Re-eval (FP32) |
|---|---|---|
| LODO-MedBIoT Phase 7 seed 42 F1 | 0.000 | 0.969 |
| LODO-MedBIoT Phase 7 seed 42 PR-AUC | NaN | 0.997 |
| LODO-MedBIoT Phase 7 seed 42 confusion | tp=0, fn=1940, tn=24, fp=1 | tp=1880, fn=60, tn=24, fp=1 |

The model's logits on MedBIoT test were very large in magnitude (extreme confidence in either direction). Under FP16 autocast, the larger logits overflow to `+inf` and the smaller ones underflow to 0; softmax of `[+inf, +inf]` → `[NaN, NaN]`; argmax of `[NaN, NaN]` falls through to class 0 in the implementation we hit. The val-time eval (which also uses AMP) was unaffected because the val data comes from the chronological tails of CTU-13 + IoT-23 — same feature distribution as train, smaller logit magnitudes, no overflow.

LODO-IoT23 was partially affected: PR-AUC and ROC-AUC were NaN under AMP (probas underflowed for benign-class samples) but F1 happened to come out correct because argmax doesn't depend on the magnitude, only the sign of the difference. LODO-CTU13 was unaffected entirely because the logit magnitudes on enterprise-traffic test data are smaller — well within FP16 range.

This is a methodological note worth including in the thesis: **AMP autocast in inference can silently flip predictions under distribution shift.** The training-time speedup is irrelevant to a single test pass; FP32 is the correct numerical choice. Patches applied:

- `scripts/train_phase5.py`, `train_phase6_sparse.py`, `train_phase7.py` — final `predict()` calls now force `use_amp=False`.

## 7. Threats to validity (revised)

- **No hyperparameter retuning per LODO.** Same as before. Retuning would conflate "architecture transfers" with "tuning was lucky for the joint pool."
- **Val set is in-distribution.** Val draws from chronological tails of the training datasets. So val F1 (consistently 0.96–0.99) does not predict test F1 under cross-dataset shift. The val-tuned threshold transfers to test better than the val-F1 number does.
- **Prevalence varies substantially.** 37% / 74% / 99%. F1@0.5 is therefore not directly comparable across directions; PR-AUC and val-tuned F1 are the metrics to use.
- **Small absolute test size in LODO-MedBIoT.** Only 1,965 nodes. The Phase 7 seed-spread (±0.056) is large relative to the other directions partly because small-sample noise dominates.
- **Per-flow feature availability is bimodal**, not graded. CTU-13 and IoT-23 supply zero of the four NaN-able features; MedBIoT supplies essentially all. Any future LODO-style experiment should establish whether the model is relying on these features by running an ablation with them zeroed in MedBIoT test (not yet done; left as follow-up).
- **GT-edge seed-replicate spread.** Mixed-pool ±0.001. LODO-CTU13 ±0.037. LODO-MedBIoT ±0.056. LODO-IoT23 ±0.001 (saturated). The high seed-spread on CTU-13 and MedBIoT suggests the LODO objective has multiple local minima, not the single sharp minimum the mixed-pool training converges to.

## 8. Reproduction

From `higt-bot-code/`:

```bash
# Train all three LODO ladders
uv run python scripts/run_lodo_ladder.py --holdout medbiot
uv run python scripts/run_lodo_ladder.py --holdout ctu13
uv run python scripts/run_lodo_ladder.py --holdout iot23

# Aggregate (Phase 5/6/7 outputs go to data/inspection_logs/ with _lodo_<x> suffix)
uv run python scripts/aggregate_lodo_results.py
```

Wall-time on RTX 5080 (16 GB) for the full ladder:
- LODO-MedBIoT: ~3 hours
- LODO-CTU13: ~3 hours
- LODO-IoT23: ~50 minutes

Manifests at `experiments/lodo/<holdout>/manifest.json` enumerate per-stage durations, exit codes, log paths.

If reproducing with checkpoints already saved (no retrain), re-evaluate in FP32:

```bash
uv run python scripts/reeval_lodo_fp32.py --all
```

## 9. Model-code fixes recorded mid-experiment

Three commits affect the model code and the eval path:

- `b56ce02` (sparse_pool/higt_bot): caller-side guard at the post-pool GNN call site — skip the GNN when SAGPool coarsens the batch to fewer than 2 nodes.
- `1aca8ee` (sparse_pool/higt_bot): self-guard inside `_GINEBlock.forward` — when input batch has fewer than 2 rows in training mode, run that one forward call in eval mode so BN uses running stats. Protects pre-pool AND post-pool, in both files.
- Pending commit (train_phase{5,6,7}.py): force `use_amp=False` for the final-eval `predict()` calls. The AMP-bug above is silently *probably present in any prior LODO-style runs done with these scripts before this fix*.

Existing mixed-pool checkpoints load and behave identically (state-dict structure unchanged; eval behavior was already FP32-correct for mixed-pool because the logit magnitudes stayed in FP16 range on in-distribution data). The 98-test pytest suite still passes after all three fixes.
