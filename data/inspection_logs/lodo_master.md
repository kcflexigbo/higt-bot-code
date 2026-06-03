# LODO master narrative

Leave-one-dataset-out generalization for HiGT-Bot. Three directions, full Phase 4–7 ladder per direction, GT-edge headline replicated across three seeds. All deep models trained from scratch — no encoder reuse from the mixed-pool checkpoint. Source data: every JSON under `data/inspection_logs/*_lodo_*.json` and `experiments/lodo/<holdout>/phase4/*.json`. Pipeline orchestrated by `scripts/run_lodo_ladder.py`; aggregated into `lodo_comparison_table.md` and the three per-holdout summaries.

## Headline

GT-edge on the canonical mixed pool: **F1 = 0.9677** (Phase 7 final).
Under leave-one-dataset-out (test F1, mean ± std across seeds {42, 1, 2}):

- **LODO-CTU13:**   F1 = 0.4352 ± 0.0365   (ROC-AUC ≈ 0.47, near-random ranking)
- **LODO-IoT23:**   F1 = 0.8500 ± 0.0009   (PR-AUC NaN — saturated all-bot output)
- **LODO-MedBIoT:** F1 = 0.0000 ± 0.0000   (collapse: tp = 0/1940 positive nodes)

The architecture does **not** transfer cross-dataset. All three directions land far from the mixed-pool number, and the three failure modes are qualitatively different — collapse-to-bot, near-random, collapse-to-benign. None of the deep models recover; the flat baselines transfer more gracefully, with GAT in particular reaching F1 = 0.683 on the LODO-CTU13 direction where every deep model is ≤ 0.43. The thesis claim about generalization is therefore *intra-dataset chronological*, not cross-dataset. Joint training is doing real work — it is not a convenience.

## Per-LODO direction

### LODO-CTU13 (train on IoT-23 + MedBIoT, test on CTU-13)

Test set = 1,560 windows across `ctu13-{1,2,3,4,6,8,9,10,13}`. Train pool = 2,709 windows.

Best deep model: T-GINE/SAGPool/GT-edge all between F1 0.379 and 0.467. GT-edge’s three seeds at F1 0.396 / 0.442 / 0.468 — high variance for an architecture whose mixed-pool seed-replicates were ±0.001. Confusion matrices show roughly half of the truly-bot nodes being recovered (tp = 4220–6701 of 10,788) but ROC-AUC ≈ 0.47–0.51 says the *ranking* is essentially random — the model is making positive predictions on roughly the right total count but for the wrong subset.

The standout in this direction is **GAT (flat), F1 = 0.683, PR-AUC = 0.874** — much higher than every other model including the deep ones. The likely reason: GAT’s attention mechanism is the only Phase-4 model that uses learnable per-edge weighting without per-feature normalization, so the topology features (degrees, fan-out patterns) that distinguish CTU-13 botnets generalize from IoT-23 better than the byte/packet statistics RF/XGB rely on or the flow-sequence features the deep models build.

### LODO-IoT23 (train on CTU-13 + MedBIoT, test on IoT-23)

Test set = 3,351 windows across all 9 IoT-23 scenarios. Train pool = 1,276 windows (smallest of the three directions).

Every model — flat and deep — converges to F1 ≈ 0.85. The deep models in particular hit tp / (tp + fn) ≈ 99.5%, so they are predicting bot for almost every node in the test set. F1 = 0.85 lands there because IoT-23 test is bot-heavy and predicting all-bot happens to score well; PR-AUC = NaN flags that the predicted probabilities are saturated (no useful ranking gradient remains). This is the “collapse to predict-modal-class” pattern. Best val F1 = 0.994–0.997 — the model fits the CTU-13 + MedBIoT training distribution near-perfectly, then defaults to the bot-rate of the test prior at inference time. The fact that flat baselines (RF F1 = 0.85, GINE F1 = 0.85) match the deep models within ±0.005 is the relevant signal: the *deep model contributes nothing* on this direction.

### LODO-MedBIoT (train on CTU-13 + IoT-23, test on MedBIoT)

Test set = 43 windows across `medbiot-{bashlite,mirai,torii}_mal_*`. Train pool = 3,921 windows (the largest because both other datasets are bigger than MedBIoT).

Catastrophic collapse for every deep model: Phase 5 T-GINE F1 = 0.087, Phase 6 SAGPool F1 = 0.000, Phase 7 GT-edge F1 = 0.000 across all three seeds. Confusion matrix: tp = 0, fn = 1,940 of 1,940 bot nodes. The model predicts every MedBIoT node as benign. Per-scenario breakdown shows the *inversion* of the flat-baseline pattern: where GINE-flat scored 0.99 on bashlite and 0.22 on torii (novel family fails), T-GINE scored 0.01 on bashlite and 0.61 on torii — the encoder has learned a CTU-13 + IoT-23-specific bot signature, so when MedBIoT shows it patterns that don’t fire that signature, it defaults to benign. Torii (more genuinely novel) actually generalizes best for the deep model precisely because it doesn’t match the over-specialized signature. This is the most quotable single result in the experiment: **adding the temporal Transformer encoder *inverts* per-family generalization**.

## Comparison vs mixed-pool baseline

| Model | Mixed-pool | LODO-CTU13 | LODO-IoT23 | LODO-MedBIoT |
|---|---|---|---|---|
| RandomForest | 0.9597 | 0.3682 | 0.8499 | 0.8558 |
| XGBoost | 0.9595 | 0.2803 | 0.8519 | 0.9074 |
| GAT (flat) | 0.9115 | **0.6831** | 0.8474 | 0.8514 |
| GIN (flat) | 0.9443 | 0.4729 | 0.8494 | 0.8239 |
| GINE (flat) | 0.9595 | 0.4820 | 0.8522 | **0.9482** |
| T-GINE + skip | 0.9661 | 0.3868 | 0.8529 | 0.0868 |
| SAGPool (sparse) | 0.9674 | 0.3794 | 0.8530 | 0.0000 |
| **GT-edge (final)** | **0.9677** | **0.4352 ± 0.0365** | **0.8500 ± 0.0009** | **0.0000 ± 0.0000** |

The mixed-pool column is the headline from `phase{4..7}_*.json` (Phase 7 GT-edge is the seeded mean across {42, 1, 2}, matching this experiment’s protocol).

## What the asymmetry says

The three LODO directions are **not** uniform degradations of a generally-transferable model. Each one fails differently, and the direction of failure is informative:

- The model has learned a bot **signature**, not a bot **concept**. When the test distribution matches the signature (LODO-IoT23: train mix includes MedBIoT which is heavily bot-flow patterns somewhat similar to IoT-23 malware), the model fires for everything and lands on the test-set prior. When the signature partially overlaps with test patterns (LODO-CTU13: enterprise traces share some structural features with IoT-23 + MedBIoT but the bot families are different), the model fires partially for the wrong subset of nodes. When the signature is absent (LODO-MedBIoT: MedBIoT bot families don’t match what was learned from CTU-13 + IoT-23), the model defaults to the majority of the *training* prior, which was bot-heavy — but predicts benign at inference because the unfamiliar features don’t activate the bot-pathway in the head.
- The encoder is what overfits. The flat baselines that bypass the temporal encoder retain reasonable per-family transfer because byte/packet/edge statistics are partially invariant across capture environments. Adding `TemporalGINE → SAGPool → TransformerConv` strictly hurts on LODO-MedBIoT and offers no improvement on LODO-IoT23.
- The Phase 7 seed-replicate variance is *larger* under LODO. Mixed-pool: ±0.001. LODO-CTU13: ±0.037. LODO-IoT23: ±0.001 (collapse is stable). LODO-MedBIoT: ±0.000 (collapse is also stable). The high variance on LODO-CTU13 says even within the same training data, three seeds find solutions whose generalization to held-out CTU-13 varies meaningfully. The collapse cases are *too* stable: the model is finding the same degenerate fixed point regardless of init.

The right framing for the thesis is therefore **not** “HiGT-Bot generalizes across capture environments.” It is: *“HiGT-Bot achieves SOTA-class F1 (0.9677) under chronological per-scenario splits across a mixed pool of three datasets, with three-seed stability of ±0.001. The chronological-split protocol already eliminates the bot-IP leakage that inflates most published GNN botnet detectors. Cross-dataset transfer (LODO) is documented as a limitation of the current architecture; we provide three reference LODO numbers (F1 = 0.000 / 0.435 / 0.850 for held-out MedBIoT / CTU-13 / IoT-23) showing that the deep encoder degrades cross-dataset transfer relative to flat baselines, which is itself an interesting design finding for future work.”*

## Threats to validity

- **No hyperparameter retuning per LODO.** Intentional — see spec §3.4. Retuning would conflate “architecture transfers” with “tuning was lucky for the joint pool.” With retuning, GT-edge might land somewhere between the flat baselines and the collapse, but the joint-training story would weaken.
- **Excluded scenarios stay excluded.** `ctu13-{5,7,11,12}` and `iot23-{20-1,21-1}` are excluded in every direction (consistent with Phase 4–7 protocol). Including the tiny CTU-13 scenarios into LODO-CTU13 test would add only ~30 windows of mostly low-bot-prevalence data and change the headline by <0.01 F1.
- **Class-prior shift across directions.** The bot fraction varies substantially: CTU-13 test is mixed (bot fraction ~0.45), IoT-23 test is bot-heavy (~0.99), MedBIoT test is bot-heavy (~0.99). PR-AUC is the prior-robust metric and is reported in every per-holdout summary, but it goes NaN whenever the model output is saturated. ROC-AUC is reported where it is finite.
- **Encoder hyperparameters held fixed.** The T-GINE encoder’s `d_model`, `nhead`, `max_flows` are inherited from `configs/phase5.yaml`. These were tuned for the mixed-pool problem. A smaller encoder might be less prone to over-specialization but we did not sweep this.
- **Sample size for LODO-MedBIoT test is small (43 windows, 1,965 nodes).** A single misclassified small graph influences F1 less than on the other directions but the *direction* of the failure (all-benign) is unambiguous.
- **The experiment is not blinded.** All test scenarios were visible during architecture/hyperparameter selection in earlier phases. A proper cross-dataset evaluation in a future paper would use *time-locked* hyperparameter selection that never sees the LODO test set.

## Reproduction

From `higt-bot-code/`:

```bash
uv run python scripts/run_lodo_ladder.py --holdout medbiot
uv run python scripts/run_lodo_ladder.py --holdout ctu13
uv run python scripts/run_lodo_ladder.py --holdout iot23
uv run python scripts/aggregate_lodo_results.py
```

Wall-time on RTX 5080 (16 GB):
- LODO-MedBIoT: ~3 hours total (Phase 5 encoder is the dominant cost at ~58 min; Phase 7 × 3 seeds another ~110 min)
- LODO-CTU13: ~3 hours total (Phase 5 ~62 min; Phase 7 × 3 ~68 min)
- LODO-IoT23: ~50 min total (smaller train pool: Phase 5 ~10 min, Phase 7 × 3 ~35 min)

Manifests at `experiments/lodo/<holdout>/manifest.json` enumerate per-stage durations, exit codes, log paths.

## Two model-code fixes were required mid-experiment

Both `src/models/sparse_pool.py` and `src/models/higt_bot.py` had a latent BatchNorm-in-training crash when the DataLoader produced a batch with fewer than 2 nodes total. The canonical mixed-pool training never tripped it because train = 2,796 is divisible by `batch_size = 4`. The LODO directions all hit it (train sizes 1,276 / 2,709 / 3,921 — only one of which is divisible by 4, and even that direction trips the K<2 case after SAGPool).

- Commit `b56ce02`: caller-side guard at the post-pool GNN call site — skip the GNN entirely when SAGPool coarsens the batch to fewer than 2 nodes.
- Commit `1aca8ee`: self-guard inside `_GINEBlock.forward` — when the input batch has fewer than 2 rows in training mode, run that one forward call in eval mode so BN uses running stats. Protects all callers (pre-pool AND post-pool, in both files).

Existing mixed-pool checkpoints load and behave identically (state-dict structure unchanged; only training-time forward differs on rare degenerate batches). Eval behavior is fully unchanged because BN already uses running stats in eval. All 98 unit tests still pass.
