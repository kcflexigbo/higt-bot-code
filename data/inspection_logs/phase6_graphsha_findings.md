# Phase 6.2 — GraphSHA-lite (Minority Manifold Mixup) Findings

**Date:** 2026-05-17
**Status:** GraphSHA does not improve aggregate F1. Abandoned in favor of
self-supervised pretraining (Phase 6.3).

## Why we tried it

After TAM (Phase 6.1) regressed aggregate F1 because of its global-prior
assumption, GraphSHA was the next shortlisted technique. Its appeal: per-graph
operation, only touches the minority class, no global prior. The hypothesis:
mixing minority-class embeddings within each graph should give iot23-35-1's
sparse positive nodes more diverse training signal without disturbing
high-bot scenarios.

## Implementation

`src/training/graphsha.py` — `minority_manifold_mixup`:
- For each graph in a batch, find class-1 nodes via `batch.batch + batch.y`.
- Pair each with another class-1 node from the same graph (random non-self shift).
- Interpolate `node_emb` AND `x` with the same λ ~ Beta(β, β).
- Gate: skip graphs where local class-1 rate > `max_local_pos_rate` (=0.5).
- Applied only at training time (val/test untouched).

Wired into `scripts/train_phase6.py` via `--graphsha`, `--graphsha-beta`,
`--graphsha-prob`. Final run: β=1.0 (uniform mix), prob=0.5.

## Results (test set)

| Run | Test F1 | iot23-35-1 | medbiot-spread | ctu13-3 | ctu13-10 |
|---|---|---|---|---|---|
| Phase 6 baseline | **0.9673** | 0.56 | 0.985 | 0.89 | 0.64 |
| TAM v1 (empirical) | 0.9577 | 0.60 | 0.010 | 0.72 | 0.59 |
| TAM v2 (balanced) | 0.9647 | 0.18 | 0.826 | 0.75 | 0.60 |
| **GraphSHA** | **0.9629** | **0.06** 💀 | **0.67** ⚠️ | **0.92** ✅ | 0.61 |

Aggregate test F1 dropped by 0.0044. Only ctu13-3 improved (+0.03). The
target scenario (iot23-35-1) collapsed to 0.06.

## Diagnosis — why GraphSHA failed where theory predicted it would help

1. **iot23-35-1 training windows have ≤2 positives.** The intra-graph mixup
   requires `n_pos ≥ 2` per graph to fire. Most iot23-35-1 training windows
   have 0 or 1 positives, so they receive *zero* augmentation. Meanwhile
   bot-rich scenarios (ctu13-9, iot23-1-1) with hundreds of positives per
   graph get heavily augmented, biasing the shared model parameters toward
   their distribution.

2. **medbiot-bashlite-spread regressed indirectly.** Even with the >50% local
   pos-rate gate skipping its training windows, augmentation on *other*
   scenarios shifted the global decision boundary away from where this
   scenario sits.

3. **iot23-35-1 F1 is hyper-sensitive.** 31 test positives means each TP/FN
   flip moves F1 by ~0.03. The variance across our four runs (0.06, 0.18,
   0.56, 0.60) reflects this, not genuine learning signal — none of the
   loss/feature-level interventions changed the *underlying* representation
   of bots-in-mostly-benign-context.

## What this tells us

Per-graph minority mixup is the wrong *scale* for our problem. The fundamental
issue is that iot23-35-1's training distribution doesn't contain enough
positive examples for *any* intra-graph augmentation to fire effectively. The
fix has to either:

- Synthesize positives *across graphs* (inter-graph SMOTE, complex with
  variable graph sizes and edge attribution), or
- Reduce the model's *dependence* on supervised positive examples by
  pretraining the encoder on unlabeled benign data (the
  anomaly-from-baseline approach).

We have ~40k benign sequences in the dataset — far more than the few hundred
labeled positives we have for the hardest scenarios. That asymmetry is what
self-supervised pretraining exploits.

## Decision

Move to self-supervised pretraining (Phase 6.3). Approach:
- Masked-flow reconstruction on all sequences (BERT-style MLM on the temporal
  flow-sequence encoder).
- Load pretrained weights into the Phase 5 T-GINE encoder, optionally
  fine-tune supervised, then re-cache embeddings, then re-train Phase 6.

## Files / artifacts

- `experiments/phase6/hgt_diffpool_sha.pt` — GraphSHA checkpoint (kept)
- `data/inspection_logs/phase6_diffpool_sha.json`
- `src/training/graphsha.py` — implementation kept; the `--graphsha` flag is
  retained as opt-in for future experiments (e.g. with inter-graph synthesis).
- `scripts/train_phase6.py` — flags retained.

## Negative results are still findings

For the thesis: both loss-level (TAM) and feature-level (GraphSHA)
augmentations failed to improve aggregate F1 over the Phase 6 baseline.
This is itself a result — it rules out the cheapest two interventions in
the imbalance-learning literature for this particular benchmark, and
motivates the move to representation-level interventions (SSL pretraining).
