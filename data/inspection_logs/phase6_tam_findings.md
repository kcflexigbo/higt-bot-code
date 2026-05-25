# Phase 6.1 — TAM (Topology-Aware Margin) Ablation Findings

**Date:** 2026-05-17
**Status:** TAM does not improve aggregate F1 on this dataset. Abandoned in
favor of GraphSHA-style minority oversampling.

## Why we tried it

Phase 6 baseline (DiffPool) hit test F1=0.9673, but two scenarios still
underperform:
- iot23-35-1: F1=0.56 (only 31 / 9,905 positives — 0.3% positive rate)
- ctu13-10: F1=0.64

Research recommendation (research subagent, 2026-05-17): try TAM (Song et al.
ICML 2022) as the cheapest first lever — single loss-function change, no
model-architecture change.

## Implementation

`src/training/tam_loss.py` — `TopologyAwareFocalLoss`:
- Compute per-node neighbor class distribution `pi_local` via PyG scatter.
- Margin: `-α * (pi_local - pi_global)` added to logits.
- Optional CCM (Class-Conditional Margin): scales margin by inverse-frequency.
- Wired into `scripts/train_phase6.py` via `--tam`, `--tam-alpha`,
  `--tam-ccm-beta`, `--tam-balanced-prior`.

## Results (test set)

| Run | Config | F1 | iot23-35-1 | medbiot-spread | ctu13-10 | ctu13-3 |
|---|---|---|---|---|---|---|
| Phase 6 baseline | no TAM | **0.9673** | 0.56 | 0.985 | 0.64 | 0.89 |
| TAM v1 | empirical π, α=1.5, ccm=0.5 | 0.9577 | **0.60** | **0.010** 💀 | 0.59 | 0.72 |
| TAM v2 | balanced π=[0.5,0.5], α=1.0, ccm=0 | 0.9647 | 0.18 | 0.826 | 0.60 | 0.75 |

**Headline:** both TAM configs regress aggregate F1 (-0.0026 to -0.0096).

## Diagnosis — why TAM is the wrong tool here

The TAM paper assumes a single global class prior that approximates every
local subgraph's prior. Our dataset violates this badly:

- **Training-set prior** is ~70% bot / 30% benign (we re-balance via focal-α
  already).
- **Per-scenario local priors span the entire [0, 1] range**: iot23-35-1
  has 0.3% positives; medbiot-bashlite-spread has 97%; medbiot-mirai-cc has
  100%; ctu13-3 has 1.3%.

**TAM v1 (empirical prior π=[0.3, 0.7])**:
- Helps iot23-35-1 (bot nodes with mostly benign neighbors get +1.05 logit
  push toward bot) — F1 0.56 → 0.60.
- Catastrophic on medbiot-bashlite-spread (97% bot scenario): every node
  gets pushed *away* from bot by ~0.4 logits — F1 0.985 → 0.010.

**TAM v2 (balanced prior π=[0.5, 0.5])**:
- Removes the global skew → medbiot survives (0.83).
- But also removes the strong push that was helping iot23-35-1 (now
  +0.5 logit instead of +1.05) → F1 collapses 0.56 → 0.18.

There is no single TAM hyperparameter that fixes both, because per-scenario
local priors span the entire [0,1] range while TAM applies one global
correction. The mechanism is correct, but the granularity is wrong — TAM
expects scenario-homogeneous topology priors, which we don't have.

## What we learned

1. **Topology-aware bias *is* the right direction** — the +0.04 win on
   iot23-35-1 (TAM v1) proves the hypothesis from the imbalance-learning
   literature applies to our setup.
2. **The correction has to be local, not global.** A single π_global can't
   simultaneously help a 0.3%-positive scenario and a 97%-positive scenario.
3. **Aggregate F1 is dominated by the bot-heavy scenarios** (iot23-1-1 alone
   is 34k nodes ≈ 58% of test set). Any technique that perturbs those
   scenarios negatively will cost more aggregate F1 than it earns on
   iot23-35-1's 9,905 nodes (which is large but mostly benign).

## Decision

Abandon TAM tuning. Move to **GraphSHA** (Li et al. KDD 2023) — synthesize
harder minority samples in embedding space via intra-class interpolation.

GraphSHA operates *per-graph* (no global prior assumption), targets only the
positive class (no risk to high-bot scenarios), and works directly on the
cached encoder embeddings (no encoder retraining).

## Files / artifacts

- `experiments/phase6/hgt_diffpool_tam.pt` — TAM v1 checkpoint (kept for posterity)
- `experiments/phase6/hgt_diffpool_tam_bal.pt` — TAM v2 checkpoint
- `data/inspection_logs/phase6_diffpool_tam.json`
- `data/inspection_logs/phase6_diffpool_tam_bal.json`
- `src/training/tam_loss.py` — implementation kept (may help future ablations
  or thesis-discussion section on what *didn't* work)
- `scripts/train_phase6.py` — `--tam` flag retained as opt-in
