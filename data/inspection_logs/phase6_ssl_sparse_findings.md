# Phase 6.3 + 6.4 — SSL Pretraining & Sparse Pooling

**Date:** 2026-05-17
**Status:** SAGPool produces a new SOTA on PR-AUC and matches DiffPool on F1.
The full SSL→supervised pipeline reveals that DiffPool destroys rare-bot signal
that the encoder otherwise learns; sparse pooling does not.

## Why we tried these

Phase 6.1 (TAM) and 6.2 (GraphSHA) both failed because they intervened at the
loss / feature level without addressing the representation gap on the hardest
scenarios. The next two shortlist items targeted the **representation** itself:
- **SSL pretraining** (Phase 6.3) — teach the encoder "what normal looks like"
  from ~40k unlabeled benign sequences.
- **Sparse pooling** (Phase 6.4) — replace DiffPool's dense soft-assignment
  with SAGPool, which keeps a sparse subset of nodes intact instead of
  averaging them into clusters.

## Phase 6.3 — SSL pretraining (full pipeline)

**Pipeline:** masked-flow reconstruction pretrain → init Phase 5 T-GINE-skip
from pretrained encoder → supervised fine-tune → cache embeddings → train
Phase 6 DiffPool on those embeddings.

Stage outputs:
- `experiments/phase6/encoder_ssl_pretrain.pt` — SSL encoder, best loss 0.0648
- `experiments/phase5/temporal_gine_ssl_init.pt` — T-GINE-skip from SSL init
- `data/flow_embeddings_ssl/`, `data/flow_embeddings_ssl_ft/` — caches
- Logs: `phase5_train_ssl_init.log`, `phase6_train_ssl.log`,
  `phase6_train_ssl_ft.log`

Scripts (separated by user request):
- `scripts/pretrain_encoder.py` — stage 1
- `scripts/cache_embeddings_ssl.py` — stage 2 (SSL-only encoder)
- `scripts/train_phase5.py --init-from-ssl ...` — Phase 5 fine-tune
- `scripts/cache_embeddings.py --checkpoint ...` — re-cache from finetuned
- `scripts/train_phase6.py --emb-dir ...` — final DiffPool train

### Result

| Model | Test F1 | PR-AUC | iot23-35-1 | ctu13-10 | medbiot-spread |
|---|---|---|---|---|---|
| Phase 5 T-GINE-skip (original) | 0.9661 | 0.9669 | 0.28 | 0.60 | 0.985 |
| **Phase 5 + SSL init** (no DiffPool) | 0.9601 | 0.9646 | **0.62** ⭐ | 0.59 | 0.39 |
| Phase 6 DiffPool baseline | **0.9673** | 0.9691 | 0.56 | **0.64** | 0.985 |
| Phase 6 + SSL alone (no FT) | 0.9643 | 0.9619 | 0.31 | 0.62 | 0.992 |
| Phase 6 + SSL→FT (full pipeline) | 0.9664 | 0.9702 | 0.27 | 0.47 | 0.985 |

### Findings

**1. SSL pretraining works — at the flat-node level.** Phase 5 with SSL
initialization scored **iot23-35-1 = 0.62**, the best we've seen on this
scenario (vs 0.28 without SSL). Hypothesis confirmed: the encoder learning a
"what does normal traffic look like" prior from unlabeled benign flows lets
the downstream classifier detect bots as anomalies-from-baseline rather than
needing 31 supervised positives to characterize them.

**2. DiffPool destroys that gain.** Going Phase 5 SSL-init (iot23-35-1 = 0.62)
→ Phase 6 SSL-FT on the same encoder (iot23-35-1 = 0.27) means **DiffPool's
soft cluster assignment averages the rare bot signal into majority-benign
super-nodes**. The encoder learned a usable representation; the pooling layer
threw it away. This is a new structural diagnosis — DiffPool has a blind spot
for ultra-rare-class scenarios that no amount of upstream representation
quality can fix.

**3. Phase 6 SSL-FT does hit best-ever PR-AUC = 0.9702**, suggesting the
ranking of bots vs benign is more discriminative even when the argmax F1
drops. Calibrating the threshold could trade some FN for FP and recover F1.

## Phase 6.4 — Sparse pooling (SAGPool)

The natural follow-up: if DiffPool's *dense averaging* is what destroys rare-
bot signal, swap in a *sparse* pooling layer that keeps a node-subset intact
instead.

**Architecture (`src/models/sparse_pool.py`):**
- Pre-pool GINE stack → per-node embedding `z` (this goes to the head).
- SAGPool (PyG built-in, learned GraphConv-based scoring, ratio=0.5) drops
  low-scoring nodes, keeping a sparse coarsened graph.
- Post-pool 1-layer GINE on the survivors → `z_pool2`.
- Graph readout = mean+max over surviving nodes per graph → `g_repr` (2·hidden).
- Per-node head sees `[z[v] || g_repr[batch[v]]]` — every node gets a
  prediction using its own pre-pool embedding plus the coarsened graph summary.
- No aux losses (link/entropy) — trained end-to-end on focal CE only.

Scripts:
- `scripts/train_phase6_sparse.py` (separate from train_phase6.py)
- Output: `experiments/phase6/hgt_sparse_baseline.pt`,
  `data/inspection_logs/phase6_sparse_baseline.json`

### Result (on the original Phase 6 baseline embeddings)

| Model | Test F1 | PR-AUC | iot23-35-1 | iot23-7-1 | ctu13-3 | ctu13-10 | medbiot-spread | FN |
|---|---|---|---|---|---|---|---|---|
| Phase 6 DiffPool | 0.9673 | 0.9691 | 0.56 | 0.67 | 0.89 | **0.64** | 0.985 | 118 |
| Phase 6 SSL-FT | 0.9664 | 0.9702 | 0.27 | 0.50 | 0.53 | 0.47 | 0.985 | 194 |
| **Phase 6.4 SAGPool** | **0.9674** | **0.9710** ⭐ | 0.55 | **0.91** ✅ | **0.98** ✅ | 0.53 | 0.986 | 132 |

### Findings

**1. SAGPool ties DiffPool on aggregate F1 and beats it on PR-AUC (+0.0019).**
Aggregate F1 0.9674 (≈ DiffPool 0.9673, +0.0001). PR-AUC 0.9710 is the best
across every model tried — sparse pooling produces a more discriminative bot
vs benign ranking.

**2. SAGPool unlocks two previously-hard scenarios.**
- iot23-7-1: 0.67 → **0.91** (+0.24)
- ctu13-3: 0.89 → **0.98** (+0.09)

Both are scenarios with few positives where DiffPool's dense averaging hurt
disproportionately. Sparse pooling keeps these nodes' identities intact.

**3. SAGPool does NOT recover iot23-35-1.** F1 stays at 0.55 (vs DiffPool 0.56).
At 3-epoch smoke iot23-35-1 hit 0.74, but that didn't survive full
convergence. The 31-positive test floor on this scenario is robust to
pooling-layer choice with the Phase 6 baseline encoder.

**4. ctu13-10 regressed (0.64 → 0.53).** Trade-off between iot23-7-1 and
ctu13-10; SAGPool kept fewer ctu13-10 bots' neighbors in the coarsened graph.

## Combined leaderboard

| Method | Phase | Test F1 | PR-AUC | iot23-35-1 |
|---|---|---|---|---|
| Phase 4 GINE | 4 | 0.9595 | 0.9610 | (not measured separately) |
| Phase 5 T-GINE-skip | 5 | 0.9661 | 0.9669 | 0.28 |
| Phase 5 + SSL init | 5 | 0.9601 | 0.9646 | **0.62** |
| Phase 6 DiffPool | 6 | 0.9673 | 0.9691 | 0.56 |
| Phase 6 + SSL-FT | 6.3 | 0.9664 | 0.9702 | 0.27 |
| **Phase 6.4 SAGPool** | **6.4** | **0.9674** | **0.9710** | 0.55 |

## Phase 6.5 — SAGPool × SSL-FT (combination)

**Hypothesis:** SAGPool preserves rare-class signal; SSL-FT encoder produces
representations that highlight rare-class anomalies. The two should compose
to give SOTA F1 *plus* best-ever iot23-35-1.

**Result:** they don't compose.

| Run | Test F1 | PR-AUC | iot23-35-1 | iot23-7-1 | ctu13-3 |
|---|---|---|---|---|---|
| Phase 6 DiffPool | 0.9673 | 0.9691 | 0.56 | 0.67 | 0.89 |
| Phase 6.4 SAGPool (baseline) | **0.9674** | **0.9710** | 0.55 | **0.91** | **0.98** |
| Phase 6 SSL-FT (DiffPool) | 0.9664 | 0.9702 | 0.27 | 0.50 | 0.53 |
| Phase 6.5 SAGPool × SSL-FT | 0.9659 | 0.9682 | **0.59** | 0.67 | 0.83 |

The combination gives the best iot23-35-1 of any Phase 6 model (0.59) but
loses the SAGPool gains on iot23-7-1 (0.91 → 0.67) and ctu13-3 (0.98 → 0.83).

**Diagnosis:** SSL-pretrained embeddings push the encoder toward
"anomaly-from-baseline" features. This is exactly what iot23-35-1 needs
(near-zero positive rate, bot looks nothing like a benign baseline) but
*hurts* iot23-7-1 / ctu13-3 — these scenarios have small but non-trivial
positive counts whose bots look *similar* to benign hosts. SSL pretraining
makes the encoder over-attend to baseline-deviation, which collapses those
similar-to-benign bots' representations toward the benign mass.

The SAGPool wins on iot23-7-1 and ctu13-3 came from sparse pooling preserving
the fine-grained representations of these subtle bots. SSL pretraining
removed the fine-grained features. The two interventions therefore conflict.

## Final ranking

| Method | Phase | Test F1 | PR-AUC | iot23-35-1 |
|---|---|---|---|---|
| Phase 4 GINE | 4 | 0.9595 | 0.9610 | (not measured separately) |
| Phase 5 T-GINE-skip | 5 | 0.9661 | 0.9669 | 0.28 |
| Phase 5 + SSL init | 5 | 0.9601 | 0.9646 | **0.62** |
| Phase 6 DiffPool | 6 | 0.9673 | 0.9691 | 0.56 |
| Phase 6 + SSL-FT | 6.3 | 0.9664 | 0.9702 | 0.27 |
| **Phase 6.4 SAGPool** | **6.4** | **0.9674** | **0.9710** | 0.55 |
| Phase 6.5 SAGPool × SSL-FT | 6.5 | 0.9659 | 0.9682 | 0.59 |

**Phase 6.4 SAGPool** is the final SOTA — best aggregate F1, best PR-AUC,
preserves easy scenarios, unlocks two previously-hard scenarios (iot23-7-1,
ctu13-3).

**For iot23-35-1 specifically**, Phase 5 SSL-init remains the strongest model
(0.62) but at the cost of -0.0073 aggregate F1. The thesis can present this
as a deployment-time choice: prioritize aggregate detection (use Phase 6.4
SAGPool) or prioritize discovery of stealth campaigns with very few hosts
(use Phase 5 SSL-init or run both models in ensemble).

## What we learned about the architecture space

1. **Loss-level imbalance interventions (TAM) fail when scenarios have
   heterogeneous local class priors.** Single global margin can't fix per-
   scenario distributions ranging from 0.3% to 100% positive.
2. **Feature-level minority oversampling (GraphSHA) fails when minority
   training count is too small per graph.** Intra-graph mixup needs ≥2 same-
   class nodes per graph; iot23-35-1 violates this.
3. **Representation-level SSL pretraining helps iot23-35-1 at the flat-GNN
   level** but the gain is *destroyed by dense pooling*.
4. **Sparse pooling (SAGPool) preserves more rare-class signal than dense
   pooling (DiffPool)** and unlocks two previously-hard scenarios.
5. **SSL + Sparse-Pool does not compose** — SSL biases features toward
   anomaly-detection which conflicts with discriminative learning for
   "subtle-bots" scenarios.

These are useful structural results for the thesis discussion: each
intervention has a clear scope of applicability, and aggregate F1 in this
benchmark is bounded around 0.967 with the current encoder/pooling stack.

Further headroom requires either:
- More training data for iot23-35-1's scenario distribution
- Per-scenario model selection / ensemble routing
- A fundamentally different graph-construction step (more contextual nodes,
  longer-range edges)
