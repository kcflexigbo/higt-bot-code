# E-GraphSAGE baseline under three labelling rules

E-GraphSAGE (Lo et al., NOMS 2022, arXiv:2103.16329) is re-implemented here as an extra published baseline. It is
trained under three label versions: v0 (any-endpoint, `data/graphs`), v1 (infected-source, `data/graphs_src`) and
v2 (corrected infected-source, `data/graphs_src2`). For each version there are 2 node-classification readouts × 3
seeds (42, 1, 2), all trained on CUDA, with no OOM and no CPU fallback.

The comparison is the tuned RF (`run_rf.py --tune-threshold`, seed 42, deterministic). It was re-run here with
`--save-preds` so that FN/FP could be computed for each subset. The v1 and v2 re-runs reproduce
`relabel/baseline_rf_tuned.json` and `v2/baseline_rf_tuned.json` exactly (test F1 0.9362 / 0.9368). The v0 tuned RF
is new.

Code:
- `src/models/egraphsage.py`
- `scripts/baselines/run_egraphsage.py`
- `scripts/queue_E_egraphsage.sh` (driver)
- `scripts/egraphsage_summary.py` (produces the tables below and `egraphsage_summary.json`)

## Implementation

- **Encoder (as in the authors' reference code, github.com/waimorris/E-GraphSAGE)**
  - The node input is all-ones, with dimension 10 (the edge-feature dimension). Our 9 engineered node features are
    not used.
  - K = 2 layers, hidden 128, ReLU, dropout 0.2 between layers.
  - Message `m_uv = W_msg[h_u || e_uv]`, neighbourhood `h_N(v) = MEAN_u m_uv`, update
    `h_v = ReLU(W_apply[h_v || h_N(v)])`.
  - Each directed host-pair edge is mirrored, with the same features, before aggregation. This reproduces the
    paper's undirected-to-directed graph, so the mean runs over all incident edges.
- **Readouts**
  - **node (b1):** `logits_v = Linear(z_v)`, trained with a node loss using the shared `train_one_model` loop.
  - **edge2node (b2):** `logits_uv = Linear([z_u || z_v])`, where u is the originator and v the responder. This is
    the paper's single-linear-layer edge predictor. A node's score is the max P(malicious) over its outgoing edges,
    or 0 if it has none.
- **Training (same conventions as `run_gnn.py`)**
  - Edge features are standardised with train statistics.
  - Class-weighted cross-entropy loss. Focal loss is available via `--focal-gamma` but was not used.
  - Adam (lr 1e-3, weight decay 1e-5), ReduceLROnPlateau, gradient clipping at 1.0.
  - Batches of 32 windows, at most 200 epochs.
  - Early stopping (patience 20) on **validation node F1 at threshold 0.5**. edge2node also early-stops on node F1.
  - The val-tuned threshold uses the same grid as RF (0.01–0.99, maximising val F1).
- **Outputs per run:** `<variant>_seed<s>.json` with `val`, `test`, `test_per_scenario`, `threshold_tuned`,
  `val_tuned`, `test_tuned` and `test_tuned_per_scenario`. `<variant>_seed<s>_preds.npz` and `..._preds_val.npz`
  hold `y`, `pred` (at 0.5), `proba`, `scenario`, `pred_tuned` and `threshold`.

### Edge labels for edge2node (derived from node labels)

Rule: `y_edge(u->v) = y_node[u]` under the active label version. An edge is malicious iff its **source
(originator)** host is a bot. The graphs store no flow-level labels, and each edge aggregates all flows of one
directed host pair within a 300 s window.

| label version | node rule | what a malicious edge means | malicious train edges | bots with no outgoing edge (train / val / test) |
|---|---|---|---|---|
| v0 | host is src OR dst of any malicious flow | originator was an endpoint of some malicious flow. This includes edges *from* victims and C2/responders; edges *to* them from benign hosts stay benign | 134,752 / 172,737 | 119,029 / 38,488 / 39,542 (96–98 % of v0 bots) |
| v1 | host is src of a malicious flow | originator itself sent malicious traffic | 134,361 / 172,737 | 0 / 0 / 0 |
| v2 | corrected infected-source | originator is a corrected infected source | 135,106 / 172,737 | 75 / 22 / 218 |

Edge labels are almost identical across the three versions (v0 and v1 differ on 391 of 172,737 train edges). The
versions differ mainly in their *node* positives. Under v0, a large set of destination-only hosts count as bots;
they are mostly IoT-23 scan and attack targets, and no outgoing edge represents them.

## Results (test split; mean ± SD over seeds 42/1/2; RF is one deterministic run)

How to read the tables:
- `thr` is the threshold that maximised validation F1, applied to test.
- Subsets:
  - **all:** 59,210 test nodes.
  - **CTU-13:** the CTU-13 tail windows plus the held-out ctu13-10.
  - **held-out:** ctu13-10, iot23-8-1 and medbiot-torii_mal_all.
- RF@0.5 uses `predict` (argmax), so ties at 0.5 go to benign.


### v0 — any-endpoint labels

**all test nodes**

| model | seeds | thr (val) | F1@0.5 | F1@thr | PR-AUC | FN@0.5 | FP@0.5 | FN@thr | FP@thr |
|---|---|---|---|---|---|---|---|---|---|
| RF (tuned, seed 42) | 1 | 0.24 | 0.960 | 0.963 | 0.960 | 663 | 2730 | 89 | 3052 |
| E-GraphSAGE node (z_v) | 3 | 0.19 ± 0.07 | 0.856 ± 0.000 | 0.857 ± 0.002 | 0.878 ± 0.016 | 787 ± 121 | 12784 ± 207 | 197 ± 11 | 13476 ± 251 |
| E-GraphSAGE edge→node (max) | 3 | 0.06 ± 0.05 | 0.062 ± 0.003 | 0.068 ± 0.000 | 0.701 ± 0.001 | 39733 ± 75 | 283 ± 154 | 39593 ± 3 | 656 ± 191 |

(n = 59,210 nodes, 41,063 bots)

**CTU-13 test nodes**

| model | seeds | thr (val) | F1@0.5 | F1@thr | PR-AUC | FN@0.5 | FP@0.5 | FN@thr | FP@thr |
|---|---|---|---|---|---|---|---|---|---|
| RF (tuned, seed 42) | 1 | 0.24 | 0.932 | 0.930 | 0.986 | 333 | 159 | 80 | 458 |
| E-GraphSAGE node (z_v) | 3 | 0.19 ± 0.07 | 0.834 ± 0.011 | 0.846 ± 0.021 | 0.826 ± 0.037 | 669 ± 83 | 531 ± 207 | 163 ± 1 | 1117 ± 204 |
| E-GraphSAGE edge→node (max) | 3 | 0.06 ± 0.05 | 0.148 ± 0.021 | 0.157 ± 0.007 | 0.462 ± 0.006 | 3367 ± 57 | 206 ± 127 | 3322 ± 2 | 523 ± 190 |

(n = 8,582 nodes, 3,679 bots)

**held-out test nodes**

| model | seeds | thr (val) | F1@0.5 | F1@thr | PR-AUC | FN@0.5 | FP@0.5 | FN@thr | FP@thr |
|---|---|---|---|---|---|---|---|---|---|
| RF (tuned, seed 42) | 1 | 0.24 | 0.942 | 0.897 | 0.990 | 20 | 77 | 1 | 184 |
| E-GraphSAGE node (z_v) | 3 | 0.19 ± 0.07 | 0.874 ± 0.011 | 0.860 ± 0.003 | 0.881 ± 0.017 | 123 ± 19 | 74 ± 30 | 71 ± 17 | 169 ± 22 |
| E-GraphSAGE edge→node (max) | 3 | 0.06 ± 0.05 | 0.504 ± 0.019 | 0.501 ± 0.014 | 0.583 ± 0.022 | 515 ± 21 | 54 ± 29 | 498 ± 3 | 113 ± 27 |

(n = 1,943 nodes, 805 bots)


### v1 — infected-source labels

**all test nodes**

| model | seeds | thr (val) | F1@0.5 | F1@thr | PR-AUC | FN@0.5 | FP@0.5 | FN@thr | FP@thr |
|---|---|---|---|---|---|---|---|---|---|
| RF (tuned, seed 42) | 1 | 0.20 | 0.770 | 0.936 | 0.978 | 509 | 12 | 81 | 96 |
| E-GraphSAGE node (z_v) | 3 | 0.89 ± 0.10 | 0.703 ± 0.074 | 0.751 ± 0.003 | 0.742 ± 0.020 | 279 ± 18 | 674 ± 342 | 361 ± 44 | 316 ± 63 |
| E-GraphSAGE edge→node (max) | 3 | 0.40 ± 0.16 | 0.801 ± 0.021 | 0.818 ± 0.052 | 0.867 ± 0.029 | 385 ± 65 | 108 ± 29 | 329 ± 161 | 130 ± 62 |

(n = 59,210 nodes, 1,380 bots)

**CTU-13 test nodes**

| model | seeds | thr (val) | F1@0.5 | F1@thr | PR-AUC | FN@0.5 | FP@0.5 | FN@thr | FP@thr |
|---|---|---|---|---|---|---|---|---|---|
| RF (tuned, seed 42) | 1 | 0.20 | 0.744 | 0.816 | 0.923 | 150 | 12 | 55 | 94 |
| E-GraphSAGE node (z_v) | 3 | 0.89 ± 0.10 | 0.558 ± 0.100 | 0.689 ± 0.036 | 0.665 ± 0.047 | 79 ± 13 | 429 ± 205 | 92 ± 3 | 174 ± 44 |
| E-GraphSAGE edge→node (max) | 3 | 0.40 ± 0.16 | 0.771 ± 0.028 | 0.781 ± 0.009 | 0.719 ± 0.056 | 107 ± 11 | 58 ± 10 | 92 ± 17 | 72 ± 29 |

(n = 8,582 nodes, 385 bots)

**held-out test nodes**

| model | seeds | thr (val) | F1@0.5 | F1@thr | PR-AUC | FN@0.5 | FP@0.5 | FN@thr | FP@thr |
|---|---|---|---|---|---|---|---|---|---|
| RF (tuned, seed 42) | 1 | 0.20 | 0.901 | 0.880 | 0.977 | 56 | 8 | 39 | 45 |
| E-GraphSAGE node (z_v) | 3 | 0.89 ± 0.10 | 0.762 ± 0.045 | 0.795 ± 0.011 | 0.750 ± 0.036 | 71 ± 17 | 101 ± 28 | 84 ± 3 | 52 ± 9 |
| E-GraphSAGE edge→node (max) | 3 | 0.40 ± 0.16 | 0.818 ± 0.020 | 0.813 ± 0.013 | 0.810 ± 0.048 | 84 ± 16 | 33 ± 10 | 84 ± 16 | 37 ± 16 |

(n = 1,943 nodes, 346 bots)


### v2 — corrected infected-source labels

**all test nodes**

| model | seeds | thr (val) | F1@0.5 | F1@thr | PR-AUC | FN@0.5 | FP@0.5 | FN@thr | FP@thr |
|---|---|---|---|---|---|---|---|---|---|
| RF (tuned, seed 42) | 1 | 0.19 | 0.783 | 0.937 | 0.962 | 545 | 44 | 66 | 142 |
| E-GraphSAGE node (z_v) | 3 | 0.96 ± 0.01 | 0.689 ± 0.047 | 0.759 ± 0.005 | 0.724 ± 0.054 | 252 ± 14 | 985 ± 300 | 408 ± 22 | 353 ± 43 |
| E-GraphSAGE edge→node (max) | 3 | 0.34 ± 0.19 | 0.765 ± 0.023 | 0.773 ± 0.019 | 0.687 ± 0.037 | 446 ± 56 | 269 ± 92 | 399 ± 6 | 310 ± 77 |

(n = 59,210 nodes, 1,607 bots)

**CTU-13 test nodes**

| model | seeds | thr (val) | F1@0.5 | F1@thr | PR-AUC | FN@0.5 | FP@0.5 | FN@thr | FP@thr |
|---|---|---|---|---|---|---|---|---|---|
| RF (tuned, seed 42) | 1 | 0.19 | 0.769 | 0.863 | 0.889 | 170 | 39 | 28 | 127 |
| E-GraphSAGE node (z_v) | 3 | 0.96 ± 0.01 | 0.575 ± 0.070 | 0.746 ± 0.018 | 0.683 ± 0.017 | 51 ± 3 | 655 ± 209 | 95 ± 5 | 193 ± 22 |
| E-GraphSAGE edge→node (max) | 3 | 0.34 ± 0.19 | 0.580 ± 0.061 | 0.604 ± 0.045 | 0.437 ± 0.060 | 234 ± 36 | 177 ± 83 | 204 ± 5 | 211 ± 75 |

(n = 8,582 nodes, 517 bots)

**held-out test nodes**

| model | seeds | thr (val) | F1@0.5 | F1@thr | PR-AUC | FN@0.5 | FP@0.5 | FN@thr | FP@thr |
|---|---|---|---|---|---|---|---|---|---|
| RF (tuned, seed 42) | 1 | 0.19 | 0.816 | 0.877 | 0.939 | 93 | 21 | 22 | 69 |
| E-GraphSAGE node (z_v) | 3 | 0.96 ± 0.01 | 0.756 ± 0.010 | 0.770 ± 0.008 | 0.776 ± 0.025 | 44 ± 3 | 152 ± 11 | 93 ± 4 | 58 ± 1 |
| E-GraphSAGE edge→node (max) | 3 | 0.34 ± 0.19 | 0.812 ± 0.011 | 0.799 ± 0.017 | 0.722 ± 0.050 | 68 ± 10 | 60 ± 15 | 68 ± 5 | 72 ± 9 |

(n = 1,943 nodes, 346 bots)


## Interpretation

**Headline: all test nodes, F1 at the val-tuned threshold**

| label | RF | E-GraphSAGE node | E-GraphSAGE edge->node | RF minus best E-GraphSAGE |
|---|---|---|---|---|
| v0 | 0.963 | 0.857 ± 0.002 | 0.068 ± 0.000 | +0.106 |
| v1 | 0.936 | 0.751 ± 0.003 | 0.818 ± 0.052 | +0.118 |
| v2 | 0.937 | 0.759 ± 0.005 | 0.773 ± 0.019 | +0.164 |

1. **The hypothesis that E-GraphSAGE beats RF under v0 and loses under v1/v2 is not supported.**
   - E-GraphSAGE loses to the tuned RF under all three label versions. It loses on every subset (all, CTU-13,
     held-out), by F1 at the tuned threshold and by PR-AUC.
   - The gap widens somewhat from v0 to v2 (+0.106 to +0.164). That is consistent with the labelling effect, but it
     is a change in size, not a sign flip.
2. **v0 inflates the absolute scores; it does not favour the graph model.**
   - Under v0, 69 % of test nodes are bots (41,063 / 59,210; 73 % on IoT-23, 98 % on MedBIoT).
   - A trivial all-bot classifier would already score F1 = 0.819, with PR-AUC equal to the prevalence, 0.69.
   - E-GraphSAGE-node's 0.857 is only slightly above that trivial score, and it carries ~13,000 FPs.
   - So v0 F1 is largely a statement about prevalence, for both models.
3. **The labelling rule flips which E-GraphSAGE readout works. This is the labelling effect that does generalise to
   the published method.**
   - The edge->node readout is the one closest to how E-GraphSAGE is used in the paper.
   - Under v0 it collapses (F1 0.07). In each window, 96 % of v0 test bots never originate an edge; they are bots
     only as responders, so no source-attributed edge score can reach them.
   - Under v1 and v2 the same readout is the better of the two (0.818 / 0.773 vs 0.751 / 0.759).
   - In other words, a flow-level classifier implicitly assumes an infected-source definition of a "bot host".
     Any-endpoint host labels and flow-level IDS labels measure different things.
4. **Calibration and threshold transfer.**
   - The node readout's val-tuned threshold is extreme under v1/v2 (0.89 / 0.96). Its F1@0.5 varies a lot across
     seeds (±0.05–0.07) and produces many FPs.
   - edge2node is better calibrated at 0.5. Under v1 at 0.5 it even beats RF@0.5 (0.801 vs 0.770).
   - However, RF@0.5 is weak only because balanced class weights shift its operating point. At the tuned threshold,
     which is the fair comparison, RF wins clearly.
5. **Held-out scenarios show the same ordering.** Under v1/v2 at 0.5, edge2node comes close to RF (0.818 vs 0.901;
   0.812 vs 0.816). It stays below RF at the tuned threshold and in PR-AUC.

**Caveats**
- E-GraphSAGE is run as published, with all-ones node input, so it sees only aggregated edge (flow) statistics. RF
  sees only the 9 engineered node features. This is a method-as-published comparison, not a same-features one.
- No hyperparameter search was done beyond the paper's settings (K = 2, hidden 128, dropout 0.2, lr 1e-3).
- There are 3 E-GraphSAGE seeds; RF is a single deterministic run.

## Deviations from the original E-GraphSAGE

- **Task.** We classify hosts (nodes), not flows (edges). This is why there are two readouts: (b1) a linear head on
  z_v, and (b2) the paper's edge predictor, trained on source-derived edge labels and max-pooled over each host's
  outgoing edges.
- **Graph construction.**
  - Edges are per-window aggregates for each directed host pair (10 features). The paper uses one edge per NetFlow
    record, with IP:port as the node identity.
  - We use one graph per 300 s window, not a single static graph.
- **Message function.** It includes the source embedding (`W_msg[h_u || e_uv]`), following the authors' released
  code. The paper's Algorithm 1 aggregates edge features only.
- **Training setup.** Edge-feature standardisation, class-weighted CE, ReduceLROnPlateau, gradient clipping and early
  stopping on validation node F1 follow this repo's `run_gnn.py` conventions, not the paper's fixed-epoch training.
