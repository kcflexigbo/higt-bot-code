# P2P botnet experiments: summary and verdict (2026-10-13)

## Verdict

1. **The P2P captures already in the thesis data do not test P2P detection.** IoT-23 1-1 (Hide and Seek) and 9-1 (Hajime) each have one infected device whose labelled traffic is a horizontal port scan. Every model, trees and GNNs alike, gets F1 1.000 on their test windows. CTU-13 scenario 12 (NSIS.ay) has three infected hosts but no bot-bot edge in any window graph (25 of 2,168 raw bot flows go between bots, and none survives graph construction as an edge between two labelled bots).
2. **Models trained on the thesis data do not transfer to unseen P2P botnets.**
   - On CTU-13 s12 the best transfer F1 is about 0.40 (XGB[x] 0.399 ± 0.088, HiGT-Bot v2 0.393 ± 0.012). Ranking is still good: PR-AUC is about 0.80 for the trees and 0.59–0.60 for the directed GINE and directed HiGT-Bot, and the thesis-validation threshold is too strict for this family.
   - On ISOT 2010 every model fails: F1 at most 0.003, PR-AUC at most 0.04, with thousands to tens of thousands of false alarms on the LBNL enterprise background. That traffic is unlike anything in the IoT training data.
3. **Within ISOT, where bots do contact each other, neighbour information helps, but explicit neighbour features help more than message passing.**
   - ISOT is the only dataset with a substantial P2P overlay inside the windows: 51.4% of bot host-windows have a bot neighbour. The comparable figures are 0% for CTU-13, 0.1% for IoT-23, and 31.5% for MedBIoT, which comes from Mirai propagation between testbed devices.
   - In window-grouped 5-fold CV, trees with mean/max neighbour features (XGB[x+nbr]) reach F1 0.808 ± 0.037 and PR-AUC 0.878 ± 0.014, against 0.534 / 0.564 for host features alone. The seed-42 ΔPR-AUC is +0.302, 95% CI [+0.018, +0.500].
   - Group similarity does not help here. HiGT-Bot v2 reaches 0.601 / 0.562, and adding grp to x+nbr lowers the scores.
   - Directed GINE (0.671 ± 0.169 / 0.563 ± 0.197) beats undirected GINE (0.451 / 0.345) on average over seeds. It is unstable across seeds, the seed-42 paired difference is not significant, and it stays well below XGB[x+nbr].
4. **What this means for the thesis.**
   - The "P2P" in the title can be supported only by ISOT. There, the hypothesis behind graph models holds in a weak form: relational evidence helps when bots talk to each other. A tree on aggregated neighbour features captures it better than a GNN does.
   - HiGT-Bot v2's group-similarity block is the right inductive bias for the centralised botnets of the main data, but not for this P2P trace.
   - These are small-sample, within-capture results: 35 positives in 18 adjacent windows, with ISOT's merged-trace artefacts (bot windows contain no LBNL background hosts). They should be reported as exploratory.

## Caveats

- ISOT bot labels come from the synthetic MACs of the dataset overview (Table 2). Storm's P2P (Overnet) traffic is labelled, but Waledac's P2P traffic is not, only its spam. External peers count as bots only if they originate marked flows.
- The paired bootstrap uses seed-42 out-of-fold scores. Seed 42 is the directed GINE's weakest seed (PR-AUC 0.341 against 0.632 and 0.716 for seeds 1 and 2), so the bootstrap rows involving the directed GINE are pessimistic for it.
- "Directed GINE (retrained)" in the transfer table was retrained on the thesis split with the P2P captures as extra held-out scenarios. The other GNN rows use the existing thesis checkpoints.
- Correction for `configs/split.yaml`: iot23-8-1 is Hakai, not Hide-and-Seek (official IoT-23 table).

Scripts: `scripts/queue_P_p2p.sh`, `src/data/parse_isot.py`, `scripts/p2p_graph_stats.py`, `scripts/p2p_family_eval.py`, `scripts/p2p_transfer_trees.py`, `scripts/p2p_gnn_eval.py`, `scripts/queue_Q_isot_cv.sh`, `scripts/p2p_isot_cv.py`, `scripts/p2p_summary.py`. Dataset provenance: `inventory.md`.

---


## 1. Do bots contact each other inside the monitored windows?

Window graphs (data/graphs_v3), corrected labels (bot = originator of bot traffic). Dataset rows pool every scenario with bots.

| Data | Bot host-windows | Bot-bot edges | Bots with a bot neighbour |
|---|---|---|---|
| CTU-13 | 1,194 | 0 | 0.0% |
| IoT-23 | 3,832 | 4 | 0.1% |
| MedBIoT | 677 | 1,272 | 31.5% |
| CTU-13 s12 (NSIS.ay, P2P) | 28 | 0 | 0.0% |
| IoT-23 1-1 (Hide and Seek, P2P) | 1,343 | 0 | 0.0% |
| IoT-23 9-1 (Hajime, P2P) | 234 | 0 | 0.0% |
| ISOT 2010 (Storm/Waledac P2P, Zeus) | 35 | 28 | 51.4% |

## 2. P2P families already in the thesis split (main-split test windows, saved predictions)

| Capture | Bots / host-windows (test) | Models | F1 | PR-AUC |
|---|---|---|---|---|
| iot23-1-1 Hide and Seek | 270 / 34,069 | HiGT-Bot v2 (x+grp), XGB[x], RF tuned, Directed GINE, Directed T-GINE, HiGT-Bot directed | 1.0 | 1.0 |
| iot23-9-1 Hajime | 49 / 153 | HiGT-Bot v2 (x+grp), XGB[x], RF tuned, Directed GINE, Directed T-GINE, HiGT-Bot directed | 1.0 | 1.0 |

Each capture has a single infected device whose labelled traffic is a horizontal port scan (raw_flow_check.txt); every model separates it perfectly. These captures do not test P2P behaviour.

## 3. Transfer: models trained on the thesis split, tested on unseen P2P captures

No retraining except the directed GINE, which was retrained on the unchanged thesis split with the P2P captures added as extra held-out test scenarios. Thresholds tuned on the thesis validation set.

| Target | Bots / host-windows | Model | F1 (thesis-val threshold) | PR-AUC | FN | FP |
|---|---|---|---|---|---|---|
| ctu13-12 | 28 / 471 | XGB[x] (3 seeds) | 0.399 ± 0.088 | 0.783 ± 0.014 | 21 | 1 |
| ctu13-12 | 28 / 471 | HiGT-Bot v2 (x+grp) (3 seeds) | 0.393 ± 0.012 | 0.802 ± 0.004 | 21 | 1 |
| ctu13-12 | 28 / 471 | RF | 0.133 | 0.800 | 26 | 0 |
| ctu13-12 | 28 / 471 | Directed GINE (retrained, seed 42) | 0.188 | 0.587 | 25 | 1 |
| ctu13-12 | 28 / 471 | Directed T-GINE (thesis checkpoint) | 0.000 | 0.318 | 28 | 2 |
| ctu13-12 | 28 / 471 | HiGT-Bot directed (thesis checkpoint) | 0.286 | 0.595 | 23 | 2 |
| isot-2010 | 35 / 102,636 | XGB[x] (3 seeds) | 0.003 ± 0.002 | 0.013 ± 0.001 | 15 | 14,816 |
| isot-2010 | 35 / 102,636 | HiGT-Bot v2 (x+grp) (3 seeds) | 0.002 ± 0.000 | 0.040 ± 0.003 | 12 | 21,181 |
| isot-2010 | 35 / 102,636 | RF | 0.003 | 0.013 | 14 | 12,746 |
| isot-2010 | 35 / 102,636 | Directed GINE (retrained, seed 42) | 0.000 | 0.000 | 25 | 42,553 |
| isot-2010 | 35 / 102,636 | Directed T-GINE (thesis checkpoint) | 0.001 | 0.000 | 11 | 56,923 |
| isot-2010 | 35 / 102,636 | HiGT-Bot directed (thesis checkpoint) | 0.002 | 0.001 | 14 | 26,353 |

## 4. Within ISOT 2010: window-grouped 5-fold cross-validation

All 35 bot host-windows lie in the last 18 of 886 windows, so a chronological split has no training bots. Windows are dealt into 5 folds stratified by bot presence (seeds 42, 1, 2); test = fold k, validation (threshold, early stopping) = fold k+1, train = the rest. ISOT merges traces recorded at different times: the bot windows contain only hosts of the honeynet/Ericsson portion (172.16.0.0/16 and their external peers), never LBNL background hosts, and adjacent bot windows are correlated. These are within-capture estimates on 35 positives and have wide uncertainty.

| Model | F1 | PR-AUC | FN | FP |
|---|---|---|---|---|
| XGB[x] | 0.534 ± 0.047 | 0.564 ± 0.023 | 16.7 | 15.7 |
| HiGT-Bot v2 (XGB[x+grp]) | 0.601 ± 0.028 | 0.562 ± 0.026 | 17.0 | 7.0 |
| XGB[x+nbr] | 0.808 ± 0.037 | 0.878 ± 0.014 | 7.3 | 5.7 |
| XGB[x+nbr+grp] | 0.674 ± 0.102 | 0.630 ± 0.086 | 14.7 | 5.0 |
| GINE (undirected) | 0.451 ± 0.022 | 0.345 ± 0.010 | 20.0 | 16.7 |
| Directed GINE | 0.671 ± 0.169 | 0.563 ± 0.197 | 10.7 | 12.0 |

Paired bootstrap over windows (2,000 resamples) of the seed-42 out-of-fold scores:

| Comparison (seed 42, out-of-fold) | ΔPR-AUC | 95% CI | one-sided p |
|---|---|---|---|
| XGB[x+nbr] − XGB[x] | +0.302 | [+0.018, +0.500] | 0.018 |
| HiGT-Bot v2 − XGB[x] | -0.045 | [-0.114, +0.090] | 0.796 |
| XGB[x+nbr] − HiGT-Bot v2 | +0.347 | [-0.028, +0.588] | 0.042 |
| Directed GINE − GINE | -0.008 | [-0.106, +0.095] | 0.613 |
| Directed GINE − XGB[x] | -0.250 | [-0.409, -0.137] | 1.000 |
| XGB[x+nbr] − Directed GINE | +0.552 | [+0.257, +0.735] | 0.000 |

## Files

graph_stats.json, raw_flow_check.txt, family_eval.json, trees_results.json, gnn_ckpt_results.json, dirgine_transfer_seed42.{json,npz}, isot_cv_*.json, isot_cv_oof_*.npz, inventory.md (dataset provenance, sha256), queue_P_status.txt, queue_Q_status.txt.
