# Phase 8.6 — Significance of HiGT-Bot (GT-edge, seed 42)

Test set: 59,210 host-window nodes in 1,232 windows. Bootstrap: 2000 resamples of whole windows (cluster bootstrap), percentile 95% CIs. McNemar: exact two-sided binomial test on discordant nodes.

## Per-model metrics (95% CI)

| Model | F1 | PR-AUC | FN |
|---|---|---|---|
| HiGT-Bot (GT-edge, seed 42) | 0.9677 [0.9657, 0.9698] | 0.9700 [0.9655, 0.9742] | 86 [53, 124] |
| HiGT-Bot (GT-edge, seed 1) | 0.9675 [0.9654, 0.9696] | 0.9692 [0.9643, 0.9740] | 86 [56, 122] |
| HiGT-Bot (GT-edge, seed 2) | 0.9654 [0.9616, 0.9686] | 0.9690 [0.9645, 0.9732] | 285 [89, 566] |
| SAGPool, no GT (Phase 6.4) | 0.9674 [0.9654, 0.9697] | 0.9710 [0.9663, 0.9755] | 132 [74, 201] |
| T-GINE + skip, no hierarchy (Phase 5) | 0.9661 [0.9627, 0.9690] | 0.9670 [0.9618, 0.9720] | 111 [76, 151] |
| GINE, flat (Phase 4, rerun) | 0.9549 [0.9461, 0.9619] | 0.9609 [0.9501, 0.9686] | 776 [393, 1337] |
| Random Forest (Phase 4, rerun) | 0.9597 [0.9577, 0.9619] | 0.9600 [0.9546, 0.9653] | 663 [561, 776] |

## HiGT-Bot (GT-edge, seed 42) minus comparator

| Comparator | ΔF1 [95% CI] | ΔPR-AUC [95% CI] | ΔFN [95% CI] | McNemar p (all) | McNemar p (bots only; ref-only / other-only detections) |
|---|---|---|---|---|---|
| HiGT-Bot (GT-edge, seed 1) | +0.0002 [-0.0002, +0.0005] | +0.0008 [-0.0006, +0.0021] | +0 [-21, +27] | 1.33e-01 | 1.00e+00 (21 / 21) |
| HiGT-Bot (GT-edge, seed 2) | +0.0023 [+0.0001, +0.0054] | +0.0009 [+0.0002, +0.0018] | -199 [-458, -22] | 7.89e-41 | 3.22e-57 (201 / 2) |
| SAGPool, no GT (Phase 6.4) | +0.0002 [-0.0005, +0.0009] | -0.0010 [-0.0023, -0.0000] | -46 [-106, -3] | 1.55e-01 | 3.81e-08 (59 / 13) |
| T-GINE + skip, no hierarchy (Phase 5) | +0.0015 [-0.0001, +0.0042] | +0.0030 [+0.0013, +0.0046] | -25 [-58, +8] | 3.81e-15 | 1.05e-02 (57 / 32) |
| GINE, flat (Phase 4, rerun) | +0.0127 [+0.0056, +0.0219] | +0.0091 [+0.0027, +0.0185] | -690 [-1254, -310] | 1.39e-221 | 4.97e-163 (733 / 43) |
| Random Forest (Phase 4, rerun) | +0.0080 [+0.0066, +0.0094] | +0.0100 [+0.0073, +0.0125] | -577 [-686, -470] | 2.42e-110 | 2.26e-118 (646 / 69) |
