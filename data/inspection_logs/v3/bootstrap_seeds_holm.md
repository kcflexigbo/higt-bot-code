# Seed x window paired bootstrap with Holm correction (2000 resamples)

Windows resampled with replacement, stratified by scenario; each model's training seed drawn at random per resample (trees: 5 seeds; directed GINE and directed HiGT-Bot: 3 seeds; T-GINE and SAGPool: 1 seed, so their training variance is not included). Point values are means over seeds. Delta = HiGT-Bot v2 minus comparator; two-sided p; Holm over all comparisons x metrics within each subset.

## all

| Model | seeds | F1 (mean ± sd) | PR-AUC |
|---|---|---|---|
| HiGT-Bot v2 (XGB x+grp) | 5 | 0.978 ± 0.001 | 0.998 |
| XGB[x] | 5 | 0.947 ± 0.004 | 0.996 |
| RF tuned | 5 | 0.937 ± 0.011 | 0.995 |
| Directed GINE | 3 | 0.950 ± 0.011 | 0.975 |
| HiGT-Bot directed | 3 | 0.940 ± 0.009 | 0.915 |
| Directed T-GINE | 1 | 0.947 | 0.918 |
| SAGPool | 1 | 0.950 | 0.959 |

| Comparison | metric | Delta [95% CI] | p (two-sided) | p (Holm) |
|---|---|---|---|---|
| v2 - XGB[x] | f1 | +0.031 [+0.015, +0.051] | < 0.0010 | < 0.0010 |
| v2 - XGB[x] | pr_auc | +0.003 [+0.001, +0.005] | 0.0010 | 0.0030 |
| v2 - RF tuned | f1 | +0.042 [+0.021, +0.065] | < 0.0010 | < 0.0010 |
| v2 - RF tuned | pr_auc | +0.004 [+0.002, +0.005] | < 0.0010 | < 0.0010 |
| v2 - Directed GINE | f1 | +0.028 [+0.009, +0.050] | < 0.0010 | < 0.0010 |
| v2 - Directed GINE | pr_auc | +0.023 [+0.008, +0.045] | < 0.0010 | < 0.0010 |
| v2 - HiGT-Bot directed | f1 | +0.038 [+0.014, +0.060] | 0.0010 | 0.0030 |
| v2 - HiGT-Bot directed | pr_auc | +0.084 [+0.024, +0.133] | < 0.0010 | < 0.0010 |
| v2 - Directed T-GINE | f1 | +0.031 [+0.009, +0.053] | 0.0010 | 0.0030 |
| v2 - Directed T-GINE | pr_auc | +0.080 [+0.046, +0.114] | < 0.0010 | < 0.0010 |
| v2 - SAGPool | f1 | +0.028 [+0.014, +0.044] | < 0.0010 | < 0.0010 |
| v2 - SAGPool | pr_auc | +0.039 [+0.019, +0.063] | < 0.0010 | < 0.0010 |

## held-out

| Model | seeds | F1 (mean ± sd) | PR-AUC |
|---|---|---|---|
| HiGT-Bot v2 (XGB x+grp) | 5 | 0.929 ± 0.006 | 0.988 |
| XGB[x] | 5 | 0.803 ± 0.019 | 0.974 |
| RF tuned | 5 | 0.773 ± 0.009 | 0.979 |
| Directed GINE | 3 | 0.849 ± 0.033 | 0.877 |
| HiGT-Bot directed | 3 | 0.902 ± 0.022 | 0.905 |
| Directed T-GINE | 1 | 0.919 | 0.977 |
| SAGPool | 1 | 0.847 | 0.871 |

| Comparison | metric | Delta [95% CI] | p (two-sided) | p (Holm) |
|---|---|---|---|---|
| v2 - XGB[x] | f1 | +0.126 [+0.064, +0.195] | < 0.0010 | < 0.0010 |
| v2 - XGB[x] | pr_auc | +0.013 [+0.001, +0.030] | 0.0340 | 0.1700 |
| v2 - RF tuned | f1 | +0.156 [+0.092, +0.223] | < 0.0010 | < 0.0010 |
| v2 - RF tuned | pr_auc | +0.008 [-0.009, +0.023] | 0.2430 | 0.9720 |
| v2 - Directed GINE | f1 | +0.081 [+0.018, +0.155] | 0.0020 | 0.0160 |
| v2 - Directed GINE | pr_auc | +0.110 [+0.041, +0.184] | < 0.0010 | < 0.0010 |
| v2 - HiGT-Bot directed | f1 | +0.027 [-0.049, +0.105] | 0.4770 | 0.9720 |
| v2 - HiGT-Bot directed | pr_auc | +0.083 [+0.013, +0.187] | 0.0090 | 0.0630 |
| v2 - Directed T-GINE | f1 | +0.010 [-0.059, +0.085] | 0.7610 | 0.9720 |
| v2 - Directed T-GINE | pr_auc | +0.011 [-0.007, +0.036] | 0.2520 | 0.9720 |
| v2 - SAGPool | f1 | +0.082 [+0.018, +0.148] | 0.0100 | 0.0630 |
| v2 - SAGPool | pr_auc | +0.116 [+0.051, +0.184] | < 0.0010 | < 0.0010 |

## ctu13-10

| Model | seeds | F1 (mean ± sd) | PR-AUC |
|---|---|---|---|
| HiGT-Bot v2 (XGB x+grp) | 5 | 0.787 ± 0.026 | 0.921 |
| XGB[x] | 5 | 0.465 ± 0.057 | 0.849 |
| RF tuned | 5 | 0.308 ± 0.043 | 0.826 |
| Directed GINE | 3 | 0.407 ± 0.160 | 0.524 |
| HiGT-Bot directed | 3 | 0.675 ± 0.099 | 0.619 |
| Directed T-GINE | 1 | 0.744 | 0.874 |
| SAGPool | 1 | 0.424 | 0.420 |

| Comparison | metric | Delta [95% CI] | p (two-sided) | p (Holm) |
|---|---|---|---|---|
| v2 - XGB[x] | f1 | +0.322 [+0.098, +0.590] | 0.0020 | 0.0160 |
| v2 - XGB[x] | pr_auc | +0.072 [-0.001, +0.178] | 0.0530 | 0.2120 |
| v2 - RF tuned | f1 | +0.479 [+0.222, +0.724] | < 0.0010 | < 0.0010 |
| v2 - RF tuned | pr_auc | +0.095 [+0.011, +0.207] | 0.0210 | 0.1260 |
| v2 - Directed GINE | f1 | +0.380 [+0.101, +0.722] | < 0.0010 | < 0.0010 |
| v2 - Directed GINE | pr_auc | +0.397 [+0.166, +0.597] | 0.0010 | 0.0090 |
| v2 - HiGT-Bot directed | f1 | +0.112 [-0.179, +0.437] | 0.4690 | 0.9380 |
| v2 - HiGT-Bot directed | pr_auc | +0.302 [+0.031, +0.650] | 0.0270 | 0.1350 |
| v2 - Directed T-GINE | f1 | +0.043 [-0.246, +0.353] | 0.7290 | 0.9380 |
| v2 - Directed T-GINE | pr_auc | +0.047 [-0.036, +0.196] | 0.2790 | 0.8370 |
| v2 - SAGPool | f1 | +0.363 [+0.072, +0.687] | 0.0140 | 0.0980 |
| v2 - SAGPool | pr_auc | +0.501 [+0.269, +0.683] | < 0.0010 | < 0.0010 |
