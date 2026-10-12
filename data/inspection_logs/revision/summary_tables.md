# Review-revision experiments: tables

## C1. Confirmatory evaluation on captures never used before (frozen models, validation thresholds)

Malicious IoT-23 captures never evaluated before: iot23-34-1, iot23-36-1, iot23-39-1, iot23-42-1, iot23-43-1, iot23-44-1, iot23-49-1, iot23-52-1 (7,355 host-windows, 743 bot host-windows). Benign-only MedBIoT captures: medbiot-bashlite_leg, medbiot-mirai_leg, medbiot-torii_leg (14,174 host-windows, no bots). Models trained once on the main training split (5 seeds), threshold from the main validation set, no retuning.

| Model | F1 (malicious captures) | PR-AUC | FN | FP | false alarms per 1,000 benign host-windows (benign captures) | alarms per host-day |
|---|---|---|---|---|---|---|
| XGB[x] | 0.965 ± 0.015 | 0.993 ± 0.001 | 40 ± 22 | 11 ± 1 | 199.94 ± 20.14 | 57.584 ± 5.800 |
| v2 XGB[x+grp] | 0.985 ± 0.001 | 0.996 ± 0.001 | 12 ± 2 | 11 ± 1 | 257.95 ± 24.08 | 74.290 ± 6.934 |
| RF tuned | 0.980 ± 0.001 | 0.993 ± 0.000 | 20 ± 2 | 10 ± 0 | 153.52 ± 17.26 | 44.214 ± 4.971 |
| XGB[x+wnorm] | 0.986 ± 0.001 | 0.997 ± 0.001 | 7 ± 1 | 13 ± 1 | 336.76 ± 18.78 | 96.986 ± 5.407 |
| BotMiner-style cohort (unsupervised) | 0.000 | 0.101 | 743 | 58 | 65.75 | 18.937 |
| Directed T-GINE (1 seed) | 0.978 | 0.972 | 7 | 26 | 311.91 | 89.830 |
| HiGT-Bot directed (3 seeds) | 0.980 ± 0.000 | 0.933 ± 0.002 | 7 ± 1 | 23 ± 1 | 304.85 ± 56.86 | 87.798 ± 16.376 |

Per-scenario recall on the malicious captures (mean over seeds):

| Scenario | bot host-windows | XGB[x] | v2 XGB[x+grp] | RF tuned | XGB[x+wnorm] |
|---|---|---|---|---|---|
| iot23-34-1 | 78 | 0.810 | 0.931 | 0.831 | 0.979 |
| iot23-36-1 | 289 | 0.934 | 0.998 | 1.000 | 1.000 |
| iot23-39-1 | 93 | 0.978 | 0.978 | 0.972 | 0.978 |
| iot23-42-1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| iot23-43-1 | 15 | 0.867 | 0.867 | 0.867 | 0.880 |
| iot23-44-1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| iot23-49-1 | 89 | 1.000 | 1.000 | 1.000 | 1.000 |
| iot23-52-1 | 177 | 1.000 | 1.000 | 1.000 | 1.000 |

Secondary set (captures listed as excluded for size; their bot counts were inspected earlier): 697 host-windows, 17 bots.

| Model | F1 | FN | FP |
|---|---|---|---|
| XGB[x] | 0.205 ± 0.011 | 6 ± 0 | 77 ± 3 |
| v2 XGB[x+grp] | 0.220 ± 0.008 | 6 ± 1 | 75 ± 2 |
| RF tuned | 0.142 ± 0.028 | 10 ± 2 | 77 ± 2 |

## C1c. Block selection on the validation set alone (seeds 42, 1, 2)

| Blocks | val PR-AUC | val F1 (val-tuned threshold) | test F1 | test PR-AUC |
|---|---|---|---|---|
| XGB[x] | 0.9999 ± 0.0000 | 0.9957 ± 0.0003 | 0.949 ± 0.004 | 0.996 ± 0.001 |
| XGB[x+seq] | 0.9998 ± 0.0000 | 0.9955 ± 0.0011 | 0.964 ± 0.002 | 0.994 ± 0.000 |
| v2 XGB[x+grp] | 0.9999 ± 0.0000 | 0.9956 ± 0.0006 | 0.979 ± 0.001 | 0.998 ± 0.000 |
| XGB[x+seq+grp] | 0.9999 ± 0.0000 | 0.9967 ± 0.0000 | 0.966 ± 0.002 | 0.997 ± 0.000 |
| XGB[x+emb] | 0.9997 ± 0.0000 | 0.9972 ± 0.0000 | 0.941 ± 0.005 | 0.990 ± 0.000 |
| XGB[x+grp+emb] | 0.9996 ± 0.0001 | 0.9968 ± 0.0003 | 0.951 ± 0.000 | 0.993 ± 0.001 |
| XGB[x+seq+grp+emb] | 0.9998 ± 0.0000 | 0.9961 ± 0.0006 | 0.957 ± 0.001 | 0.995 ± 0.000 |
| XGB[x+nbr+seq+grp] | 0.9999 ± 0.0000 | 0.9970 ± 0.0003 | 0.968 ± 0.002 | 0.998 ± 0.000 |

Best by validation PR-AUC: **XGB[x+nbr+seq+grp]**; best by validation F1: **XGB[x+emb]**. Note: the validation set contains no held-out scenario.

## M2. Cohort baselines and decomposition of the group block (main test split, 5 seeds)

| Model | F1 | PR-AUC | FN | FP | held-out F1 | ctu13-10 F1 |
|---|---|---|---|---|---|---|
| XGB[x] | 0.947 ± 0.004 | 0.996 ± 0.000 | 150 ± 14 | 13 ± 3 | 0.803 ± 0.019 | 0.465 ± 0.057 |
| v2 XGB[x+grp] | 0.978 ± 0.001 | 0.998 ± 0.000 | 49 ± 5 | 20 ± 2 | 0.929 ± 0.006 | 0.787 ± 0.026 |
| RF tuned | 0.937 ± 0.011 | 0.995 ± 0.000 | 177 ± 32 | 17 ± 2 | 0.773 ± 0.009 | 0.308 ± 0.043 |
| XGB[x+wnorm] | 0.970 ± 0.001 | 0.995 ± 0.000 | 66 ± 6 | 29 ± 4 | 0.899 ± 0.009 | 0.664 ± 0.027 |
| XGB[x+grp_prof] | 0.977 ± 0.002 | 0.998 ± 0.000 | 48 ± 8 | 25 ± 4 | 0.930 ± 0.011 | 0.780 ± 0.044 |
| XGB[x+grp_share] | 0.969 ± 0.006 | 0.997 ± 0.000 | 85 ± 21 | 13 ± 2 | 0.877 ± 0.029 | 0.683 ± 0.036 |
| XGB[x+wnorm+grp] | 0.972 ± 0.002 | 0.997 ± 0.000 | 65 ± 8 | 24 ± 10 | 0.909 ± 0.009 | 0.721 ± 0.030 |
| BotMiner-style cohort (unsupervised) | 0.261 | 0.234 | 1283 | 554 | 0.250 | 0.813 |

Feature importance of the x+grp model (seed 42): gain share, and drop in validation PR-AUC when the feature is permuted (validation PR-AUC 0.9999).

| Feature | gain share | permutation drop |
|---|---|---|
| x0 | 0.723 | 0.4702 |
| x4 | 0.181 | 0.0024 |
| x2 | 0.020 | 0.0042 |
| x1 | 0.014 | 0.0001 |
| grp_share_r2 | 0.013 | 0.0000 |
| x5 | 0.008 | 0.0022 |
| x8 | 0.007 | 0.0004 |
| grp_count_r2 | 0.006 | -0.0000 |
| x7 | 0.006 | 0.0001 |
| grp_n_cotarget_hosts | 0.005 | -0.0000 |
| grp_nn_mean_dist | 0.003 | 0.0001 |
| grp_nn5_dist | 0.003 | 0.0002 |
| x3 | 0.003 | 0.0002 |
| grp_max_target_jaccard | 0.002 | 0.0000 |
| x6 | 0.002 | 0.0002 |
| grp_nn1_dist | 0.002 | 0.0000 |
| grp_share_r1 | 0.001 | -0.0000 |

Block permutation (validation PR-AUC drop): all x 0.9350; all grp 0.0006; grp profile (g1-6) 0.0004; grp shared-target (g7-8) 0.0000

## M6. Recall by number of infected hosts in the window (main test split, means over seeds)

| Bots in window | bot host-windows | XGB[x] | v2 XGB[x+grp] | XGB[x+wnorm] | BotMiner-style |
|---|---|---|---|---|---|
| 1 (single infected host) | 1079 | 0.964 | 0.991 | 0.989 | 0.034 |
| 2-3 | 7 | 0.000 | 0.171 | 0.429 | 0.286 |
| 4-9 | 75 | 0.747 | 0.960 | 0.875 | 0.867 |
| >= 10 | 446 | 0.809 | 0.932 | 0.910 | 0.493 |

## M8. Macro-averaged metrics and end-to-end recall (main test split; means over seeds)

Macro = unweighted mean over the 23 test scenarios with at least one bot host-window. End-to-end recall counts infected host-windows dropped by graph construction as misses; for split scenarios the dropped count in the test period is estimated from the scenario's kept fraction (blind_spot_counts), for held-out scenarios (all windows in test) it is exact.

| Model | pooled F1 | macro F1 | macro PR-AUC | CTU-13 F1 | IoT-23 F1 | MedBIoT F1 | recall (in graph) | end-to-end recall | held-out end-to-end recall | FP per 1,000 benign host-windows |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 XGB[x+grp] (5 seeds) | 0.978 ± 0.001 | 0.967 ± 0.002 | 0.992 ± 0.001 | 0.945 ± 0.004 | 0.994 ± 0.002 | 0.994 ± 0.002 | 0.970 ± 0.003 | 0.704 ± 0.002 | 0.594 ± 0.008 | 0.35 ± 0.03 |
| XGB[x] (5 seeds) | 0.947 ± 0.004 | 0.942 ± 0.004 | 0.988 ± 0.001 | 0.903 ± 0.005 | 0.975 ± 0.003 | 0.943 ± 0.006 | 0.907 ± 0.008 | 0.659 ± 0.006 | 0.456 ± 0.019 | 0.22 ± 0.05 |
| RF tuned (5 seeds) | 0.937 ± 0.011 | 0.920 ± 0.017 | 0.984 ± 0.001 | 0.882 ± 0.003 | 0.975 ± 0.001 | 0.917 ± 0.060 | 0.890 ± 0.020 | 0.646 ± 0.015 | 0.426 ± 0.009 | 0.30 ± 0.04 |
| Directed GINE (3 seeds) | 0.950 ± 0.011 | 0.919 ± 0.016 | 0.955 ± 0.006 | 0.880 ± 0.021 | 0.987 ± 0.003 | 0.965 ± 0.019 | 0.914 ± 0.018 | 0.664 ± 0.013 | 0.506 ± 0.033 | 0.27 ± 0.06 |
| HiGT-Bot directed (3 seeds) | 0.940 ± 0.009 | 0.906 ± 0.020 | 0.936 ± 0.011 | 0.931 ± 0.014 | 0.989 ± 0.002 | 0.801 ± 0.058 | 0.912 ± 0.021 | 0.662 ± 0.015 | 0.576 ± 0.033 | 0.80 ± 0.28 |
| Directed T-GINE (1 seed) | 0.947 | 0.922 | 0.933 | 0.939 | 0.985 | 0.856 | 0.945 | 0.686 | 0.592 | 1.39 |
| SAGPool (1 seed) | 0.950 | 0.902 | 0.940 | 0.895 | 0.979 | 0.965 | 0.923 | 0.670 | 0.511 | 0.56 |

Benign host-windows in the test split: 57,603. One host-window = 300 s, so alarms per host-day = (alarms per 1,000 host-windows) x 288/1000.

## M9. CTU-13 leave-one-scenario-out, quantile scaler refitted per fold (3 seeds)

Note: the ctu13-10 fold trains on the same scenarios as the main split (ctu13-10 is held out there too), so it largely repeats the main held-out result.

| Left-out | model | PR-AUC | F1 at validation threshold | F1 at best threshold (oracle) | recall at 1% FPR |
|---|---|---|---|---|---|
| ctu13-1 | XGB[x] | 1.000 ± 0.000 | 0.986 ± 0.017 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| ctu13-1 | v2 XGB[x+grp] | 1.000 ± 0.000 | 0.978 ± 0.013 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| ctu13-10 | XGB[x] | 0.851 ± 0.019 | 0.492 ± 0.061 | 0.790 ± 0.006 | 0.571 ± 0.113 |
| ctu13-10 | v2 XGB[x+grp] | 0.921 ± 0.004 | 0.789 ± 0.032 | 0.898 ± 0.026 | 0.791 ± 0.087 |
| ctu13-13 | XGB[x] | 0.986 ± 0.000 | 0.929 ± 0.046 | 0.980 ± 0.002 | 1.000 ± 0.000 |
| ctu13-13 | v2 XGB[x+grp] | 0.970 ± 0.001 | 0.691 ± 0.071 | 0.965 ± 0.006 | 1.000 ± 0.000 |
| ctu13-2 | XGB[x] | 1.000 ± 0.000 | 0.988 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| ctu13-2 | v2 XGB[x+grp] | 1.000 ± 0.000 | 0.992 ± 0.007 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| ctu13-3 | XGB[x] | 0.999 ± 0.000 | 0.976 ± 0.004 | 0.991 ± 0.000 | 1.000 ± 0.000 |
| ctu13-3 | v2 XGB[x+grp] | 0.997 ± 0.001 | 0.947 ± 0.004 | 0.976 ± 0.011 | 1.000 ± 0.000 |
| ctu13-4 | XGB[x] | 0.843 ± 0.007 | 0.809 ± 0.018 | 0.837 ± 0.006 | 0.800 ± 0.000 |
| ctu13-4 | v2 XGB[x+grp] | 0.804 ± 0.006 | 0.739 ± 0.010 | 0.776 ± 0.021 | 0.787 ± 0.023 |
| ctu13-6 | XGB[x] | 0.994 ± 0.002 | 0.936 ± 0.021 | 0.980 ± 0.000 | 1.000 ± 0.000 |
| ctu13-6 | v2 XGB[x+grp] | 0.992 ± 0.002 | 0.941 ± 0.000 | 0.980 ± 0.000 | 1.000 ± 0.000 |
| ctu13-8 | XGB[x] | 0.521 ± 0.010 | 0.336 ± 0.030 | 0.379 ± 0.011 | 0.263 ± 0.004 |
| ctu13-8 | v2 XGB[x+grp] | 0.488 ± 0.007 | 0.351 ± 0.006 | 0.386 ± 0.002 | 0.275 ± 0.007 |
| ctu13-9 | XGB[x] | 0.998 ± 0.000 | 0.981 ± 0.006 | 0.986 ± 0.001 | 0.997 ± 0.000 |
| ctu13-9 | v2 XGB[x+grp] | 0.998 ± 0.000 | 0.983 ± 0.002 | 0.987 ± 0.001 | 0.997 ± 0.000 |

| Model | mean PR-AUC | mean F1 (val thr) | mean F1 (oracle thr) | mean recall at 1% FPR |
|---|---|---|---|---|
| XGB[x] | 0.910 | 0.826 | 0.882 | 0.848 |
| v2 XGB[x+grp] | 0.908 | 0.823 | 0.885 | 0.872 |