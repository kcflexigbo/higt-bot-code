# Paired window-level bootstrap (2000 resamples, stratified by scenario)

Delta = first minus second model; p = share of resamples where the delta is <= 0.

## all

| Comparison | dF1 [95% CI] | p | dPR-AUC [95% CI] | p |
|---|---|---|---|---|
| HiGT-Bot v2 (x+grp) - XGB[x] (v3 features) | +0.013 [+0.004, +0.024] | 0.000 | +0.003 [+0.001, +0.005] | 0.000 |
| HiGT-Bot v2 (x+nbr+seq+grp) - XGB[x] (v3 features) | +0.002 [-0.010, +0.014] | 0.356 | +0.002 [+0.000, +0.004] | 0.015 |
| HiGT-Bot v2 (x+grp) - RF tuned (v2 features) | +0.043 [+0.031, +0.055] | 0.000 | +0.037 [+0.028, +0.046] | 0.000 |
| HiGT-Bot v2 (x+grp) - Directed T-GINE (v2) | +0.062 [+0.046, +0.079] | 0.000 | +0.013 [+0.007, +0.021] | 0.000 |
| HiGT-Bot v2 (x+grp) - HiGT-Bot directed (v2) | +0.039 [+0.026, +0.053] | 0.000 | +0.050 [+0.030, +0.073] | 0.000 |
| XGB[x] (v3 features) - RF tuned (v2 features) | +0.030 [+0.017, +0.043] | 0.000 | +0.034 [+0.026, +0.042] | 0.000 |
| HiGT-Bot v2 (x+grp) - RF tuned (v3 features) | +0.035 [+0.021, +0.050] | 0.000 | +0.004 [+0.002, +0.005] | 0.000 |
| HiGT-Bot v2 (x+grp) - Directed GINE (v3) | +0.033 [+0.018, +0.050] | 0.000 | +0.020 [+0.007, +0.037] | 0.000 |
| HiGT-Bot v2 (x+grp) - Directed T-GINE (v3) | +0.033 [+0.011, +0.055] | 0.000 | +0.080 [+0.046, +0.114] | 0.000 |
| HiGT-Bot v2 (x+grp) - HiGT-Bot directed (v3) | +0.044 [+0.031, +0.058] | 0.000 | +0.099 [+0.075, +0.126] | 0.000 |

## held-out

| Comparison | dF1 [95% CI] | p | dPR-AUC [95% CI] | p |
|---|---|---|---|---|
| HiGT-Bot v2 (x+grp) - XGB[x] (v3 features) | +0.028 [+0.001, +0.068] | 0.019 | +0.018 [+0.005, +0.034] | 0.000 |
| HiGT-Bot v2 (x+nbr+seq+grp) - XGB[x] (v3 features) | -0.045 [-0.097, +0.000] | 0.975 | +0.005 [-0.010, +0.022] | 0.276 |
| HiGT-Bot v2 (x+grp) - RF tuned (v2 features) | +0.060 [+0.016, +0.102] | 0.004 | +0.051 [+0.029, +0.076] | 0.000 |
| HiGT-Bot v2 (x+grp) - Directed T-GINE (v2) | +0.143 [+0.078, +0.212] | 0.000 | +0.035 [+0.012, +0.060] | 0.000 |
| HiGT-Bot v2 (x+grp) - HiGT-Bot directed (v2) | +0.085 [+0.031, +0.145] | 0.000 | +0.150 [+0.077, +0.218] | 0.000 |
| XGB[x] (v3 features) - RF tuned (v2 features) | +0.032 [-0.008, +0.069] | 0.062 | +0.033 [+0.009, +0.056] | 0.004 |
| HiGT-Bot v2 (x+grp) - RF tuned (v3 features) | +0.154 [+0.093, +0.224] | 0.000 | +0.009 [-0.002, +0.023] | 0.044 |
| HiGT-Bot v2 (x+grp) - Directed GINE (v3) | +0.104 [+0.045, +0.171] | 0.000 | +0.100 [+0.043, +0.158] | 0.000 |
| HiGT-Bot v2 (x+grp) - Directed T-GINE (v3) | +0.018 [-0.054, +0.090] | 0.324 | +0.013 [-0.001, +0.037] | 0.037 |
| HiGT-Bot v2 (x+grp) - HiGT-Bot directed (v3) | +0.059 [-0.003, +0.128] | 0.030 | +0.136 [+0.057, +0.215] | 0.000 |

## CTU-13

| Comparison | dF1 [95% CI] | p | dPR-AUC [95% CI] | p |
|---|---|---|---|---|
| HiGT-Bot v2 (x+grp) - XGB[x] (v3 features) | +0.017 [-0.001, +0.043] | 0.031 | +0.014 [+0.002, +0.025] | 0.000 |
| HiGT-Bot v2 (x+nbr+seq+grp) - XGB[x] (v3 features) | -0.025 [-0.060, +0.006] | 0.939 | +0.007 [-0.004, +0.017] | 0.100 |
| HiGT-Bot v2 (x+grp) - RF tuned (v2 features) | +0.087 [+0.056, +0.117] | 0.000 | +0.105 [+0.078, +0.132] | 0.000 |
| HiGT-Bot v2 (x+grp) - Directed T-GINE (v2) | +0.093 [+0.051, +0.138] | 0.000 | +0.041 [+0.014, +0.071] | 0.000 |
| HiGT-Bot v2 (x+grp) - HiGT-Bot directed (v2) | +0.052 [+0.016, +0.093] | 0.001 | +0.109 [+0.043, +0.178] | 0.000 |
| XGB[x] (v3 features) - RF tuned (v2 features) | +0.070 [+0.040, +0.098] | 0.000 | +0.091 [+0.065, +0.119] | 0.000 |
| HiGT-Bot v2 (x+grp) - RF tuned (v3 features) | +0.068 [+0.032, +0.108] | 0.000 | +0.015 [+0.006, +0.024] | 0.000 |
| HiGT-Bot v2 (x+grp) - Directed GINE (v3) | +0.080 [+0.041, +0.124] | 0.000 | +0.080 [+0.035, +0.128] | 0.000 |
| HiGT-Bot v2 (x+grp) - Directed T-GINE (v3) | +0.012 [-0.033, +0.062] | 0.300 | +0.019 [+0.005, +0.037] | 0.004 |
| HiGT-Bot v2 (x+grp) - HiGT-Bot directed (v3) | +0.035 [-0.005, +0.081] | 0.047 | +0.112 [+0.039, +0.186] | 0.000 |

## ctu13-10

| Comparison | dF1 [95% CI] | p | dPR-AUC [95% CI] | p |
|---|---|---|---|---|
| HiGT-Bot v2 (x+grp) - XGB[x] (v3 features) | +0.075 [-0.026, +0.210] | 0.113 | +0.085 [+0.013, +0.175] | 0.009 |
| HiGT-Bot v2 (x+nbr+seq+grp) - XGB[x] (v3 features) | -0.225 [-0.439, -0.041] | 0.992 | -0.073 [-0.185, +0.012] | 0.956 |
| HiGT-Bot v2 (x+grp) - RF tuned (v2 features) | +0.127 [-0.032, +0.286] | 0.054 | +0.349 [+0.215, +0.478] | 0.000 |
| HiGT-Bot v2 (x+grp) - Directed T-GINE (v2) | +0.744 [+0.550, +0.905] | 0.000 | +0.322 [+0.166, +0.482] | 0.000 |
| HiGT-Bot v2 (x+grp) - HiGT-Bot directed (v2) | +0.374 [+0.164, +0.611] | 0.000 | +0.534 [+0.309, +0.708] | 0.000 |
| XGB[x] (v3 features) - RF tuned (v2 features) | +0.053 [-0.090, +0.207] | 0.219 | +0.264 [+0.155, +0.380] | 0.000 |
| HiGT-Bot v2 (x+grp) - RF tuned (v3 features) | +0.477 [+0.240, +0.727] | 0.000 | +0.096 [+0.019, +0.196] | 0.004 |
| HiGT-Bot v2 (x+grp) - Directed GINE (v3) | +0.491 [+0.237, +0.799] | 0.000 | +0.433 [+0.247, +0.624] | 0.000 |
| HiGT-Bot v2 (x+grp) - Directed T-GINE (v3) | +0.062 [-0.202, +0.354] | 0.318 | +0.058 [-0.016, +0.191] | 0.067 |
| HiGT-Bot v2 (x+grp) - HiGT-Bot directed (v3) | +0.244 [-0.009, +0.517] | 0.032 | +0.499 [+0.233, +0.715] | 0.000 |
