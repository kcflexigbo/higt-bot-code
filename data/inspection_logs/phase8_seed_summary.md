# Phase 8.1 — Seed Sweep Summary

Final HiGT-Bot (GT-edge) trained with three seeds. Reported as mean ± std across available seeds. Higher seed count gives tighter bars; even a 2-seed pair lets us check stability.

## Aggregate metrics

| Metric | seed=42 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| f1 | 0.9677 | 0.9675 | 0.9654 | 0.9669 ± 0.0010 |
| precision | 0.9392 | 0.9389 | 0.9392 | 0.9391 ± 0.0002 |
| recall | 0.9979 | 0.9979 | 0.9931 | 0.9963 ± 0.0023 |
| pr_auc | 0.9700 | 0.9692 | 0.9690 | 0.9694 ± 0.0004 |
| roc_auc | 0.9504 | 0.9494 | 0.9493 | 0.9497 ± 0.0005 |
| fn | 86 | 86 | 285 | 152.3333 ± 93.8095 |

## Per-scenario test F1

| Scenario | seed=42 | seed=1 | seed=2 | mean ± std |
|---|---|---|---|---|
| iot23-35-1 | 0.5957 | 0.5652 | 0.3077 | 0.4896 ± 0.1292 |
| ctu13-10 | 0.6957 | 0.6519 | 0.6406 | 0.6627 ± 0.0238 |
| iot23-7-1 | 0.7368 | 0.7368 | 0.7368 | 0.7368 ± 0.0000 |
| ctu13-3 | 0.8889 | 0.9796 | 0.7500 | 0.8728 ± 0.0944 |
| medbiot-bashlite_mal_spread_all | 0.9848 | 0.9898 | 0.9848 | 0.9864 ± 0.0024 |

Per-scenario stability is the headline check — variance >0.02 on iot23-35-1 (31 positives) is expected, but easy scenarios should be tight across seeds.