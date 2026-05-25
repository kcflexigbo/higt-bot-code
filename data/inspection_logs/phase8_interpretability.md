# Phase 8.5 — Interpretability Artefacts

Per-scenario inspection of the final GT-edge model:
  - what fraction of bots survive SAGPool coarsening?
  - does the SAGPool score actually separate bots from benigns?

| Scenario | n | n_pos | bot survival | benign survival | TP | FN | FP |
|---|---|---|---|---|---|---|---|
| iot23-35-1 | 146 | 3 | 0.00 | 0.51 | 2 | 1 | 0 |
| ctu13-10 | 12 | 2 | 0.50 | 0.50 | 0 | 2 | 0 |
| iot23-7-1 | 2 | 2 | 0.50 | 0.00 | 0 | 2 | 0 |
| ctu13-9 | 217 | 172 | 0.37 | 1.00 | 172 | 0 | 0 |
| medbiot-bashlite_mal_spread_all | 400 | 393 | 0.49 | 1.00 | 393 | 0 | 7 |

See `figures/phase8_sagpool_<scenario>.png` for histograms.