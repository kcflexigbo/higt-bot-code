# Phase 8.3 — Efficiency Report

Measured on the held-out test graphs (median over 1000 graphs).

| Model | Params | Peak VRAM (MB) | Median ms/graph | p95 ms/graph |
|---|---|---|---|---|
| Phase 5 T-GINE+skip | 144,325 | 172.13 | 0.752 | 1.529 |
| Phase 6 DiffPool | 225,818 | 34.05 | 0.661 | 1.006 |
| Phase 6.4 SAGPool | 211,335 | 13.84 | 0.565 | 0.713 |
| Phase 7 GT-edge | 1,355,399 | 26.81 | 0.943 | 1.405 |
| Phase 7 GT-global | 607,879 | 18.02 | 0.918 | 1.414 |
| Phase 7 GT-hybrid 2L | 981,639 | 20.86 | 1.016 | 1.451 |