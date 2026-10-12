# Phase 8.7 — SAGPool survival vs. detection (whole test set, final model)

Nodes: 59,210. Bot survival 0.495; benign survival 0.525.

| Bot group | n | detected | recall |
|---|---|---|---|
| kept by SAGPool | 20,327 | 20,296 | 0.9985 |
| dropped by SAGPool | 20,736 | 20,681 | 0.9973 |

False positives among benign hosts: kept 1,328, dropped 1,324.
