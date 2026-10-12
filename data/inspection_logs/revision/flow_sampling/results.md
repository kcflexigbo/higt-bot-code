# Flow-record loss (records dropped before features and edges are computed; node set fixed)

| Records dropped | XGB[x] F1 | v2 XGB[x+grp] F1 | XGB[x] PR-AUC | v2 XGB[x+grp] PR-AUC |
|---|---|---|---|---|
| 0% | 0.949 ± 0.004 | 0.979 ± 0.001 | 0.9957 | 0.9982 |
| 5% | 0.946 ± 0.005 | 0.978 ± 0.001 | 0.9954 | 0.9982 |
| 10% | 0.943 ± 0.007 | 0.975 ± 0.001 | 0.9948 | 0.9978 |
| 20% | 0.919 ± 0.015 | 0.972 ± 0.001 | 0.9918 | 0.9964 |
| 30% | 0.888 ± 0.016 | 0.962 ± 0.002 | 0.9887 | 0.9952 |