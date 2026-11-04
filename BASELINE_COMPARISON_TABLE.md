| Method (all on the same 25 splits) | A: notebook protocol AUC / AUPR | B: per-fold recomputed AUC / AUPR | final recipe vs method in B (paired Wilcoxon) |
|---|---|---|---|
| SGAT-CLMDA, plain end-to-end (notebook model) | 0.899 ± 0.027 / 0.894 | 0.667 ± 0.075 / 0.654 | +0.219, p=6.0e-08, wins 25/25 |
| GCN-only encoder (same framework) | 0.912 ± 0.029 / 0.908 | 0.661 ± 0.071 / 0.652 | +0.225, p=6.0e-08, wins 25/25 |
| GAT-only encoder (same framework) | 0.875 ± 0.030 / 0.877 | 0.672 ± 0.094 / 0.672 | +0.213, p=6.0e-08, wins 25/25 |
| GCATCMDA-style serial GCN->GAT (same framework) | 0.863 ± 0.050 / 0.866 | 0.672 ± 0.082 / 0.665 | +0.214, p=6.0e-08, wins 25/25 |
| GCATCMDA replica, paper settings (lr 1e-5) | 0.729 ± 0.092 / 0.713 | 0.577 ± 0.095 / 0.604 | +0.308, p=6.0e-08, wins 25/25 |
| GCATCMDA replica, lr 1e-3 | 0.916 ± 0.027 / 0.919 | 0.591 ± 0.112 / 0.643 | +0.295, p=6.0e-08, wins 25/25 |
| KATZ-style (re-implemented) | 0.899 ± 0.016 / 0.909 | 0.866 ± 0.019 / 0.877 | +0.020, p=3.2e-05, wins 23/25 |
| NTSHMDA-style RWR (re-implemented) | 0.963 ± 0.013 / 0.967 | 0.908 ± 0.019 / 0.916 | -0.022, p=5.4e-05, wins 3/25 |
| Item-CF (re-implemented) | 0.921 ± 0.019 / 0.940 | 0.855 ± 0.022 / 0.877 | +0.031, p=6.0e-08, wins 25/25 |
| SGAT-CLMDA + masked-edge training | n/a | 0.847 ± 0.031 / 0.822 | +0.038, p=6.0e-08, wins 25/25 |
| SGAT-CLMDA + masked + CF blend (final recipe) | n/a | 0.886 ± 0.023 / 0.893 | (reference) |
| GCN-only + masked + CF blend (same recipe) | n/a | 0.884 ± 0.018 / 0.891 | +0.002, p=7.1e-01, wins 12/25 |
| GAT-only + masked + CF blend (same recipe) | n/a | 0.880 ± 0.024 / 0.889 | +0.006, p=1.2e-01, wins 13/25 |
| GCATCMDA-style serial + masked + CF blend (same recipe) | n/a | 0.862 ± 0.023 / 0.878 | +0.023, p=3.0e-07, wins 24/25 |
| NTSHMDA-style RWR, from run_rwr.py (cross-check) | n/a | 0.908 ± 0.019 / 0.916 | -0.022, p=5.4e-05, wins 3/25 |
