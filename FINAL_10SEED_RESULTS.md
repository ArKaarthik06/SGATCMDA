### ALL 10 seeds (50 runs)

| Model | A: notebook protocol (leaky) AUC / AUPR | B1: per-fold, plain AUC | B2: per-fold + masked AUC | B3: per-fold + masked + CF blend AUC / AUPR |
|---|---|---|---|---|
| SGAT-CLMDA (parallel) | 0.902 ± 0.025 / 0.897 | 0.671 ± 0.082 | 0.849 ± 0.031 | 0.885 ± 0.025 / 0.894 |
| GCATCMDA-style serial | 0.862 ± 0.051 / 0.863 | 0.685 ± 0.096 | 0.827 ± 0.035 | 0.863 ± 0.029 / 0.879 |
| GCN | 0.911 ± 0.025 / 0.909 | 0.677 ± 0.077 | 0.845 ± 0.027 | 0.880 ± 0.026 / 0.889 |
| GAT | 0.881 ± 0.030 / 0.882 | 0.690 ± 0.093 | 0.847 ± 0.031 | 0.879 ± 0.027 / 0.889 |

- SGAT-CLMDA minus GCATCMDA-style serial (recipe B3): +0.022, paired Wilcoxon p=7.6e-14, wins 48/50
- SGAT-CLMDA minus GCN (recipe B3): +0.005, paired Wilcoxon p=3.2e-02, wins 31/50
- SGAT-CLMDA minus GAT (recipe B3): +0.006, paired Wilcoxon p=2.4e-02, wins 29/50

### Pre-specified first 5 seeds (25 runs)

| Model | A: notebook protocol (leaky) AUC / AUPR | B1: per-fold, plain AUC | B2: per-fold + masked AUC | B3: per-fold + masked + CF blend AUC / AUPR |
|---|---|---|---|---|
| SGAT-CLMDA (parallel) | 0.899 ± 0.027 / 0.894 | 0.667 ± 0.075 | 0.847 ± 0.031 | 0.886 ± 0.023 / 0.893 |
| GCATCMDA-style serial | 0.863 ± 0.050 / 0.866 | 0.672 ± 0.082 | 0.820 ± 0.030 | 0.862 ± 0.023 / 0.878 |
| GCN | 0.912 ± 0.029 / 0.908 | 0.661 ± 0.071 | 0.849 ± 0.025 | 0.884 ± 0.018 / 0.891 |
| GAT | 0.875 ± 0.030 / 0.877 | 0.672 ± 0.094 | 0.847 ± 0.031 | 0.880 ± 0.024 / 0.889 |

- SGAT-CLMDA minus GCATCMDA-style serial (recipe B3): +0.023, paired Wilcoxon p=3.0e-07, wins 24/25
- SGAT-CLMDA minus GCN (recipe B3): +0.002, paired Wilcoxon p=7.1e-01, wins 12/25
- SGAT-CLMDA minus GAT (recipe B3): +0.006, paired Wilcoxon p=1.2e-01, wins 13/25

### Per-seed mean AUC (every seed shown)

| Model / protocol | 42 | 123 | 456 | 789 | 2024 | 11 | 22 | 33 | 44 | 55 |
|---|---|---|---|---|---|---|---|---|---|---|
| SGAT-CLMDA (parallel) / leaky | 0.903 | 0.907 | 0.890 | 0.893 | 0.903 | 0.904 | 0.896 | 0.909 | 0.911 | 0.908 |
| SGAT-CLMDA (parallel) / per-fold plain | 0.708 | 0.596 | 0.695 | 0.675 | 0.660 | 0.696 | 0.607 | 0.667 | 0.682 | 0.723 |
| SGAT-CLMDA (parallel) / per-fold recipe | 0.889 | 0.884 | 0.884 | 0.892 | 0.879 | 0.881 | 0.888 | 0.882 | 0.894 | 0.879 |
| GCATCMDA-style serial / leaky | 0.830 | 0.854 | 0.890 | 0.869 | 0.869 | 0.852 | 0.845 | 0.870 | 0.886 | 0.850 |
| GCATCMDA-style serial / per-fold plain | 0.700 | 0.632 | 0.669 | 0.681 | 0.676 | 0.757 | 0.684 | 0.726 | 0.631 | 0.697 |
| GCATCMDA-style serial / per-fold recipe | 0.875 | 0.853 | 0.857 | 0.867 | 0.858 | 0.862 | 0.865 | 0.862 | 0.875 | 0.859 |
| GCN / leaky | 0.926 | 0.901 | 0.911 | 0.917 | 0.905 | 0.907 | 0.894 | 0.916 | 0.913 | 0.919 |
| GCN / per-fold plain | 0.671 | 0.625 | 0.636 | 0.672 | 0.701 | 0.720 | 0.659 | 0.682 | 0.698 | 0.710 |
| GCN / per-fold recipe | 0.882 | 0.877 | 0.890 | 0.892 | 0.878 | 0.872 | 0.874 | 0.875 | 0.891 | 0.873 |
| GAT / leaky | 0.876 | 0.861 | 0.885 | 0.877 | 0.875 | 0.895 | 0.876 | 0.885 | 0.889 | 0.894 |
| GAT / per-fold plain | 0.729 | 0.593 | 0.679 | 0.698 | 0.663 | 0.765 | 0.702 | 0.660 | 0.726 | 0.682 |
| GAT / per-fold recipe | 0.881 | 0.879 | 0.879 | 0.884 | 0.876 | 0.881 | 0.874 | 0.871 | 0.884 | 0.883 |

### GCATCMDA replica, 10 seeds x 5 folds (50 runs, from the paper description)

| Setting | full-matrix kernels (A) | per-fold kernels (B) |
|---|---|---|
| paper settings (lr 1e-5) | 0.701 ± 0.118 | 0.554 ± 0.094 |
| lr 1e-3 | 0.906 ± 0.031 | 0.598 ± 0.109 |
