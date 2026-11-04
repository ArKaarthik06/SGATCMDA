## Table A. Matched protocols, identical folds / negatives / seeds (5 folds x 5 seeds = 25 runs per cell; mean ± sd)

| Model | Leaky AUC | Honest AUC | gap | p (leaky vs honest) | Honest+masked AUC | Honest+masked+CF AUC | Honest+masked+CF AUPR |
|---|---|---|---|---|---|---|---|
| SGAT-CLMDA (parallel GCN || SAGA) | 0.8992 ± 0.0266 | 0.6668 ± 0.0748 | +0.232 | 6.0e-08 | 0.8474 ± 0.0307 | **0.8857 ± 0.0230** | 0.8930 ± 0.0302 |
| GCATCMDA-style (serial GCN->GAT) | 0.8626 ± 0.0498 | 0.6717 ± 0.0823 | +0.191 | 1.4e-05 | 0.8196 ± 0.0296 | **0.8622 ± 0.0235** | 0.8777 ± 0.0220 |
| GCN-only encoder | 0.9121 ± 0.0290 | 0.6609 ± 0.0709 | +0.251 | 6.0e-08 | 0.8490 ± 0.0250 | **0.8836 ± 0.0182** | 0.8913 ± 0.0238 |
| GAT-only encoder | 0.8748 ± 0.0301 | 0.6722 ± 0.0936 | +0.203 | 6.0e-08 | 0.8466 ± 0.0312 | **0.8799 ± 0.0237** | 0.8894 ± 0.0246 |
| SGAT-CLMDA without View B / contrastive | 0.8857 ± 0.0240 | 0.7117 ± 0.0793 | +0.174 | 6.0e-08 | 0.8582 ± 0.0274 | **0.8851 ± 0.0205** | 0.8930 ± 0.0233 |
| Item-based CF, no learning (A_train only) | - | 0.8548 ± 0.0223 | - | - | - | 0.8548 ± 0.0223 | 0.8769 ± 0.0214 |
| KATZ-style (honest, re-implemented) | - | 0.8659 ± 0.0195 | - | - | - | - | 0.8766 |
| RWR-style (r=0.3) (honest, re-implemented) | - | 0.6878 ± 0.0435 | - | - | - | - | 0.6511 |
| RWR-style (r=0.1) (honest, re-implemented) | - | 0.4956 ± 0.0376 | - | - | - | - | 0.4811 |
| RWR-style (r=0.5) (honest, re-implemented) | - | 0.7177 ± 0.0419 | - | - | - | - | 0.6797 |
| GRNMF-style (honest, re-implemented) | - | 0.8130 ± 0.0307 | - | - | - | - | 0.8025 |

Definitions: *Leaky* = GIP/cosine similarities from the full matrix. *Honest* = recomputed from A_train in every fold. *Masked* = honest + masked-edge training (mask ratio 0.5, fixed a priori). *+CF* = 0.5 x rank(model) + 0.5 x rank(item-CF), weight fixed a priori. Nothing was tuned on test folds.

## Table B. Paired Wilcoxon tests on AUC (25 paired runs)

| Comparison | mean diff | p | wins/25 |
|---|---|---|---|
| Final (parallel, masked+CF) vs item-CF alone | +0.0309 | 6.0e-08 | 25 |
| Final (parallel, masked+CF) vs masked parallel model alone | +0.0383 | 6.0e-08 | 25 |
| Final vs KATZ-style (honest) | +0.0197 | 3.2e-05 | 23 |
| masked+CF: parallel vs GCATCMDA-style (serial GCN->GAT) | +0.0234 | 3.0e-07 | 24 |
| masked only: parallel vs GCATCMDA-style (serial GCN->GAT) | +0.0278 | 1.8e-05 | 20 |
| masked+CF: parallel vs GCN-only encoder | +0.0020 | 7.1e-01 | 12 |
| masked only: parallel vs GCN-only encoder | -0.0015 | 7.3e-01 | 11 |
| masked+CF: parallel vs GAT-only encoder | +0.0058 | 1.2e-01 | 13 |
| masked only: parallel vs GAT-only encoder | +0.0008 | 8.1e-01 | 10 |
| masked+CF: parallel vs SGAT-CLMDA without View B / contrastive | +0.0005 | 8.5e-01 | 12 |
| masked only: parallel vs SGAT-CLMDA without View B / contrastive | -0.0108 | 1.7e-02 | 7 |
| masked only: parallel vs View-A-only (no View B) | -0.0108 | 1.7e-02 | 7 |
| SGAT-CLMDA (parallel GCN || SAGA): honest+masked+CF vs honest plain | +0.2189 | 6.0e-08 | 25 |
| GCATCMDA-style (serial GCN->GAT): honest+masked+CF vs honest plain | +0.1905 | 6.0e-08 | 25 |
| GCN-only encoder: honest+masked+CF vs honest plain | +0.2227 | 6.0e-08 | 25 |
| GAT-only encoder: honest+masked+CF vs honest plain | +0.2077 | 6.0e-08 | 25 |
| SGAT-CLMDA without View B / contrastive: honest+masked+CF vs honest plain | +0.1734 | 6.0e-08 | 25 |

## Table C. Cost (CPU, 3 threads, one fold; plain / masked training)

| Model | Parameters | ms/epoch plain | ms/epoch masked | peak RSS MB (process) |
|---|---|---|---|---|
| SGAT-CLMDA (parallel GCN || SAGA) | 1,171,020 | 173 | 210 | 528 |
| GCATCMDA-style (serial GCN->GAT) | 895,686 | 167 | 206 | 518 |
| GCN-only encoder | 499,014 | 93 | 133 | 488 |
| GAT-only encoder | 499,014 | 110 | 149 | 501 |
| SGAT-CLMDA without View B / contrastive | 421,572 | 92 | 133 | 497 |

## Table D. Non-learning probes (Reviewer 4 #2) and leakage audit, 25 runs

| Probe | AUC | AUPR |
|---|---|---|
| model_leaky_features | 0.8985 ± 0.0248 | 0.8933 |
| model_honest_features | 0.6605 ± 0.0786 | 0.6505 |
| heur_a_raw_A | 1.0000 ± 0.0000 | 1.0000 |
| heur_b_full_GIP_projection | 0.9621 ± 0.0217 | 0.9256 |
| heur_c_fold_GIP_projection | 0.3700 ± 0.0345 | 0.5175 |
| cf_leaky_(A_full x S_M_full) | 0.9212 ± 0.0193 | 0.9409 |
| cf_honest_(A_tr x S_M_tr) | 0.8548 ± 0.0223 | 0.8769 |

## Table E. Ablations (honest + masked training, 25 runs each; mean AUC)

| Variant | AUC | AUPR |
|---|---|---|
| SGAT-CLMDA parallel + masked-edge training (r=0.5) | 0.8484 ± 0.0297 | 0.8245 |
| Serial GCN->GAT + masked-edge training (r=0.5) | 0.8204 ± 0.0317 | 0.8122 |
| tau frozen at 0.1 | 0.8452 ± 0.0269 | 0.8232 |
| tau frozen at 0.3 | 0.8469 ± 0.0280 | 0.8258 |
| similarity: GIP only | 0.8515 ± 0.0344 | 0.8355 |
| similarity: cosine only | 0.7910 ± 0.0433 | 0.7877 |
| CL none | 0.8482 ± 0.0273 | 0.8256 |
| CL disease only | 0.8497 ± 0.0262 | 0.8242 |
| CL microbe only | 0.8445 ± 0.0285 | 0.8286 |
| CL both, tau_cl=0.1 | 0.8465 ± 0.0250 | 0.8259 |
| CL both, tau_cl=0.2 | 0.8482 ± 0.0239 | 0.8257 |
| CL both, tau_cl=1.0 | 0.8452 ± 0.0236 | 0.8221 |
| View A only (no View B, no CL) | 0.8582 ± 0.0274 | 0.8459 |
