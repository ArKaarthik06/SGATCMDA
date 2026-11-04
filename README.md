# SGAT-CLMDA: code, fold definitions, seeds and logs

Code for the manuscript on SGAT-CLMDA and a leakage-controlled benchmark for microbe–disease association prediction on HMDAD
(450 associations, 39 diseases, 292 microbes). This repository contains the complete code, the seeded fold definitions, the per-run result files
and the training logs behind **every table and figure in the revised manuscript**, including the runs that do not favour the proposed model
(plain end-to-end training, ablations with no measurable effect, single-branch baselines, the published-method reproductions).

## Protocol
* 5 disjoint folds of 90 positives (`sklearn.model_selection.KFold`, shuffled, seeded). Test set per fold = 90 held-out positives + 90 sampled unrecorded pairs;
  training = 360 positives + 360 sampled unrecorded pairs. Repeated for seeds 42, 123, 456, 789, 2024, 11, 22, 33, 44, 55 (all reported).
* **Original protocol:** GIP/cosine similarities computed once from the full association matrix (the submitted version of the paper).
* **Per-fold protocol:** every association-derived similarity, feature and graph is recomputed from `A_train` only.
* Final training procedure: masked-edge training (50 % of training associations hidden each epoch) + rank blend with a training-only item-CF score (fixed weight 0.5).
  Per-fold results *without* these two additions (plain training) are stored under the key `honest` in `sgat_clmda_outputs/matched_*.json` and are lower.

## Setup
```bash
pip install -r requirements.txt
python download_disbiome.py      # only for the Disbiome check of the case study
```
CPU is sufficient. `HMDAD.txt` is the public HMDAD table (http://www.cuilab.cn/hmdad).

## Reproducing the manuscript
| Manuscript item | Command | Output |
|---|---|---|
| Table 1, Fig. 1 (per-seed, original vs per-fold, 10 seeds) | `for a in parallel serial gcn_only gat_only; do python run_matched_baselines.py $a; SEEDS=11,22,33,44,55 SUFFIX=_s2 python run_matched_baselines.py $a; done` then `python aggregate_10seed.py` | `matched_*.json`, `matched_*_s2.json`, `FINAL_10SEED_RESULTS.md` |
| Table 2, Fig. 2 (SGAT-CLMDA, GCN, GAT, GCATCMDA-style) | same as above, then `python run_baseline_table.py` | `BASELINE_COMPARISON_TABLE.md`, `baseline_table.json` |
| Table 2, Fig. 2 (RKGATMDA on our folds) | `python run_rkgatmda_ours.py <seed> released <path to RKGATMDA data dir>` (seeds 42 123 456 789 2024; code from github.com/hwlacc/RKGATMDA) | `rkgatmda_ours_released_*.json` |
| Table 3, Fig. 4 (ablations) | `python run_masked_ablation.py`; `python run_extra_ablations.py`; `python run_cl_study.py <none|ntxent|grace_ab|grace_aug|grace_both>` | `masked_ablation.json`, `extra_ablations.json`, `cl_study_*.json` |
| Table 4, Fig. 3 (scoring rules without learning) | `python run_leakage_multiseed.py` | `leakage_multiseed.json` |
| Case study (Disbiome check) | `python run_case_study_real.py` | `case_study_summary.json`, `case_studies_disbiome.csv` |
| Figures 1–4 | `python make_manuscript_figures.py` | `figures/ms_fig*.png` |
| GCATCMDA replica from the published description | `python run_gcatcmda_replica.py <original|perfold> [lr]` | `gcatcmda_replica_*.json` |
| Other analyses used in the study | `run_classical_baselines.py`, `run_rwr.py`, `run_robustness.py`, `run_disbiome_study.py`, `run_gatmda_port.py`, `run_gatmda_protocol_test.py`, `run_rkgatmda_probe.py`, `run_lr_probe.py`, `run_hybrid.py` | `sgat_clmda_outputs/` |
| GPU / Colab notebook (10 seeds, all models, both protocols) | `SGAT_CLMDA_Leakage_Audit_Colab.ipynb` (regenerate with `make_leakage_notebook.py`) | `leakage_audit_*.csv` |

All scripts are resumable (finished runs are cached in the JSON files). Raw logs of our runs are in `logs/`.

## Layout
`sgat_clmda_framework/` — `data/` (loader, similarity engines for both protocols, graph builder), `models/` (parallel and serial encoders, gates),
`training/` (trainer, masked-edge trainer, losses, evaluator), `analysis/`. `SGAT_CLMDA_Final.ipynb` is the original notebook.

## Notes and caveats
* KATZ-, RWR- and GRNMF-style methods, the GCATCMDA-style and replica models, and the RKGATMDA CPU port are **our implementations or ports**, not the original authors' code
  (the GCATCMDA, MADGAN and DBGCNMDA code is not available to us).
* The contrastive-learning and gate-threshold ablations show no measurable effect on accuracy; the parallel encoder is 0.022 AUC above the serial one after blending and level with GCN-only and GAT-only.
* Runs are not bit-wise deterministic across machines; expect small run-to-run differences.
