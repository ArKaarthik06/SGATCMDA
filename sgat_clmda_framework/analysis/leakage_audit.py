import numpy as np
import torch
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score

from ..data.dataset_loader import DatasetLoader
from ..data.similarity import SimilarityEngine
from ..data.graph_builder import GraphBuilder
from ..models.sgat_model import SGAT_CLMDA
from ..training.trainer import Trainer
from ..training.evaluator import Evaluator

class LeakageAuditor:
    """
    Addresses Reviewer Comment 1 & Reviewer Comment 2:
    1. Evaluates non-learning heuristic baselines (Comment 2):
       (a) Original A rows / raw entries
       (b) Full GIP profile similarity
       (c) Fold-recalculated GIP profile similarity
    2. Compares Full similarity features vs. Recalculated similarity features in SGAT-CLMDA (Comment 1).
    """
    def __init__(self, dataset: DatasetLoader, device: torch.device):
        self.dataset = dataset
        self.device = device

    def audit_non_learning_heuristics(self, k_folds: int = 5) -> Dict[str, Dict]:
        """
        Evaluate predictive scores WITHOUT any learning model (Reviewer Comment 2):
          (a) Original A[i, j] lookup directly
          (b) Full GIP profile similarity
          (c) Fold-recalculated GIP similarity
        """
        splits = self.dataset.get_kfold_splits(k_folds=k_folds)
        A_full = self.dataset.A
        SD_full_gip = SimilarityEngine.gip_similarity(A_full)
        SM_full_gip = SimilarityEngine.gip_similarity(A_full.T)

        results = {
            '(a) Raw A Lookup': {'auc': [], 'aupr': []},
            '(b) Full GIP Score': {'auc': [], 'aupr': []},
            '(c) Per-Fold GIP Score': {'auc': [], 'aupr': []}
        }

        for split in splits:
            te_pd, te_pm = split['te_pd'], split['te_pm']
            te_neg_d, te_neg_m = split['te_neg_d'], split['te_neg_m']
            A_tr = split['A_tr']

            y_true = np.concatenate([np.ones(len(te_pd)), np.zeros(len(te_neg_d))])

            # (a) Raw A Lookup: Predicts A_full[i, j]
            # Since test positives have A_full[d, m] = 1 and test negatives have A_full[d, m] = 0,
            # this yields AUC = 1.0 directly from label lookup!
            score_pos_a = A_full[te_pd, te_pm]
            score_neg_a = A_full[te_neg_d, te_neg_m]
            y_score_a = np.concatenate([score_pos_a, score_neg_a])
            results['(a) Raw A Lookup']['auc'].append(roc_auc_score(y_true, y_score_a))
            results['(a) Raw A Lookup']['aupr'].append(average_precision_score(y_true, y_score_a))

            # (b) Full GIP score:
            # Score(d, m) is inner product / projection of d's GIP profile onto known microbes via A_full
            # S_GIP = SD_full_gip @ A_full
            S_full_pred = SD_full_gip @ A_full
            score_pos_b = S_full_pred[te_pd, te_pm]
            score_neg_b = S_full_pred[te_neg_d, te_neg_m]
            y_score_b = np.concatenate([score_pos_b, score_neg_b])
            results['(b) Full GIP Score']['auc'].append(roc_auc_score(y_true, y_score_b))
            results['(b) Full GIP Score']['aupr'].append(average_precision_score(y_true, y_score_b))

            # (c) Per-fold recalculated GIP score:
            # Computed strictly using A_tr (no test information present)
            SD_fold_gip = SimilarityEngine.gip_similarity(A_tr)
            S_fold_pred = SD_fold_gip @ A_tr
            score_pos_c = S_fold_pred[te_pd, te_pm]
            score_neg_c = S_fold_pred[te_neg_d, te_neg_m]
            y_score_c = np.concatenate([score_pos_c, score_neg_c])
            results['(c) Per-Fold GIP Score']['auc'].append(roc_auc_score(y_true, y_score_c))
            results['(c) Per-Fold GIP Score']['aupr'].append(average_precision_score(y_true, y_score_c))

        summary = {}
        for k, v in results.items():
            summary[k] = {
                'auc_mean': float(np.mean(v['auc'])),
                'auc_std': float(np.std(v['auc'])),
                'aupr_mean': float(np.mean(v['aupr'])),
                'aupr_std': float(np.std(v['aupr'])),
                'per_fold_auc': [float(x) for x in v['auc']]
            }
        return summary

    def compare_full_vs_recalculated_features(self, k_folds: int = 5, epochs: int = 150) -> Dict:
        """
        Ablation Study (Reviewer Comment 1):
        Full similarity features (SD/SM pre-computed on full A)
        vs.
        Strictly recalculated similarity features (SD/SM computed per-fold on A_tr).
        Evaluated on the exact same folds and model architecture.
        """
        splits = self.dataset.get_kfold_splits(k_folds=k_folds)
        SD_full, SM_full = SimilarityEngine.compute_full_similarities(self.dataset.A)

        results = {
            'Full Similarity Features (Leaky)': {'auc': [], 'aupr': []},
            'Recalculated Fold Features (Honest)': {'auc': [], 'aupr': []}
        }

        for split in splits:
            fold = split['fold']
            A_tr = split['A_tr']
            tr_pd, tr_pm = split['tr_pd'], split['tr_pm']
            te_pd, te_pm = split['te_pd'], split['te_pm']
            te_neg_d, te_neg_m = split['te_neg_d'], split['te_neg_m']

            # Training pairs (balanced 1:1)
            rng = np.random.RandomState(42 + fold)
            neg_sel = rng.choice(len(self.dataset.neg_d_all), size=len(tr_pd), replace=False)
            train_d = torch.tensor(np.concatenate([tr_pd, self.dataset.neg_d_all[neg_sel]]), dtype=torch.long, device=self.device)
            train_m = torch.tensor(np.concatenate([tr_pm, self.dataset.neg_m_all[neg_sel]]), dtype=torch.long, device=self.device)
            train_y = torch.tensor(np.concatenate([np.ones(len(tr_pd)), np.zeros(len(tr_pd))]), dtype=torch.float32, device=self.device)

            # Evaluate each feature protocol
            for mode, (SD_cur, SM_cur) in [
                ('Full Similarity Features (Leaky)', (SD_full, SM_full)),
                ('Recalculated Fold Features (Honest)', SimilarityEngine.compute_fold_similarities(A_tr))
            ]:
                X, ei_h = GraphBuilder.build_hetero_network(A_tr, SD_cur, SM_cur)
                ei_d, ei_m = GraphBuilder.build_sim_networks(SD_cur, SM_cur)
                FD = torch.tensor(SD_cur, dtype=torch.float32)
                FM = torch.tensor(SM_cur, dtype=torch.float32)

                X, ei_h = X.to(self.device), ei_h.to(self.device)
                FD, FM = FD.to(self.device), FM.to(self.device)
                ei_d, ei_m = ei_d.to(self.device), ei_m.to(self.device)

                model = SGAT_CLMDA(self.dataset.nd, self.dataset.nm, feat_dim=self.dataset.nd + self.dataset.nm).to(self.device)
                trainer = Trainer(model, self.device, max_epochs=epochs, patience=30)
                trainer.fit(X, ei_h, FD, FM, ei_d, ei_m, train_d, train_m, train_y)

                eval_res = Evaluator.evaluate_model(model, X, ei_h, FD, FM, ei_d, ei_m, te_pd, te_pm, te_neg_d, te_neg_m)
                results[mode]['auc'].append(eval_res['auc'])
                results[mode]['aupr'].append(eval_res['aupr'])

        summary = {}
        for k, v in results.items():
            summary[k] = {
                'auc_mean': float(np.mean(v['auc'])),
                'auc_std': float(np.std(v['auc'])),
                'aupr_mean': float(np.mean(v['aupr'])),
                'aupr_std': float(np.std(v['aupr'])),
                'per_fold_auc': [float(x) for x in v['auc']],
                'per_fold_aupr': [float(x) for x in v['aupr']]
            }
        summary['leakage_gap_auc'] = float(summary['Full Similarity Features (Leaky)']['auc_mean'] - summary['Recalculated Fold Features (Honest)']['auc_mean'])
        summary['leakage_gap_aupr'] = float(summary['Full Similarity Features (Leaky)']['aupr_mean'] - summary['Recalculated Fold Features (Honest)']['aupr_mean'])
        return summary
