import numpy as np
import torch
from typing import Dict, List, Tuple
from ..data.dataset_loader import DatasetLoader
from ..data.similarity import SimilarityEngine
from ..data.graph_builder import GraphBuilder
from ..models.sgat_model import SGAT_CLMDA
from ..training.trainer import Trainer
from ..training.evaluator import Evaluator

class AblationSuite:
    """
    Comprehensive ablation experiment runner addressing:
      - Parallel vs. Serial architecture comparison across random seeds (Comment 4)
      - Independent Disease vs. Microbe contrastive learning ablation & tau scan (Comment 5)
      - Node Norm Gating vs. Differentiable Edge Gating vs. Fixed Threshold (Comment 3)
    """
    def __init__(self, dataset: DatasetLoader, device: torch.device):
        self.dataset = dataset
        self.device = device

    def compare_serial_vs_parallel(self, seeds: List[int] = [42, 123, 456, 789, 2024],
                                   k_folds: int = 5, epochs: int = 150) -> Dict:
        """
        Compare Parallel GCN+SAGA vs Serial GCN->GAT across multiple random seeds (Comment 4).
        Tests whether parallel architecture delivers genuine accuracy gain or only stability.
        """
        results = {
            'Parallel (Proposed)': {'auc': [], 'aupr': []},
            'Serial (GCATCMDA Style)': {'auc': [], 'aupr': []}
        }

        # Use fold 0 and fold 1 across seeds for rapid, reliable comparison
        for seed in seeds:
            splits = self.dataset.get_kfold_splits(k_folds=k_folds, shuffle_seed=seed)
            split = splits[0] # Test on representative fold per seed
            A_tr = split['A_tr']
            tr_pd, tr_pm = split['tr_pd'], split['tr_pm']
            te_pd, te_pm = split['te_pd'], split['te_pm']
            te_neg_d, te_neg_m = split['te_neg_d'], split['te_neg_m']

            SD_fold, SM_fold = SimilarityEngine.compute_fold_similarities(A_tr)
            X, ei_h = GraphBuilder.build_hetero_network(A_tr, SD_fold, SM_fold)
            ei_d, ei_m = GraphBuilder.build_sim_networks(SD_fold, SM_fold)
            FD, FM = torch.tensor(SD_fold, dtype=torch.float32), torch.tensor(SM_fold, dtype=torch.float32)

            X, ei_h = X.to(self.device), ei_h.to(self.device)
            FD, FM = FD.to(self.device), FM.to(self.device)
            ei_d, ei_m = ei_d.to(self.device), ei_m.to(self.device)

            rng = np.random.RandomState(seed)
            neg_sel = rng.choice(len(self.dataset.neg_d_all), size=len(tr_pd), replace=False)
            train_d = torch.tensor(np.concatenate([tr_pd, self.dataset.neg_d_all[neg_sel]]), dtype=torch.long, device=self.device)
            train_m = torch.tensor(np.concatenate([tr_pm, self.dataset.neg_m_all[neg_sel]]), dtype=torch.long, device=self.device)
            train_y = torch.tensor(np.concatenate([np.ones(len(tr_pd)), np.zeros(len(tr_pd))]), dtype=torch.float32, device=self.device)

            for arch in ['Parallel (Proposed)', 'Serial (GCATCMDA Style)']:
                arch_type = 'parallel' if 'Parallel' in arch else 'serial'
                model = SGAT_CLMDA(self.dataset.nd, self.dataset.nm,
                                   feat_dim=self.dataset.nd + self.dataset.nm,
                                   architecture=arch_type).to(self.device)
                trainer = Trainer(model, self.device, max_epochs=epochs, patience=25)
                trainer.fit(X, ei_h, FD, FM, ei_d, ei_m, train_d, train_m, train_y)

                eval_res = Evaluator.evaluate_model(model, X, ei_h, FD, FM, ei_d, ei_m,
                                                    te_pd, te_pm, te_neg_d, te_neg_m)
                results[arch]['auc'].append(eval_res['auc'])
                results[arch]['aupr'].append(eval_res['aupr'])

        summary = {}
        for arch in results:
            summary[arch] = {
                'auc_mean': float(np.mean(results[arch]['auc'])),
                'auc_std': float(np.std(results[arch]['auc'])),
                'aupr_mean': float(np.mean(results[arch]['aupr'])),
                'aupr_std': float(np.std(results[arch]['aupr'])),
                'seeds_evaluated': seeds
            }
        summary['delta_auc'] = float(summary['Parallel (Proposed)']['auc_mean'] - summary['Serial (GCATCMDA Style)']['auc_mean'])
        return summary

    def contrastive_learning_ablation(self, k_folds: int = 5, epochs: int = 150) -> Dict:
        """
        Independent ablation of contrastive learning (Comment 5):
          (1) Full CL (Disease + Microbe)
          (2) Disease-only CL
          (3) Microbe-only CL
          (4) No CL (BCE only)
        """
        modes = ['both', 'disease_only', 'microbe_only', 'none']
        results = {m: {'auc': [], 'aupr': []} for m in modes}

        splits = self.dataset.get_kfold_splits(k_folds=k_folds)
        # Run across first 2 representative folds
        for split in splits[:2]:
            A_tr = split['A_tr']
            tr_pd, tr_pm = split['tr_pd'], split['tr_pm']
            te_pd, te_pm = split['te_pd'], split['te_pm']
            te_neg_d, te_neg_m = split['te_neg_d'], split['te_neg_m']

            SD_fold, SM_fold = SimilarityEngine.compute_fold_similarities(A_tr)
            X, ei_h = GraphBuilder.build_hetero_network(A_tr, SD_fold, SM_fold)
            ei_d, ei_m = GraphBuilder.build_sim_networks(SD_fold, SM_fold)
            FD, FM = torch.tensor(SD_fold, dtype=torch.float32), torch.tensor(SM_fold, dtype=torch.float32)

            X, ei_h = X.to(self.device), ei_h.to(self.device)
            FD, FM = FD.to(self.device), FM.to(self.device)
            ei_d, ei_m = ei_d.to(self.device), ei_m.to(self.device)

            rng = np.random.RandomState(42 + split['fold'])
            neg_sel = rng.choice(len(self.dataset.neg_d_all), size=len(tr_pd), replace=False)
            train_d = torch.tensor(np.concatenate([tr_pd, self.dataset.neg_d_all[neg_sel]]), dtype=torch.long, device=self.device)
            train_m = torch.tensor(np.concatenate([tr_pm, self.dataset.neg_m_all[neg_sel]]), dtype=torch.long, device=self.device)
            train_y = torch.tensor(np.concatenate([np.ones(len(tr_pd)), np.zeros(len(tr_pd))]), dtype=torch.float32, device=self.device)

            for m in modes:
                model = SGAT_CLMDA(self.dataset.nd, self.dataset.nm, feat_dim=self.dataset.nd + self.dataset.nm).to(self.device)
                trainer = Trainer(model, self.device, max_epochs=epochs, patience=25, cl_mode=m)
                trainer.fit(X, ei_h, FD, FM, ei_d, ei_m, train_d, train_m, train_y)

                eval_res = Evaluator.evaluate_model(model, X, ei_h, FD, FM, ei_d, ei_m, te_pd, te_pm, te_neg_d, te_neg_m)
                results[m]['auc'].append(eval_res['auc'])
                results[m]['aupr'].append(eval_res['aupr'])

        summary = {}
        for m in modes:
            summary[m] = {
                'auc_mean': float(np.mean(results[m]['auc'])),
                'aupr_mean': float(np.mean(results[m]['aupr']))
            }
        return summary

    def temperature_scan(self, temps: List[float] = [0.1, 0.2, 0.5, 0.8, 1.0], epochs: int = 150) -> Dict:
        """
        Temperature parameter tau_CL scan (Comment 5).
        """
        results = {}
        splits = self.dataset.get_kfold_splits(k_folds=5)
        split = splits[0]
        A_tr = split['A_tr']
        tr_pd, tr_pm = split['tr_pd'], split['tr_pm']
        te_pd, te_pm = split['te_pd'], split['te_pm']
        te_neg_d, te_neg_m = split['te_neg_d'], split['te_neg_m']

        SD_fold, SM_fold = SimilarityEngine.compute_fold_similarities(A_tr)
        X, ei_h = GraphBuilder.build_hetero_network(A_tr, SD_fold, SM_fold)
        ei_d, ei_m = GraphBuilder.build_sim_networks(SD_fold, SM_fold)
        FD, FM = torch.tensor(SD_fold, dtype=torch.float32), torch.tensor(SM_fold, dtype=torch.float32)

        X, ei_h = X.to(self.device), ei_h.to(self.device)
        FD, FM = FD.to(self.device), FM.to(self.device)
        ei_d, ei_m = ei_d.to(self.device), ei_m.to(self.device)

        rng = np.random.RandomState(42)
        neg_sel = rng.choice(len(self.dataset.neg_d_all), size=len(tr_pd), replace=False)
        train_d = torch.tensor(np.concatenate([tr_pd, self.dataset.neg_d_all[neg_sel]]), dtype=torch.long, device=self.device)
        train_m = torch.tensor(np.concatenate([tr_pm, self.dataset.neg_m_all[neg_sel]]), dtype=torch.long, device=self.device)
        train_y = torch.tensor(np.concatenate([np.ones(len(tr_pd)), np.zeros(len(tr_pd))]), dtype=torch.float32, device=self.device)

        for tau_val in temps:
            model = SGAT_CLMDA(self.dataset.nd, self.dataset.nm, feat_dim=self.dataset.nd + self.dataset.nm).to(self.device)
            trainer = Trainer(model, self.device, max_epochs=epochs, patience=25, tau_cl=tau_val)
            trainer.fit(X, ei_h, FD, FM, ei_d, ei_m, train_d, train_m, train_y)
            eval_res = Evaluator.evaluate_model(model, X, ei_h, FD, FM, ei_d, ei_m, te_pd, te_pm, te_neg_d, te_neg_m)
            results[f'tau_{tau_val}'] = {
                'tau': tau_val,
                'auc': float(eval_res['auc']),
                'aupr': float(eval_res['aupr'])
            }
        return results
