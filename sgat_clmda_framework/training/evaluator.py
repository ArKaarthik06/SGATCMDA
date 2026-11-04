import numpy as np
import torch
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

class Evaluator:
    """
    Evaluator for link-prediction metrics (AUC-ROC, AUPR-PRC, curves).
    """

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
        auc = float(roc_auc_score(y_true, y_score))
        aupr = float(average_precision_score(y_true, y_score))
        fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
        precision, recall, pr_thresh = precision_recall_curve(y_true, y_score)
        return {
            'auc': auc,
            'aupr': aupr,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }

    @classmethod
    def evaluate_model(cls, model: torch.nn.Module,
                       X: torch.Tensor, ei_h: torch.Tensor,
                       FD: torch.Tensor, FM: torch.Tensor,
                       ei_d: torch.Tensor, ei_m: torch.Tensor,
                       te_pd: np.ndarray, te_pm: np.ndarray,
                       te_neg_d: np.ndarray, te_neg_m: np.ndarray) -> Dict:
        model.eval()
        with torch.no_grad():
            score_mat, *_ = model(X, ei_h, FD, FM, ei_d, ei_m)
            score_np = score_mat.cpu().numpy()
            
        pos_scores = score_np[te_pd, te_pm]
        neg_scores = score_np[te_neg_d, te_neg_m]
        
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_score = np.concatenate([pos_scores, neg_scores])
        
        res = cls.compute_metrics(y_true, y_score)
        res['pos_scores'] = pos_scores
        res['neg_scores'] = neg_scores
        res['score_matrix'] = score_np
        return res
