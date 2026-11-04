import numpy as np
import torch
from typing import Dict, List, Tuple

class RepresentationSimilarityAuditor:
    """
    Computes Centered Kernel Alignment (CKA) and Cosine Similarity between
    the GCN branch and SAGA branch representations at each layer (Addressing Reviewer Comment 4).
    """

    @staticmethod
    def centering(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    @classmethod
    def linear_hsic(cls, X: np.ndarray, Y: np.ndarray) -> float:
        K = X @ X.T
        L = Y @ Y.T
        K_c = cls.centering(K)
        L_c = cls.centering(L)
        return float(np.sum(K_c * L_c))

    @classmethod
    def linear_cka(cls, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Linear Centered Kernel Alignment between representations X and Y.
        Output is bounded in [0, 1].
        Values well below 1.0 demonstrate distinct, non-redundant feature spaces.
        """
        hsic_xy = cls.linear_hsic(X, Y)
        hsic_xx = cls.linear_hsic(X, X)
        hsic_yy = cls.linear_hsic(Y, Y)
        denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
        return float(hsic_xy / denom)

    @staticmethod
    def mean_cosine_similarity(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Mean row-wise cosine similarity between aligned node representations.
        """
        norm_x = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        norm_y = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8
        X_normed = X / norm_x
        Y_normed = Y / norm_y
        row_cos = np.sum(X_normed * Y_normed, axis=1)
        return float(np.mean(row_cos))

    @classmethod
    def audit_model_branches(cls, model: torch.nn.Module, X: torch.Tensor, ei_h: torch.Tensor) -> Dict[str, Dict]:
        """
        Extracts representations from all layers and computes CKA & Cosine similarity.
        """
        model.eval()
        with torch.no_grad():
            reps = model.get_branch_representations(X, ei_h)
            
        layer_metrics = {}
        for layer_idx, (h_gcn_t, h_gat_t) in enumerate(reps):
            h_gcn = h_gcn_t.cpu().numpy()
            h_gat = h_gat_t.cpu().numpy()
            
            cka_val = cls.linear_cka(h_gcn, h_gat)
            cos_val = cls.mean_cosine_similarity(h_gcn, h_gat)
            
            layer_metrics[f'Layer_{layer_idx+1}'] = {
                'linear_cka': float(cka_val),
                'mean_cosine': float(cos_val),
                'feature_dim': int(h_gcn.shape[-1]),
                'interpretation': 'Distinct / Complementary' if cka_val < 0.85 else 'Highly Correlated'
            }
        return layer_metrics
