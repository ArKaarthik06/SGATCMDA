import numpy as np
from typing import Tuple

class SimilarityEngine:
    """
    Engine for calculating Gaussian Interaction Profile (GIP) kernel similarity,
    Cosine similarity, and Integrated similarity.
    
    Provides strict per-fold recomputation to prevent data leakage (Addressing Reviewer Comment 1).
    """

    @staticmethod
    def gip_similarity(profiles: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian Interaction Profile (GIP) kernel similarity.
        profiles: shape (n_samples, feature_dim)
        """
        norms = np.sum(profiles**2, axis=1)
        mean_norm = np.mean(norms)
        lam = 1.0 / (mean_norm + 1e-8) if mean_norm > 0 else 1.0
        diff = profiles[:, None, :] - profiles[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        return np.exp(-lam * dist_sq).astype(np.float32)

    @staticmethod
    def cosine_similarity(profiles: np.ndarray) -> np.ndarray:
        """
        Compute Cosine similarity. Handles zero-norm profiles safely.
        profiles: shape (n_samples, feature_dim)
        """
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        # Avoid division by zero for unassociated nodes in training fold
        non_zero_mask = (norms > 1e-8)
        normed = np.zeros_like(profiles)
        np.divide(profiles, norms, out=normed, where=non_zero_mask)
        
        sim = normed @ normed.T
        sim = np.clip(sim.astype(np.float32), 0.0, 1.0)
        # Self-similarity should always be 1.0
        np.fill_diagonal(sim, 1.0)
        return sim

    @classmethod
    def integrated_similarity(cls, profiles: np.ndarray) -> np.ndarray:
        """
        Average of GIP kernel similarity and Cosine similarity.
        """
        gip = cls.gip_similarity(profiles)
        cos = cls.cosine_similarity(profiles)
        return ((gip + cos) / 2.0).astype(np.float32)

    @classmethod
    def compute_fold_similarities(cls, A_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        STRICTLY LEAKAGE-FREE PROTOCOL:
        Computes disease similarity SD and microbe similarity SM using ONLY A_train.
        """
        SD_fold = cls.integrated_similarity(A_train)
        SM_fold = cls.integrated_similarity(A_train.T)
        return SD_fold, SM_fold

    @classmethod
    def compute_full_similarities(cls, A_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        LEAKY PROTOCOL:
        Computes disease similarity SD and microbe similarity SM using the full A.
        Used for ablation comparison against the honest protocol.
        """
        SD_full = cls.integrated_similarity(A_full)
        SM_full = cls.integrated_similarity(A_full.T)
        return SD_full, SM_full
