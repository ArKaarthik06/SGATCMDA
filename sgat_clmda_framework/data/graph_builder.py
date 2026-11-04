import torch
import numpy as np
from typing import Tuple

class GraphBuilder:
    """
    Constructs dual-view graphs:
      View A: Heterogeneous Association Network (331 nodes: 39 diseases + 292 microbes)
      View B: Disjoint Disease Similarity Network & Microbe Similarity Network
    """

    @staticmethod
    def build_hetero_network(A_mat: np.ndarray, SD_mat: np.ndarray, SM_mat: np.ndarray,
                             threshold: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
        nd = SD_mat.shape[0]
        nm = SM_mat.shape[0]
        n_total = nd + nm
        
        # Heterogeneous block-structured feature matrix Y
        Y = np.zeros((n_total, n_total), dtype=np.float32)
        Y[:nd, :nd] = SD_mat
        Y[:nd, nd:] = A_mat
        Y[nd:, :nd] = A_mat.T
        Y[nd:, nd:] = SM_mat
        
        src, dst = [], []
        
        # 1. Disease-Microbe association edges
        dr, mc = np.where(A_mat == 1)
        for d, m in zip(dr, mc):
            src.extend([int(d), int(nd + m)])
            dst.extend([int(nd + m), int(d)])
            
        # 2. Disease-Disease similarity edges
        r_d, c_d = np.where((SD_mat >= threshold) & (np.eye(nd) == 0))
        for i, j in zip(r_d, c_d):
            src.append(int(i))
            dst.append(int(j))
            
        # 3. Microbe-Microbe KNN similarity edges (top 10 with sim >= 0.05)
        for i in range(nm):
            s = SM_mat[i].copy()
            s[i] = -1.0
            top = np.argsort(s)[-10:]
            for j in top:
                if SM_mat[i, j] >= 0.05:
                    src.append(int(nd + i))
                    dst.append(int(nd + j))
                    
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        X = torch.tensor(Y, dtype=torch.float32)
        return X, edge_index

    @staticmethod
    def build_sim_networks(SD_mat: np.ndarray, SM_mat: np.ndarray,
                           threshold: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
        nd = SD_mat.shape[0]
        nm = SM_mat.shape[0]
        
        # Disease similarity graph
        r, c = np.where((SD_mat >= threshold) & (np.eye(nd) == 0))
        if len(r) == 0:
            r = c = np.arange(nd)
        ei_d = torch.tensor(np.stack([r, c]), dtype=torch.long)
        
        # Microbe similarity graph
        src_m, dst_m = [], []
        for i in range(nm):
            s = SM_mat[i].copy()
            s[i] = -1.0
            top = np.argsort(s)[-10:]
            for j in top:
                if SM_mat[i, j] >= 0.05:
                    src_m.append(i)
                    dst_m.append(int(j))
        if len(src_m) == 0:
            src_m = dst_m = list(range(nm))
        ei_m = torch.tensor([src_m, dst_m], dtype=torch.long)
        
        return ei_d, ei_m
