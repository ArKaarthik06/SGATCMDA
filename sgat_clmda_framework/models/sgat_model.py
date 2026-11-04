import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .encoders import ParallelCrossLevelEncoder, SerialCrossLevelEncoder, GCNOnlyEncoder, GATOnlyEncoder

class SGAT_CLMDA(nn.Module):
    """
    Main SGAT-CLMDA Framework:
    Combines:
      - Dual-view graph encoding (View A: Heterogeneous, View B: Disjoint Similarity)
      - Parallel GCN + SAGA Dual-Branch architecture (or configurable Serial / Ablation mode)
      - Dual-View Contrastive Learning (NT-Xent)
      - Adaptive node norm gating or differentiable edge gating
    """
    def __init__(self, nd: int, nm: int, feat_dim: int,
                 hidden: int = 256, out_dim: int = 64,
                 n_layers: int = 2, heads: int = 4, dropout: float = 0.4,
                 edge_gating_mode: str = "node_norm",
                 architecture: str = "parallel",
                 encoder_type: str = "full",
                 init_tau: float = 0.1):
        super().__init__()
        self.nd = nd
        self.nm = nm
        self.edge_gating_mode = edge_gating_mode
        self.architecture = architecture
        self.encoder_type = encoder_type

        # Select encoder variant
        if encoder_type == "gcn_only":
            enc_fn = lambda in_d: GCNOnlyEncoder(in_d, hidden, out_dim, n_layers, dropout)
        elif encoder_type == "gat_only":
            enc_fn = lambda in_d: GATOnlyEncoder(in_d, hidden, out_dim, n_layers, heads, dropout)
        elif architecture == "serial":
            enc_fn = lambda in_d: SerialCrossLevelEncoder(in_d, hidden, out_dim, n_layers, heads, dropout)
        else:
            enc_fn = lambda in_d: ParallelCrossLevelEncoder(
                in_d, hidden, out_dim, n_layers, heads, dropout,
                edge_gating_mode=edge_gating_mode, init_tau=init_tau)

        self.enc_A = enc_fn(feat_dim)
        self.enc_B_d = enc_fn(nd)
        self.enc_B_m = enc_fn(nm)

        self.combine_d = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
        self.combine_m = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def encode(self, X: torch.Tensor, ei_h: torch.Tensor,
               FD: torch.Tensor, FM: torch.Tensor,
               ei_d: torch.Tensor, ei_m: torch.Tensor):
        z_A = self.enc_A(X, ei_h)
        zd_A = z_A[:self.nd]
        zm_A = z_A[self.nd:]
        zd_B = self.enc_B_d(FD, ei_d)
        zm_B = self.enc_B_m(FM, ei_m)
        
        zd = self.combine_d(torch.cat([zd_A, zd_B], dim=-1))
        zm = self.combine_m(torch.cat([zm_A, zm_B], dim=-1))
        return zd, zm, zd_A, zm_A, zd_B, zm_B

    def decode(self, zd: torch.Tensor, zm: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(zd @ zm.T)

    def forward(self, X: torch.Tensor, ei_h: torch.Tensor,
                FD: torch.Tensor, FM: torch.Tensor,
                ei_d: torch.Tensor, ei_m: torch.Tensor):
        zd, zm, zd_A, zm_A, zd_B, zm_B = self.encode(X, ei_h, FD, FM, ei_d, ei_m)
        scores = self.decode(zd, zm)
        return scores, zd, zm, zd_A, zm_A, zd_B, zm_B

    def get_tau_history(self) -> Dict[str, float]:
        taus = {}
        if hasattr(self.enc_A, 'gat_layers'):
            for enc_name, enc in [('A', self.enc_A), ('B_d', self.enc_B_d), ('B_m', self.enc_B_m)]:
                for li, layer in enumerate(enc.gat_layers):
                    if hasattr(layer, 'get_tau'):
                        taus[f'{enc_name}_L{li}'] = layer.get_tau()
        return taus

    def get_branch_representations(self, X: torch.Tensor, ei_h: torch.Tensor):
        if hasattr(self.enc_A, 'get_branch_representations'):
            return self.enc_A.get_branch_representations(X, ei_h)
        return []

    def get_edge_sparsity(self, threshold: float = 0.5) -> float:
        if hasattr(self.enc_A, 'get_edge_sparsity'):
            return self.enc_A.get_edge_sparsity(threshold)
        return 0.0
