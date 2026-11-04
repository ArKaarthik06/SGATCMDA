import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from typing import List, Tuple, Dict
from .layers import AdaptiveNodeGatedGAT, DifferentiableEdgeGatedGAT

class ParallelCrossLevelEncoder(nn.Module):
    """
    Parallel GCN + SAGA Dual-Branch Encoder with Adaptive Gated Cross-Branch Fusion (AG-Fusion).
    Both branches run in parallel on the same hidden state h at every layer,
    adaptively fused via channel-wise gating and stabilized by LayerNorm residual connections.
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 n_layers: int = 2, heads: int = 4, dropout: float = 0.4,
                 edge_gating_mode: str = "node_norm", init_tau: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.gcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.fusion_gates = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        self.res_norms = nn.ModuleList()
        
        gat_cls = DifferentiableEdgeGatedGAT if edge_gating_mode == "differentiable_edge" else AdaptiveNodeGatedGAT
        
        for i in range(n_layers):
            d_out = out_dim if i == n_layers - 1 else hidden
            self.gcn_layers.append(GCNConv(hidden, d_out))
            self.gat_layers.append(gat_cls(hidden, d_out, heads=heads, dropout=dropout, init_tau=init_tau))
            self.bn_layers.append(nn.LayerNorm(d_out))
            # Adaptive channel-wise fusion gate between GCN and GAT branches
            self.fusion_gates.append(nn.Sequential(
                nn.Linear(d_out * 2, d_out),
                nn.Sigmoid()
            ))
            self.proj_layers.append(nn.Sequential(
                nn.Linear(d_out, out_dim),
                nn.LayerNorm(out_dim)
            ))
            self.res_norms.append(nn.LayerNorm(d_out))
            
        self.level_w = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.drop = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        w = F.softmax(self.level_w, dim=0)
        out = []
        for i in range(self.n_layers):
            h_gcn = F.relu(self.bn_layers[i](self.gcn_layers[i](h, edge_index)))
            h_gcn = F.dropout(h_gcn, p=self.drop, training=self.training)
            h_gat = self.gat_layers[i](h, edge_index)
            
            # Adaptive Gated Cross-Branch Fusion
            gate = self.fusion_gates[i](torch.cat([h_gcn, h_gat], dim=-1))
            h_fused = gate * h_gcn + (1.0 - gate) * h_gat
            
            p = self.proj_layers[i]
            out.append(w[i] * F.elu(p(h_fused)))
            
            # Normalized residual connection prevents representation explosion
            if h_fused.shape[-1] == h.shape[-1]:
                h = self.res_norms[i](h + h_fused)
            else:
                h = self.res_norms[i](h_fused)
        return sum(out)

    def get_branch_representations(self, x: torch.Tensor, edge_index: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extracts (h_gcn, h_gat) for each layer to compute CKA and Cosine similarity.
        (Addressing Reviewer Comment 4)
        """
        h = self.proj(x)
        reps = []
        for i in range(self.n_layers):
            h_gcn = F.relu(self.bn_layers[i](self.gcn_layers[i](h, edge_index)))
            h_gat = self.gat_layers[i](h, edge_index)
            reps.append((h_gcn.detach(), h_gat.detach()))
            gate = self.fusion_gates[i](torch.cat([h_gcn, h_gat], dim=-1))
            h_fused = gate * h_gcn + (1.0 - gate) * h_gat
            if h_fused.shape[-1] == h.shape[-1]:
                h = self.res_norms[i](h + h_fused)
            else:
                h = self.res_norms[i](h_fused)
        return reps

    def get_tau_values(self) -> List[float]:
        return [layer.get_tau() for layer in self.gat_layers]

    def get_edge_sparsity(self, threshold: float = 0.5) -> float:
        sparsities = []
        for layer in self.gat_layers:
            if hasattr(layer, 'get_edge_sparsity'):
                sparsities.append(layer.get_edge_sparsity(threshold))
        return float(np.mean(sparsities)) if sparsities else 0.0


class SerialCrossLevelEncoder(nn.Module):
    """
    Serial GCN -> GAT Architecture (GCATCMDA baseline style).
    GAT processes the output of GCN hierarchically (Addressing Reviewer Comment 4).
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 n_layers: int = 2, heads: int = 4, dropout: float = 0.4):
        super().__init__()
        self.n_layers = n_layers
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.gcn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        for i in range(n_layers):
            d_out = out_dim if i == n_layers - 1 else hidden
            assert d_out % heads == 0
            self.gcn_layers.append(GCNConv(hidden, hidden))
            self.gat_layers.append(GATConv(hidden, d_out // heads, heads=heads, concat=True,
                                           dropout=dropout, add_self_loops=True))
            self.bn_layers.append(nn.LayerNorm(d_out))
            self.proj_layers.append(nn.Linear(d_out, out_dim))
        self.level_w = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.drop = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        w = F.softmax(self.level_w, dim=0)
        out = []
        for i in range(self.n_layers):
            # Serial: GCN first, then GAT on GCN output
            h_gcn = F.relu(self.gcn_layers[i](h, edge_index))
            h_gcn = F.dropout(h_gcn, p=self.drop, training=self.training)
            h_gat = F.elu(self.bn_layers[i](self.gat_layers[i](h_gcn, edge_index)))
            h_gat = F.dropout(h_gat, p=self.drop, training=self.training)
            
            p = self.proj_layers[i]
            out.append(w[i] * p(h_gat))
            if h_gat.shape[-1] == h.shape[-1]:
                h = h_gat + h
            else:
                h = h_gat
        return sum(out)


class GCNOnlyEncoder(nn.Module):
    """Ablation: GCN-only encoder."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2))
        self.gcn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.level_w = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.drop = dropout
        for i in range(n_layers):
            d_out = out_dim if i == n_layers - 1 else hidden
            self.gcn_layers.append(GCNConv(hidden, d_out))
            self.bn_layers.append(nn.LayerNorm(d_out))
        self.out_proj = nn.ModuleList([nn.Linear(hidden if i < n_layers - 1 else out_dim, out_dim) for i in range(n_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        w = F.softmax(self.level_w, dim=0)
        out = []
        for i in range(len(self.gcn_layers)):
            h = F.relu(self.bn_layers[i](self.gcn_layers[i](h, edge_index)))
            h = F.dropout(h, p=self.drop, training=self.training)
            out.append(w[i] * self.out_proj[i](h))
        return sum(out)


class GATOnlyEncoder(nn.Module):
    """Ablation: GAT-only encoder."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 2, heads: int = 4, dropout: float = 0.4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2))
        self.gat_layers = nn.ModuleList()
        self.level_w = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.drop = dropout
        for i in range(n_layers):
            d_out = out_dim if i == n_layers - 1 else hidden
            assert d_out % heads == 0
            self.gat_layers.append(GATConv(hidden, d_out // heads, heads=heads,
                                           concat=True, dropout=dropout, add_self_loops=True))
        self.out_proj = nn.ModuleList([nn.Linear(hidden if i < n_layers - 1 else out_dim, out_dim) for i in range(n_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        w = F.softmax(self.level_w, dim=0)
        out = []
        for i, layer in enumerate(self.gat_layers):
            h = F.elu(layer(h, edge_index))
            h = F.dropout(h, p=self.drop, training=self.training)
            out.append(w[i] * self.out_proj[i](h))
        return sum(out)
