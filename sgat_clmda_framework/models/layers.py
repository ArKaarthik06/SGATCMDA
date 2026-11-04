import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import Optional, Tuple

class AdaptiveNodeGatedGAT(nn.Module):
    """
    SAGA Variant A: Adaptive Node Representation Gating / Normalization.
    Gate: g_v = sigmoid(||h_v||_2 - |tau|)
    Attenuates noisy or low-norm node representations with a learnable threshold tau.
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.4, init_tau: float = 0.1):
        super().__init__()
        assert out_dim % heads == 0, f"out_dim ({out_dim}) must be divisible by heads ({heads})"
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, concat=True,
                           dropout=dropout, add_self_loops=True)
        self.bn = nn.LayerNorm(out_dim)
        self.drop = dropout
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gat(x, edge_index)
        gate = torch.sigmoid(h.norm(dim=-1, keepdim=True) - torch.abs(self.tau))
        h = h * gate
        h = F.elu(self.bn(h))
        return F.dropout(h, p=self.drop, training=self.training)

    def get_tau(self) -> float:
        return float(self.tau.item())


class DifferentiableEdgeGatedGAT(nn.Module):
    """
    SAGA Variant B (Addressing Reviewer Comment 3):
    Differentiable Edge-Level Gating with Learnable Sparsity Threshold tau.
    
    Computes an edge relevance score:
      s_{uv} = MLP([h_u || h_v])
      e_{uv} = sigmoid(s_{uv} - |tau|)
      
    During message passing, edge weights e_{uv} dynamically denoise and sparsify graph edges.
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.4, init_tau: float = 0.1):
        super().__init__()
        assert out_dim % heads == 0, f"out_dim ({out_dim}) must be divisible by heads ({heads})"
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, concat=True,
                           dropout=dropout, add_self_loops=True, edge_dim=1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.bn = nn.LayerNorm(out_dim)
        self.drop = dropout
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        self.last_edge_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([x[src], x[dst]], dim=-1)
        edge_logits = self.edge_mlp(edge_feat)
        edge_gate = torch.sigmoid(edge_logits - torch.abs(self.tau))  # Shape [E, 1]
        self.last_edge_weights = edge_gate.squeeze(-1).detach()
        
        # Edge-gated attention message passing (edge_dim=1)
        h = self.gat(x, edge_index, edge_attr=edge_gate)
        h = F.elu(self.bn(h))
        return F.dropout(h, p=self.drop, training=self.training)

    def get_tau(self) -> float:
        return float(self.tau.item())

    def get_edge_sparsity(self, threshold: float = 0.5) -> float:
        """Percentage of edges attenuated below threshold."""
        if self.last_edge_weights is None:
            return 0.0
        return float((self.last_edge_weights < threshold).float().mean().item() * 100)
