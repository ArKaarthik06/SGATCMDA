import torch
import torch.nn.functional as F
from typing import Tuple

def bce_loss(pred: torch.Tensor, target: torch.Tensor, max_pos_weight: float = 15.0) -> torch.Tensor:
    """
    Weighted Binary Cross Entropy loss to handle extreme class imbalance (3.95% positive density).
    """
    n_pos = target.sum().clamp(min=1.0)
    n_neg = (1.0 - target).sum().clamp(min=1.0)
    pw = (n_neg / n_pos).clamp(max=max_pos_weight)
    weights = torch.where(target == 1.0, pw * torch.ones_like(target), torch.ones_like(target))
    return F.binary_cross_entropy(pred, target, weight=weights)

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) contrastive loss.
    Aligns representation of identical nodes across View A and View B.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    n = z1.shape[0]
    logits = (z1 @ z2.T) / tau
    labels = torch.arange(n, device=z1.device)
    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.T, labels)
    return (loss_1 + loss_2) / 2.0

def total_loss(pred: torch.Tensor, target: torch.Tensor,
               zd_A: torch.Tensor, zd_B: torch.Tensor,
               zm_A: torch.Tensor, zm_B: torch.Tensor,
               lam_cl: float = 0.1, tau_cl: float = 0.5,
               cl_mode: str = "both") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined classification and contrastive loss.
    Supports modular CL ablation modes (Addressing Reviewer Comment 5):
      - 'both'         : Full CL (Disease CL + Microbe CL) / 2
      - 'disease_only' : Contrastive alignment on disease graph only
      - 'microbe_only' : Contrastive alignment on microbe graph only
      - 'none'         : BCE loss only
    """
    l_bce = bce_loss(pred, target)
    
    if cl_mode == "none" or lam_cl <= 0.0:
        return l_bce, l_bce, torch.tensor(0.0, device=pred.device)
    
    l_cl_d = nt_xent_loss(zd_A, zd_B, tau=tau_cl) if cl_mode in ["both", "disease_only"] else torch.tensor(0.0, device=pred.device)
    l_cl_m = nt_xent_loss(zm_A, zm_B, tau=tau_cl) if cl_mode in ["both", "microbe_only"] else torch.tensor(0.0, device=pred.device)
    
    if cl_mode == "both":
        l_cl = (l_cl_d + l_cl_m) / 2.0
    elif cl_mode == "disease_only":
        l_cl = l_cl_d
    elif cl_mode == "microbe_only":
        l_cl = l_cl_m
    else:
        l_cl = torch.tensor(0.0, device=pred.device)
        
    l_total = l_bce + lam_cl * l_cl
    return l_total, l_bce, l_cl
