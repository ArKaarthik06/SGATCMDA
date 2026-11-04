import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, List, Optional, Tuple
import numpy as np

from .losses import total_loss

class Trainer:
    """
    Training manager for SGAT-CLMDA models.
    Supports CosineAnnealingWarmRestarts, early stopping, and tau trajectory tracking.
    """
    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 patience: int = 50, max_epochs: int = 400,
                 lam_cl: float = 0.1, tau_cl: float = 0.5,
                 cl_mode: str = "both"):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_epochs = max_epochs
        self.lam_cl = lam_cl
        self.tau_cl = tau_cl
        self.cl_mode = cl_mode

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=80, T_mult=2, eta_min=1e-5
        )

    def fit(self, X: torch.Tensor, ei_h: torch.Tensor,
            FD: torch.Tensor, FM: torch.Tensor,
            ei_d: torch.Tensor, ei_m: torch.Tensor,
            train_d: torch.Tensor, train_m: torch.Tensor, train_y: torch.Tensor,
            verbose: bool = False) -> Dict:
        
        losses = []
        bce_losses = []
        cl_losses = []
        tau_history = []
        best_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(self.max_epochs):
            self.optimizer.zero_grad()
            scores, zd, zm, zd_A, zm_A, zd_B, zm_B = self.model(X, ei_h, FD, FM, ei_d, ei_m)
            preds = scores[train_d, train_m]
            
            l_tot, l_bce, l_cl = total_loss(
                preds, train_y, zd_A, zd_B, zm_A, zm_B,
                lam_cl=self.lam_cl, tau_cl=self.tau_cl, cl_mode=self.cl_mode
            )
            l_tot.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step(epoch + 1)

            losses.append(l_tot.item())
            bce_losses.append(l_bce.item())
            cl_losses.append(l_cl.item())

            # Track tau values from GAT layers
            taus = list(self.model.get_tau_history().values())
            if taus:
                tau_history.append(float(np.mean(taus)))

            if l_tot.item() < best_loss - 1e-4:
                best_loss = l_tot.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if verbose:
                    print(f"Early stopped at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 100 == 0:
                mean_t = np.mean(taus) if taus else 0.0
                print(f"Epoch {epoch+1:3d} | Loss: {l_tot.item():.4f} | BCE: {l_bce.item():.4f} | CL: {l_cl.item():.4f} | Tau: {mean_t:.4f}")

        return {
            'losses': losses,
            'bce_losses': bce_losses,
            'cl_losses': cl_losses,
            'tau_history': tau_history,
            'final_epoch': len(losses)
        }
