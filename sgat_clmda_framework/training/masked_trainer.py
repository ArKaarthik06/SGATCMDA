import numpy as np
import torch
import torch.nn as nn
from .trainer import Trainer
from .losses import total_loss
from ..data.similarity import SimilarityEngine
from ..data.graph_builder import GraphBuilder


class MaskedEdgeTrainer(Trainer):
    """
    Removes the train-time label shortcut: in the plain trainer every supervised positive pair is also a
    message-passing edge / non-zero entry of the input features, so the model can learn "edge present => positive",
    which never holds for held-out pairs. Here, at every epoch a random fraction `mask_ratio` of the TRAINING
    associations is hidden from the graph, from the A-blocks of the node features, and from the similarity matrices
    (which are recomputed from the remaining edges), while the loss still supervises ALL training pairs.
    Held-out test associations are never touched. At inference the full A_train is used.
    """
    def __init__(self, *args, mask_ratio: float = 0.5, sim_threshold: float = 0.3, seed: int = 0, **kw):
        super().__init__(*args, **kw)
        self.mask_ratio, self.sim_threshold = mask_ratio, sim_threshold
        self.rng = np.random.RandomState(seed)

    def fit(self, A_tr, nd, nm, train_d, train_m, train_y, verbose=False):
        pd_, pm_ = np.where(A_tr == 1)
        losses, best, pat = [], float('inf'), 0
        self.model.train()
        for epoch in range(self.max_epochs):
            keep = self.rng.rand(len(pd_)) >= self.mask_ratio
            A_msg = np.zeros_like(A_tr); A_msg[pd_[keep], pm_[keep]] = 1.0
            SD, SM = SimilarityEngine.compute_fold_similarities(A_msg)
            X, ei = GraphBuilder.build_hetero_network(A_msg, SD, SM, self.sim_threshold)
            ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, self.sim_threshold)
            FD, FM = torch.tensor(SD), torch.tensor(SM)
            self.optimizer.zero_grad()
            scores, zd, zm, zd_A, zm_A, zd_B, zm_B = self.model(X, ei, FD, FM, ei_d, ei_m)
            l_tot, l_bce, l_cl = total_loss(scores[train_d, train_m], train_y, zd_A, zd_B, zm_A, zm_B,
                                            lam_cl=self.lam_cl, tau_cl=self.tau_cl, cl_mode=self.cl_mode)
            l_tot.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step(); self.scheduler.step(epoch + 1)
            losses.append(l_tot.item())
            if l_tot.item() < best - 1e-4: best, pat = l_tot.item(), 0
            else: pat += 1
            if pat >= self.patience: break
        return {'losses': losses, 'final_epoch': len(losses)}
