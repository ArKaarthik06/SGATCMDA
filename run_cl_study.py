"""Contrastive-learning study (usage: python run_cl_study.py <variant>), leakage-free recipe (per-fold similarities, masked-edge training 0.5), 5 folds x 5 seeds = 25 runs per variant,
the same splits, training negatives and initial seeds as the earlier ablation (so `none` and `ntxent` reproduce the existing table).
Variants (fixed in advance; every variant is reported, none is chosen on test folds):
  none     : BCE only
  ntxent   : the original loss - NT-Xent between the View A and View B embeddings of the same node, no projection head, lambda 0.1, tau 0.5
  grace_ab : GRACE-style loss as specified for GCATCMDA (Zhu et al. 2020): shared 2-layer projection head, positives = same node in the other view,
             negatives = all other nodes in BOTH views (inter- and intra-view), View A <-> View B, GCATCMDA's published lambda 0.2 and tau 1.0
  grace_aug: GRACE loss between two independently edge-masked augmentations of View A (graph-augmentation contrastive learning), lambda 0.2, tau 0.5
  grace_both: grace_ab + grace_aug
Metrics: test AUC/AUPR of the masked model alone (as in the existing table) and AUC after the fixed 0.5/0.5 rank blend with training-only item-CF."""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine as SE
from sgat_clmda_framework.data.graph_builder import GraphBuilder
from sgat_clmda_framework.models.sgat_model import SGAT_CLMDA
from sgat_clmda_framework.training.masked_trainer import MaskedEdgeTrainer
from sgat_clmda_framework.training.losses import total_loss
from sgat_clmda_framework.training.evaluator import Evaluator

variant = sys.argv[1]; torch.set_num_threads(int(os.environ.get('NT', 3)))
cfg = Config(); ds = DatasetLoader(cfg.data_path, 42); nd, nm = ds.nd, ds.nm
OUT = cfg.output_dir + f'/cl_study_{variant}{os.environ.get("SUFFIX", "")}.json'; res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def grace(u, v, tau):
    u, v = F.normalize(u, dim=-1), F.normalize(v, dim=-1); n = u.shape[0]; eye = torch.eye(n, dtype=torch.bool)
    def one(a, b):
        inter, intra = torch.exp(a @ b.T / tau), torch.exp(a @ a.T / tau)
        return -torch.log(inter.diag() / (inter.diag() + inter.masked_fill(eye, 0).sum(1) + intra.masked_fill(eye, 0).sum(1)))
    return 0.5 * (one(u, v) + one(v, u)).mean()
class Head(nn.Module):
    def __init__(self, d=64): super().__init__(); self.f = nn.Sequential(nn.Linear(d, d), nn.ELU(), nn.Linear(d, d))
    def forward(self, z): return self.f(z)

class CLTrainer(MaskedEdgeTrainer):
    def __init__(self, *a, mode='none', **k):
        k.update(dict(cl_mode='none' if mode in ('none', 'grace_ab', 'grace_aug', 'grace_both') else 'both')); super().__init__(*a, **k)
        self.mode = mode; self.head_d, self.head_m = Head(), Head(); self.opt_h = torch.optim.Adam(list(self.head_d.parameters()) + list(self.head_m.parameters()), lr=1e-3, weight_decay=1e-4)
    def graphs(self, A_msg):
        SD, SM = SE.compute_fold_similarities(A_msg); X, ei = GraphBuilder.build_hetero_network(A_msg, SD, SM, self.sim_threshold)
        ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, self.sim_threshold); return X, ei, torch.tensor(SD), torch.tensor(SM), ei_d, ei_m
    def fit(self, A_tr, nd, nm, train_d, train_m, train_y, verbose=False):
        pd_, pm_ = np.where(A_tr == 1); best, pat = float('inf'), 0; self.model.train()
        def masked():
            keep = self.rng.rand(len(pd_)) >= self.mask_ratio; A = np.zeros_like(A_tr); A[pd_[keep], pm_[keep]] = 1.0; return A
        for epoch in range(self.max_epochs):
            g1 = self.graphs(masked()); self.optimizer.zero_grad(); self.opt_h.zero_grad()
            scores, zd, zm, zd_A, zm_A, zd_B, zm_B = self.model(*g1)
            loss, _, _ = total_loss(scores[train_d, train_m], train_y, zd_A, zd_B, zm_A, zm_B, lam_cl=self.lam_cl, tau_cl=self.tau_cl, cl_mode=self.cl_mode)
            if self.mode in ('grace_ab', 'grace_both'):
                loss = loss + 0.2 * 0.5 * (grace(self.head_d(zd_A), self.head_d(zd_B), 1.0) + grace(self.head_m(zm_A), self.head_m(zm_B), 1.0))
            if self.mode in ('grace_aug', 'grace_both'):
                g2 = self.graphs(masked()); z2 = self.model.enc_A(g2[0], g2[1])
                loss = loss + 0.2 * 0.5 * (grace(self.head_d(zd_A), self.head_d(z2[:nd]), 0.5) + grace(self.head_m(zm_A), self.head_m(z2[nd:]), 0.5))
            loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step(); self.opt_h.step(); self.scheduler.step(epoch + 1)
            if loss.item() < best - 1e-4: best, pat = loss.item(), 0
            else: pat += 1
            if pat >= self.patience: break

def pr(S, A):
    m = (A == 0); r = np.zeros_like(S, dtype=float); r[m] = rankdata(S[m]) / m.sum(); return r
def ev(sp, S):
    y = np.r_[np.ones(len(sp['te_pd'])), np.zeros(len(sp['te_neg_d']))]; s = np.r_[S[sp['te_pd'], sp['te_pm']], S[sp['te_neg_d'], sp['te_neg_m']]]; return float(roc_auc_score(y, s))
for seed in [int(x) for x in os.environ.get('SEEDS', '42,123,456,789,2024').split(',')]:
    for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
        key = f"{seed}|{sp['fold']}"
        if key in res: continue
        A = sp['A_tr']; SD, SM = SE.compute_fold_similarities(A); X, ei = GraphBuilder.build_hetero_network(A, SD, SM, cfg.sim_threshold); ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold)
        rng = np.random.RandomState(seed * 10 + sp['fold']); ns = rng.choice(len(ds.neg_d_all), len(sp['tr_pd']), replace=False)
        td = torch.tensor(np.r_[sp['tr_pd'], ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[sp['tr_pm'], ds.neg_m_all[ns]]); ty = torch.tensor(np.r_[np.ones(len(sp['tr_pd'])), np.zeros(len(ns))], dtype=torch.float32)
        torch.manual_seed(seed + sp['fold']); model = SGAT_CLMDA(nd, nm, nd + nm); t0 = time.time()
        CLTrainer(model, torch.device('cpu'), max_epochs=120, patience=25, mask_ratio=0.5, seed=seed + sp['fold'], mode=variant).fit(A, nd, nm, td, tm, ty)
        e = Evaluator.evaluate_model(model, X, ei, torch.tensor(SD), torch.tensor(SM), ei_d, ei_m, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m'])
        sm = SM.copy(); np.fill_diagonal(sm, 0); cf = A @ sm; blend = ev(sp, .5 * pr(e['score_matrix'], A) + .5 * pr(cf, A))
        res[key] = dict(auc=e['auc'], aupr=e['aupr'], blend_auc=blend, sec=time.time() - t0); json.dump(res, open(OUT, 'w'))
        print(variant, key, {k: round(v, 4) for k, v in res[key].items()}, flush=True)
