"""NTSHMDA-style random walk with restart, built from A_train only (usage: python run_rwr.py <seed>).
Same splits, negatives and masked model (ratio 0.5, same seeds) as run_robustness.py. Variants (all leakage-free):
  rwr                : NTSHMDA-style RWR alone (eta=0.7, phi=0.9, delta=0.3, 10 iterations - the published values)
  cf / cf+rwr        : item-CF reference and its rank blend with RWR
  model              : masked-edge SGAT-CLMDA (as in the main table)
  model+rwr, model+cf, model+cf+rwr : tie-averaged rank blends (equal weights, fixed a priori)
  fused              : the SAME architecture with the RWR rank prior added to its decoder logit,
                       score = sigmoid(z_d . z_m + s * (rank_prior - 0.5)), s learned. The prior is recomputed from the
                       masked message-passing matrix at every epoch and is neutral (0) for pairs that are message-passing
                       edges, so train and test see the prior under the same condition. Test pairs are never used.
  fused+cf           : rank blend of the fused model with item-CF
Reading of an under-specified step: NTSHMDA gives the converged probability matrix P; we score (d, m) as
P[m <- seed d] + P[d <- seed m]."""
import os, sys, json, numpy as np, torch, torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine as SE
from sgat_clmda_framework.data.graph_builder import GraphBuilder
from sgat_clmda_framework.models.sgat_model import SGAT_CLMDA
from sgat_clmda_framework.training.masked_trainer import MaskedEdgeTrainer
from sgat_clmda_framework.training.losses import total_loss

seed = int(sys.argv[1]); torch.set_num_threads(int(os.environ.get('NT', 3)))
cfg = Config(); ds = DatasetLoader(cfg.data_path, 42); nd, nm = ds.nd, ds.nm
OUT = os.path.join(cfg.output_dir, f'rwr_{seed}.json'); res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def safe_div(a, b): return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)
def rwr(A, SD, SM, eta=0.7, phi=0.9, delta=0.3, iters=10):
    A = A.astype(float); a_ms, a_ds = A @ SM, SD @ A                      # Eq. 7-8
    num_m, num_d = A * a_ms, A * a_ds
    W_dm = phi * safe_div(num_m, num_m.sum(1, keepdims=True))             # disease i -> microbe j   (Eq. 9)
    W_md = phi * safe_div(num_d, num_d.sum(0, keepdims=True))             # microbe j -> disease i   (Eq. 10)
    cd = np.where(A.sum(1) > 0, 1 - phi, 1.0)[:, None]; cm = np.where(A.sum(0) > 0, 1 - phi, 1.0)[:, None]
    W_d = cd * safe_div(SD, SD.sum(1, keepdims=True)); W_m = cm * safe_div(SM, SM.sum(1, keepdims=True))   # Eq. 11-12
    T = np.zeros((nd + nm, nd + nm))                                      # column-stochastic: T[target, source]
    T[:nd, :nd], T[nd:, nd:], T[nd:, :nd], T[:nd, nd:] = W_d.T, W_m.T, W_dm.T, W_md
    P0 = np.diag(np.r_[np.full(nd, 1 - delta), np.full(nm, delta)]); P = P0.copy()
    for _ in range(iters): P = (1 - eta) * T @ P + eta * P0               # Eq. 13
    return P[nd:, :nd].T + P[:nd, nd:]
def cf_scores(A, SM): sm = SM.copy(); np.fill_diagonal(sm, 0); return A @ sm
def pr(S, A):
    m = (A == 0); r = np.zeros_like(S, dtype=float); r[m] = rankdata(S[m]) / m.sum(); return r
def metrics(sp, S):
    y = np.r_[np.ones(len(sp['te_pd'])), np.zeros(len(sp['te_neg_d']))]; s = np.r_[S[sp['te_pd'], sp['te_pm']], S[sp['te_neg_d'], sp['te_neg_m']]]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
def prior(A, SD, SM):
    """rank prior in [-0.5, 0.5] over non-edge pairs, 0 on message-passing edges"""
    return np.where(A == 0, pr(rwr(A, SD, SM), A) - 0.5, 0.0)

class FusedTrainer(MaskedEdgeTrainer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.s = nn.Parameter(torch.tensor(3.0)); self.opt_s = torch.optim.Adam([self.s], lr=0.05)
    def fit(self, A_tr, nd, nm, train_d, train_m, train_y):
        pd_, pm_ = np.where(A_tr == 1); best, pat = float('inf'), 0; self.model.train()
        for epoch in range(self.max_epochs):
            keep = self.rng.rand(len(pd_)) >= self.mask_ratio
            A_msg = np.zeros_like(A_tr); A_msg[pd_[keep], pm_[keep]] = 1.0
            SD, SM = SE.compute_fold_similarities(A_msg); X, ei = GraphBuilder.build_hetero_network(A_msg, SD, SM, self.sim_threshold)
            ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, self.sim_threshold); Pr = torch.tensor(prior(A_msg, SD, SM), dtype=torch.float32)
            self.optimizer.zero_grad(); self.opt_s.zero_grad()
            _, zd, zm, zd_A, zm_A, zd_B, zm_B = self.model(X, ei, torch.tensor(SD), torch.tensor(SM), ei_d, ei_m)
            pred = torch.sigmoid(zd @ zm.T + self.s * Pr)
            l, _, _ = total_loss(pred[train_d, train_m], train_y, zd_A, zd_B, zm_A, zm_B, lam_cl=self.lam_cl, tau_cl=self.tau_cl, cl_mode=self.cl_mode)
            l.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step(); self.opt_s.step(); self.scheduler.step(epoch + 1)
            if l.item() < best - 1e-4: best, pat = l.item(), 0
            else: pat += 1
            if pat >= self.patience: break

def train(A, rs, fused):
    SD, SM = SE.compute_fold_similarities(A); X, ei = GraphBuilder.build_hetero_network(A, SD, SM, cfg.sim_threshold)
    ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold); pd_, pm_ = np.where(A == 1)
    rng = np.random.RandomState(rs); ns = rng.choice(len(ds.neg_d_all), len(pd_), replace=False)
    td = torch.tensor(np.r_[pd_, ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[pm_, ds.neg_m_all[ns]]); ty = torch.tensor(np.r_[np.ones(len(pd_)), np.zeros(len(ns))], dtype=torch.float32)
    torch.manual_seed(rs); m = SGAT_CLMDA(nd, nm, nd + nm)
    T = (FusedTrainer if fused else MaskedEdgeTrainer)(m, torch.device('cpu'), max_epochs=120, patience=25, mask_ratio=0.5, seed=rs)
    T.fit(A, nd, nm, td, tm, ty); m.eval()
    with torch.no_grad():
        _, zd, zm, *_ = m(X, ei, torch.tensor(SD), torch.tensor(SM), ei_d, ei_m)
        L = (zd @ zm.T).numpy()
        if fused: L = L + T.s.item() * prior(A, SD, SM)
    return 1 / (1 + np.exp(-L)), SD, SM, (T.s.item() if fused else None)

for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
    key = f"{seed}|{sp['fold']}"
    if key in res: continue
    A = sp['A_tr']; ev = lambda S: metrics(sp, S); rs = seed * 1000 + sp['fold']
    S_m, SD, SM, _ = train(A, rs, False); S_f, _, _, s_val = train(A, rs, True)
    R, CF = rwr(A, SD, SM), cf_scores(A, SM); pm_, pf_, pc_, pfu = pr(S_m, A), pr(R, A), pr(CF, A), pr(S_f, A)
    out = {'rwr': ev(R), 'cf': ev(CF), 'cf+rwr': ev(.5 * pc_ + .5 * pf_), 'model': ev(S_m), 'model+cf': ev(.5 * pm_ + .5 * pc_),
           'model+rwr': ev(.5 * pm_ + .5 * pf_), 'model+cf+rwr': ev((pm_ + pc_ + pf_) / 3), 'fused': ev(S_f), 'fused+cf': ev(.5 * pfu + .5 * pc_), 'fused_s': s_val}
    res[key] = out; json.dump(res, open(OUT, 'w'))
    print(key, {k: round(v[0], 3) if isinstance(v, (list, tuple)) else round(v, 2) for k, v in out.items()}, flush=True)
