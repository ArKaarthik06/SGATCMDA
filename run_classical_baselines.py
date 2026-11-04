"""Honest-protocol re-implementations (NOT the original authors' code) of classical MDA predictors, 5 folds x 5 seeds.
All similarities come from A_train only. Hyper-parameters are fixed a priori and not tuned on test folds.
  KATZ-style : S = sum_{l=1..4} beta^(l-1) W^l on the heterogeneous matrix W=[[S_D,A],[A^T,S_M]] (zero diagonal), beta=0.01
  RWR-style  : random walk with restart on column-normalised W, restart prob r (headline r=0.3; 0.1/0.5 shown as sensitivity)
  GRNMF-style: graph-regularised NMF  A ~ U V^T, k=10, lambda=0.1, 200 multiplicative updates
"""
import json, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine as SE
ds = DatasetLoader(Config().data_path, 42); nd, nm = ds.nd, ds.nm
def hetero(A, SD, SM):
    W = np.zeros((nd + nm, nd + nm)); sd, sm = SD.copy(), SM.copy(); np.fill_diagonal(sd, 0); np.fill_diagonal(sm, 0)
    W[:nd, :nd] = sd; W[nd:, nd:] = sm; W[:nd, nd:] = A; W[nd:, :nd] = A.T; return W
def katz(W, beta=0.01, L=4):
    S = np.zeros_like(W); P = np.eye(len(W))
    for l in range(1, L + 1): P = P @ W; S += beta ** (l - 1) * P
    return S[:nd, nd:]
def rwr(W, r):
    cs = W.sum(0); cs[cs == 0] = 1; Wn = W / cs
    R = r * np.linalg.inv(np.eye(len(W)) - (1 - r) * Wn); return R[nd:, :nd].T      # column d = steady state of walk restarting at disease d
def grnmf(A, SD, SM, k=10, lam=0.1, it=200, seed=0):
    rng = np.random.RandomState(seed); U = rng.rand(nd, k) + 0.1; V = rng.rand(nm, k) + 0.1
    sd, sm = SD.copy(), SM.copy(); np.fill_diagonal(sd, 0); np.fill_diagonal(sm, 0); Dd, Dm = np.diag(sd.sum(1)), np.diag(sm.sum(1)); e = 1e-9
    for _ in range(it):
        U *= (A @ V + lam * sd @ U) / (U @ (V.T @ V) + lam * Dd @ U + e)
        V *= (A.T @ U + lam * sm @ V) / (V @ (U.T @ U) + lam * Dm @ V + e)
    return U @ V.T
def ev(sp, S):
    y = np.r_[np.ones(len(sp['te_pd'])), np.zeros(len(sp['te_neg_d']))]
    s = np.r_[S[sp['te_pd'], sp['te_pm']], S[sp['te_neg_d'], sp['te_neg_m']]]; return roc_auc_score(y, s), average_precision_score(y, s)
R = {k: [] for k in ['KATZ-style', 'RWR-style (r=0.3)', 'RWR-style (r=0.1)', 'RWR-style (r=0.5)', 'GRNMF-style']}
for seed in [42, 123, 456, 789, 2024]:
    for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
        A = sp['A_tr']; SD, SM = SE.compute_fold_similarities(A); W = hetero(A, SD, SM)
        R['KATZ-style'].append(ev(sp, katz(W)))
        for r in (0.1, 0.3, 0.5): R[f'RWR-style (r={r})'].append(ev(sp, rwr(W, r)))
        R['GRNMF-style'].append(ev(sp, grnmf(A, SD, SM, seed=seed + sp['fold'])))
out = {k: dict(auc=float(np.mean([a for a, _ in v])), auc_sd=float(np.std([a for a, _ in v], ddof=1)), aupr=float(np.mean([b for _, b in v])), per_run_auc=[float(a) for a, _ in v], n=len(v)) for k, v in R.items()}
json.dump(out, open('sgat_clmda_outputs/classical_baselines.json', 'w'), indent=1)
for k, v in out.items(): print(f"{k:20s} AUC {v['auc']:.4f} ± {v['auc_sd']:.4f}  AUPR {v['aupr']:.4f}")
