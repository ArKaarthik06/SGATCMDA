"""Robustness checks for the final recipe (honest protocol; usage: python run_robustness.py <seed>).
Rank-blend variants over candidate pairs (A_train == 0), all built from A_train only:
  cf+katz          : fairness check - classical methods get the same blending
  model+katz, model+cf+katz
  model+cf (fixed) : the headline recipe (mask 0.5, weight 0.5 fixed a priori)
  model+cf (inner) : mask ratio in {0.25, 0.5} and model weight in {0.25, 0.5, 0.75} chosen by validation INSIDE the training fold
                     (20% of the training positives + equal sampled unobserved pairs held out); test pairs are never used for selection."""
import os, sys, json, numpy as np, torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine as SE
from sgat_clmda_framework.data.graph_builder import GraphBuilder
from sgat_clmda_framework.models.sgat_model import SGAT_CLMDA
from sgat_clmda_framework.training.masked_trainer import MaskedEdgeTrainer
from sgat_clmda_framework.training.evaluator import Evaluator

seed = int(sys.argv[1]); torch.set_num_threads(3)
cfg = Config(); ds = DatasetLoader(cfg.data_path, 42); nd, nm = ds.nd, ds.nm
OUT = os.path.join(cfg.output_dir, f'robustness_{seed}.json'); res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def hetero(A, SD, SM):
    W = np.zeros((nd + nm, nd + nm)); sd, sm = SD.copy(), SM.copy(); np.fill_diagonal(sd, 0); np.fill_diagonal(sm, 0)
    W[:nd, :nd] = sd; W[nd:, nd:] = sm; W[:nd, nd:] = A; W[nd:, :nd] = A.T; return W
def katz(W, beta=0.01, L=4):
    S = np.zeros_like(W); P = np.eye(len(W))
    for l in range(1, L + 1): P = P @ W; S += beta ** (l - 1) * P
    return S[:nd, nd:]
def cf_scores(A, SM): sm = SM.copy(); np.fill_diagonal(sm, 0); return A @ sm
def pr(S, A):
    m = (A == 0); r = np.zeros_like(S, dtype=float); r[m] = rankdata(S[m]) / m.sum(); return r
def metrics(pd_, pm_, nd_, nm_, S):
    y = np.r_[np.ones(len(pd_)), np.zeros(len(nd_))]; s = np.r_[S[pd_, pm_], S[nd_, nm_]]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))

def train_masked(A, ratio, rs):
    """Masked-edge model trained on the positives of A (similarities/graphs from A only); returns the dense score matrix."""
    SD, SM = SE.compute_fold_similarities(A); X, ei = GraphBuilder.build_hetero_network(A, SD, SM, cfg.sim_threshold)
    ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold); pd_, pm_ = np.where(A == 1)
    rng = np.random.RandomState(rs); ns = rng.choice(len(ds.neg_d_all), len(pd_), replace=False)
    td = torch.tensor(np.r_[pd_, ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[pm_, ds.neg_m_all[ns]]); ty = torch.tensor(np.r_[np.ones(len(pd_)), np.zeros(len(ns))], dtype=torch.float32)
    torch.manual_seed(rs); m = SGAT_CLMDA(nd, nm, nd + nm)
    MaskedEdgeTrainer(m, torch.device('cpu'), max_epochs=120, patience=25, mask_ratio=ratio, seed=rs).fit(A, nd, nm, td, tm, ty)
    m.eval()
    with torch.no_grad(): return m(X, ei, torch.tensor(SD), torch.tensor(SM), ei_d, ei_m)[0].numpy(), SM, SD

for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
    key = f"{seed}|{sp['fold']}"
    if key in res: continue
    A = sp['A_tr']; ev = lambda S: metrics(sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m'], S)
    # ---- inner validation: choose (mask ratio, model weight) using only the training fold ----
    rng = np.random.RandomState(seed * 100 + sp['fold']); pos = np.arange(len(sp['tr_pd'])); rng.shuffle(pos); nv = len(pos) // 5
    vi, ti = pos[:nv], pos[nv:]
    A_in = np.zeros_like(A); A_in[sp['tr_pd'][ti], sp['tr_pm'][ti]] = 1
    cand_d, cand_m = np.where(A == 0); vn = rng.choice(len(cand_d), nv, replace=False)
    vpd, vpm, vnd, vnm = sp['tr_pd'][vi], sp['tr_pm'][vi], cand_d[vn], cand_m[vn]
    best = (-1, None, None)
    for ratio in (0.25, 0.5):
        S_in, SM_in, _ = train_masked(A_in, ratio, seed * 1000 + sp['fold']); cf_in = cf_scores(A_in, SM_in)
        for w in (0.25, 0.5, 0.75):
            auc = metrics(vpd, vpm, vnd, vnm, w * pr(S_in, A_in) + (1 - w) * pr(cf_in, A_in))[0]
            if auc > best[0]: best = (auc, ratio, w)
    _, r_star, w_star = best
    # ---- outer: models trained on the full training fold ----
    SDh, SMh = SE.compute_fold_similarities(A); cf = cf_scores(A, SMh); kz = katz(hetero(A, SDh, SMh))
    S05, _, _ = train_masked(A, 0.5, seed * 1000 + sp['fold']); S_star = S05 if r_star == 0.5 else train_masked(A, r_star, seed * 1000 + sp['fold'])[0]
    R = {'model (masked 0.5)': ev(S05), 'item-CF': ev(cf), 'KATZ-style': ev(kz),
         'cf+katz': ev(.5 * pr(cf, A) + .5 * pr(kz, A)),
         'model+katz': ev(.5 * pr(S05, A) + .5 * pr(kz, A)),
         'model+cf+katz': ev((pr(S05, A) + pr(cf, A) + pr(kz, A)) / 3),
         'model+cf (fixed 0.5/0.5)': ev(.5 * pr(S05, A) + .5 * pr(cf, A)),
         'model+cf (inner-validated)': ev(w_star * pr(S_star, A) + (1 - w_star) * pr(cf, A))}
    res[key] = dict(R, chosen=dict(ratio=r_star, w=w_star, inner_val_auc=best[0])); json.dump(res, open(OUT, 'w'))
    print(key, {k: round(v[0], 3) for k, v in R.items()}, 'chosen', r_star, w_star, flush=True)
