"""Leakage-free hybrid: masked-edge SGAT-CLMDA score blended with item-CF score (both use A_train only).
Blend = weighted average of percentile ranks over candidate pairs (entries not in A_train). Weight 0.5 fixed a priori;
0.25/0.75 reported as sensitivity only. 5 folds x 5 seeds, framework model (as in other honest experiments)."""
import os, json, numpy as np, torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine as SE
from sgat_clmda_framework.data.graph_builder import GraphBuilder
from sgat_clmda_framework.models.sgat_model import SGAT_CLMDA
from sgat_clmda_framework.training.masked_trainer import MaskedEdgeTrainer
from sgat_clmda_framework.training.evaluator import Evaluator
cfg = Config(); ds = DatasetLoader(cfg.data_path, 42); OUT = cfg.output_dir + '/hybrid.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def pr(S, A):
    m = (A == 0); r = np.zeros_like(S, dtype=float); r[m] = rankdata(S[m]) / m.sum(); return r
def ev(sp, S):
    y = np.r_[np.ones(len(sp['te_pd'])), np.zeros(len(sp['te_neg_d']))]
    s = np.r_[S[sp['te_pd'], sp['te_pm']], S[sp['te_neg_d'], sp['te_neg_m']]]; return roc_auc_score(y, s), average_precision_score(y, s)
for seed in [42, 123, 456, 789, 2024]:
    for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
        key = f"{seed}|{sp['fold']}"
        if key in res: continue
        A = sp['A_tr']; SD, SM = SE.compute_fold_similarities(A)
        X, ei = GraphBuilder.build_hetero_network(A, SD, SM, cfg.sim_threshold); ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold)
        rng = np.random.RandomState(seed * 10 + sp['fold']); ns = rng.choice(len(ds.neg_d_all), len(sp['tr_pd']), replace=False)
        td = torch.tensor(np.r_[sp['tr_pd'], ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[sp['tr_pm'], ds.neg_m_all[ns]])
        ty = torch.tensor(np.r_[np.ones(len(sp['tr_pd'])), np.zeros(len(ns))], dtype=torch.float32)
        torch.manual_seed(seed + sp['fold']); model = SGAT_CLMDA(ds.nd, ds.nm, ds.nd + ds.nm)
        MaskedEdgeTrainer(model, torch.device('cpu'), max_epochs=120, patience=25, mask_ratio=0.5, seed=seed + sp['fold']).fit(A, ds.nd, ds.nm, td, tm, ty)
        S_m = Evaluator.evaluate_model(model, X, ei, torch.tensor(SD), torch.tensor(SM), ei_d, ei_m, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m'])['score_matrix']
        sm = SM.copy(); np.fill_diagonal(sm, 0); S_cf = A @ sm
        R_m, R_c = pr(S_m, A), pr(S_cf, A); r = {}
        for name, S in [('model', R_m), ('cf', R_c), ('hybrid_w0.5', .5 * R_m + .5 * R_c), ('hybrid_w0.25', .25 * R_m + .75 * R_c), ('hybrid_w0.75', .75 * R_m + .25 * R_c)]:
            r[name] = ev(sp, S)
        res[key] = r; json.dump(res, open(OUT, 'w'))
        print(key, {k: round(v[0], 3) for k, v in r.items()}, flush=True)
