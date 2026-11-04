import json, numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine as SE
from sgat_clmda_framework.data.graph_builder import GraphBuilder
from sgat_clmda_framework.models.sgat_model import SGAT_CLMDA
from sgat_clmda_framework.training.trainer import Trainer
from sgat_clmda_framework.training.evaluator import Evaluator
cfg = Config(); ds = DatasetLoader(cfg.data_path, 42)
SDf, SMf = SE.compute_full_similarities(ds.A)
R = {k: [] for k in ['model_leaky_features', 'model_honest_features', 'heur_a_raw_A', 'heur_b_full_GIP_projection', 'heur_c_fold_GIP_projection',
                     'cf_leaky_(A_full x S_M_full)', 'cf_honest_(A_tr x S_M_tr)']}
def m(sp, sc):
    y = np.r_[np.ones(len(sp['te_pd'])), np.zeros(len(sp['te_neg_d']))]
    s = np.r_[sc[sp['te_pd'], sp['te_pm']], sc[sp['te_neg_d'], sp['te_neg_m']]]; return roc_auc_score(y, s), average_precision_score(y, s)
for seed in [42, 123, 456, 789, 2024]:
    for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
        A = sp['A_tr']; SDh, SMh = SE.compute_fold_similarities(A)
        R['heur_a_raw_A'].append(m(sp, ds.A))
        R['heur_b_full_GIP_projection'].append(m(sp, SE.gip_similarity(ds.A) @ ds.A))
        R['heur_c_fold_GIP_projection'].append(m(sp, SE.gip_similarity(A) @ A))
        sm = SMf.copy(); np.fill_diagonal(sm, 0); R['cf_leaky_(A_full x S_M_full)'].append(m(sp, ds.A @ sm))
        sm = SMh.copy(); np.fill_diagonal(sm, 0); R['cf_honest_(A_tr x S_M_tr)'].append(m(sp, A @ sm))
        rng = np.random.RandomState(seed * 10 + sp['fold']); ns = rng.choice(len(ds.neg_d_all), len(sp['tr_pd']), replace=False)
        td = torch.tensor(np.r_[sp['tr_pd'], ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[sp['tr_pm'], ds.neg_m_all[ns]])
        ty = torch.tensor(np.r_[np.ones(len(sp['tr_pd'])), np.zeros(len(ns))], dtype=torch.float32)
        for k, (SD, SM) in [('model_leaky_features', (SDf, SMf)), ('model_honest_features', (SDh, SMh))]:
            X, ei = GraphBuilder.build_hetero_network(A, SD, SM, cfg.sim_threshold); ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold)
            FD, FM = torch.tensor(SD), torch.tensor(SM); torch.manual_seed(seed + sp['fold'])
            mod = SGAT_CLMDA(ds.nd, ds.nm, ds.nd + ds.nm); Trainer(mod, torch.device('cpu'), max_epochs=120, patience=25).fit(X, ei, FD, FM, ei_d, ei_m, td, tm, ty)
            e = Evaluator.evaluate_model(mod, X, ei, FD, FM, ei_d, ei_m, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m']); R[k].append((e['auc'], e['aupr']))
        print(seed, sp['fold'], {k: round(v[-1][0], 3) for k, v in R.items()}, flush=True)
out = {k: dict(auc=float(np.mean([a for a, _ in v])), auc_sd=float(np.std([a for a, _ in v])), aupr=float(np.mean([b for _, b in v])), aupr_sd=float(np.std([b for _, b in v])),
               per_run_auc=[float(a) for a, _ in v]) for k, v in R.items()}
json.dump(out, open('sgat_clmda_outputs/leakage_multiseed.json', 'w'), indent=1)
for k, v in out.items(): print(f"{k:34s} AUC={v['auc']:.4f}±{v['auc_sd']:.4f} AUPR={v['aupr']:.4f}")
