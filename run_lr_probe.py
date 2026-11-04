"""Probe: does the honest-protocol collapse come from over-training? Plain training, per-fold (honest) vs full-matrix (leaky) similarities,
SGAT-CLMDA with the notebook settings (lr 1e-3, 120 ep) vs GCATCMDA's published optimisation (lr 1e-5, 100 ep, wd 1e-3). 5 folds x 3 seeds."""
import json, numpy as np, torch
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine as SE
from sgat_clmda_framework.data.graph_builder import GraphBuilder
from sgat_clmda_framework.models.sgat_model import SGAT_CLMDA
from sgat_clmda_framework.training.trainer import Trainer
from sgat_clmda_framework.training.evaluator import Evaluator
torch.set_num_threads(8); cfg = Config(); ds = DatasetLoader(cfg.data_path, 42); SDf, SMf = SE.compute_full_similarities(ds.A)
CFG = {'notebook (lr 1e-3, 120 ep)': dict(lr=1e-3, weight_decay=1e-4, max_epochs=120, patience=25),
       'GCATCMDA-style optim (lr 1e-5, 100 ep, wd 1e-3)': dict(lr=1e-5, weight_decay=1e-3, max_epochs=100, patience=100)}
R = {(c, p): [] for c in CFG for p in ('leaky', 'honest')}
for seed in [42, 123, 456]:
    for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
        A = sp['A_tr']; SDh, SMh = SE.compute_fold_similarities(A)
        rng = np.random.RandomState(seed * 10 + sp['fold']); ns = rng.choice(len(ds.neg_d_all), len(sp['tr_pd']), replace=False)
        td = torch.tensor(np.r_[sp['tr_pd'], ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[sp['tr_pm'], ds.neg_m_all[ns]])
        ty = torch.tensor(np.r_[np.ones(len(sp['tr_pd'])), np.zeros(len(ns))], dtype=torch.float32)
        for p, (SD, SM) in [('leaky', (SDf, SMf)), ('honest', (SDh, SMh))]:
            X, ei = GraphBuilder.build_hetero_network(A, SD, SM, cfg.sim_threshold); ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold)
            FD, FM = torch.tensor(SD), torch.tensor(SM)
            for c, kw in CFG.items():
                torch.manual_seed(seed + sp['fold']); m = SGAT_CLMDA(ds.nd, ds.nm, ds.nd + ds.nm)
                Trainer(m, torch.device('cpu'), **kw).fit(X, ei, FD, FM, ei_d, ei_m, td, tm, ty)
                e = Evaluator.evaluate_model(m, X, ei, FD, FM, ei_d, ei_m, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m']); R[(c, p)].append((e['auc'], e['aupr']))
        print(seed, sp['fold'], {f'{c[:8]}|{p}': round(v[-1][0], 3) for (c, p), v in R.items()}, flush=True)
out = {f'{c}|{p}': dict(auc=float(np.mean([a for a, _ in v])), sd=float(np.std([a for a, _ in v])), aupr=float(np.mean([b for _, b in v])), n=len(v)) for (c, p), v in R.items()}
json.dump(out, open('sgat_clmda_outputs/lr_probe.json', 'w'), indent=1)
for k, v in out.items(): print(f"{k:70s} AUC={v['auc']:.4f}±{v['sd']:.4f} AUPR={v['aupr']:.4f} n={v['n']}")
