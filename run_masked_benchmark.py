"""Honest protocol + masked-edge training (mask ratios fixed a priori: 0.25, 0.5). 5 folds x 5 seeds."""
import os, json, time, numpy as np, torch
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine
from sgat_clmda_framework.data.graph_builder import GraphBuilder
from sgat_clmda_framework.models.sgat_model import SGAT_CLMDA
from sgat_clmda_framework.training.masked_trainer import MaskedEdgeTrainer
from sgat_clmda_framework.training.evaluator import Evaluator
cfg = Config(); ds = DatasetLoader(cfg.data_path, 42)
OUT = os.path.join(cfg.output_dir, 'masked_benchmarks.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
ARCH = {'SGAT-CLMDA parallel': 'parallel', 'Serial GCN->GAT': 'serial'}
for ratio in [0.5, 0.25]:
  for seed in [42, 123, 456, 789, 2024]:
    for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
      A = sp['A_tr']; SD, SM = SimilarityEngine.compute_fold_similarities(A)
      X, ei = GraphBuilder.build_hetero_network(A, SD, SM, cfg.sim_threshold)
      ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold)
      FD, FM = torch.tensor(SD), torch.tensor(SM)
      rng = np.random.RandomState(seed * 10 + sp['fold'])
      ns = rng.choice(len(ds.neg_d_all), len(sp['tr_pd']), replace=False)
      td = torch.tensor(np.r_[sp['tr_pd'], ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[sp['tr_pm'], ds.neg_m_all[ns]])
      ty = torch.tensor(np.r_[np.ones(len(sp['tr_pd'])), np.zeros(len(ns))], dtype=torch.float32)
      for name, arch in ARCH.items():
        key = f"{name}|mask{ratio}|{seed}|{sp['fold']}"
        if key in res: continue
        torch.manual_seed(seed + sp['fold'])
        model = SGAT_CLMDA(ds.nd, ds.nm, ds.nd + ds.nm, architecture=arch)
        tr = MaskedEdgeTrainer(model, torch.device('cpu'), max_epochs=120, patience=25, mask_ratio=ratio, seed=seed + sp['fold'])
        t0 = time.perf_counter(); fit = tr.fit(A, ds.nd, ds.nm, td, tm, ty); dt = time.perf_counter() - t0
        ev = Evaluator.evaluate_model(model, X, ei, FD, FM, ei_d, ei_m, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m'])
        res[key] = dict(config=f'{name} + masked-edge training (r={ratio})', seed=seed, fold=sp['fold'], auc=ev['auc'], aupr=ev['aupr'],
                        epochs=fit['final_epoch'], ms_per_epoch=dt / fit['final_epoch'] * 1000)
        print(key, f"AUC={ev['auc']:.4f} AUPR={ev['aupr']:.4f} {res[key]['ms_per_epoch']:.0f}ms/ep", flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)
