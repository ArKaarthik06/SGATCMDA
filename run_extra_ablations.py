"""Extra ablations (R3-C2, R4-C3): learnable vs frozen gate threshold tau, and similarity-fusion (GIP only / cosine only / average).
Masked-edge training r=0.5, honest per-fold similarities, 5 folds x 5 seeds."""
import os, json, time, numpy as np, torch
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine as SE
from sgat_clmda_framework.data.graph_builder import GraphBuilder
from sgat_clmda_framework.models.sgat_model import SGAT_CLMDA
from sgat_clmda_framework.training.masked_trainer import MaskedEdgeTrainer
from sgat_clmda_framework.training.evaluator import Evaluator
cfg = Config(); ds = DatasetLoader(cfg.data_path, 42); OUT = cfg.output_dir + '/extra_ablations.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
orig = SE.integrated_similarity.__func__
VAR = {'tau frozen at 0.1': dict(tau=0.1, sim='avg'), 'tau frozen at 0.3': dict(tau=0.3, sim='avg'),
       'similarity: GIP only': dict(tau=None, sim='gip'), 'similarity: cosine only': dict(tau=None, sim='cos')}
def setsim(mode):
    if mode == 'avg': SE.integrated_similarity = classmethod(orig)
    elif mode == 'gip': SE.integrated_similarity = classmethod(lambda cls, p: cls.gip_similarity(p))
    else: SE.integrated_similarity = classmethod(lambda cls, p: cls.cosine_similarity(p))
for seed in [42, 123, 456, 789, 2024]:
    splits = ds.get_kfold_splits(5, shuffle_seed=seed)
    for name, v in VAR.items():
        setsim(v['sim'])
        for sp in splits:
            key = f"{name}|{seed}|{sp['fold']}"
            if key in res: continue
            A = sp['A_tr']; SD, SM = SE.compute_fold_similarities(A)
            X, ei = GraphBuilder.build_hetero_network(A, SD, SM, cfg.sim_threshold); ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold)
            rng = np.random.RandomState(seed * 10 + sp['fold']); ns = rng.choice(len(ds.neg_d_all), len(sp['tr_pd']), replace=False)
            td = torch.tensor(np.r_[sp['tr_pd'], ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[sp['tr_pm'], ds.neg_m_all[ns]])
            ty = torch.tensor(np.r_[np.ones(len(sp['tr_pd'])), np.zeros(len(ns))], dtype=torch.float32)
            torch.manual_seed(seed + sp['fold']); model = SGAT_CLMDA(ds.nd, ds.nm, ds.nd + ds.nm)
            if v['tau'] is not None:
                for n, p in model.named_parameters():
                    if n.endswith('.tau'): p.data.fill_(v['tau']); p.requires_grad = False
            fit = MaskedEdgeTrainer(model, torch.device('cpu'), max_epochs=120, patience=25, mask_ratio=0.5, seed=seed + sp['fold']).fit(A, ds.nd, ds.nm, td, tm, ty)
            ev = Evaluator.evaluate_model(model, X, ei, torch.tensor(SD), torch.tensor(SM), ei_d, ei_m, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m'])
            res[key] = dict(config=name, seed=seed, fold=sp['fold'], auc=ev['auc'], aupr=ev['aupr']); json.dump(res, open(OUT, 'w'))
            print(key, f"AUC={ev['auc']:.4f}", flush=True)
setsim('avg')
