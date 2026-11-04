"""Matched-protocol baselines and ablations (Reviewer 1 #5, Reviewer 3 #2/#4, Reviewer 4 #4).
usage: python run_matched_baselines.py <arch>      arch in {serial, gcn_only, gat_only, viewA_only, parallel}
Every architecture sees identical folds, identical training/test negatives (seed*10+fold) and the same trainer settings as the
SGAT-CLMDA runs (120 epochs, patience 25). Per (seed, fold) it records:
  leaky   : plain training, similarities from the FULL matrix (original protocol)
  honest  : plain training, similarities recomputed from A_train only
  masked  : honest + masked-edge training (mask ratio 0.5, fixed a priori)
  cf      : training-only item-based collaborative filtering (A_tr x S_M_tr), no learning
  hybrid  : 0.5 * rank(masked model) + 0.5 * rank(cf)  (weight fixed a priori)
plus parameter count, ms/epoch and peak resident memory."""
import os, sys, json, time, numpy as np, torch, torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine as SE
from sgat_clmda_framework.data.graph_builder import GraphBuilder
from sgat_clmda_framework.models.sgat_model import SGAT_CLMDA
from sgat_clmda_framework.training.trainer import Trainer
from sgat_clmda_framework.training.masked_trainer import MaskedEdgeTrainer
from sgat_clmda_framework.training.evaluator import Evaluator
try: import psutil
except ImportError: psutil = None

arch = sys.argv[1]; torch.set_num_threads(int(os.environ.get('NT', 3)))
cfg = Config(); ds = DatasetLoader(cfg.data_path, 42)
OUT = os.path.join(cfg.output_dir, f'matched_{arch}{os.environ.get("SUFFIX", "")}.json'); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
SDf, SMf = SE.compute_full_similarities(ds.A)

class ViewAOnly(SGAT_CLMDA):
    """No View B: node embeddings come from the heterogeneous-graph encoder alone (no similarity-graph encoders, no contrastive term)."""
    def __init__(self, nd, nm, feat_dim, **kw):
        super().__init__(nd, nm, feat_dim, **kw)
        od = kw.get('out_dim', 64)
        self.enc_B_d = self.enc_B_m = None
        mk = lambda: nn.Sequential(nn.Linear(od, od), nn.LayerNorm(od), nn.ReLU())
        self.combine_d, self.combine_m = mk(), mk()
    def encode(self, X, ei_h, FD, FM, ei_d, ei_m):
        z = self.enc_A(X, ei_h); zd_A, zm_A = z[:self.nd], z[self.nd:]
        return self.combine_d(zd_A), self.combine_m(zm_A), zd_A, zm_A, zd_A, zm_A
    def get_tau_history(self): return {}

def make():
    n = ds.nd + ds.nm
    if arch == 'serial': return SGAT_CLMDA(ds.nd, ds.nm, n, architecture='serial')
    if arch in ('gcn_only', 'gat_only'): return SGAT_CLMDA(ds.nd, ds.nm, n, encoder_type=arch)
    if arch == 'viewA_only': return ViewAOnly(ds.nd, ds.nm, n)
    return SGAT_CLMDA(ds.nd, ds.nm, n)
CL = 'none' if arch == 'viewA_only' else 'both'
LAM = 0.0 if arch == 'viewA_only' else 0.1

def pr(S, A):
    m = (A == 0); r = np.zeros_like(S, dtype=float); r[m] = rankdata(S[m]) / m.sum(); return r
def ev(sp, S):
    y = np.r_[np.ones(len(sp['te_pd'])), np.zeros(len(sp['te_neg_d']))]
    s = np.r_[S[sp['te_pd'], sp['te_pm']], S[sp['te_neg_d'], sp['te_neg_m']]]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))

for seed in [int(x) for x in os.environ.get('SEEDS', '42,123,456,789,2024').split(',')]:
    for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
        key = f"{arch}|{seed}|{sp['fold']}"
        if key in res: continue
        A = sp['A_tr']; SDh, SMh = SE.compute_fold_similarities(A)
        rng = np.random.RandomState(seed * 10 + sp['fold']); ns = rng.choice(len(ds.neg_d_all), len(sp['tr_pd']), replace=False)
        td = torch.tensor(np.r_[sp['tr_pd'], ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[sp['tr_pm'], ds.neg_m_all[ns]])
        ty = torch.tensor(np.r_[np.ones(len(sp['tr_pd'])), np.zeros(len(ns))], dtype=torch.float32)
        r = {}
        def graphs(SD, SM):
            X, ei = GraphBuilder.build_hetero_network(A, SD, SM, cfg.sim_threshold); ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold)
            return X, ei, torch.tensor(SD), torch.tensor(SM), ei_d, ei_m
        for name, (SD, SM) in [('leaky', (SDf, SMf)), ('honest', (SDh, SMh))]:
            g = graphs(SD, SM); torch.manual_seed(seed + sp['fold']); mod = make()
            t0 = time.perf_counter(); fit = Trainer(mod, torch.device('cpu'), max_epochs=120, patience=25, lam_cl=LAM, cl_mode=CL).fit(*g, td, tm, ty); dt = time.perf_counter() - t0
            e = Evaluator.evaluate_model(mod, *g, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m']); r[name] = (e['auc'], e['aupr'])
            if name == 'honest': r['params'] = sum(p.numel() for p in mod.parameters()); r['ms_per_epoch_plain'] = dt / fit['final_epoch'] * 1000
        g = graphs(SDh, SMh); torch.manual_seed(seed + sp['fold']); mod = make()
        t0 = time.perf_counter()
        fit = MaskedEdgeTrainer(mod, torch.device('cpu'), max_epochs=120, patience=25, mask_ratio=0.5, seed=seed + sp['fold'], lam_cl=LAM, cl_mode=CL).fit(A, ds.nd, ds.nm, td, tm, ty)
        r['ms_per_epoch_masked'] = (time.perf_counter() - t0) / fit['final_epoch'] * 1000
        e = Evaluator.evaluate_model(mod, *g, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m']); r['masked'] = (e['auc'], e['aupr'])
        sm = SMh.copy(); np.fill_diagonal(sm, 0); S_cf = A @ sm
        r['cf'] = ev(sp, S_cf); r['hybrid'] = ev(sp, .5 * pr(e['score_matrix'], A) + .5 * pr(S_cf, A))
        if psutil: r['rss_mb'] = psutil.Process().memory_info().rss / 2**20
        res[key] = r; json.dump(res, open(OUT, 'w'))
        print(key, {k: round(v[0], 3) for k, v in r.items() if isinstance(v, (tuple, list))}, flush=True)
