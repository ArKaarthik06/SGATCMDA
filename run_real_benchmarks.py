"""
Unified, honest benchmark harness (5 folds x 5 seeds, per-fold similarities from A_train only).
Every number written to sgat_clmda_outputs/real_benchmarks.json is produced here; nothing is typed by hand.
Resumable: finished (config, seed, fold) runs are cached in the JSON.
"""
import os, sys, json, time
import numpy as np
import torch
import psutil

from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine
from sgat_clmda_framework.data.graph_builder import GraphBuilder
from sgat_clmda_framework.models.sgat_model import SGAT_CLMDA
from sgat_clmda_framework.training.trainer import Trainer
from sgat_clmda_framework.training.evaluator import Evaluator

CONFIGS = {
    'SGAT-CLMDA (parallel, ANR)':   dict(architecture='parallel', encoder_type='full', edge_gating_mode='node_norm', cl_mode='both'),
    'Parallel + DEG (edge gate)':   dict(architecture='parallel', encoder_type='full', edge_gating_mode='differentiable_edge', cl_mode='both'),
    'Serial GCN->GAT (GCATCMDA-style)': dict(architecture='serial', encoder_type='full', edge_gating_mode='node_norm', cl_mode='both'),
    'GCN-only':                     dict(architecture='parallel', encoder_type='gcn_only', edge_gating_mode='node_norm', cl_mode='both'),
    'GAT-only':                     dict(architecture='parallel', encoder_type='gat_only', edge_gating_mode='node_norm', cl_mode='both'),
    'CL: none (BCE only)':          dict(architecture='parallel', encoder_type='full', edge_gating_mode='node_norm', cl_mode='none'),
    'CL: disease view only':        dict(architecture='parallel', encoder_type='full', edge_gating_mode='node_norm', cl_mode='disease_only'),
    'CL: microbe view only':        dict(architecture='parallel', encoder_type='full', edge_gating_mode='node_norm', cl_mode='microbe_only'),
}
SEEDS = [42, 123, 456, 789, 2024]
EPOCHS, PATIENCE = 120, 25
OUT = os.path.join(Config().output_dir, 'real_benchmarks.json')


def add_noise(ei, n, ratio, seed):
    rng = np.random.RandomState(seed)
    k = int(ei.shape[1] * ratio)
    noise = torch.tensor(np.stack([rng.randint(0, n, k), rng.randint(0, n, k)]), dtype=torch.long)
    return torch.cat([ei, noise], 1)


def main():
    dev = torch.device('cpu')
    cfg = Config()
    ds = DatasetLoader(cfg.data_path, seed=42)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    proc = psutil.Process()
    for seed in SEEDS:
        splits = ds.get_kfold_splits(5, shuffle_seed=seed)
        for sp in splits:
            A_tr = sp['A_tr']
            SD, SM = SimilarityEngine.compute_fold_similarities(A_tr)
            X, ei_h = GraphBuilder.build_hetero_network(A_tr, SD, SM, cfg.sim_threshold)
            ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold)
            FD, FM = torch.tensor(SD), torch.tensor(SM)
            rng = np.random.RandomState(seed * 10 + sp['fold'])
            ns = rng.choice(len(ds.neg_d_all), size=len(sp['tr_pd']), replace=False)
            td = torch.tensor(np.concatenate([sp['tr_pd'], ds.neg_d_all[ns]]), dtype=torch.long)
            tm = torch.tensor(np.concatenate([sp['tr_pm'], ds.neg_m_all[ns]]), dtype=torch.long)
            ty = torch.tensor(np.r_[np.ones(len(sp['tr_pd'])), np.zeros(len(sp['tr_pd']))], dtype=torch.float32)
            ei_noisy = add_noise(ei_h, ds.nd + ds.nm, 0.15, seed + sp['fold'])
            for name, c in CONFIGS.items():
                key = f"{name}|{seed}|{sp['fold']}"
                if key in res:
                    continue
                torch.manual_seed(seed + sp['fold']); np.random.seed(seed + sp['fold'])
                model = SGAT_CLMDA(ds.nd, ds.nm, ds.nd + ds.nm, architecture=c['architecture'],
                                   encoder_type=c['encoder_type'], edge_gating_mode=c['edge_gating_mode'])
                nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
                tr = Trainer(model, dev, max_epochs=EPOCHS, patience=PATIENCE, cl_mode=c['cl_mode'])
                rss0 = proc.memory_info().rss
                t0 = time.perf_counter()
                fit = tr.fit(X, ei_h, FD, FM, ei_d, ei_m, td, tm, ty)
                dt = time.perf_counter() - t0
                rss1 = proc.memory_info().rss
                ev = Evaluator.evaluate_model(model, X, ei_h, FD, FM, ei_d, ei_m, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m'])
                evn = Evaluator.evaluate_model(model, X, ei_noisy, FD, FM, ei_d, ei_m, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m'])
                res[key] = dict(config=name, seed=seed, fold=sp['fold'], auc=ev['auc'], aupr=ev['aupr'],
                                auc_noisy=evn['auc'], params=nparam, epochs=fit['final_epoch'],
                                ms_per_epoch=dt / fit['final_epoch'] * 1000, rss_mb=rss1 / 2**20, rss_delta_mb=(rss1 - rss0) / 2**20,
                                sparsity=float(model.get_edge_sparsity(0.5)) if c['edge_gating_mode'] == 'differentiable_edge' else None)
                print(key, f"AUC={ev['auc']:.4f} AUPR={ev['aupr']:.4f} ep={fit['final_epoch']} {res[key]['ms_per_epoch']:.0f}ms/ep", flush=True)
                json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
