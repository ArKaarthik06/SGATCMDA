"""Second dataset: Disbiome (human host, genus level). usage: python run_disbiome_study.py <seed>
Association = a disease and a bacterial genus with at least one Elevated or Reduced finding (direction ignored). Genus = first token of the organism name.
Iterative degree filter (done once, before any split, as for curated HMDAD): diseases with >= 10 genera and genera in >= 3 diseases.
Protocols (identical to HMDAD): leaky = similarities from full A; honest = recomputed from A_train; masked = honest + masked-edge training (0.5);
final = masked model rank-blended 0.5/0.5 with training-only item-CF. Same hyper-parameters as on HMDAD (nothing re-tuned for this dataset)."""
import os, sys, re, json, collections, numpy as np, pandas as pd, torch
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

seed = int(sys.argv[1]); torch.set_num_threads(int(os.environ.get('NT', 4))); cfg = Config()
SUBSET = os.path.join(cfg.output_dir, 'disbiome_subset.txt')
if not os.path.exists(SUBSET):
    rows = json.load(open('external_data/disbiome/disbiome_experiments.json', encoding='utf-8'))
    genus = lambda n: (re.sub(r'[^a-z ]', '', n.lower()).split() or [''])[0]
    P = {(r['disease_name'], genus(r['organism_name'])) for r in rows if r['host_type'] == 'Human' and r['organism_name']
         and r['qualitative_outcome'] in ('Elevated', 'Reduced') and genus(r['organism_name']) not in ('', 'unclassified', 'uncultured')}
    for _ in range(10):
        D, M = collections.Counter(p[0] for p in P), collections.Counter(p[1] for p in P); P = {p for p in P if D[p[0]] >= 10 and M[p[1]] >= 3}
    pd.DataFrame(sorted(P), columns=['Disease', 'Microbe']).to_csv(SUBSET, sep='\t', index=False)
ds = DatasetLoader(SUBSET, 42); nd, nm = ds.nd, ds.nm
OUT = os.path.join(cfg.output_dir, f'disbiome_{seed}.json'); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
SDf, SMf = SE.compute_full_similarities(ds.A)
if seed == 42: print('Disbiome subset:', ds.summary(), flush=True)
def pr(S, A):
    m = (A == 0); r = np.zeros_like(S, dtype=float); r[m] = rankdata(S[m]) / m.sum(); return r
def ev(sp, S):
    y = np.r_[np.ones(len(sp['te_pd'])), np.zeros(len(sp['te_neg_d']))]; s = np.r_[S[sp['te_pd'], sp['te_pm']], S[sp['te_neg_d'], sp['te_neg_m']]]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
    key = f"{seed}|{sp['fold']}"
    if key in res: continue
    A = sp['A_tr']; SDh, SMh = SE.compute_fold_similarities(A)
    rng = np.random.RandomState(seed * 10 + sp['fold']); ns = rng.choice(len(ds.neg_d_all), len(sp['tr_pd']), replace=False)
    td = torch.tensor(np.r_[sp['tr_pd'], ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[sp['tr_pm'], ds.neg_m_all[ns]])
    ty = torch.tensor(np.r_[np.ones(len(sp['tr_pd'])), np.zeros(len(ns))], dtype=torch.float32); r = {}
    for name, (SD, SM) in [('leaky', (SDf, SMf)), ('honest', (SDh, SMh))]:
        X, ei = GraphBuilder.build_hetero_network(A, SD, SM, cfg.sim_threshold); ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold)
        g = (X, ei, torch.tensor(SD), torch.tensor(SM), ei_d, ei_m); torch.manual_seed(seed + sp['fold']); m = SGAT_CLMDA(nd, nm, nd + nm)
        Trainer(m, torch.device('cpu'), max_epochs=120, patience=25).fit(*g, td, tm, ty)
        e = Evaluator.evaluate_model(m, *g, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m']); r[name] = (e['auc'], e['aupr'])
    g = (X, ei, torch.tensor(SDh), torch.tensor(SMh), ei_d, ei_m)           # honest graph for evaluation of the masked model
    torch.manual_seed(seed + sp['fold']); m = SGAT_CLMDA(nd, nm, nd + nm)
    MaskedEdgeTrainer(m, torch.device('cpu'), max_epochs=120, patience=25, mask_ratio=0.5, seed=seed + sp['fold']).fit(A, nd, nm, td, tm, ty)
    S_m = Evaluator.evaluate_model(m, *g, sp['te_pd'], sp['te_pm'], sp['te_neg_d'], sp['te_neg_m'])['score_matrix']
    sm = SMh.copy(); np.fill_diagonal(sm, 0); S_cf = A @ sm
    r['masked'] = ev(sp, S_m); r['cf'] = ev(sp, S_cf); r['final'] = ev(sp, .5 * pr(S_m, A) + .5 * pr(S_cf, A))
    res[key] = {k: (v[0], v[1]) for k, v in r.items()}; json.dump(res, open(OUT, 'w'))
    print(key, {k: round(v[0], 3) for k, v in r.items()}, flush=True)
