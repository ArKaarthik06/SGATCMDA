"""Case study with blind validation against Disbiome (downloaded API dump, no author-chosen literature).
Final model trained on all HMDAD associations (no held-out set, so full-A similarity is legitimate here).
Top-10 unobserved microbes per disease, 5 seeds; compared to a popularity baseline and random pairs."""
import json, re, numpy as np, pandas as pd, torch
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine
from sgat_clmda_framework.data.graph_builder import GraphBuilder
from sgat_clmda_framework.models.sgat_model import SGAT_CLMDA
from sgat_clmda_framework.training.masked_trainer import MaskedEdgeTrainer
from sgat_clmda_framework.analysis.graph_proximity import GraphProximityAuditor

cfg = Config(); ds = DatasetLoader(cfg.data_path, 42)
DIS = json.load(open('external_data/disbiome/disbiome_experiments.json', encoding='utf-8'))
MAP = {'Asthma': ['asthma'], 'COPD': ['chronic obstructive pulmonary', 'copd'], 'Type 2 diabetes': ['type 2 diabetes'],
       'Obesity': ['obesity'], 'Inflammatory bowel disease(IBD)': ['inflammatory bowel', "crohn", 'ulcerative colitis']}
def genus(n): return re.sub(r'[^a-z ]', '', n.lower()).split()[0] if n.strip() else ''
disb = {}
for d, keys in MAP.items():
    disb[d] = {genus(x['organism_name']) for x in DIS if any(k in x['disease_name'].lower() for k in keys) and x['organism_name']}
def validated(d, m):
    g = genus(ds.i2m[m]); return g in disb[d] or any(g == x for x in disb[d])

SD, SM = SimilarityEngine.compute_fold_similarities(ds.A)
X, ei = GraphBuilder.build_hetero_network(ds.A, SD, SM, cfg.sim_threshold)
ei_d, ei_m = GraphBuilder.build_sim_networks(SD, SM, cfg.sim_threshold)
FD, FM = torch.tensor(SD), torch.tensor(SM)
S = []
for seed in [42, 123, 456, 789, 2024]:
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    ns = rng.choice(len(ds.neg_d_all), len(ds.pos_d_all), replace=False)
    td = torch.tensor(np.r_[ds.pos_d_all, ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[ds.pos_m_all, ds.neg_m_all[ns]])
    ty = torch.tensor(np.r_[np.ones(len(ds.pos_d_all)), np.zeros(len(ns))], dtype=torch.float32)
    model = SGAT_CLMDA(ds.nd, ds.nm, ds.nd + ds.nm)
    MaskedEdgeTrainer(model, torch.device('cpu'), max_epochs=120, patience=25, mask_ratio=0.5, seed=seed).fit(ds.A, ds.nd, ds.nm, td, tm, ty)
    model.eval()
    with torch.no_grad(): S.append(model(X, ei, FD, FM, ei_d, ei_m)[0].numpy())
    print('seed', seed, 'done', flush=True)
S = np.stack(S); mean, std = S.mean(0), S.std(0)
prox = GraphProximityAuditor(ds); deg = ds.A.sum(0)
rows, summ = [], {}
for d in MAP:
    di = ds.d2i[d]; cand = np.where(ds.A[di] == 0)[0]
    order = cand[np.argsort(-mean[di, cand])]
    top = order[:10]
    # stability: Jaccard of per-seed top-10 sets vs consensus
    jac = [len(set(np.argsort(-np.where(ds.A[di] == 0, S[s, di], -1))[:10]) & set(top)) / 10 for s in range(5)]
    pop = cand[np.argsort(-deg[cand])][:10]            # popularity baseline
    rnd = np.random.RandomState(0).choice(cand, 10, replace=False)
    pool_rate = np.mean([validated(d, m) for m in cand])  # base rate over entire candidate pool
    for r, m in enumerate(top, 1):
        p = prox.analyze_pair_proximity(di, m)
        rows.append(dict(Disease=d, Rank=r, Microbe=ds.i2m[m], MeanScore=float(mean[di, m]), SeedStd=float(std[di, m]),
                         PoolPercentile=float((cand.size - r) / cand.size * 100), ShortestPath=p['shortest_path_length'],
                         MicrobeKnownDiseases=p['target_microbe_known_diseases'], Disbiome_Genus_Match=bool(validated(d, m))))
    summ[d] = dict(disbiome_genera=len(disb[d]), top10_validated=int(sum(validated(d, m) for m in top)),
                   popularity_top10_validated=int(sum(validated(d, m) for m in pop)),
                   random10_validated=int(sum(validated(d, m) for m in rnd)),
                   pool_base_rate=float(pool_rate), top10_seed_stability_jaccard_mean=float(np.mean(jac)),
                   top10_mean_microbe_degree=float(deg[top].mean()), pool_mean_microbe_degree=float(deg[cand].mean()))
pd.DataFrame(rows).to_csv('sgat_clmda_outputs/case_studies_disbiome.csv', index=False)
json.dump(summ, open('sgat_clmda_outputs/case_study_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))
