import json, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine
ds = DatasetLoader(Config().data_path, 42); out = {}
def ev(score, sp):
    y = np.r_[np.ones(len(sp['te_pd'])), np.zeros(len(sp['te_neg_d']))]
    s = np.r_[score[sp['te_pd'], sp['te_pm']], score[sp['te_neg_d'], sp['te_neg_m']]]
    return roc_auc_score(y, s), average_precision_score(y, s)
res = {'Microbe popularity (train degree)': [], 'Disease-similarity CF (S_D x A_train, off-diagonal)': [], 'Microbe-similarity CF (A_train x S_M, off-diagonal)': [], 'Random': []}
for seed in [42, 123, 456, 789, 2024]:
    for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
        A = sp['A_tr']; SD, SM = SimilarityEngine.compute_fold_similarities(A)
        pop = np.tile(A.sum(0), (ds.nd, 1))
        sd = SD.copy(); np.fill_diagonal(sd, 0); sm = SM.copy(); np.fill_diagonal(sm, 0)
        rnd = np.random.RandomState(seed + sp['fold']).rand(*A.shape)
        for k, sc in zip(res, [pop + 1e-6 * rnd, sd @ A + 1e-6 * rnd, A @ sm + 1e-6 * rnd, rnd]):
            res[k].append(ev(sc, sp))
for k, v in res.items():
    v = np.array(v); out[k] = dict(auc=v[:, 0].mean(), auc_sd=v[:, 0].std(), aupr=v[:, 1].mean(), aupr_sd=v[:, 1].std(), n=len(v))
    print(f"{k:55s} AUC={v[:,0].mean():.4f}±{v[:,0].std():.4f} AUPR={v[:,1].mean():.4f}")
json.dump(out, open('sgat_clmda_outputs/trivial_baselines.json', 'w'), indent=1)
