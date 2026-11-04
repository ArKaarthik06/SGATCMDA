"""Effect of the GATMDA code's evaluation protocol on HMDAD (usage: python run_gatmda_protocol_test.py).
GATMDA's main.py trains a random forest on [train positives + ALL unknown pairs] and tests on [test positives + the SAME unknown pairs].
Here: identical features (similarities from A_train only, no label leak) and identical RF (350 trees), only the negatives differ:
  same-negatives : test negatives = the unknown pairs the forest was trained on as label 0   (GATMDA code protocol)
  held-out       : the unknown pairs are split 5 ways too; test negatives are never seen in training"""
import json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine as SE
cfg = Config(); ds = DatasetLoader(cfg.data_path, 42); nd, nm = ds.nd, ds.nm; out = {}
unk_d, unk_m = np.where(ds.A == 0)
for seed in (42, 123):
    rng = np.random.RandomState(seed); perm = rng.permutation(len(unk_d)); nfold = np.array_split(perm, 5)
    for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
        SD, SM = SE.compute_fold_similarities(sp['A_tr']); f = lambda d, m: np.hstack([SD[d], SM[m]])
        Xp_tr, Xp_te = f(sp['tr_pd'], sp['tr_pm']), f(sp['te_pd'], sp['te_pm'])
        te_idx = nfold[sp['fold']]; tr_idx = np.setdiff1d(perm, te_idx)
        res = {}
        for name, neg_tr, neg_te in [('same-negatives', perm, perm), ('held-out', tr_idx, te_idx)]:
            Xn_tr, Xn_te = f(unk_d[neg_tr], unk_m[neg_tr]), f(unk_d[neg_te], unk_m[neg_te])
            clf = RandomForestClassifier(n_estimators=350, n_jobs=-1, random_state=seed).fit(np.vstack([Xp_tr, Xn_tr]), np.r_[np.ones(len(Xp_tr)), np.zeros(len(Xn_tr))])
            y = np.r_[np.ones(len(Xp_te)), np.zeros(len(Xn_te))]; s = clf.predict_proba(np.vstack([Xp_te, Xn_te]))[:, 1]
            res[name] = (float(roc_auc_score(y, s)), float(average_precision_score(y, s)))
        out[f"{seed}|{sp['fold']}"] = res; print(seed, sp['fold'], {k: round(v[0], 3) for k, v in res.items()}, flush=True)
json.dump(out, open(cfg.output_dir + '/gatmda_protocol_test.json', 'w'))
for k in ('same-negatives', 'held-out'): a = np.array([v[k][0] for v in out.values()]); print(k, 'AUC', a.mean().round(3), '+-', a.std().round(3))
