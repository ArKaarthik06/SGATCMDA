"""Leakage probe for the RKGATMDA data pipeline (usage: python run_rkgatmda_probe.py <rkgatmda_data_dir>). No repository code is executed; the feature
construction of RKGATMDA/utils.py (gengip, fusionfea_nopro, load_data) is re-implemented here and fed with the shipped data files.
Observation from reading utils.py / train.py: `disSim` is loaded ONCE before the 100 x 5-fold loop and `fusionfea_nopro` writes into it in place
(dSim[y, z] = dGIP[y, z] for every zero entry; 96% of the shipped file is zero). From the second call on, those entries hold the GIP of the FIRST fold's training matrix.
Probe (no learning): score(d, m) = sum_d' dFea[d, d'] * A_train[d', m]  (the disease-side feature as a similarity-weighted vote), test positives vs sampled unknown pairs.
  clean        : dSim re-read from file at every fold (what a per-fold implementation would do)
  as_released  : dSim persists and is mutated across folds and repetitions (what the released code does)
  full_matrix  : dFea built from GIP of the FULL matrix (reference for a complete leak)"""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
D = sys.argv[1].rstrip('/\\') + '/'
lab = np.loadtxt(D + 'label.txt').astype(int); dSim0 = np.loadtxt(D + 'disease_similarity.txt'); nd, nm = 39, 292
A = np.zeros((nd, nm)); A[lab[:, 0] - 1, lab[:, 1] - 1] = 1
def gip_rows(P):                     # rows of P are profiles; sigma as calsigma(): 1 / mean squared norm
    lam = 1.0 / np.mean((P ** 2).sum(1)); sq = (P ** 2).sum(1); return np.exp(-lam * np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0))
def fuse(dGIP, dSim, mutate):
    S = dSim if mutate else dSim.copy(); y, z = np.where(S == 0); S[y, z] = dGIP[y, z]; return 0.5 * (dGIP + S)
rng = np.random.RandomState(0); persistent = dSim0.copy()
pos = np.arange(450); res = {k: [[] for _ in range(5)] for k in ('clean', 'as_released', 'full_matrix')}
for rep in range(40):
    perm = rng.permutation(450); folds = np.array_split(perm, 5)
    for k, te in enumerate(folds):
        tr = np.setdiff1d(pos, te); Atr = np.zeros_like(A); Atr[lab[tr, 0] - 1, lab[tr, 1] - 1] = 1
        te_d, te_m = lab[te, 0] - 1, lab[te, 1] - 1
        cand_d, cand_m = np.where((A == 0)); ni = rng.choice(len(cand_d), len(te), replace=False)
        y = np.r_[np.ones(len(te)), np.zeros(len(te))]; dd, mm = np.r_[te_d, cand_d[ni]], np.r_[te_m, cand_m[ni]]
        g = gip_rows(Atr)                                              # disease GIP from the training matrix (profiles = rows of Atr)
        variants = {'clean': fuse(g, dSim0, False), 'as_released': fuse(g, persistent, True), 'full_matrix': fuse(gip_rows(A), dSim0, False)}
        for name, F in variants.items():
            S = F @ Atr; res[name][k].append(roc_auc_score(y, S[dd, mm]))
print('Probe AUC (no learning), mean over 40 random repetitions; position = fold index inside each repetition')
print('variant        fold1  fold2  fold3  fold4  fold5   mean(folds 2-5)')
for name in res:
    m = [np.mean(res[name][k]) for k in range(5)]; print(f'{name:13s}', '  '.join(f'{x:.3f}' for x in m), f'   {np.mean(m[1:]):.3f}')
print('share of zero entries in the shipped disease_similarity.txt:', round(float((dSim0 == 0).mean()), 3), '| zeros left in persistent matrix after first fold:', int((persistent == 0).sum()))
