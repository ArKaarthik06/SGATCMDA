"""Reviewer comment 5: every method on IDENTICAL splits (5 seeds x 5 folds = 25 runs, same fold partitions, same sampled negatives,
same test negatives, same metrics), under two protocols:
  A "notebook" : similarities from the FULL association matrix (what the original notebook and the compared GNN papers do)
  B "per-fold" : every association-derived similarity recomputed from A_train only (leakage-free)
Neural models come from matched_*.json / gcatcmda_replica_*.json (same partitions and negative draws by construction);
KATZ-style, NTSHMDA-style RWR and item-CF are computed here for BOTH protocols. usage: python run_baseline_table.py"""
import json, numpy as np
from scipy.stats import wilcoxon, rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader
from sgat_clmda_framework.data.similarity import SimilarityEngine as SE
O = 'sgat_clmda_outputs/'; SEEDS = [42, 123, 456, 789, 2024]
ds = DatasetLoader(Config().data_path, 42); nd, nm = ds.nd, ds.nm; SDf, SMf = SE.compute_full_similarities(ds.A)
def sdiv(a, b): return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)
def hetero(A, SD, SM):
    W = np.zeros((nd + nm, nd + nm)); sd, sm = SD.copy(), SM.copy(); np.fill_diagonal(sd, 0); np.fill_diagonal(sm, 0)
    W[:nd, :nd] = sd; W[nd:, nd:] = sm; W[:nd, nd:] = A; W[nd:, :nd] = A.T; return W
def katz(W, beta=0.01, L=4):
    S = np.zeros_like(W); P = np.eye(len(W))
    for l in range(1, L + 1): P = P @ W; S += beta ** (l - 1) * P
    return S[:nd, nd:]
def nts(A, SD, SM, eta=.7, phi=.9, delta=.3, iters=10):          # NTSHMDA-style RWR, published parameters (see run_rwr.py)
    A = A.astype(float); nm_, nd_ = A @ SM, SD @ A; a, b = A * nm_, A * nd_
    Wdm, Wmd = phi * sdiv(a, a.sum(1, keepdims=True)), phi * sdiv(b, b.sum(0, keepdims=True))
    cd = np.where(A.sum(1) > 0, 1 - phi, 1.0)[:, None]; cm = np.where(A.sum(0) > 0, 1 - phi, 1.0)[:, None]
    T = np.zeros((nd + nm, nd + nm)); T[:nd, :nd] = (cd * sdiv(SD, SD.sum(1, keepdims=True))).T; T[nd:, nd:] = (cm * sdiv(SM, SM.sum(1, keepdims=True))).T
    T[nd:, :nd], T[:nd, nd:] = Wdm.T, Wmd; P0 = np.diag(np.r_[np.full(nd, 1 - delta), np.full(nm, delta)]); P = P0.copy()
    for _ in range(iters): P = (1 - eta) * T @ P + eta * P0
    return P[nd:, :nd].T + P[:nd, nd:]
def cf(A, SM): s = SM.copy(); np.fill_diagonal(s, 0); return A @ s
def ev(sp, S):
    y = np.r_[np.ones(len(sp['te_pd'])), np.zeros(len(sp['te_neg_d']))]; s = np.r_[S[sp['te_pd'], sp['te_pm']], S[sp['te_neg_d'], sp['te_neg_m']]]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
C = {k: {'A': [], 'B': []} for k in ('KATZ-style', 'NTSHMDA-style RWR', 'Item-CF')}; keys = []
for seed in SEEDS:
    for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
        keys.append(f"{seed}|{sp['fold']}"); SDh, SMh = SE.compute_fold_similarities(sp['A_tr'])
        for pr_, (SD, SM) in (('A', (SDf, SMf)), ('B', (SDh, SMh))):
            C['KATZ-style'][pr_].append(ev(sp, katz(hetero(sp['A_tr'], SD, SM)))); C['NTSHMDA-style RWR'][pr_].append(ev(sp, nts(sp['A_tr'], SD, SM)))
            C['Item-CF'][pr_].append(ev(sp, cf(sp['A_tr'], SM)))
MT = lambda a: json.load(open(f'{O}matched_{a}.json'))
def neural(a, f, i=0): d = MT(a); return np.array([d[f"{a}|{k}"][f][i] for k in keys])
def replica(name, i='auc'): d = json.load(open(f'{O}gcatcmda_replica_{name}.json')); return np.array([d[k][i] for k in keys])
R = {}   # name -> (A (notebook protocol), B (per-fold)), each (auc, aupr) arrays
def put(name, A=None, B=None): R[name] = (A, B)
two = lambda f: (lambda a: (a, None))
put('SGAT-CLMDA, plain end-to-end (notebook model)', (neural('parallel', 'leaky'), neural('parallel', 'leaky', 1)), (neural('parallel', 'honest'), neural('parallel', 'honest', 1)))
for a, n in (('gcn_only', 'GCN-only encoder (same framework)'), ('gat_only', 'GAT-only encoder (same framework)'), ('serial', 'GCATCMDA-style serial GCN->GAT (same framework)')):
    put(n, (neural(a, 'leaky'), neural(a, 'leaky', 1)), (neural(a, 'honest'), neural(a, 'honest', 1)))
put('GCATCMDA replica, paper settings (lr 1e-5)', (replica('original'), replica('original', 'aupr')), (replica('perfold'), replica('perfold', 'aupr')))
put('GCATCMDA replica, lr 1e-3', (replica('original_lr0.001'), replica('original_lr0.001', 'aupr')), (replica('perfold_lr0.001'), replica('perfold_lr0.001', 'aupr')))
for n, v in C.items(): put(n + ' (re-implemented)', tuple(np.array([x[i] for x in v['A']]) for i in (0, 1)), tuple(np.array([x[i] for x in v['B']]) for i in (0, 1)))
put('SGAT-CLMDA + masked-edge training', None, (neural('parallel', 'masked'), neural('parallel', 'masked', 1)))
put('SGAT-CLMDA + masked + CF blend (final recipe)', None, (neural('parallel', 'hybrid'), neural('parallel', 'hybrid', 1)))
for a, n in (('gcn_only', 'GCN-only + masked + CF blend (same recipe)'), ('gat_only', 'GAT-only + masked + CF blend (same recipe)'), ('serial', 'GCATCMDA-style serial + masked + CF blend (same recipe)')):
    put(n, None, (neural(a, 'hybrid'), neural(a, 'hybrid', 1)))
# keys in the replica files exist for seeds 42..2024 x 5 folds; only those are used
ours = R['SGAT-CLMDA + masked + CF blend (final recipe)'][1][0]
ms = lambda x: f'{np.mean(x):.3f} ± {np.std(x):.3f}'
lines = ['| Method (all on the same 25 splits) | A: notebook protocol AUC / AUPR | B: per-fold recomputed AUC / AUPR | final recipe vs method in B (paired Wilcoxon) |', '|---|---|---|---|']
for n, (A, B) in R.items():
    a = f'{ms(A[0])} / {np.mean(A[1]):.3f}' if A is not None else 'n/a'; b = f'{ms(B[0])} / {np.mean(B[1]):.3f}'
    t = '(reference)' if B[0] is ours else f'{np.mean(ours - B[0]):+.3f}, p={wilcoxon(ours, B[0]).pvalue:.1e}, wins {int((ours > B[0]).sum())}/25'
    lines.append(f'| {n} | {a} | {b} | {t} |')
rw = {}
for s in SEEDS: rw.update(json.load(open(f'{O}rwr_{s}.json')))
r_nts = np.array([rw[k]['rwr'][0] for k in keys]); r_nts_p = np.array([rw[k]['rwr'][1] for k in keys])
lines.append(f'| NTSHMDA-style RWR, from run_rwr.py (cross-check) | n/a | {ms(r_nts)} / {r_nts_p.mean():.3f} | {np.mean(ours - r_nts):+.3f}, p={wilcoxon(ours, r_nts).pvalue:.1e}, wins {int((ours > r_nts).sum())}/25 |')
open('BASELINE_COMPARISON_TABLE.md', 'w', encoding='utf-8').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
json.dump({n: {'A': None if A is None else [A[0].tolist(), A[1].tolist()], 'B': [B[0].tolist(), B[1].tolist()]} for n, (A, B) in R.items()}, open(O + 'baseline_table.json', 'w'))
