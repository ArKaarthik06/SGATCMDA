"""Faithful, memory-light port of the released GATMDA pipeline (ghli16/GATMDA main.py + GAT/gat.py) run on its own miRNA-disease data
(usage: python run_gatmda_port.py <seed> <repo_dir>). Pipeline kept as released: pair feature = [SM@miRNA-lncRNA, SD@disease-lncRNA, 40-d GAT embedding of each],
GAT = 4 heads x 68 -> 40, UNTRAINED, dropout left on (the module is never put in eval mode), adjacency = training miRNA-disease edges only,
random forest 350 trees -> reduced to 100 and negatives subsampled to a fixed pool (the full 188k x 1802 matrix does not fit in memory here).
The released code builds `adj` with np.empty (uninitialised); zeros are used here.
Configurations (5-fold over positives; test positives are removed from the GAT graph in all of them, as in the released code):
  released        : similarity files as shipped, test negatives = the pool the forest was trained on (as in main.py)
  heldout_negs    : same similarity files, test negatives never used in training
  cos_full+heldout: miRNA similarity = cosine of the FULL association matrix (label-derived, not recomputed), held-out negatives
  cos_fold+heldout: miRNA similarity = cosine of the TRAINING matrix only (recomputed per fold), held-out negatives
The last two differ ONLY in whether the label-derived similarity sees the test associations."""
import os, sys, json, numpy as np, torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
seed = int(sys.argv[1]); repo = sys.argv[2]; NNEG, NTREE = 30000, 100
sys.path.insert(0, repo); from GAT import GAT_layer
torch.set_num_threads(8); d2 = os.path.join(repo, 'data2')
R = np.loadtxt(f'{d2}/midis.csv', delimiter=','); SMf = np.loadtxt(f'{d2}/miRNA-miRNA.csv', delimiter=','); SDf = np.loadtxt(f'{d2}/disease-disease.csv', delimiter=',')
LD = np.loadtxt(f'{d2}/lncdis.csv', delimiter=',').T; ML = np.loadtxt(f'{d2}/milnc.csv', delimiter=',')
nmi, ndi = R.shape; rng = np.random.RandomState(seed)
def cosine(P):
    n = np.linalg.norm(P, axis=1, keepdims=True); Q = np.divide(P, n, out=np.zeros_like(P), where=n > 0); S = Q @ Q.T; np.fill_diagonal(S, 1.0); return S
def embed(SM, SD, Atr):
    gat_feature = torch.tensor(np.vstack([SM @ ML, SD @ LD]), dtype=torch.float32)
    net = np.zeros((nmi + ndi, nmi + ndi)); net[:nmi, nmi:] = Atr; net = np.maximum(net, net.T)
    torch.manual_seed(seed); model = GAT_layer.GAT(861, 68, 40, 0.2, 0.2, 4)       # train mode (dropout on), never trained - as released
    with torch.no_grad(): E = model(gat_feature, torch.tensor(net, dtype=torch.float32)).numpy()
    return E[:nmi], E[nmi:]
def feats(SM, SD, Em, Ed, mi, di): return np.hstack([(SM @ ML)[mi], (SD @ LD)[di], Em[mi], Ed[di]]).astype(np.float32)
def evaluate(Xtr, ytr, Xte, yte):
    clf = RandomForestClassifier(n_estimators=NTREE, n_jobs=-1, random_state=seed).fit(Xtr, ytr); s = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, s)), float(average_precision_score(yte, s))

pm, pd_ = np.where(R == 1); order = rng.permutation(len(pm)); pfold = np.array_split(order, 5)
zm, zd = np.where(R == 0); pool = rng.choice(len(zm), NNEG, replace=False); nfold = np.array_split(pool, 5)
OUT = f'sgat_clmda_outputs/gatmda_port_{seed}.json'; res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for k in range(5):
    if str(k) in res: continue
    te = pfold[k]; tr = np.setdiff1d(order, te); Atr = np.zeros_like(R); Atr[pm[tr], pd_[tr]] = 1
    cfgs = {'released': (SMf, None, 'shared'), 'heldout_negs': (SMf, None, 'held'), 'cos_full+heldout': (cosine(R), None, 'held'), 'cos_fold+heldout': (cosine(Atr), None, 'held')}
    r = {}
    for name, (SM, _, mode) in cfgs.items():
        Em, Ed = embed(SM, SDf, Atr); Xp_tr, Xp_te = feats(SM, SDf, Em, Ed, pm[tr], pd_[tr]), feats(SM, SDf, Em, Ed, pm[te], pd_[te])
        if mode == 'shared': ntr = nte = pool
        else: nte = nfold[k]; ntr = np.setdiff1d(pool, nte)
        Xn_tr, Xn_te = feats(SM, SDf, Em, Ed, zm[ntr], zd[ntr]), feats(SM, SDf, Em, Ed, zm[nte], zd[nte])
        r[name] = evaluate(np.vstack([Xp_tr, Xn_tr]), np.r_[np.ones(len(Xp_tr)), np.zeros(len(Xn_tr))], np.vstack([Xp_te, Xn_te]), np.r_[np.ones(len(Xp_te)), np.zeros(len(Xn_te))])
    res[str(k)] = r; json.dump(res, open(OUT, 'w')); print(seed, k, {a: round(b[0], 3) for a, b in r.items()}, flush=True)
