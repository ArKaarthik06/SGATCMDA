"""RKGATMDA (Liu et al., BMC Med Inform Decis Mak 2025; code: github.com/hwlacc/RKGATMDA) evaluated on OUR folds. usage: python run_rkgatmda_ours.py <seed> <variant> <rkgatmda_data_dir> [max_folds]
Port of the released model/training code (RKGATMDA/aggrlayer.py, gatmda.py, utils.py, train.py), run on CPU; the released files are not executed.
Kept as released: GIP from the training matrix, fusion with disease_similarity.txt, 3-head AggrLayer + output AggrLayer, KL + variance regularisers, softmax 2-class decoder,
400 epochs, lr 1e-3, weight decay 1e-2, a new Adam optimiser at every epoch (as in train()), negatives re-sampled every epoch from pairs that are not positives in the FULL matrix,
inference on the complement adjacency (1 for every non-training pair).  Differences: CPU instead of cuda(1); disSim re-read for every fold (the released code mutates it across folds).
variant = released : exactly the released _adj_knn (its inner loop re-uses the loop variable `i`, so each node's attention row is written to row len(idx)-1)
variant = fixed    : the same code with the loop variable renamed, so each row is written to its own node.
Our folds: DatasetLoader.get_kfold_splits(5, shuffle_seed=seed); our test negatives; also reports RKGATMDA's own metric (each test positive ranked against all unobserved pairs)."""
import os, sys, json, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader

seed, variant, DATA = int(sys.argv[1]), sys.argv[2], sys.argv[3].rstrip('/\\') + '/'
MAXF = int(sys.argv[4]) if len(sys.argv) > 4 else 5
torch.set_num_threads(int(os.environ.get('NT', 2))); DEV = torch.device('cpu'); ND, NM = 39, 292
OUT = f'sgat_clmda_outputs/rkgatmda_ours_{variant}_{seed}.json'; res = json.load(open(OUT)) if os.path.exists(OUT) else {}

class AggrLayer(nn.Module):
    def __init__(self, in_feature, out_feature, dropout):
        super().__init__(); self.dropout, self.nd, self.nm = dropout, ND, NM
        self.W = nn.Parameter(torch.empty(size=(in_feature, out_feature))); nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_feature, 1))); nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a_adj_nd = nn.Parameter(torch.empty(size=(ND, 1))); nn.init.normal_(self.a_adj_nd.data, mean=11, std=2.828)
        self.a_adj_nm = nn.Parameter(torch.empty(size=(NM, 1))); nn.init.normal_(self.a_adj_nm.data, mean=11, std=2.828)
        self.linear1 = nn.Linear(out_feature, out_feature); self.linear2 = nn.Linear(2 * out_feature, out_feature); self.bn = nn.BatchNorm1d(out_feature); self.out_feature = out_feature
    def forward(self, adj, h):
        Wh = torch.matmul(h, self.W); e = F.leaky_relu((Wh @ self.a[:self.out_feature]) * (Wh @ self.a[self.out_feature:]).T)
        adj_ = F.dropout(self._adj_to_attention(adj, e), self.dropout, training=self.training)
        h_prime = self.bn(torch.matmul(adj_, Wh))
        return torch.add(F.elu(self.linear1(torch.add(Wh, h_prime))), F.elu(self.linear2(torch.cat((Wh, h_prime), dim=1))))
    def _adj_to_attention(self, adj, e):
        adj_dm, adj_md, e_dm, e_md = adj[:ND, ND:], adj[ND:, :ND], e[:ND, ND:], e[ND:, :ND]
        n_nd, n_nm = self._a_adj(adj[:ND, :], self.a_adj_nd), self._a_adj(adj[ND:, :], self.a_adj_nm)
        neg = lambda x: -1e15 * torch.ones_like(x)
        k_dm = self._adj_knn(torch.where(adj_dm > 0, e_dm, neg(e_dm)), adj_dm, e_dm, n_nd); k_md = self._adj_knn(torch.where(adj_md > 0, e_md, neg(e_md)), adj_md, e_md, n_nm)
        return torch.cat((torch.cat((torch.zeros((ND, ND)), k_dm), dim=1), torch.cat((k_md, torch.zeros((NM, NM))), dim=1)), dim=0)
    def _a_adj(self, adj, adj_a):
        adj_n = torch.sum(adj, dim=1); adj_n[adj_n == 0] = 1
        return torch.mul(1 / torch.pow(adj_n, 3), torch.abs(torch.squeeze(adj_a))).int()
    def _adj_knn(self, adj_e, adj_bool, adj_all_att, adj_num):
        adj_knn = torch.zeros_like(adj_e)
        for i in range(adj_e.shape[0]):
            adj_e_i = adj_e[i, :].clone(); ass_num = adj_bool[i, :]; emp = torch.zeros(adj_all_att.shape[1])
            r = i if variant == 'fixed' else None          # released: the inner loop below re-binds i
            if ass_num.sum() <= adj_num[i]:
                adj_e_i = adj_num[i] * adj_e_i; top = adj_num[i] - (ass_num.sum()).int(); val, idx = torch.topk(adj_all_att[i, :], top)
                for j in range(idx.shape[0]):
                    emp[idx[j]] = val[j]
                    if variant != 'fixed': i = j
                adj_knn[i if variant != 'fixed' else r, :] = torch.add(torch.mul(adj_e_i, ass_num), emp)
            else:
                val, idx = torch.topk(adj_e_i, adj_num[i])
                for j in range(idx.shape[0]):
                    emp[idx[j]] = val[j]
                    if variant != 'fixed': i = j
                adj_knn[i if variant != 'fixed' else r, :] = torch.mul(emp, ass_num)
        return adj_knn

class Gatmda(nn.Module):
    def __init__(self, in_feature, out_feature, hid, dropout, nheads, nd):
        super().__init__(); self.dropout, self.hid, self.nd = dropout, hid, nd
        self.bn = nn.BatchNorm1d(out_feature); self.kl = nn.KLDivLoss(reduction='batchmean')
        self.linear1, self.linear2, self.bn2 = nn.Linear(out_feature, 32), nn.Linear(32, 2), nn.BatchNorm1d(2)
        self.aggregation = [AggrLayer(in_feature, hid, dropout) for _ in range(nheads)]
        for i, a in enumerate(self.aggregation): self.add_module(f'aggr_{i}', a)
        self.out_aggr = AggrLayer(hid, out_feature, dropout)
    def forward(self, adj, x, fea_pca):
        x = F.dropout(x, self.dropout, training=self.training); y = torch.zeros((331, self.hid))
        for a in self.aggregation: y = torch.add(y, a(adj, x))
        logits = self.bn(self.out_aggr(adj, F.dropout(y, self.dropout, training=self.training)))
        kl = self.kl(F.log_softmax(logits, dim=1), fea_pca); loss_var = torch.var(logits[:self.nd]) + torch.var(logits[self.nd:])
        U, V = logits[:self.nd], logits[self.nd:]
        uv = torch.cat([-torch.abs(U[i].unsqueeze(0) - V) for i in range(U.shape[0])], dim=0)          # (39*292, 64), row = d*292+m
        out = self.bn2(F.elu(self.linear2(F.dropout(F.elu(self.linear1(uv)), self.dropout, training=self.training))))
        return out, kl, loss_var

def gip(P):                                   # rows of P = profiles; sigma as calsigma(): 1 / mean squared norm
    sq = (P ** 2).sum(1); lam = 1.0 / np.mean(sq); return np.exp(-lam * np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0))
def build(Atr, dSim0):
    dG, mG = gip(Atr), gip(Atr.T); dS = dSim0.copy(); z = dS == 0; dS[z] = dG[z]; dF = 0.5 * (dG + dS)
    feat = np.zeros((331, 331)); feat[:ND, :ND] = dF; feat[ND:, ND:] = mG
    deg = feat.sum(1); deg[deg == 0] = np.inf; feat = feat / deg[:, None]
    comp = np.ones((ND, NM)) - Atr
    adj = lambda B: np.vstack((np.hstack((np.zeros((ND, ND)), B)), np.hstack((B.T, np.zeros((NM, NM))))))
    return feat, adj(Atr), adj(comp)

ds = DatasetLoader(Config().data_path, 42); dSim0 = np.loadtxt(DATA + 'disease_similarity.txt')
their = {n: i for i, n in enumerate(sorted(ds.diseases, key=str.lower))}; P = np.array([their[n] for n in ds.diseases])     # our disease index -> RKGATMDA index
Afull = np.zeros((ND, NM)); Afull[P[ds.pos_d_all], ds.pos_m_all] = 1
unk_flat = np.where(Afull.reshape(-1) == 0)[0]
t_all = time.time()
for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
    key = f"{seed}|{sp['fold']}"
    if key in res or sp['fold'] >= MAXF: continue
    t0 = time.time(); torch.manual_seed(seed + sp['fold']); random.seed(seed + sp['fold']); rng = np.random.RandomState(seed * 10 + sp['fold'])
    Atr = np.zeros((ND, NM)); Atr[P[sp['tr_pd']], sp['tr_pm']] = 1
    feat, adj_tr, adj_te = build(Atr, dSim0); pca = PCA(n_components=64); fea_pca = F.softmax(torch.tensor(pca.fit_transform(feat), dtype=torch.float32), dim=1)
    fea, adj_tr_t, adj_te_t = (torch.tensor(a, dtype=torch.float32) for a in (feat, adj_tr, adj_te))
    train_mask = torch.tensor(Atr.reshape(-1), dtype=torch.float32); npos = int(Atr.sum())
    model = Gatmda(331, 64, 128, 0.3, 3, ND); sm = nn.Softmax(dim=1)
    for ep in range(400):
        neg = np.zeros(ND * NM); neg[rng.choice(unk_flat, npos, replace=False)] = 1; neg_mask = torch.tensor(neg, dtype=torch.float32)
        model.train(); out, kl, lv = model(adj_tr_t, fea, fea_pca); out = sm(out)
        loss = -(torch.dot(train_mask, torch.log(out[:, 1])) + torch.dot(neg_mask, torch.log(out[:, 0]))) / (2 * npos) + kl + 0.01 * lv
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2); opt.zero_grad(); loss.backward(); opt.step()      # fresh optimiser every epoch, as in train()
    model.eval()
    with torch.no_grad(): S = sm(model(adj_te_t, fea, fea_pca)[0])[:, 1].numpy().reshape(ND, NM)
    te_d, te_m = P[sp['te_pd']], sp['te_pm']; nd_, nm_ = P[sp['te_neg_d']], sp['te_neg_m']
    y = np.r_[np.ones(len(te_d)), np.zeros(len(nd_))]; s = np.r_[S[te_d, te_m], S[nd_, nm_]]
    unk = np.sort(S.reshape(-1)[unk_flat]); pos = S[te_d, te_m]                     # RKGATMDA's metric: rank each test positive among all unobserved pairs
    g_auc = float(np.mean((np.searchsorted(unk, pos, 'left') + 0.5 * (np.searchsorted(unk, pos, 'right') - np.searchsorted(unk, pos, 'left'))) / len(unk)))
    res[key] = dict(auc=float(roc_auc_score(y, s)), aupr=float(average_precision_score(y, s)), global_rank_auc=g_auc, sec=time.time() - t0, final_loss=float(loss))
    json.dump(res, open(OUT, 'w')); print(variant, key, {k: round(v, 4) for k, v in res[key].items()}, flush=True)
