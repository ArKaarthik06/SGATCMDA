"""Replica of GCATCMDA built ONLY from the description in Jiang et al., Front. Microbiol. 2024, 15:1483983 (code is not public).
usage: python run_gcatcmda_replica.py <protocol> [lr]      protocol in {original, perfold}
  original : Gaussian-kernel similarities computed ONCE from the full association matrix (our original protocol)
  perfold  : Gaussian-kernel similarities recomputed from A_train inside every fold (what the paper states it does)
In both, the association-network edges are the TRAINING associations only; test associations never enter the graph.

Specified in the paper (used here): Gaussian kernel with bandwidth 1/mean||IP|| (Eq. 1-4); similarity-network edges where K >= t=0.4 (Eq. 5-6);
association network G=[[0,A],[A^T,0]] with block-diagonal kernel features H0=diag(KM,KD) (Eq. 7-8); linear projection to F=128; per layer GCN then
multi-head GAT (2 heads, averaged); L=3 layers; dual fusion Z=(Z1||..||ZN)Wa+(Z1*..*ZN)Wb (Eq. 12); three encoders (association / microbe-similarity /
disease-similarity networks); GRACE-style contrastive loss with intra- and inter-view negatives (Eq. 13-16), tau=1, lambda=0.2; Z=[Z_A||Z_S];
inner-product decoder + sigmoid; BCE on positives and an equal number of sampled unknown pairs; Adam lr=1e-5, wd=1e-3, 100 epochs.
NOT specified (our choices): activation (ReLU after GCN, ELU after GAT), no dropout, how the similarity-network node features are initialised
(we use the kernel matrix rows), whether layer outputs are chained (we chain), loss = BCE + lambda*contrast, initialisation (PyG defaults)."""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import GCNConv, GATConv
from sgat_clmda_framework.config import Config
from sgat_clmda_framework.data.dataset_loader import DatasetLoader

protocol = sys.argv[1]; LR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
torch.set_num_threads(int(os.environ.get('NT', 4)))
FD, HEADS, LAYERS, TAU, LAM, THR, WD, EPOCHS = 128, 2, 3, 1.0, 0.2, 0.4, 1e-3, 100
cfg = Config(); ds = DatasetLoader(cfg.data_path, 42); nd, nm = ds.nd, ds.nm
tag = '' if LR == 1e-5 else f'_lr{LR:g}'
OUT = os.path.join(cfg.output_dir, f'gcatcmda_replica_{protocol}{tag}.json'); res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def gaussian_kernel(P):                      # rows of P are interaction profiles; Eq. 1-4 with lambda' = 1
    lam = 1.0 / (np.linalg.norm(P, axis=1).mean() + 1e-8)
    sq = (P ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0)
    return np.exp(-lam * d2).astype(np.float32)

def thr_edges(K, t=THR):                     # Eq. 5-6 (self loops are added by the GCN/GAT layers)
    r, c = np.where((K >= t) & (np.eye(len(K)) == 0))
    if len(r) == 0: r = c = np.arange(len(K))
    return torch.tensor(np.stack([r, c]), dtype=torch.long)

class GCAT(nn.Module):                       # Fig. 2 / Eq. 9-12
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, FD, bias=False)
        self.gcn = nn.ModuleList([GCNConv(FD, FD) for _ in range(LAYERS)])
        self.gat = nn.ModuleList([GATConv(FD, FD, heads=HEADS, concat=False) for _ in range(LAYERS)])   # heads averaged
        self.Wa = nn.Linear(LAYERS * FD, FD, bias=False); self.Wb = nn.Linear(FD, FD, bias=False)
    def forward(self, x, ei):
        h, Z = self.proj(x), []
        for g, a in zip(self.gcn, self.gat):
            h = F.elu(a(F.relu(g(h, ei)), ei)); Z.append(h)
        prod = Z[0]
        for z in Z[1:]: prod = prod * z
        return self.Wa(torch.cat(Z, -1)) + self.Wb(prod)

class GCATCMDA(nn.Module):
    def __init__(self):
        super().__init__(); self.enc_A, self.enc_Sd, self.enc_Sm = GCAT(nd + nm), GCAT(nd), GCAT(nm)
    def forward(self, H0, G_ei, KD, ed, KM, em):
        zA = self.enc_A(H0, G_ei); zdA, zmA = zA[:nd], zA[nd:]
        zdS, zmS = self.enc_Sd(KD, ed), self.enc_Sm(KM, em)
        zd, zm = torch.cat([zdA, zdS], -1), torch.cat([zmA, zmS], -1)
        return torch.sigmoid(zd @ zm.T), zdA, zdS, zmA, zmS

def grace(u, v, tau=TAU):                    # Eq. 13-14 (one direction pair): intra- and inter-view negatives
    u, v = F.normalize(u, dim=-1), F.normalize(v, dim=-1); n = u.shape[0]
    def one(a, b):
        inter = torch.exp(a @ b.T / tau); intra = torch.exp(a @ a.T / tau); eye = torch.eye(n, device=a.device, dtype=torch.bool)
        pos = inter.diag(); neg = inter.masked_fill(eye, 0).sum(1) + intra.masked_fill(eye, 0).sum(1)
        return -torch.log(pos / (pos + neg))
    return 0.5 * (one(u, v) + one(v, u)).mean()

def ev(sp, S):
    y = np.r_[np.ones(len(sp['te_pd'])), np.zeros(len(sp['te_neg_d']))]; s = np.r_[S[sp['te_pd'], sp['te_pm']], S[sp['te_neg_d'], sp['te_neg_m']]]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))

SEEDS = [42, 123, 456, 789, 2024, 11, 22, 33, 44, 55]            # paper: 10 repetitions of 5-fold CV
for seed in SEEDS:
    for sp in ds.get_kfold_splits(5, shuffle_seed=seed):
        key = f"{seed}|{sp['fold']}"
        if key in res: continue
        A = sp['A_tr']; P_d, P_m = (ds.A if protocol == 'original' else A), (ds.A.T if protocol == 'original' else A.T)
        KD, KM = gaussian_kernel(P_d), gaussian_kernel(P_m)           # disease (nd,nd), microbe (nm,nm)
        n = nd + nm; H0 = np.zeros((n, n), np.float32); H0[:nd, :nd] = KD; H0[nd:, nd:] = KM            # Eq. 8
        dr, mc = np.where(A == 1); G_ei = torch.tensor(np.stack([np.r_[dr, nd + mc], np.r_[nd + mc, dr]]), dtype=torch.long)   # Eq. 7, train edges only
        rng = np.random.RandomState(seed * 10 + sp['fold']); ns = rng.choice(len(ds.neg_d_all), len(sp['tr_pd']), replace=False)
        td = torch.tensor(np.r_[sp['tr_pd'], ds.neg_d_all[ns]]); tm = torch.tensor(np.r_[sp['tr_pm'], ds.neg_m_all[ns]])
        ty = torch.tensor(np.r_[np.ones(len(sp['tr_pd'])), np.zeros(len(ns))], dtype=torch.float32)
        T = lambda a: torch.tensor(a); H0t, KDt, KMt, ed, em = T(H0), T(KD), T(KM), thr_edges(KD), thr_edges(KM)
        torch.manual_seed(seed + sp['fold']); model = GCATCMDA(); opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        t0 = time.perf_counter(); model.train()
        for ep in range(EPOCHS):
            opt.zero_grad(); S, zdA, zdS, zmA, zmS = model(H0t, G_ei, KDt, ed, KMt, em)
            loss = F.binary_cross_entropy(S[td, tm], ty) + LAM * (grace(zdA, zdS) + grace(zmA, zmS))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): S = model(H0t, G_ei, KDt, ed, KMt, em)[0].numpy()
        a, p = ev(sp, S); res[key] = dict(auc=a, aupr=p, final_loss=float(loss), sec=time.perf_counter() - t0, edges_sim=[int(ed.shape[1]), int(em.shape[1])])
        json.dump(res, open(OUT, 'w')); print(protocol, key, f'AUC={a:.4f} AUPR={p:.4f} loss={loss.item():.3f} sim_edges(d,m)={ed.shape[1]},{em.shape[1]}', flush=True)
au = np.array([v['auc'] for v in res.values()]); ap = np.array([v['aupr'] for v in res.values()])
print(f'{protocol} lr={LR:g}: AUC {au.mean():.4f} ± {au.std():.4f}  AUPR {ap.mean():.4f} ± {ap.std():.4f}  (n={len(au)})')
