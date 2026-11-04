"""Builds SGAT_CLMDA_Leakage_Audit_Colab.ipynb (Reviewer 1 response experiments)."""
import json, os

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md(r'''
# SGAT-CLMDA: Leakage audit (Reviewer 1 response)

**Question answered:** if the *same* model is trained on identical folds, negatives and seeds, how much of the reported
performance depends on computing GIP/cosine similarities from the **full** association matrix (leaky) versus recomputing
them from the **training-fold associations only** (leakage-free)?

**Protocols** (only the similarity inputs differ; everything else is identical):
| Protocol | SD, SM used for node features, disease/microbe similarity graphs and View B |
|---|---|
| `leaky`  | computed once from the full matrix `A` (original notebook) |
| `honest` | recomputed inside every fold from `A_train` (test associations zeroed) |

**Models** (all trained and evaluated under both protocols):
`SGAT-CLMDA` (original code, unchanged), `GCN`, `GAT`, `GCATCMDA-style` (serial GCN -> GAT inside our framework) and a `GCATCMDA-replica` built only from the published description (3 encoders, Gaussian kernel t=0.4, F=128, L=3, 2 heads, GRACE loss, tau=1, lambda=0.2, 100 epochs; run with the paper's lr 1e-5 and with lr 1e-3). Neither GCATCMDA variant is the authors' code (not public).
Two no-learning collaborative-filtering references are also included: `CF-reference` (item-based, `A_train @ S_M`, the partner of the final rank blend) and `CF-user+item` (a stronger normalised user+item variant, reported separately and **not** used in the blend).

**Seeds:** 10 pre-specified seeds x 5 folds = 50 runs per cell; every seed is reported. A second table uses only the first five seeds (fixed in advance). Seeds are never selected on results.

**Upload:** only `HMDAD.txt` (tab-separated: Disease, Microbe, Position, Evidence, PMID). Use *Runtime > Change runtime type > GPU*.
''')

code(r'''
!pip install torch-geometric scipy -q
import torch; print(torch.__version__, torch.cuda.is_available())
''')

code(r'''
import os, json, time, random, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon, rankdata
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else '')

# ---- experiment configuration -------------------------------------------------
QUICK_TEST = False                  # True: 1 seed, 2 folds, 40 epochs (smoke test)
SEEDS      = [42, 123, 456, 789, 2024, 11, 22, 33, 44, 55]   # 10 pre-specified seeds, ALL reported (no seed selection)
PRESPECIFIED_5 = SEEDS[:5]       # five seeds fixed before any result was seen (used only for the 5-seed table)
K_FOLDS    = 5
MODELS     = ['SGAT-CLMDA', 'GCATCMDA-style', 'GCN', 'GAT']
PROTOCOLS  = ['leaky', 'honest', 'honest+masked']   # + 'honest+masked+CF' (blend) is derived automatically
MASK_RATIO = 0.5                    # fraction of TRAIN associations hidden each epoch (fixed a priori)
BLEND_W    = 0.5                    # weight of the model in the rank blend with item-CF (fixed a priori)
# ---- hyper-parameters: identical to the original SGAT-CLMDA notebook ---------------
EPOCHS, LR, HIDDEN, OUT_DIM = 400, 1e-3, 256, 64
N_LAYERS, HEADS, DROPOUT    = 2, 4, 0.5
THRESHOLD, LAM_CL, TAU      = 0.3, 0.1, 0.5
PATIENCE = 50
if QUICK_TEST:
    SEEDS, EPOCHS, PATIENCE = [42], 40, 20
RESULT_FILE = 'leakage_audit_results.json'
''')

code(r'''
try:
    from google.colab import files
    if not os.path.exists('HMDAD.txt'):
        print('Upload HMDAD.txt ...'); up = files.upload(); DATA_PATH = list(up.keys())[0]
    else: DATA_PATH = 'HMDAD.txt'
except Exception:
    DATA_PATH = 'HMDAD.txt'

df = pd.read_csv(DATA_PATH, sep='\t').drop_duplicates(subset=['Disease', 'Microbe'])
diseases, microbes = sorted(df.Disease.unique()), sorted(df.Microbe.unique())
d2i, m2i = {d: i for i, d in enumerate(diseases)}, {m: i for i, m in enumerate(microbes)}
nd, nm = len(diseases), len(microbes)
A = np.zeros((nd, nm), dtype=np.float32)
for d, m in zip(df.Disease, df.Microbe): A[d2i[d], m2i[m]] = 1.0
print(f'Diseases {nd} | Microbes {nm} | Associations {int(A.sum())} | Density {A.mean()*100:.2f}%')
pos_d_all, pos_m_all = np.where(A == 1)
neg_d_all, neg_m_all = np.where(A == 0)
FEAT_DIM = nd + nm
''')

code(r'''
# ---- similarity + graph construction (same definitions as the original notebook; vectorised) ----
def gip_similarity(P):
    sq = (P ** 2).sum(1)
    lam = 1.0 / (sq.mean() + 1e-8)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0)
    return np.exp(-lam * d2).astype(np.float32)

def cosine_similarity(P):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    return np.clip(Pn @ Pn.T, 0, 1).astype(np.float32)

def integrated_similarity(P):
    return (gip_similarity(P) + cosine_similarity(P)) / 2.0

def knn_edges(S, k=10, floor=0.05):
    S = S.copy(); np.fill_diagonal(S, -1)
    top = np.argsort(S, axis=1)[:, -k:]
    src = np.repeat(np.arange(S.shape[0]), k); dst = top.ravel()
    keep = S[src, dst] >= floor
    return src[keep], dst[keep]

def build_hetero_network(A_mat, SD, SM, thr=THRESHOLD):
    n = nd + nm
    Y = np.zeros((n, n), dtype=np.float32)
    Y[:nd, :nd], Y[:nd, nd:], Y[nd:, :nd], Y[nd:, nd:] = SD, A_mat, A_mat.T, SM
    dr, mc = np.where(A_mat == 1)
    r, c = np.where((SD >= thr) & (np.eye(nd) == 0))
    ms, md_ = knn_edges(SM)
    src = np.concatenate([dr, nd + mc, r, nd + ms]); dst = np.concatenate([nd + mc, dr, c, nd + md_])
    return torch.tensor(Y), torch.tensor(np.stack([src, dst]), dtype=torch.long)

def build_sim_networks(SD, SM, thr=THRESHOLD):
    r, c = np.where((SD >= thr) & (np.eye(nd) == 0))
    if len(r) == 0: r = c = np.arange(nd)
    ms, md_ = knn_edges(SM)
    if len(ms) == 0: ms = md_ = np.arange(nm)
    return (torch.tensor(np.stack([r, c]), dtype=torch.long), torch.tensor(np.stack([ms, md_]), dtype=torch.long))

SD_FULL, SM_FULL = integrated_similarity(A), integrated_similarity(A.T)   # leaky protocol: computed once
print('Full-matrix similarities ready:', SD_FULL.shape, SM_FULL.shape)
''')

code(r'''
# ---- models -------------------------------------------------------------------------------
class SparseAdaptiveGAT(nn.Module):             # verbatim from the original notebook
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.4, gate=True):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, concat=True, dropout=dropout, add_self_loops=True)
        self.bn, self.drop, self.use_gate = nn.LayerNorm(out_dim), dropout, gate
        self.tau = nn.Parameter(torch.tensor(0.1))
    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        if self.use_gate:
            h = h * torch.sigmoid(h.norm(dim=-1, keepdim=True) - torch.abs(self.tau))
        return F.dropout(F.elu(self.bn(h)), p=self.drop, training=self.training)

class CrossLevelEncoder(nn.Module):             # SGAT-CLMDA parallel GCN || SAGA encoder (original)
    def __init__(self, in_dim, hidden, out_dim, n_layers=2, heads=4, dropout=0.4):
        super().__init__()
        self.n_layers, self.out_dim, self.drop = n_layers, out_dim, dropout
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2))
        self.gcn_layers, self.gat_layers = nn.ModuleList(), nn.ModuleList()
        self.bn_layers, self.proj_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(n_layers):
            d_out = out_dim if i == n_layers - 1 else hidden
            self.gcn_layers.append(GCNConv(hidden, d_out))
            self.gat_layers.append(SparseAdaptiveGAT(hidden, d_out, heads=heads, dropout=dropout))
            self.bn_layers.append(nn.LayerNorm(d_out))
            self.proj_layers.append(nn.Linear(d_out, out_dim))
        self.level_w = nn.Parameter(torch.ones(n_layers) / n_layers)
    def forward(self, x, edge_index):
        h, w, out = self.proj(x), F.softmax(self.level_w, 0), []
        for i in range(self.n_layers):
            h_gcn = F.dropout(F.relu(self.bn_layers[i](self.gcn_layers[i](h, edge_index))), p=self.drop, training=self.training)
            h_gat = self.gat_layers[i](h, edge_index)
            out.append(w[i] * F.elu(self.proj_layers[i](h_gcn) + self.proj_layers[i](h_gat)))
            h = h_gcn + h_gat + h if h_gcn.shape[-1] == h.shape[-1] else h_gcn + h_gat
        return sum(out)

class SerialEncoder(nn.Module):                 # GCATCMDA-style: GCN layer(s) followed by GAT layer(s), no gate
    def __init__(self, in_dim, hidden, out_dim, n_layers=2, heads=4, dropout=0.4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2))
        self.gcn = GCNConv(hidden, hidden); self.bn = nn.LayerNorm(hidden); self.drop = dropout
        self.gat = SparseAdaptiveGAT(hidden, out_dim, heads=heads, dropout=dropout, gate=False)
    def forward(self, x, ei):
        h = self.proj(x)
        h = F.dropout(F.relu(self.bn(self.gcn(h, ei))), p=self.drop, training=self.training)
        return self.gat(h, ei)

class PlainGNN(nn.Module):                      # single-view GCN / GAT baseline on the heterogeneous graph
    def __init__(self, kind, in_dim, hidden, out_dim, heads=4, dropout=0.5):
        super().__init__()
        self.drop = dropout
        if kind == 'GCN': self.c1, self.c2 = GCNConv(in_dim, hidden), GCNConv(hidden, out_dim)
        else: self.c1, self.c2 = GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout), GATConv(hidden, out_dim, heads=1, dropout=dropout)
    def forward(self, x, ei):
        h = F.dropout(F.elu(self.c1(x, ei)), p=self.drop, training=self.training)
        return self.c2(h, ei)

class DualViewModel(nn.Module):                 # View A (hetero graph) + View B (similarity graphs) + NT-Xent
    uses_cl = True
    def __init__(self, arch):                   # arch: 'parallel' (SGAT-CLMDA) | 'serial' (GCATCMDA-style)
        super().__init__()
        Enc = CrossLevelEncoder if arch == 'parallel' else SerialEncoder
        self.enc_A = Enc(FEAT_DIM, HIDDEN, OUT_DIM, N_LAYERS, HEADS, DROPOUT)
        self.enc_B_d = Enc(nd, HIDDEN, OUT_DIM, N_LAYERS, HEADS, DROPOUT)
        self.enc_B_m = Enc(nm, HIDDEN, OUT_DIM, N_LAYERS, HEADS, DROPOUT)
        self.combine_d = nn.Sequential(nn.Linear(OUT_DIM * 2, OUT_DIM), nn.LayerNorm(OUT_DIM), nn.ReLU())
        self.combine_m = nn.Sequential(nn.Linear(OUT_DIM * 2, OUT_DIM), nn.LayerNorm(OUT_DIM), nn.ReLU())
    def forward(self, X, ei, FD, FM, ei_d, ei_m):
        zA = self.enc_A(X, ei); zd_A, zm_A = zA[:nd], zA[nd:]
        zd_B, zm_B = self.enc_B_d(FD, ei_d), self.enc_B_m(FM, ei_m)
        zd = self.combine_d(torch.cat([zd_A, zd_B], -1)); zm = self.combine_m(torch.cat([zm_A, zm_B], -1))
        return torch.sigmoid(zd @ zm.T), zd_A, zm_A, zd_B, zm_B

class SingleViewModel(nn.Module):
    uses_cl = False
    def __init__(self, kind):
        super().__init__(); self.net = PlainGNN(kind, FEAT_DIM, HIDDEN, OUT_DIM, HEADS, DROPOUT)
    def forward(self, X, ei, FD, FM, ei_d, ei_m):
        z = self.net(X, ei); return torch.sigmoid(z[:nd] @ z[nd:].T), None, None, None, None

def make_model(name):
    return {'SGAT-CLMDA': lambda: DualViewModel('parallel'), 'GCATCMDA-style': lambda: DualViewModel('serial'),
            'GCN': lambda: SingleViewModel('GCN'), 'GAT': lambda: SingleViewModel('GAT')}[name]().to(DEVICE)

def bce_loss(pred, y):                           # identical to the original
    pw = ((1 - y).sum().clamp(min=1) / y.sum().clamp(min=1)).clamp(max=15.0)
    return F.binary_cross_entropy(pred, y, weight=torch.where(y == 1, pw * torch.ones_like(y), torch.ones_like(y)))

def nt_xent(z1, z2, tau=TAU):
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / tau; lab = torch.arange(z1.shape[0], device=z1.device)
    return (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
print('Models defined:', MODELS)
''')

code(r'''
# ---- one experiment = (model, protocol, seed, fold) ------------------------------------------
def make_fold(seed, fold, tr_idx, te_idx):
    tr_pd, tr_pm, te_pd, te_pm = pos_d_all[tr_idx], pos_m_all[tr_idx], pos_d_all[te_idx], pos_m_all[te_idx]
    A_tr = np.zeros((nd, nm), np.float32); A_tr[tr_pd, tr_pm] = 1
    rng = np.random.RandomState(seed * 1000 + fold)                       # same negatives for every model/protocol
    ns = rng.choice(len(neg_d_all), len(tr_pd), replace=False)
    A_te = np.zeros((nd, nm), np.float32); A_te[te_pd, te_pm] = 1
    ntd, ntm = np.where((A == 0) & (A_te == 0))
    ni = np.random.RandomState(seed * 1000 + fold + 100).choice(len(ntd), len(te_pd), replace=False)
    return dict(A_tr=A_tr, tr=(np.r_[tr_pd, neg_d_all[ns]], np.r_[tr_pm, neg_m_all[ns]], np.r_[np.ones(len(tr_pd)), np.zeros(len(ns))]),
                te_pos=(te_pd, te_pm), te_neg=(ntd[ni], ntm[ni]))

def fold_inputs(f, protocol):
    if protocol == 'leaky': SD, SM = SD_FULL, SM_FULL                       # uses test associations
    else: SD, SM = integrated_similarity(f['A_tr']), integrated_similarity(f['A_tr'].T)   # train-only
    X, ei = build_hetero_network(f['A_tr'], SD, SM)                         # A block is always train-only
    ei_d, ei_m = build_sim_networks(SD, SM)
    t = lambda a: torch.tensor(a, dtype=torch.float32).to(DEVICE)
    return X.to(DEVICE), ei.to(DEVICE), t(SD), t(SM), ei_d.to(DEVICE), ei_m.to(DEVICE), SD, SM

def evaluate_scores(sc, f):
    s = np.r_[sc[f['te_pos']], sc[f['te_neg']]]; y = np.r_[np.ones(len(f['te_pos'][0])), np.zeros(len(f['te_neg'][0]))]
    return roc_auc_score(y, s), average_precision_score(y, s)

def to_dev(SD, SM, A_msg):                       # graph inputs built from whatever associations are visible to message passing
    X, ei = build_hetero_network(A_msg, SD, SM); ei_d, ei_m = build_sim_networks(SD, SM)
    t = lambda a: torch.tensor(a, dtype=torch.float32).to(DEVICE)
    return X.to(DEVICE), ei.to(DEVICE), t(SD), t(SM), ei_d.to(DEVICE), ei_m.to(DEVICE)

def run_one(name, protocol, seed, fold, f, inp):
    torch.manual_seed(seed * 100 + fold); np.random.seed(seed * 100 + fold)
    masked = protocol == 'honest+masked'
    mrng = np.random.RandomState(seed * 100 + fold)
    pd_, pm_ = np.where(f['A_tr'] == 1)
    X, ei, FD, FM, ei_d, ei_m = inp[:6]
    d, m, y = (torch.tensor(f['tr'][0]).to(DEVICE), torch.tensor(f['tr'][1]).to(DEVICE), torch.tensor(f['tr'][2], dtype=torch.float32).to(DEVICE))
    model = make_model(name)
    opt = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=80, T_mult=2, eta_min=1e-5)
    best, pat, t0 = float('inf'), 0, time.time()
    model.train()
    for ep in range(EPOCHS):
        if masked:                                # hide a random subset of TRAIN associations from graph, A-blocks and similarities
            keep = mrng.rand(len(pd_)) >= MASK_RATIO
            A_msg = np.zeros_like(f['A_tr']); A_msg[pd_[keep], pm_[keep]] = 1
            Xm, eim, FDm, FMm, ei_dm, ei_mm = to_dev(integrated_similarity(A_msg), integrated_similarity(A_msg.T), A_msg)
        opt.zero_grad()
        sc, zdA, zmA, zdB, zmB = model(Xm, eim, FDm, FMm, ei_dm, ei_mm) if masked else model(X, ei, FD, FM, ei_d, ei_m)
        loss = bce_loss(sc[d, m], y)
        if model.uses_cl: loss = loss + LAM_CL * (nt_xent(zdA, zdB) + nt_xent(zmA, zmB)) / 2
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(ep + 1)
        if loss.item() < best - 1e-4: best, pat = loss.item(), 0
        else: pat += 1
        if pat >= PATIENCE: break
    model.eval()
    with torch.no_grad(): sc = model(X, ei, FD, FM, ei_d, ei_m)[0].cpu().numpy()
    auc, aupr = evaluate_scores(sc, f)
    out = [dict(model=name, protocol=protocol, seed=seed, fold=fold, auc=auc, aupr=aupr, epochs=ep + 1, sec=time.time() - t0,
                params=sum(p.numel() for p in model.parameters()))]
    if masked:                                    # rank blend of the model with training-only item-CF, over candidate (unobserved) pairs
        cf = cf_matrix(f, inp[6], inp[7]); cand = f['A_tr'] == 0
        def rk(S):                                # tie-averaged percentile rank over candidate pairs (sigmoid scores saturate and tie)
            r = np.zeros_like(S, dtype=float); r[cand] = rankdata(S[cand]) / cand.sum(); return r
        a2, p2 = evaluate_scores(BLEND_W * rk(sc) + (1 - BLEND_W) * rk(cf), f)
        out.append(dict(model=name, protocol='honest+masked+CF', seed=seed, fold=fold, auc=a2, aupr=p2, epochs=ep + 1, sec=out[0]['sec'], params=out[0]['params']))
    return out

def cf_matrix(f, SD, SM):                       # item-based CF (A_train @ S_M): the blend partner used in all main experiments
    S2 = SM.copy(); np.fill_diagonal(S2, 0)
    return f['A_tr'] @ S2

def cf_user_item(f, SD, SM):                    # stronger no-learning reference: normalised user+item CF (reported separately, not used in the blend)
    S1 = SD.copy(); np.fill_diagonal(S1, 0); S2 = SM.copy(); np.fill_diagonal(S2, 0)
    return S1 @ f['A_tr'] / (S1.sum(1, keepdims=True) + 1e-8) + f['A_tr'] @ S2 / (S2.sum(0, keepdims=True) + 1e-8)

def cf_reference(f, SD, SM): return evaluate_scores(cf_matrix(f, SD, SM), f)
def cf_ui_reference(f, SD, SM): return evaluate_scores(cf_user_item(f, SD, SM), f)
''')

code(r'''
# ---- run everything (resumable: results are appended to a JSON file) --------------------------------
results = json.load(open(RESULT_FILE)) if os.path.exists(RESULT_FILE) else []
done = {(r['model'], r['protocol'], r['seed'], r['fold']) for r in results}
T0 = time.time()
for seed in SEEDS:
    kf = KFold(K_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, te_idx) in enumerate(kf.split(np.arange(len(pos_d_all)))):
        if QUICK_TEST and fold >= 2: break
        f = make_fold(seed, fold, tr_idx, te_idx)
        for protocol in PROTOCOLS:
            inp = fold_inputs(f, 'honest' if protocol == 'honest+masked' else protocol)
            if protocol != 'honest+masked' and ('CF-reference', protocol, seed, fold) not in done:
                a, p = cf_reference(f, inp[6], inp[7])
                results.append(dict(model='CF-reference', protocol=protocol, seed=seed, fold=fold, auc=a, aupr=p, epochs=0, sec=0))
                a, p = cf_ui_reference(f, inp[6], inp[7])
                results.append(dict(model='CF-user+item', protocol=protocol, seed=seed, fold=fold, auc=a, aupr=p, epochs=0, sec=0))
            for name in MODELS:
                if (name, protocol, seed, fold) in done: continue
                for r in run_one(name, protocol, seed, fold, f, inp):
                    results.append(r)
                    print(f"seed {seed} fold {fold} {r['protocol']:17s} {name:15s} AUC={r['auc']:.4f} AUPR={r['aupr']:.4f} ({r['epochs']} ep, {r['sec']:.0f}s)", flush=True)
                json.dump(results, open(RESULT_FILE, 'w'))
print(f'Finished in {(time.time()-T0)/60:.1f} min')
''')

code(r'''
# ---- GCATCMDA replica (from the published description; not the authors' code) ------------------------------
R_F, R_HEADS, R_L, R_TAU, R_LAM, R_THR, R_WD, R_EPOCHS = 128, 2, 3, 1.0, 0.2, 0.4, 1e-3, 100
REPLICAS = [('GCATCMDA-replica (paper lr 1e-5)', 1e-5), ('GCATCMDA-replica (lr 1e-3)', 1e-3)]
def gaussian_kernel(P):
    lam = 1.0 / (np.linalg.norm(P, axis=1).mean() + 1e-8); sq = (P ** 2).sum(1)
    return np.exp(-lam * np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0)).astype(np.float32)
def thr_edges(K, t=R_THR):
    r, c = np.where((K >= t) & (np.eye(len(K)) == 0))
    if len(r) == 0: r = c = np.arange(len(K))
    return torch.tensor(np.stack([r, c]), dtype=torch.long)
class GCAT(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, R_F, bias=False)
        self.gcn = nn.ModuleList([GCNConv(R_F, R_F) for _ in range(R_L)])
        self.gat = nn.ModuleList([GATConv(R_F, R_F, heads=R_HEADS, concat=False) for _ in range(R_L)])
        self.Wa = nn.Linear(R_L * R_F, R_F, bias=False); self.Wb = nn.Linear(R_F, R_F, bias=False)
    def forward(self, x, ei):
        h, Z = self.proj(x), []
        for g, a in zip(self.gcn, self.gat): h = F.elu(a(F.relu(g(h, ei)), ei)); Z.append(h)
        prod = Z[0]
        for z in Z[1:]: prod = prod * z
        return self.Wa(torch.cat(Z, -1)) + self.Wb(prod)
class GCATCMDA(nn.Module):
    def __init__(self):
        super().__init__(); self.enc_A, self.enc_Sd, self.enc_Sm = GCAT(nd + nm), GCAT(nd), GCAT(nm)
    def forward(self, H0, G_ei, KD, ed, KM, em):
        zA = self.enc_A(H0, G_ei); zdA, zmA = zA[:nd], zA[nd:]; zdS, zmS = self.enc_Sd(KD, ed), self.enc_Sm(KM, em)
        return torch.sigmoid(torch.cat([zdA, zdS], -1) @ torch.cat([zmA, zmS], -1).T), zdA, zdS, zmA, zmS
def grace(u, v, tau=R_TAU):
    u, v = F.normalize(u, dim=-1), F.normalize(v, dim=-1); n = u.shape[0]
    def one(a, b):
        inter, intra = torch.exp(a @ b.T / tau), torch.exp(a @ a.T / tau); eye = torch.eye(n, device=a.device, dtype=torch.bool)
        return -torch.log(inter.diag() / (inter.diag() + inter.masked_fill(eye, 0).sum(1) + intra.masked_fill(eye, 0).sum(1)))
    return 0.5 * (one(u, v) + one(v, u)).mean()
def run_replica(label, lr, protocol, seed, fold, f):
    P_d, P_m = (A, A.T) if protocol == 'leaky' else (f['A_tr'], f['A_tr'].T)          # kernels: full matrix vs training fold only
    KD, KM = gaussian_kernel(P_d), gaussian_kernel(P_m); n = nd + nm
    H0 = np.zeros((n, n), np.float32); H0[:nd, :nd] = KD; H0[nd:, nd:] = KM
    dr, mc = np.where(f['A_tr'] == 1); G_ei = torch.tensor(np.stack([np.r_[dr, nd + mc], np.r_[nd + mc, dr]]), dtype=torch.long).to(DEVICE)
    T = lambda a: torch.tensor(a).to(DEVICE); H0t, KDt, KMt, ed, em = T(H0), T(KD), T(KM), thr_edges(KD).to(DEVICE), thr_edges(KM).to(DEVICE)
    d, m, y = (torch.tensor(f['tr'][0]).to(DEVICE), torch.tensor(f['tr'][1]).to(DEVICE), torch.tensor(f['tr'][2], dtype=torch.float32).to(DEVICE))
    torch.manual_seed(seed * 100 + fold); model = GCATCMDA().to(DEVICE); opt = Adam(model.parameters(), lr=lr, weight_decay=R_WD); t0 = time.time(); model.train()
    for ep in range(R_EPOCHS):
        opt.zero_grad(); S, zdA, zdS, zmA, zmS = model(H0t, G_ei, KDt, ed, KMt, em)
        (F.binary_cross_entropy(S[d, m], y) + R_LAM * (grace(zdA, zdS) + grace(zmA, zmS))).backward(); opt.step()
    model.eval()
    with torch.no_grad(): sc = model(H0t, G_ei, KDt, ed, KMt, em)[0].cpu().numpy()
    auc, aupr = evaluate_scores(sc, f)
    return dict(model=label, protocol=protocol, seed=seed, fold=fold, auc=auc, aupr=aupr, epochs=R_EPOCHS, sec=time.time() - t0, params=sum(p.numel() for p in model.parameters()))

done = {(r['model'], r['protocol'], r['seed'], r['fold']) for r in results}
for seed in SEEDS:
    kf = KFold(K_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, te_idx) in enumerate(kf.split(np.arange(len(pos_d_all)))):
        if QUICK_TEST and fold >= 2: break
        f = make_fold(seed, fold, tr_idx, te_idx)
        for label, lr in REPLICAS:
            for protocol in ['leaky', 'honest']:
                if (label, protocol, seed, fold) in done: continue
                r = run_replica(label, lr, protocol, seed, fold, f); results.append(r)
                print(f"seed {seed} fold {fold} {protocol:6s} {label:34s} AUC={r['auc']:.4f} AUPR={r['aupr']:.4f}", flush=True)
        json.dump(results, open(RESULT_FILE, 'w'))
''')

code(r'''
# ---- summary tables ------------------------------------------------------------------------------
R = pd.DataFrame(results)
order = ['SGAT-CLMDA', 'GCATCMDA-style', 'GCATCMDA-replica (paper lr 1e-5)', 'GCATCMDA-replica (lr 1e-3)', 'GCN', 'GAT', 'CF-reference', 'CF-user+item']
rows = []
ALL_P = PROTOCOLS + ['honest+masked+CF']
for m in order:
    row = {'Model': m}
    for p in ALL_P:
        s = R[(R.model == m) & (R.protocol == p)]
        if len(s) == 0: continue
        row[f'{p} AUC'] = f'{s.auc.mean():.4f} ± {s.auc.std():.4f}'
        row[f'{p} AUPR'] = f'{s.aupr.mean():.4f} ± {s.aupr.std():.4f}'
    a = R[(R.model == m) & (R.protocol == 'leaky')].sort_values(['seed', 'fold']).auc.values
    b = R[(R.model == m) & (R.protocol == 'honest')].sort_values(['seed', 'fold']).auc.values
    ok = len(a) == len(b) > 5
    row['AUC gap (leaky-honest)'] = f'{(a - b).mean():+.4f}' if ok else 'n/a'
    row['Wilcoxon p'] = f'{wilcoxon(a, b).pvalue:.2e}' if ok and np.any(a != b) else 'n/a'
    rows.append(row)
summary = pd.DataFrame(rows); display(summary)
n_runs = R.groupby(['model', 'protocol']).size().iloc[0]
print(f'(mean ± std over {n_runs} runs = {len(SEEDS)} seeds x folds; paired Wilcoxon over identical (seed, fold) units)')

print('\nPer-fold AUC, seed', SEEDS[0], '(rows = model/protocol, columns = fold):')
display(R[R.seed == SEEDS[0]].pivot_table(index=['model', 'protocol'], columns='fold', values='auc').round(4))

for proto in ['honest+masked', 'honest+masked+CF']:
    print(f'\nPaired Wilcoxon on AUC, SGAT-CLMDA minus other model, protocol = {proto}:')
    base = R[(R.model == 'SGAT-CLMDA') & (R.protocol == proto)].sort_values(['seed', 'fold']).auc.values
    for m in [x for x in order[1:] if not x.startswith('CF')]:
        o = R[(R.model == m) & (R.protocol == proto)].sort_values(['seed', 'fold']).auc.values
        if len(o) == len(base) > 5 and np.any(base != o): print(f'  vs {m:34s}: {np.mean(base - o):+.4f}  p={wilcoxon(base, o).pvalue:.3g}')
summary.to_csv('leakage_audit_summary.csv', index=False); R.to_csv('leakage_audit_all_runs.csv', index=False)
''')

code(r'''
# ---- (i) the five pre-specified seeds only, (ii) per-seed means so variability is visible (no seed is dropped) ----
R5 = R[R.seed.isin(PRESPECIFIED_5)]
t5 = R5.groupby(['model', 'protocol']).auc.agg(['mean', 'std', 'count']).round(4); print('AUC over the pre-specified seeds', PRESPECIFIED_5); display(t5)
per_seed = R.groupby(['model', 'protocol', 'seed']).auc.mean().unstack('seed').round(3); print('\nPer-seed mean AUC (every seed shown):'); display(per_seed)
R5.groupby(['model', 'protocol'])[['auc', 'aupr']].mean().round(4).to_csv('leakage_audit_5seed_summary.csv')
''')

code(r'''
# ---- figure -----------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
LAB = {'leaky': ('Leaky (full-A similarities)', '#d95f02'), 'honest': ('Leakage-free, plain training', '#7570b3'),
       'honest+masked': ('Leakage-free + masked-edge training', '#1b9e77'), 'honest+masked+CF': ('Leakage-free + masked + CF blend (final)', '#e7298a')}
x, w = np.arange(len(order)), 0.2
for k, metric in enumerate(['auc', 'aupr']):
    for j, p in enumerate(ALL_P):
        g = [R[(R.model == m) & (R.protocol == p)][metric] for m in order]
        ax[k].bar(x + (j - 1.5) * w, [s.mean() if len(s) else 0 for s in g], w, yerr=[s.std() if len(s) else 0 for s in g], capsize=2, color=LAB[p][1], label=LAB[p][0])
    ax[k].set_xticks(x); ax[k].set_xticklabels(order, rotation=20); ax[k].set_ylabel(metric.upper()); ax[k].set_ylim(0.4, 1.0)
    ax[k].axhline(0.5, ls=':', c='grey'); ax[k].set_title(f'{metric.upper()} ({K_FOLDS}-fold CV, {len(SEEDS)} seeds)')
ax[0].legend(fontsize=7, loc='lower right'); plt.tight_layout(); plt.savefig('leakage_audit.png', dpi=200); plt.show()
try:
    from google.colab import files
    import shutil; shutil.make_archive('leakage_audit_outputs', 'zip', '.', 'leakage_audit.png')
    for fn in ['leakage_audit_results.json', 'leakage_audit_summary.csv', 'leakage_audit_all_runs.csv', 'leakage_audit.png']: files.download(fn)
except Exception: pass
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SGAT_CLMDA_Leakage_Audit_Colab.ipynb")
json.dump(nb, open(out, "w", encoding="utf8"), indent=1)
print("wrote", out)
