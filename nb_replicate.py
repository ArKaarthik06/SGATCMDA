"""Executes the ORIGINAL notebook code (cells 2,3,4,5,6,7) verbatim, then runs the Cell-8 CV loop with a switch:
   MODE='original' : SD,SM from full A (exactly Cell 8)
   MODE='perfold'  : SD,SM,graphs recomputed from A_tr only (everything else identical)
   MODE='perfold_masked': perfold + masked-edge training (r=0.5)"""
import json, sys, time
nb = json.load(open('SGAT_CLMDA_Final.ipynb', encoding='utf-8'))
g = {}
for i in [2, 3, 4, 5, 6, 7]:
    exec(''.join(nb['cells'][i]['source']), g)
np, torch, nn, F = g['np'], g['torch'], g['nn'], g['F']
MODE = sys.argv[1]; SEEDS = [int(x) for x in sys.argv[2].split(',')]; MASK = 0.5
nd, nm, A = g['nd'], g['nm'], g['A']
res = []
for SEED in SEEDS:
    g['random'].seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    kf = g['KFold'](n_splits=5, shuffle=True, random_state=SEED)
    pos_idx = np.arange(len(g['pos_d_all']))
    for fold, (tr_idx, te_idx) in enumerate(kf.split(pos_idx)):
        tr_pd, tr_pm = g['pos_d_all'][tr_idx], g['pos_m_all'][tr_idx]; te_pd, te_pm = g['pos_d_all'][te_idx], g['pos_m_all'][te_idx]
        A_tr = np.zeros((nd, nm), dtype=np.float32); A_tr[tr_pd, tr_pm] = 1.0
        def make(A_graph, SDx, SMx):
            X, ei_h = g['build_hetero_network'](A_graph, SDx, SMx, 0.3); ei_d, ei_m = g['build_sim_networks'](SDx, SMx, 0.3)
            return X, ei_h, torch.tensor(SDx), torch.tensor(SMx), ei_d, ei_m
        if MODE == 'original': SDx, SMx = g['SD'], g['SM']
        else: SDx, SMx = g['integrated_similarity'](A_tr), g['integrated_similarity'](A_tr.T)
        inp = make(A_tr, SDx, SMx)
        np.random.seed(SEED + fold); neg_sel = np.random.choice(len(g['neg_d_all']), size=len(tr_pd), replace=False)
        all_d = torch.tensor(np.concatenate([tr_pd, g['neg_d_all'][neg_sel]]), dtype=torch.long); all_m = torch.tensor(np.concatenate([tr_pm, g['neg_m_all'][neg_sel]]), dtype=torch.long)
        all_y = torch.tensor(np.r_[np.ones(len(tr_pd)), np.zeros(len(tr_pd))], dtype=torch.float32)
        torch.manual_seed(SEED + fold)
        model = g['SGAT_CLMDA'](nd, nm, nd + nm, 256, 64, 2, 4, 0.5)
        opt = g['Adam'](model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=80, T_mult=2, eta_min=1e-5)
        best, pat = float('inf'), 0; rng = np.random.RandomState(SEED * 7 + fold)
        for ep in range(400):
            if MODE == 'perfold_masked':
                keep = rng.rand(len(tr_pd)) >= MASK; Am = np.zeros_like(A_tr); Am[tr_pd[keep], tr_pm[keep]] = 1.0
                cur = make(Am, g['integrated_similarity'](Am), g['integrated_similarity'](Am.T))
            else: cur = inp
            model.train(); opt.zero_grad()
            sc, zd, zm, zdA, zmA, zdB, zmB = model(*cur)
            loss = g['total_loss'](sc[all_d, all_m], all_y, zdA, zdB, zmA, zmB, 0.1, 0.5, use_contrastive=True)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(ep + 1)
            if loss.item() < best - 1e-4: best, pat = loss.item(), 0
            else: pat += 1
            if pat >= 50: break
        model.eval()
        with torch.no_grad(): sc_np = model(*inp)[0].numpy()
        A_te = np.zeros((nd, nm), dtype=np.float32); A_te[te_pd, te_pm] = 1.0
        ntd, ntm = np.where((A == 0) & (A_te == 0)); np.random.seed(SEED + fold + 100)
        ni = np.random.choice(len(ntd), size=len(te_pd), replace=False)
        yt = np.r_[np.ones(len(te_pd)), np.zeros(len(ni))]; ys = np.r_[sc_np[te_pd, te_pm], sc_np[ntd[ni], ntm[ni]]]
        auc, aupr = g['roc_auc_score'](yt, ys), g['average_precision_score'](yt, ys)
        res.append(dict(mode=MODE, seed=SEED, fold=fold, auc=auc, aupr=aupr, epochs=ep + 1))
        print(f'{MODE} seed={SEED} fold={fold+1} AUC={auc:.4f} AUPR={aupr:.4f} epochs={ep+1}', flush=True)
        json.dump(res, open(f'sgat_clmda_outputs/nb_{MODE}_{"-".join(map(str,SEEDS))}.json', 'w'), indent=1)
import statistics as st
print(f'{MODE}: AUC {np.mean([r["auc"] for r in res]):.4f} ± {np.std([r["auc"] for r in res]):.4f}  AUPR {np.mean([r["aupr"] for r in res]):.4f} ± {np.std([r["aupr"] for r in res]):.4f}')
