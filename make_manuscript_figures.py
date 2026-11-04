"""Four replacement figures for the revised manuscript (all numbers read from the result files)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
O = 'sgat_clmda_outputs/'; S10 = [42, 123, 456, 789, 2024, 11, 22, 33, 44, 55]; S5 = S10[:5]
M = {}
for a in ('parallel', 'serial', 'gcn_only', 'gat_only'):
    d = json.load(open(f'{O}matched_{a}.json')); d.update(json.load(open(f'{O}matched_{a}_s2.json'))); M[a] = d
g = lambda a, f, seeds=S10, i=0: np.array([M[a][f'{a}|{s}|{k}'][f][i] for s in seeds for k in range(5)])
RK = {}
for s in S5: RK.update(json.load(open(f'{O}rkgatmda_ours_released_{s}.json')))
rk_auc = np.array([v['auc'] for v in RK.values()])
C = {'orig': '#c8553d', 'fold': '#2a6f97', 'grey': '#6c757d'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

# Fig 1: per-seed AUC of SGAT-CLMDA, original vs per-fold
fig, ax = plt.subplots(figsize=(6.2, 3.2)); x = np.arange(10)
o = [g('parallel', 'leaky', [s]).mean() for s in S10]; h = [g('parallel', 'hybrid', [s]).mean() for s in S10]
ax.plot(x, o, 'o-', c=C['orig'], label=f'Original protocol (full-matrix similarities), mean {np.mean(o):.3f}'); ax.plot(x, h, 's-', c=C['fold'], label=f'Per-fold recomputation, mean {np.mean(h):.3f}')
ax.set_xticks(x); ax.set_xticklabels(S10); ax.set_xlabel('Random seed'); ax.set_ylabel('AUC (mean of five folds)'); ax.set_ylim(0.86, 0.93); ax.legend(frameon=False, fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig('figures/ms_fig1_per_seed.png', dpi=200); plt.close()

# Fig 2: four GNN models + RKGATMDA, AUC and AUPR, both protocols
names = [('parallel', 'SGAT-CLMDA'), ('gcn_only', 'GCN'), ('gat_only', 'GAT'), ('serial', 'GCATCMDA-style')]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, (idx, lab) in zip(axs, ((0, 'AUC'), (1, 'AUPR'))):
    xs = np.arange(len(names) + 1); w = 0.38
    om = [g(a, 'leaky', S10, idx).mean() for a, _ in names] + [np.nan]; osd = [g(a, 'leaky', S10, idx).std() for a, _ in names] + [0]
    fm = [g(a, 'hybrid', S10, idx).mean() for a, _ in names] + [np.mean([v['auc' if idx == 0 else 'aupr'] for v in RK.values()])]
    fsd = [g(a, 'hybrid', S10, idx).std() for a, _ in names] + [np.std([v['auc' if idx == 0 else 'aupr'] for v in RK.values()])]
    ax.bar(xs - w / 2, om, w, yerr=osd, capsize=2, color=C['orig'], label='Original protocol'); ax.bar(xs + w / 2, fm, w, yerr=fsd, capsize=2, color=C['fold'], label='Per-fold recomputation')
    ax.set_xticks(xs); ax.set_xticklabels([n for _, n in names] + ['RKGATMDA'], rotation=25, ha='right'); ax.set_ylim(0.75, 0.95); ax.set_ylabel(lab)
axs[0].plot([len(names) - 0.45, len(names) + 0.45], [0.8906, 0.8906], '--', c='black', lw=1.2); axs[0].text(len(names) + 0.45, 0.8946, 'published 0.8906', fontsize=6.5, ha='right', va='bottom')
axs[0].legend(frameon=False, fontsize=7, loc='upper right'); plt.tight_layout(); plt.savefig('figures/ms_fig2_models.png', dpi=200); plt.close()

# Fig 3: no-learning scoring rules
LM = json.load(open(O + 'leakage_multiseed.json')); keys = [('heur_a_raw_A', '(a) A[i,j]'), ('heur_b_full_GIP_projection', '(b) projection,\nfull-matrix GIP'), ('heur_c_fold_GIP_projection', '(c) projection,\nper-fold GIP')]
fig, ax = plt.subplots(figsize=(4.6, 3.2)); v = [LM[k]['auc'] for k, _ in keys]; e = [LM[k]['auc_sd'] for k, _ in keys]
ax.bar(range(3), v, yerr=e, capsize=3, color=[C['orig'], C['orig'], C['fold']]); ax.axhline(0.5, ls=':', c=C['grey']); ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in keys]); ax.set_ylabel('AUC (no learning)'); ax.set_ylim(0, 1.08)
for i, y in enumerate(v): ax.text(i, y + 0.03, f'{y:.3f}', ha='center', fontsize=8)
plt.tight_layout(); plt.savefig('figures/ms_fig3_probes.png', dpi=200); plt.close()

# Fig 4: ablation (per-fold similarities + masked-edge training, 25 runs each)
MA = json.load(open(O + 'masked_ablation.json')); EA = json.load(open(O + 'extra_ablations.json'))
def grp(d):
    out = {}
    for v in d.values(): out.setdefault(v['config'], []).append(v['auc'])
    return out
mg, eg = grp(MA), grp(EA); keys5 = [f'{s}|{k}' for s in S5 for k in range(5)]
full = np.array([M['parallel'][f'parallel|{s}|{k}']['masked'][0] for s in S5 for k in range(5)])
rows = [('Full model (contrastive learning on both sides, learnable tau)', full)] + [(n, np.array(mg[c])) for n, c in [('No contrastive learning', 'CL none'), ('Contrastive learning, disease side only', 'CL disease only'), ('Contrastive learning, microbe side only', 'CL microbe only'), ('Both sides, temperature 0.1', 'CL both, tau_cl=0.1'), ('Both sides, temperature 0.2', 'CL both, tau_cl=0.2'), ('Both sides, temperature 1.0', 'CL both, tau_cl=1.0')]]
rows += [(n, np.array(eg[c])) for n, c in [('tau frozen at 0.1', 'tau frozen at 0.1'), ('tau frozen at 0.3', 'tau frozen at 0.3'), ('Similarity: GIP only', 'similarity: GIP only'), ('Similarity: cosine only', 'similarity: cosine only')]]
rows += [(n, np.array([M[a][f'{a}|{s}|{k}']['masked'][0] for s in S5 for k in range(5)])) for a, n in [('serial', 'Serial GCN-to-GAT encoder'), ('gcn_only', 'GCN-only encoder'), ('gat_only', 'GAT-only encoder')]]
json.dump({n: [float(x.mean()), float(x.std())] for n, x in rows}, open(O + 'ablation_table_values.json', 'w'))
fig, ax = plt.subplots(figsize=(6.6, 4.4)); y = np.arange(len(rows))[::-1]
ax.barh(y, [x.mean() for _, x in rows], xerr=[x.std() for _, x in rows], color=[C['fold']] + [C['grey']] * (len(rows) - 1), capsize=2); ax.axvline(full.mean(), ls=':', c=C['orig'])
ax.set_yticks(y); ax.set_yticklabels([n for n, _ in rows], fontsize=7.5); ax.set_xlim(0.7, 0.92); ax.set_xlabel('AUC (mean ± SD, five folds × five seeds)')
plt.tight_layout(); plt.savefig('figures/ms_fig4_ablation.png', dpi=200); plt.close()
print('figures done'); [print(f'{n:62s} {x.mean():.3f} ± {x.std():.3f}') for n, x in rows]
