"""10-seed (5-fold each) aggregation: seeds 42,123,456,789,2024 (matched_*.json) + 11,22,33,44,55 (matched_*_s2.json). All seeds reported."""
import json, numpy as np
from scipy.stats import wilcoxon
O = 'sgat_clmda_outputs/'; S5 = [42, 123, 456, 789, 2024]; S10 = S5 + [11, 22, 33, 44, 55]
ARCH = [('parallel', 'SGAT-CLMDA (parallel)'), ('serial', 'GCATCMDA-style serial'), ('gcn_only', 'GCN'), ('gat_only', 'GAT')]
D = {}
for a, _ in ARCH:
    d = json.load(open(f'{O}matched_{a}.json')); d.update(json.load(open(f'{O}matched_{a}_s2.json'))); D[a] = d
def get(a, f, seeds, i=0): return np.array([D[a][f'{a}|{s}|{k}'][f][i] for s in seeds for k in range(5)])
ms = lambda x: f'{x.mean():.3f} ± {x.std():.3f}'
out = []
for title, seeds in (('ALL 10 seeds (50 runs)', S10), ('Pre-specified first 5 seeds (25 runs)', S5)):
    out += [f'### {title}', '', '| Model | A: notebook protocol (leaky) AUC / AUPR | B1: per-fold, plain AUC | B2: per-fold + masked AUC | B3: per-fold + masked + CF blend AUC / AUPR |', '|---|---|---|---|---|']
    for a, n in ARCH:
        out.append(f"| {n} | {ms(get(a,'leaky',seeds))} / {get(a,'leaky',seeds,1).mean():.3f} | {ms(get(a,'honest',seeds))} | {ms(get(a,'masked',seeds))} | {ms(get(a,'hybrid',seeds))} / {get(a,'hybrid',seeds,1).mean():.3f} |")
    out.append('')
    ref = get('parallel', 'hybrid', seeds)
    for a, n in ARCH[1:]:
        o = get(a, 'hybrid', seeds); out.append(f'- SGAT-CLMDA minus {n} (recipe B3): {np.mean(ref-o):+.3f}, paired Wilcoxon p={wilcoxon(ref,o).pvalue:.1e}, wins {int((ref>o).sum())}/{len(ref)}')
    out.append('')
out += ['### Per-seed mean AUC (every seed shown)', '', '| Model / protocol | ' + ' | '.join(map(str, S10)) + ' |', '|---|' + '---|' * len(S10)]
for a, n in ARCH:
    for f, lab in (('leaky', 'leaky'), ('honest', 'per-fold plain'), ('hybrid', 'per-fold recipe')):
        out.append(f'| {n} / {lab} | ' + ' | '.join(f'{get(a,f,[s]).mean():.3f}' for s in S10) + ' |')
rp = {k: json.load(open(f'{O}gcatcmda_replica_{k}.json')) for k in ('original', 'perfold', 'original_lr0.001', 'perfold_lr0.001')}
out += ['', '### GCATCMDA replica, 10 seeds x 5 folds (50 runs, from the paper description)', '', '| Setting | full-matrix kernels (A) | per-fold kernels (B) |', '|---|---|---|']
for lab, ka, kb in (('paper settings (lr 1e-5)', 'original', 'perfold'), ('lr 1e-3', 'original_lr0.001', 'perfold_lr0.001')):
    f = lambda k: np.array([v['auc'] for v in rp[k].values()])
    out.append(f'| {lab} | {ms(f(ka))} | {ms(f(kb))} |')
open('FINAL_10SEED_RESULTS.md', 'w', encoding='utf-8').write('\n'.join(out) + '\n'); print('\n'.join(out))
