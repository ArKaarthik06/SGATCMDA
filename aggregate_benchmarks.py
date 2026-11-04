import json, numpy as np, pandas as pd
from scipy import stats
r = json.load(open('sgat_clmda_outputs/real_benchmarks.json'))
df = pd.DataFrame(r.values())
rows = []
for c, g in df.groupby('config', sort=False):
    seedmeans = g.groupby('seed')[['auc', 'aupr']].mean()
    rows.append(dict(config=c, n=len(g), AUC=g.auc.mean(), AUC_sd_folds=g.auc.std(), AUC_sd_seeds=seedmeans.auc.std(),
                     AUPR=g.aupr.mean(), AUPR_sd_folds=g.aupr.std(), noisy_AUC=g.auc_noisy.mean(),
                     params=int(g.params.iloc[0]), ms_per_epoch=g.ms_per_epoch.mean(), epochs=g.epochs.mean(),
                     peak_rss_mb=g.rss_mb.max(), sparsity=g.sparsity.mean() if g.sparsity.notna().any() else np.nan))
t = pd.DataFrame(rows); t.to_csv('sgat_clmda_outputs/benchmark_table.csv', index=False)
print(t.round(4).to_string(index=False))
base = df[df.config == 'SGAT-CLMDA (parallel, ANR)'].set_index(['seed', 'fold']).auc
print('\nPaired Wilcoxon (per seed x fold) vs SGAT-CLMDA:')
out = {}
for c in df.config.unique():
    if c == 'SGAT-CLMDA (parallel, ANR)': continue
    o = df[df.config == c].set_index(['seed', 'fold']).auc
    idx = base.index.intersection(o.index)
    if len(idx) < 5: continue
    d = base[idx] - o[idx]
    p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
    out[c] = dict(mean_diff=float(d.mean()), p=float(p), n=len(idx)); print(f'{c:38s} diff={d.mean():+.4f} p={p:.4f} n={len(idx)}')
json.dump(out, open('sgat_clmda_outputs/paired_tests.json', 'w'), indent=1)
